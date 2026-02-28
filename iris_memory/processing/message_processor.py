"""
消息处理模块

封装 LLM Hook、消息装饰和普通消息处理逻辑。
"""

from typing import Optional, Any, List, TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent

from iris_memory.utils.event_utils import get_group_id, get_sender_name, extract_reply_info
from iris_memory.utils.persona_utils import get_event_persona_id
from iris_memory.utils.command_utils import SessionKeyBuilder, MessageFilter
from iris_memory.core.constants import (
    InputValidationConfig,
    ConfigKeys,
    PROACTIVE_EXTRA_KEY,
    PROACTIVE_CONTEXT_KEY,
)

if TYPE_CHECKING:
    from iris_memory.services.memory_service import MemoryService


class MessageProcessor:
    """
    消息处理器

    封装所有消息相关的处理逻辑：
    - LLM 请求前的上下文注入
    - LLM 响应后的记忆捕获
    - 普通消息的分层处理
    """

    def __init__(self, service: "MemoryService") -> None:
        """
        初始化消息处理器

        Args:
            service: MemoryService 实例
        """
        self._service = service

    async def prepare_llm_context(
        self,
        event: AstrMessageEvent,
        req: Any
    ) -> None:
        """
        在 LLM 请求前注入上下文

        注入的上下文层次（按优先级排序）：
        1. 近期聊天记录 - 让AI了解当前话题
        2. 相关记忆 - 长期记忆检索结果
        3. 图片分析 - 当前消息中的图片描述
        4. 行为指导 - 防止重复/过度反问
        5. 主动回复指令 - 仅在主动回复时附加

        Args:
            event: 消息事件对象
            req: LLM 请求对象
        """
        if not getattr(self._service, 'is_initialized', False):
            req.system_prompt += "\n\n[系统提示：记忆插件正在初始化，暂时无法提供服务]\n"
            return

        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_inject:
            return

        if not self._service.is_embedding_ready():
            req.system_prompt += "\n\n[系统提示：记忆系统正在初始化，暂时无法提供历史记忆参考]\n"
            self._service.logger.info("Embedding model not ready, skipping memory injection")
            return

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        query = event.message_str
        sender_name = get_sender_name(event)

        is_proactive = event.get_extra(PROACTIVE_EXTRA_KEY, False)

        if self._service.member_identity and not is_proactive:
            await self._service.member_identity.resolve_tag(
                user_id, sender_name, group_id
            )

        await self._service.activate_session(user_id, group_id)

        image_context = ""
        if self._service.image_analyzer and not is_proactive:
            try:
                llm_ctx, _ = await self._service.analyze_images(
                    message_chain=event.message_obj.message,
                    user_id=user_id,
                    group_id=group_id,
                    context_text=query,
                    umo=event.unified_msg_origin,
                    session_id=SessionKeyBuilder.build(user_id, group_id)
                )
                image_context = llm_ctx
            except Exception as e:
                self._service.logger.warning(f"Image analysis in LLM hook failed: {e}")

        reply_context = ""
        if not is_proactive:
            reply_info = extract_reply_info(event)
            if reply_info:
                reply_context = (
                    "【引用消息上下文】\n"
                    "用户在回复/引用另一条消息，请结合被引用内容理解用户意图：\n"
                    f"{reply_info.format_for_prompt()}\n"
                    f"用户的新消息: {query}"
                )
                self._service.logger.debug(
                    f"Reply context extracted: sender={reply_info.sender_nickname}, "
                    f"content_len={len(reply_info.content or '')}"
                )

        raw_persona_id = get_event_persona_id(event)
        query_persona = self._service.cfg.get_persona_id_for_query(raw_persona_id, "memory")
        context = await self._service.prepare_llm_context(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context,
            sender_name=sender_name,
            reply_context=reply_context,
            persona_id=query_persona,
        )

        if context:
            req.system_prompt += f"\n\n{context}\n"

        if is_proactive:
            proactive_ctx = event.get_extra(PROACTIVE_CONTEXT_KEY, {})
            proactive_directive = self._build_proactive_directive(proactive_ctx)
            req.system_prompt += f"\n\n{proactive_directive}\n"
            self._service.logger.info(
                f"Proactive reply context injected for user={user_id}"
            )

    async def handle_llm_response(
        self,
        event: AstrMessageEvent,
        resp: Any
    ) -> None:
        """
        在 LLM 响应后处理

        1. 记录 Bot 的回复到聊天缓冲区
        2. 自动捕获新记忆（主动回复时跳过用户消息捕获）

        Args:
            event: 消息事件对象
            resp: LLM 响应对象
        """
        if not getattr(self._service, 'is_initialized', False):
            return

        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_memory:
            return

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        sender_name = get_sender_name(event)

        is_proactive = event.get_extra(PROACTIVE_EXTRA_KEY, False)

        bot_reply = ""
        if hasattr(resp, 'completion_text'):
            bot_reply = resp.completion_text or ""
        elif hasattr(resp, 'text'):
            bot_reply = resp.text or ""
        elif isinstance(resp, str):
            bot_reply = resp

        if bot_reply:
            await self._service.record_chat_message(
                sender_id="bot",
                sender_name=None,
                content=bot_reply,
                group_id=group_id,
                is_bot=True,
                session_user_id=user_id
            )
            
            if not is_proactive:
                proactive_mgr = getattr(self._service, 'proactive_manager', None)
                if proactive_mgr:
                    proactive_mgr.clear_pending_tasks_for_session(user_id, group_id)

        self._service.update_session_activity(user_id, group_id)

        if is_proactive:
            self._service.logger.info(
                f"Proactive reply completed for user={user_id}, "
                f"reply_len={len(bot_reply)}"
            )
            return

        capture_message = message
        reply_info = extract_reply_info(event)
        if reply_info and reply_info.content:
            capture_message = f"{message}\n{reply_info.format_for_buffer()}"

        raw_persona_id = get_event_persona_id(event)
        store_persona = self._service.cfg.get_persona_id_for_storage(raw_persona_id)
        memory = await self._service.capture_and_store_memory(
            message=capture_message,
            user_id=user_id,
            group_id=group_id,
            sender_name=sender_name,
            persona_id=store_persona,
        )

        if memory:
            self._service.logger.debug(f"Memory captured: {memory.id}")

    async def process_normal_message(
        self,
        event: AstrMessageEvent
    ) -> Optional[str]:
        """
        处理普通消息

        职责：
        1. 记录消息到聊天缓冲区（供LLM上下文注入）
        2. 分层处理：immediate/batch/discard
        3. 主动回复事件检测与 LLM 请求转发

        Args:
            event: 消息事件对象

        Returns:
            Optional[str]: 如果需要 LLM 请求，返回 prompt；否则返回 None
        """
        if not getattr(self._service, 'is_initialized', False):
            return None

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        sender_name = get_sender_name(event)

        if event.get_extra(PROACTIVE_EXTRA_KEY, False):
            self._service.logger.info(
                f"Proactive reply event detected for user={user_id}, "
                f"group={group_id}"
            )
            return message

        if MessageFilter.is_command(message):
            return None

        if len(message) > InputValidationConfig.MAX_MESSAGE_LENGTH:
            message = message[:InputValidationConfig.MAX_MESSAGE_LENGTH]

        if self._service.member_identity:
            await self._service.member_identity.resolve_tag(
                user_id, sender_name, group_id
            )

        reply_info = extract_reply_info(event)
        reply_kw = {}
        if reply_info:
            reply_kw = {
                "reply_sender_name": reply_info.sender_nickname,
                "reply_sender_id": reply_info.sender_id,
                "reply_content": reply_info.content,
            }
        await self._service.record_chat_message(
            sender_id=user_id,
            sender_name=sender_name,
            content=message,
            group_id=group_id,
            is_bot=False,
            **reply_kw
        )

        self._service.update_session_activity(user_id, group_id)

        if not self._service.batch_processor:
            return None

        image_description = ""
        if self._service.image_analyzer:
            try:
                _, mem_format = await self._service.analyze_images(
                    message_chain=event.message_obj.message,
                    user_id=user_id,
                    group_id=group_id,
                    context_text=message,
                    umo=event.unified_msg_origin,
                    session_id=SessionKeyBuilder.build(user_id, group_id)
                )
                image_description = mem_format
            except Exception as e:
                self._service.logger.warning(f"Image analysis failed: {e}")

        context = await self._build_message_context(user_id, group_id)
        context["sender_name"] = sender_name

        enriched_message = message
        if reply_info and reply_info.content:
            enriched_message = f"{message}\n{reply_info.format_for_buffer()}"

        raw_persona_id = get_event_persona_id(event)
        store_persona = self._service.cfg.get_persona_id_for_storage(raw_persona_id)
        await self._service.process_message_batch(
            message=enriched_message,
            user_id=user_id,
            group_id=group_id,
            context=context,
            umo=event.unified_msg_origin,
            image_description=image_description,
            persona_id=store_persona,
        )

        return None

    async def _build_message_context(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> dict[str, Any]:
        """
        构建消息上下文

        Args:
            user_id: 用户ID
            group_id: 群聊ID

        Returns:
            Dict[str, Any]: 上下文字典
        """
        session_key = SessionKeyBuilder.build(user_id, group_id)
        session = None

        if self._service.session_manager:
            session = self._service.session_manager.get_session(session_key)

        return {
            "session_key": session_key,
            "session_message_count": session.get("message_count", 0) if session else 0,
            "user_persona": self._service.get_or_create_user_persona(user_id),
            "emotional_state": self._service._get_or_create_emotional_state(user_id)
        }

    def _build_proactive_directive(self, proactive_ctx: dict) -> str:
        """
        构建主动回复的特殊系统指令

        告诉 LLM 这是一次主动回复场景，提供触发原因和行为指导。

        Args:
            proactive_ctx: 主动回复上下文，包含触发原因、近期消息等

        Returns:
            str: 主动回复系统指令文本
        """
        reason = proactive_ctx.get("reason", "检测到对话信号")
        recent_messages = proactive_ctx.get("recent_messages", [])
        emotion_summary = proactive_ctx.get("emotion_summary", "")
        target_user = proactive_ctx.get("target_user", "用户")

        recent_text = ""
        if recent_messages:
            recent_lines = []
            for msg in recent_messages[-5:]:
                name = msg.get("sender_name", "未知")
                content = msg.get("content", "")
                recent_lines.append(f"  {name}: {content}")
            recent_text = "\n".join(recent_lines)

        directive = (
            "【主动回复场景】\n"
            "你正在主动向用户发起对话，而不是回复用户的消息。\n"
            f"触发原因：{reason}\n"
        )

        if recent_text:
            directive += f"\n近期对话记录：\n{recent_text}\n"

        if emotion_summary:
            directive += f"\n用户情绪状态：{emotion_summary}\n"

        directive += (
            f"\n对话对象：{target_user}\n"
            "\n行为指导：\n"
            "- 你的消息应该自然、简短，像是你忽然想到了什么而发起的对话\n"
            "- 不要提及'系统检测'、'主动回复'等元信息\n"
            "- 结合你对用户的记忆和近期话题来开启对话\n"
            "- 避免重复之前已经讨论过的内容\n"
            "- 语气要符合你的人格设定\n"
            "- 不要重复提及用户刚才已经说过的话题或事件\n"
            "- 如果是群聊环境，注意适度存在感，不要过度介入\n"
            "- 避免机械式回应，要有个性化的互动\n"
        )

        return directive


class ErrorFriendlyProcessor:
    """
    错误消息友好化处理器

    在消息发送前拦截框架错误消息，替换为友好提示。
    """

    def __init__(self, config: Any) -> None:
        """
        初始化错误友好化处理器

        Args:
            config: 插件配置对象
        """
        self._config = config

    def should_process(self, event: AstrMessageEvent) -> bool:
        """
        检查是否需要处理该事件

        Args:
            event: 消息事件对象

        Returns:
            bool: 是否需要处理
        """
        return self._is_error_friendly_enabled()

    def process_result(self, result: Any) -> None:
        """
        处理消息结果，替换错误消息

        Args:
            result: 消息结果对象
        """
        if not result:
            return

        text = self._get_result_plain_text(result)
        if not text:
            return

        if self._is_framework_error(text):
            friendly_msg = self._get_friendly_error_message(text)
            result.chain.clear()
            result.message(friendly_msg)

    def _is_error_friendly_enabled(self) -> bool:
        """检查错误消息友好化功能是否启用"""
        try:
            return self._config.get(ConfigKeys.ERROR_FRIENDLY_ENABLE, True)
        except Exception:
            return True

    def _get_result_plain_text(self, result: Any) -> str:
        """
        获取消息结果的纯文本内容

        Args:
            result: 消息结果对象

        Returns:
            纯文本内容，无法获取时返回空字符串
        """
        if hasattr(result, 'get_plain_text'):
            return result.get_plain_text() or ""
        return ""

    def _is_framework_error(self, text: str) -> bool:
        """
        检测是否为 AstrBot 框架错误消息

        Args:
            text: 消息文本

        Returns:
            是否为框架错误消息
        """
        from iris_memory.core.constants import ErrorFriendlyMessages

        text_lower = text.lower()

        match_count = sum(
            1 for pattern in ErrorFriendlyMessages.ERROR_PATTERNS
            if pattern.lower() in text_lower
        )
        if match_count >= 2:
            return True

        if "400" in text or "bad request" in text_lower:
            if any(keyword in text_lower for keyword in ["请求", "request", "error", "failed"]):
                return True

        error_keywords = [
            "error", "failed", "exception", "traceback",
            "请求失败", "错误", "异常"
        ]
        framework_indicators = [
            "platform", "api", "http", "status code",
            "平台", "框架"
        ]

        has_error_keyword = any(kw in text_lower for kw in error_keywords)
        has_framework_indicator = any(ind in text_lower for ind in framework_indicators)

        return has_error_keyword and has_framework_indicator

    def _get_friendly_error_message(self, text: str) -> str:
        """
        根据错误内容返回合适的友好消息

        Args:
            text: 错误消息文本

        Returns:
            友好的错误提示
        """
        from iris_memory.core.constants import ErrorFriendlyMessages

        text_lower = text.lower()

        if "400" in text or "bad request" in text_lower:
            return ErrorFriendlyMessages.BAD_REQUEST_MSG

        if any(kw in text_lower for kw in ["network", "timeout", "连接", "网络", "timeout"]):
            return ErrorFriendlyMessages.NETWORK_ERROR_MSG

        if any(kw in text_lower for kw in ["rate", "limit", "限流", "频率", "频繁"]):
            return ErrorFriendlyMessages.RATE_LIMIT_MSG

        return ErrorFriendlyMessages.DEFAULT_FRIENDLY_MSG
