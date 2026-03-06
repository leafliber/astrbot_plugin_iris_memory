"""
主动回复发送器

负责将主动回复通过完整的 LLM 流程发送出去，确保经过与被动回复相同的处理：
- 记忆上下文注入
- 用户画像注入
- 行为指导注入
- LLM 调用生成回复
- 通过平台 API 发送消息
- 记录 Bot 回复到聊天缓冲区

设计原则：
- 通过回调/依赖注入获取 MemoryService 能力，避免循环依赖
- ProactiveManager 持有 sender 实例，由服务层创建并注入
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Optional

from iris_memory.proactive.models import ProactiveReplyResult
from iris_memory.utils.llm_helper import call_llm
from iris_memory.core.provider_utils import get_default_provider
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.reply_sender")

# 回调类型定义
PrepareLLMContextCallback = Callable[
    ...,
    Coroutine[Any, Any, str],
]

RecordChatCallback = Callable[
    ...,
    Coroutine[Any, Any, None],
]


class ProactiveReplySender:
    """主动回复发送器

    将主动回复通过完整的 LLM + 平台发送流程发出。
    确保主动回复经历与被动回复相同的上下文注入流程。

    外部依赖通过构造函数注入：
    - astrbot_context: AstrBot Context，用于 llm_generate 和 send_message
    - prepare_llm_context: MemoryService.prepare_llm_context 回调
    - record_chat_message: MemoryService.record_chat_message 回调
    - get_group_umo: 获取群组 UMO 的回调
    """

    def __init__(
        self,
        astrbot_context: Any,
        prepare_llm_context: PrepareLLMContextCallback,
        record_chat_message: RecordChatCallback,
        get_group_umo: Callable[[str], Optional[str]],
        llm_provider: Optional[Any] = None,
        llm_provider_id: Optional[str] = None,
        configured_provider_id: Optional[str] = None,
    ) -> None:
        self._context = astrbot_context
        self._prepare_llm_context = prepare_llm_context
        self._record_chat_message = record_chat_message
        self._get_group_umo = get_group_umo
        self._llm_provider = llm_provider
        self._llm_provider_id = llm_provider_id
        # 延迟解析用的配置 provider_id（AstrBot 先加载插件后加载 provider）
        self._configured_provider_id = configured_provider_id or llm_provider_id

    async def send_reply(
        self,
        result: ProactiveReplyResult,
    ) -> Optional[str]:
        """发送主动回复

        完整流程：
        1. 获取目标群组的 UMO
        2. 准备 LLM 上下文（记忆 + 画像 + 行为指导）
        3. 组合系统指令 + 主动回复指令
        4. 懒加载解析 LLM provider（按 umo 获取当前对话的活跃 provider）
        5. 调用 LLM 生成回复文本
        6. 通过平台 API 发送消息
        7. 记录 Bot 回复到聊天缓冲区

        Args:
            result: 主动回复结果，包含 trigger_prompt、目标信息等

        Returns:
            Bot 回复文本，失败返回 None
        """
        group_id = result.group_id
        user_id = result.target_user

        if not group_id:
            logger.warning("ProactiveReplySender: no group_id in result, skipping")
            return None

        # 1. 获取 UMO
        umo = self._get_group_umo(group_id)
        if not umo:
            logger.warning(
                f"ProactiveReplySender: no UMO for group {group_id}, "
                f"cannot send reply"
            )
            return None

        try:
            # 2. 准备 LLM 上下文（记忆 + 画像 + 行为指导）
            memory_context = await self._prepare_llm_context(
                query="",
                user_id=user_id,
                group_id=group_id,
            )

            # 3. 组合完整 prompt
            full_prompt = self._build_full_prompt(
                memory_context=memory_context,
                trigger_prompt=result.trigger_prompt,
            )

            # 4. 解析 LLM provider（懒加载：优先用缓存值，否则按配置或默认解析）
            provider = self._llm_provider
            provider_id = self._llm_provider_id
            if not provider and not provider_id:
                # 首次使用时懒加载解析 provider（此时 AstrBot 的 provider 已加载完毕）
                if self._configured_provider_id:
                    from iris_memory.core.provider_utils import get_provider_by_id
                    provider, provider_id = get_provider_by_id(
                        self._context, self._configured_provider_id
                    )
                if not provider:
                    provider, provider_id = get_default_provider(self._context, umo)
                if not provider and not provider_id:
                    # 最后兜底：不带 umo 的全局默认 provider
                    provider, provider_id = get_default_provider(self._context)
                if provider_id:
                    # 缓存解析结果，避免每次发送都重新解析
                    self._llm_provider = provider
                    self._llm_provider_id = provider_id
                    logger.debug(
                        f"ProactiveReplySender: lazily resolved provider "
                        f"for group={group_id}: {provider_id}"
                    )
                else:
                    logger.warning(
                        f"ProactiveReplySender: no LLM provider available "
                        f"for group={group_id}, reply will be skipped"
                    )

            # 5. 调用 LLM
            llm_result = await call_llm(
                context=self._context,
                provider=provider,
                provider_id=provider_id,
                prompt=full_prompt,
            )

            if not llm_result.success or not llm_result.content:
                logger.warning(
                    f"ProactiveReplySender: LLM call failed for "
                    f"group={group_id}: {llm_result.error}"
                )
                return None

            reply_text = llm_result.content.strip()
            if not reply_text:
                logger.debug("ProactiveReplySender: LLM returned empty reply")
                return None

            # 6. 发送消息
            await self._send_message(umo, reply_text)

            # 7. 记录 Bot 回复
            await self._record_chat_message(
                sender_id="bot",
                sender_name=None,
                content=reply_text,
                group_id=group_id,
                is_bot=True,
                session_user_id=user_id,
            )

            logger.info(
                f"Proactive reply sent: group={group_id}, "
                f"user={user_id}, source={result.source}, "
                f"reply_len={len(reply_text)}"
            )
            return reply_text

        except Exception as e:
            logger.error(
                f"ProactiveReplySender: failed to send reply for "
                f"group={group_id}: {e}"
            )
            return None

    async def send_text(self, umo: str, text: str) -> None:
        """通过 AstrBot 平台 API 发送纯文本消息（不经过 LLM）

        Args:
            umo: unified_msg_origin，目标会话标识
            text: 消息文本
        """
        await self._send_message(umo, text)

    async def _send_message(self, umo: str, text: str) -> None:
        """通过 AstrBot 平台 API 发送消息

        Args:
            umo: unified_msg_origin，目标会话标识
            text: 消息文本
        """
        try:
            from astrbot.core.message.components import Plain
            from astrbot.core.message.message_event_result import MessageChain

            message_chain = MessageChain(chain=[Plain(text)])
            await self._context.send_message(umo, message_chain)
        except ImportError as e:
            logger.error(
                f"Cannot import AstrBot message types, "
                f"message sending unavailable: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to send message via AstrBot API: {e}")
            raise

    @staticmethod
    def _build_full_prompt(
        memory_context: str,
        trigger_prompt: str,
    ) -> str:
        """构建完整的 LLM prompt

        将记忆上下文和主动回复指令合并为一个完整的 prompt。
        记忆上下文包含用户画像、相关记忆、行为指导等。

        Args:
            memory_context: 记忆系统注入的上下文
            trigger_prompt: 主动回复场景指令

        Returns:
            完整的 prompt 字符串
        """
        parts = []

        if memory_context:
            parts.append(memory_context)

        parts.append(trigger_prompt)

        parts.append(
            "\n请根据以上背景信息和场景指令，自然地生成一条回复。"
            "只输出回复内容本身，不要包含任何元信息或解释。"
        )

        return "\n\n".join(parts)
