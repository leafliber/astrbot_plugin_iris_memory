"""
Iris Memory Plugin - 主入口（重构版）
基于 companion-memory 框架的三层记忆插件

架构：
- Handler 层（本文件）：指令路由、权限检查、消息回发
- Service 层（services/memory_service.py）：业务逻辑封装
- Utils 层（utils/command_utils.py）：工具函数

职责分离原则：
- 本文件只负责 AstrBot 事件处理和响应，不直接操作底层组件
- 所有业务逻辑委托给 MemoryService
"""
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator, Any, List

# 将插件根目录添加到Python路径
plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from iris_memory.services.memory_service import MemoryService
from iris_memory.utils.event_utils import get_group_id, get_sender_name
from iris_memory.utils.logger import init_logging_from_config
from iris_memory.utils.command_utils import (
    CommandParser, DeleteScopeParser, StatsFormatter,
    SessionKeyBuilder, MessageFilter, UnifiedDeleteScopeParser, DeleteMainScope
)
from iris_memory.core.constants import (
    CommandPrefix, ErrorMessages, SuccessMessages,
    DeleteScope, NumericDefaults, LogTemplates,
    ErrorFriendlyMessages, ConfigKeys
)


@register("iris_memory", "YourName", "基于companion-memory框架的三层记忆插件", "1.3.0")
class IrisMemoryPlugin(Star):
    """
    Iris记忆插件 - Handler层
    
    实现三层记忆模型：
    - 工作记忆：会话内临时存储
    - 情景记忆：基于RIF评分动态管理
    - 语义记忆：永久保存用户画像
    
    支持私聊和群聊的完全隔离。
    """
    
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置对象
        """
        super().__init__(context)
        self.context = context
        self.config = config
        
        # 插件名称
        self.name = "iris_memory"
        
        # 插件数据目录
        data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        
        # 初始化业务服务层
        self._service = MemoryService(context, config, data_path)
    
    async def initialize(self) -> None:
        """异步初始化插件"""
        # 初始化日志系统
        init_logging_from_config(self.config, self._service.plugin_data_path)
        
        # 初始化业务服务
        await self._service.initialize()
        
        # 加载持久化数据
        await self._service.load_from_kv(self.get_kv_data)
    
    # ========== 权限检查 ==========
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """
        检查用户是否为管理员
        
        Args:
            event: 消息事件对象
            
        Returns:
            bool: 是否为管理员
        """
        return event.is_admin()
    
    def _check_private_only(self, event: AstrMessageEvent) -> bool:
        """检查是否在私聊场景"""
        return get_group_id(event) is None
    
    def _check_group_only(self, event: AstrMessageEvent) -> Optional[str]:
        """
        检查是否在群聊场景
        
        Returns:
            Optional[str]: 群聊ID，私聊返回None
        """
        return get_group_id(event)
    
    # ========== 指令处理器 ==========
    
    @filter.command("memory_save")
    async def save_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        手动保存记忆指令
        
        用法：/memory_save <内容>
        """
        # 解析指令
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SAVE)
        
        if not parsed.has_content:
            yield event.plain_result(ErrorMessages.EMPTY_CONTENT)
            return
        
        # 获取上下文信息
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        sender_name = get_sender_name(event)
        
        # 执行业务逻辑
        memory = await self._service.capture_and_store_memory(
            message=parsed.content,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=True,
            sender_name=sender_name
        )
        
        # 响应结果
        if memory:
            result = SuccessMessages.MEMORY_SAVED.format(
                memory_type=memory.type.value,
                confidence=memory.confidence
            )
            # 保存最后保存时间
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await self.put_kv_data(kv_key, memory.created_time.isoformat())
        else:
            result = ErrorMessages.CAPTURE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_search")
    async def search_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        搜索记忆指令
        
        用法：/memory_search <查询内容>
        """
        # 解析指令
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SEARCH)
        
        if not parsed.has_content:
            yield event.plain_result(ErrorMessages.EMPTY_QUERY)
            return
        
        # 获取上下文信息
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        memories = await self._service.search_memories(
            query=parsed.content,
            user_id=user_id,
            group_id=group_id,
            top_k=NumericDefaults.TOP_K_SEARCH
        )
        
        # 格式化并响应
        result = StatsFormatter.format_search_results(memories)
        yield event.plain_result(result)
    
    @filter.command("memory_clear")
    async def clear_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        清除当前会话记忆指令（memory_delete current 的别名）

        用法：/memory_clear
        """
        # 直接复用 delete_memory 的 current 逻辑
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        success = await self._service.clear_memories(user_id, group_id)

        if success:
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await self.delete_kv_data(kv_key)
            yield event.plain_result(SuccessMessages.MEMORY_CLEARED)
        else:
            yield event.plain_result(ErrorMessages.DELETE_FAILED)
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        success = await self._service.clear_memories(user_id, group_id)
        
        if success:
            # 删除保存时间记录
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await self.delete_kv_data(kv_key)
            result = SuccessMessages.MEMORY_CLEARED
        else:
            result = ErrorMessages.DELETE_FAILED
        
        yield event.plain_result(result)

    @filter.command("memory_stats")
    async def memory_stats(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        记忆统计指令
        
        用法：/memory_stats
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        stats = await self._service.get_memory_stats(user_id, group_id)
        
        # 格式化响应
        image_stats = ""
        if stats.get("image_analyzed", 0) > 0:
            image_stats = f"\n- 图片分析：{stats['image_analyzed']} 张\n- 缓存命中：{stats['cache_hits']} 次"
        
        result = SuccessMessages.STATS_TEMPLATE.format(
            working_count=stats.get("working_count", 0),
            episodic_count=stats.get("episodic_count", 0),
            image_stats=image_stats
        )
        
        yield event.plain_result(result)

    @filter.command("memory_delete")
    async def delete_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        统一删除记忆指令

        用法：
        /memory_delete              - 删除当前会话记忆
        /memory_delete current      - 删除当前会话记忆
        /memory_delete private      - 删除我的私聊记忆
        /memory_delete group [scope] - 删除群聊记忆（管理员，群聊场景）
        /memory_delete all confirm   - 删除所有记忆（超管）
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        # 解析参数
        parsed = CommandParser.parse_with_slash(event.message_str, "memory_delete")

        # 检查 all confirm 参数
        has_confirm = (
            len(parsed.args) >= 2 and
            parsed.args[-1].lower() == NumericDefaults.CONFIRM_VALUE
        )

        # 解析范围
        args_for_parser = parsed.args[:-1] if has_confirm else parsed.args
        result = UnifiedDeleteScopeParser.parse(args_for_parser, has_confirm)

        if not result.is_valid:
            yield event.plain_result(result.error_message)
            return

        # 根据主范围执行不同的删除逻辑
        if result.main_scope == DeleteMainScope.CURRENT:
            # 删除当前会话记忆（无权限限制）
            success = await self._service.clear_memories(user_id, group_id)
            if success:
                kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
                await self.delete_kv_data(kv_key)
                yield event.plain_result(SuccessMessages.MEMORY_CLEARED)
            else:
                yield event.plain_result(ErrorMessages.DELETE_FAILED)

        elif result.main_scope == DeleteMainScope.PRIVATE:
            # 删除私聊记忆（无场景限制，任何地方都可删除自己的私聊记忆）
            success, count = await self._service.delete_private_memories(user_id)
            kv_key = SessionKeyBuilder.build_for_kv(user_id, None)
            await self.delete_kv_data(kv_key)
            if success:
                yield event.plain_result(SuccessMessages.PRIVATE_DELETED.format(count=count))
            else:
                yield event.plain_result(ErrorMessages.DELETE_FAILED)

        elif result.main_scope == DeleteMainScope.GROUP:
            # 删除群聊记忆（需要群聊场景 + 管理员权限）
            if not group_id:
                yield event.plain_result(ErrorMessages.GROUP_ONLY)
                return

            if not self._is_admin(event):
                yield event.plain_result(ErrorMessages.GROUP_ADMIN_REQUIRED)
                return

            success, count = await self._service.delete_group_memories(
                group_id=group_id,
                scope_filter=result.scope_filter,
                user_id=user_id
            )
            if success:
                yield event.plain_result(
                    SuccessMessages.GROUP_DELETED.format(count=count, scope_desc=result.scope_desc)
                )
            else:
                yield event.plain_result(ErrorMessages.DELETE_FAILED)

        elif result.main_scope == DeleteMainScope.ALL:
            # 删除所有记忆（需要超管权限 + confirm 参数）
            if not self._is_admin(event):
                yield event.plain_result(ErrorMessages.ADMIN_REQUIRED)
                return

            if not has_confirm:
                yield event.plain_result(ErrorMessages.DELETE_CONFIRM_REQUIRED)
                return

            success, count = await self._service.delete_all_memories()
            if success:
                yield event.plain_result(SuccessMessages.ALL_DELETED.format(count=count))
            else:
                yield event.plain_result(ErrorMessages.DELETE_FAILED)

    @filter.command("proactive_reply")
    async def proactive_reply_control(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        群聊主动回复开关指令（仅管理员，仅群聊）
        
        用法：
        /proactive_reply on      - 开启当前群的主动回复
        /proactive_reply off     - 关闭当前群的主动回复
        /proactive_reply status  - 查看当前群的主动回复状态
        /proactive_reply list    - 查看所有已开启主动回复的群聊
        """
        # 权限检查：管理员
        if not self._is_admin(event):
            yield event.plain_result(ErrorMessages.ADMIN_REQUIRED)
            return
        
        # 解析参数
        parsed = CommandParser.parse_with_slash(event.message_str, "proactive_reply")
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else "status"
        
        # 检查主动回复是否启用
        proactive_mgr = self._service.proactive_manager
        if not proactive_mgr:
            yield event.plain_result("主动回复功能未启用，请先在配置中开启 proactive_reply.enable")
            return
        
        # 检查白名单模式是否开启
        if not proactive_mgr.group_whitelist_mode:
            yield event.plain_result(
                "群聊白名单模式未开启，请先在配置中开启 proactive_reply.group_whitelist_mode"
            )
            return
        
        # list 子命令不要求群聊场景
        if sub_cmd == "list":
            whitelist = proactive_mgr.get_whitelist()
            if whitelist:
                group_list = "\n".join(f"- {gid}" for gid in whitelist)
                yield event.plain_result(f"已开启主动回复的群聊：\n{group_list}")
            else:
                yield event.plain_result("当前没有群聊开启主动回复")
            return
        
        # 以下子命令需要群聊场景
        group_id = self._check_group_only(event)
        if not group_id:
            yield event.plain_result(ErrorMessages.GROUP_ONLY)
            return
        
        if sub_cmd == "on":
            added = proactive_mgr.add_group_to_whitelist(group_id)
            if added:
                # 持久化
                await self._service.save_to_kv(self.put_kv_data)
                yield event.plain_result("已开启当前群聊的主动回复功能")
            else:
                yield event.plain_result("当前群聊已开启主动回复，无需重复操作")
                
        elif sub_cmd == "off":
            removed = proactive_mgr.remove_group_from_whitelist(group_id)
            if removed:
                # 持久化
                await self._service.save_to_kv(self.put_kv_data)
                yield event.plain_result("已关闭当前群聊的主动回复功能")
            else:
                yield event.plain_result("当前群聊未开启主动回复，无需操作")
                
        elif sub_cmd == "status":
            is_enabled = proactive_mgr.is_group_in_whitelist(group_id)
            status_text = "已开启" if is_enabled else "未开启"
            yield event.plain_result(f"当前群聊主动回复状态：{status_text}")
            
        else:
            yield event.plain_result(
                "用法：/proactive_reply <on|off|status|list>\n"
                "- on: 开启当前群的主动回复\n"
                "- off: 关闭当前群的主动回复\n"
                "- status: 查看当前群的状态\n"
                "- list: 查看所有已开启的群聊"
            )
    
    @filter.command("activity_status")
    async def activity_status(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        查看群活跃度状态指令
        
        用法：
        /activity_status          - 查看当前群的活跃度状态
        /activity_status all      - 查看所有群的活跃度概览（管理员）
        """
        group_id = get_group_id(event)
        parsed = CommandParser.parse_with_slash(event.message_str, "activity_status")
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else ""
        
        # 检查自适应系统是否可用
        provider = self._service.activity_provider
        if not provider or not provider.enabled:
            yield event.plain_result("场景自适应系统未启用")
            return
        
        level_labels = {
            "quiet": "🌙 安静",
            "moderate": "☀️ 中等",
            "active": "🔥 活跃",
            "intensive": "⚡ 超活跃",
        }
        
        if sub_cmd == "all":
            # 管理员查看所有群
            if not self._is_admin(event):
                yield event.plain_result(ErrorMessages.ADMIN_REQUIRED)
                return
            
            summaries = provider.get_all_activity_summaries()
            if not summaries:
                yield event.plain_result("暂无群活跃度数据")
                return
            
            lines = ["📊 群活跃度概览：\n"]
            for s in summaries:
                label = level_labels.get(s["activity_level"], s["activity_level"])
                lines.append(
                    f"  群 {s['group_id']}: {label} "
                    f"({s['messages_per_hour']:.0f} 条/时)"
                )
            yield event.plain_result("\n".join(lines))
        else:
            # 查看当前群
            if not group_id:
                yield event.plain_result("此指令仅限群聊使用")
                return
            
            summary = provider.get_group_activity_summary(group_id)
            level = summary["activity_level"]
            label = level_labels.get(level, level)
            mph = summary["messages_per_hour"]
            cfg = summary["config"]
            
            result_lines = [
                f"📊 当前群活跃度：{label}",
                f"消息频率：约 {mph:.0f} 条/小时\n",
                "当前自适应配置：",
                f"  • 主动回复冷却：{cfg.get('cooldown_seconds', '?')}秒",
                f"  • 每日回复上限：{cfg.get('max_daily_replies', '?')}次",
                f"  • 批处理阈值：{cfg.get('batch_threshold_count', '?')}条",
                f"  • 处理间隔：{cfg.get('batch_threshold_interval', '?')}秒",
                f"  • 图片分析预算：{cfg.get('daily_analysis_budget', '?')}次/日",
                f"  • 上下文条数：{cfg.get('chat_context_count', '?')}条",
            ]
            yield event.plain_result("\n".join(result_lines))
    
    # ========== 消息装饰钩子 ==========
    
    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """
        消息发送前拦截，替换框架错误消息为友好提示
        
        钩子特性：
        - 与其他插件的 on_decorating_result 钩子顺序执行
        - 通过 event.get_result() 获取/修改消息结果
        - 直接修改 result 对象，无需返回值
        - 可通过 event.stop() 阻止后续钩子执行
        
        Args:
            event: 消息事件对象
        """
        # 功能开关检查
        if not self._is_error_friendly_enabled():
            return
        
        # 获取消息结果
        result = event.get_result()
        if not result:
            return
        
        # 获取消息文本
        text = self._get_result_plain_text(result)
        if not text:
            return
        
        # 检测是否为框架错误消息
        if self._is_framework_error(text):
            friendly_msg = ErrorFriendlyMessages.DEFAULT_FRIENDLY_MSG
            result.chain.clear()
            result.message(friendly_msg)
            self._service.logger.info("Replaced framework error message with friendly text")
            # 注意：不调用 event.stop()，允许其他插件继续处理
    
    def _is_error_friendly_enabled(self) -> bool:
        """检查错误消息友好化功能是否启用"""
        try:
            return self.config.get(ConfigKeys.ERROR_FRIENDLY_ENABLE, True)
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
        text_lower = text.lower()
        # 检查是否包含多个错误特征
        match_count = sum(
            1 for pattern in ErrorFriendlyMessages.ERROR_PATTERNS
            if pattern.lower() in text_lower
        )
        # 至少匹配2个特征才判定为框架错误消息
        return match_count >= 2
    
    # ========== LLM Hook ==========
    
    @filter.on_llm_request()
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        req: Any
    ) -> None:
        """
        在LLM请求前注入上下文

        注入的上下文层次（按优先级排序）：
        1. 近期聊天记录 - 让AI了解当前话题
        2. 相关记忆 - 长期记忆检索结果
        3. 图片分析 - 当前消息中的图片描述
        4. 行为指导 - 防止重复/过度反问
        5. 主动回复指令 - 仅在主动回复时附加
        """
        # 功能开关检查
        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_inject:
            return
        
        # 检查记忆系统是否就绪（embedding 模型可能还在后台加载）
        if not self._service.is_embedding_ready():
            # 跳过记忆注入，在系统提示中添加状态信息
            req.system_prompt += "\n\n[系统提示：记忆系统正在初始化，暂时无法提供历史记忆参考]\n"
            self._service.logger.info("Embedding model not ready, skipping memory injection")
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        query = event.message_str
        sender_name = get_sender_name(event)
        
        is_proactive = event.get_extra("iris_proactive", False)
        
        # 确保成员身份信息最新（LLM请求时也更新一次）
        if self._service.member_identity and not is_proactive:
            await self._service.member_identity.resolve_tag(
                user_id, sender_name, group_id
            )
        
        # 激活会话
        await self._service.activate_session(user_id, group_id)
        
        # 注意：@Bot 的消息已在 on_all_messages（先于本 Hook 执行）中
        # 记录到聊天缓冲区，此处不再重复记录。
        
        # 图片分析（主动回复时跳过，合成事件无真实图片）
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
        
        # 准备LLM上下文（聊天记录 + 记忆 + 图片 + 行为指导）
        # 主动回复时：使用触发提示作为 query 检索记忆
        context = await self._service.prepare_llm_context(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context,
            sender_name=sender_name
        )
        
        # 注入上下文
        if context:
            req.system_prompt += f"\n\n{context}\n"
        
        # 主动回复场景：附加特殊系统指令
        if is_proactive:
            proactive_ctx = event.get_extra("iris_proactive_context", {})
            proactive_directive = self._build_proactive_directive(proactive_ctx)
            req.system_prompt += f"\n\n{proactive_directive}\n"
            self._service.logger.info(
                f"Proactive reply context injected for user={user_id}"
            )
    
    @filter.on_llm_response()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        resp: Any
    ) -> None:
        """
        在LLM响应后：
        1. 记录Bot的回复到聊天缓冲区
        2. 自动捕获新记忆（主动回复时跳过用户消息捕获）
        """
        # 功能开关检查
        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_memory:
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        sender_name = get_sender_name(event)
        
        is_proactive = event.get_extra("iris_proactive", False)
        
        # 记录Bot回复到聊天缓冲区（主动回复也要记录Bot的回复）
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
                session_user_id=user_id  # 归入对话用户的缓冲区
            )
        
        # 更新会话活动
        self._service.update_session_activity(user_id, group_id)
        
        # 主动回复时：跳过用户消息的记忆捕获
        # （合成事件的 message_str 是触发提示，不是真实用户消息）
        if is_proactive:
            self._service.logger.info(
                f"Proactive reply completed for user={user_id}, "
                f"reply_len={len(bot_reply)}"
            )
            return
        
        # 捕获记忆（仅正常消息流程）
        memory = await self._service.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            sender_name=sender_name
        )
        
        if memory:
            self._service.logger.debug(LogTemplates.MEMORY_CAPTURED.format(memory_id=memory.id))
    
    # ========== 普通消息处理器 ==========
    
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_messages(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        统一处理所有普通消息 - 分层处理策略
        
        职责：
        1. 记录消息到聊天缓冲区（供LLM上下文注入）
        2. 分层处理：immediate/batch/discard
        3. 主动回复事件检测与 LLM 请求转发
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        sender_name = get_sender_name(event)
        
        # ========== 主动回复事件处理 ==========
        # 检测合成事件标记，转入完整 LLM 流程
        if event.get_extra("iris_proactive", False):
            self._service.logger.info(
                f"Proactive reply event detected for user={user_id}, "
                f"group={group_id}"
            )
            # 通过 yield event.request_llm() 将请求注入 ProcessStage
            # 后续经过 build_main_agent（人格+技能）→ OnLLMRequestEvent
            # （记忆注入）→ LLM 生成 → OnLLMResponseEvent → 装饰 → 发送
            yield event.request_llm(prompt=message)
            return
        
        # ========== 普通消息处理 ==========
        
        # 过滤指令消息
        if MessageFilter.is_command(message):
            return
        
        # 更新成员身份信息（名称追踪、活跃度、群归属）
        if self._service.member_identity:
            await self._service.member_identity.resolve_tag(
                user_id, sender_name, group_id
            )
        
        # 记录消息到聊天缓冲区（无论批量处理器是否就绪都记录）
        await self._service.record_chat_message(
            sender_id=user_id,
            sender_name=sender_name,
            content=message,
            group_id=group_id,
            is_bot=False
        )
        
        # 以下为记忆捕获流程，需要批量处理器
        if not self._service.batch_processor:
            return
        
        # 更新会话活动
        self._service.update_session_activity(user_id, group_id)
        
        # 图片分析
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
        
        # 构建上下文
        context = await self._build_message_context(user_id, group_id)
        context["sender_name"] = sender_name  # 传递发送者名称
        
        # 处理消息批次
        await self._service.process_message_batch(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context=context,
            umo=event.unified_msg_origin,
            image_description=image_description
        )
    
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
        
        告诉LLM这是一次主动回复场景，提供触发原因和行为指导。
        
        Args:
            proactive_ctx: 主动回复上下文，包含触发原因、近期消息等
            
        Returns:
            str: 主动回复系统指令文本
        """
        reason = proactive_ctx.get("reason", "检测到对话信号")
        recent_messages = proactive_ctx.get("recent_messages", [])
        emotion_summary = proactive_ctx.get("emotion_summary", "")
        target_user = proactive_ctx.get("target_user", "用户")
        
        # 构建近期消息摘要
        recent_text = ""
        if recent_messages:
            recent_lines = []
            for msg in recent_messages[-5:]:  # 最多展示5条
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
        )
        
        return directive
    
    # ========== 生命周期方法 ==========
    
    async def terminate(self) -> None:
        """插件销毁"""
        # 保存数据
        await self._service.save_to_kv(self.put_kv_data)
        
        # 销毁服务
        await self._service.terminate()
