"""
命令处理器模块

封装所有 AstrBot 插件命令的处理逻辑，与 main.py 解耦。

设计原则：
- 每个命令处理器返回字符串结果，由 main.py 负责发送
- 权限检查通过 PermissionChecker 完成
- 业务逻辑委托给 MemoryService
"""

from typing import Optional, Any, TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent

from iris_memory.commands.permissions import PermissionChecker
from iris_memory.utils.event_utils import get_group_id, get_sender_name
from iris_memory.utils.persona_utils import get_event_persona_id
from iris_memory.utils.command_utils import (
    CommandParser,
    StatsFormatter,
    SessionKeyBuilder,
    UnifiedDeleteScopeParser,
)
from iris_memory.core.constants import (
    CommandPrefix,
    ErrorMessages,
    SuccessMessages,
    NumericDefaults,
    DeleteMainScope,
    InputValidationConfig,
    KVStoreKeys,
)

if TYPE_CHECKING:
    from iris_memory.services.memory_service import MemoryService


class CommandHandlers:
    """
    命令处理器集合

    封装所有命令的处理逻辑，提供统一的处理接口。
    每个处理器返回响应字符串，由调用方负责发送。
    """

    def __init__(self, service: "MemoryService") -> None:
        """
        初始化命令处理器

        Args:
            service: MemoryService 实例
        """
        self._service = service
        self._perms = PermissionChecker()

    async def handle_save_memory(self, event: AstrMessageEvent) -> str:
        """
        处理 /memory_save 命令

        用法：/memory_save <内容>

        Args:
            event: 消息事件对象

        Returns:
            str: 响应消息
        """
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SAVE)

        if not parsed.has_content:
            return ErrorMessages.EMPTY_CONTENT

        if len(parsed.content) > InputValidationConfig.MAX_SAVE_CONTENT_LENGTH:
            return f"内容过长（最大 {InputValidationConfig.MAX_SAVE_CONTENT_LENGTH} 字符）"

        content = InputValidationConfig.sanitize_input(parsed.content)
        if not content:
            return ErrorMessages.EMPTY_CONTENT

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        sender_name = get_sender_name(event)

        raw_persona_id = get_event_persona_id(event)
        store_persona = self._service.cfg.get_persona_id_for_storage(raw_persona_id)

        memory = await self._service.capture_and_store_memory(
            message=content,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=True,
            sender_name=sender_name,
            persona_id=store_persona,
        )

        if memory:
            result = SuccessMessages.MEMORY_SAVED.format(
                memory_type=memory.type.value,
                confidence=memory.confidence
            )
            return result
        return ErrorMessages.CAPTURE_FAILED

    async def handle_search_memory(self, event: AstrMessageEvent) -> str:
        """
        处理 /memory_search 命令

        用法：/memory_search <查询内容>

        Args:
            event: 消息事件对象

        Returns:
            str: 响应消息
        """
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SEARCH)

        if not parsed.has_content:
            return ErrorMessages.EMPTY_QUERY

        if len(parsed.content) > InputValidationConfig.MAX_QUERY_LENGTH:
            return f"查询内容过长（最大 {InputValidationConfig.MAX_QUERY_LENGTH} 字符）"

        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        raw_persona_id = get_event_persona_id(event)
        query_persona = self._service.cfg.get_persona_id_for_query(raw_persona_id, "memory")

        memories = await self._service.search_memories(
            query=parsed.content,
            user_id=user_id,
            group_id=group_id,
            top_k=NumericDefaults.TOP_K_SEARCH,
            persona_id=query_persona,
        )

        return StatsFormatter.format_search_results(memories)

    async def handle_clear_memory(self, event: AstrMessageEvent) -> str:
        """
        处理 /memory_clear 命令（memory_delete current 的别名）

        用法：/memory_clear

        Args:
            event: 消息事件对象

        Returns:
            str: 响应消息
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        success = await self._service.clear_memories(user_id, group_id)

        if success:
            return SuccessMessages.MEMORY_CLEARED
        return ErrorMessages.DELETE_FAILED

    async def handle_memory_stats(self, event: AstrMessageEvent) -> str:
        """
        处理 /memory_stats 命令

        用法：/memory_stats

        Args:
            event: 消息事件对象

        Returns:
            str: 响应消息
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        stats = await self._service.get_memory_stats(user_id, group_id)

        image_stats = ""
        if stats.get("image_analyzed", 0) > 0:
            image_stats = f"\n- 图片分析：{stats['image_analyzed']} 张\n- 缓存命中：{stats['cache_hits']} 次"

        return SuccessMessages.STATS_TEMPLATE.format(
            working_count=stats.get("working_count", 0),
            episodic_count=stats.get("episodic_count", 0),
            image_stats=image_stats
        )

    async def handle_delete_memory(
        self,
        event: AstrMessageEvent,
        delete_kv_func: Any
    ) -> str:
        """
        处理 /memory_delete 命令

        用法：
        /memory_delete              - 删除当前会话记忆
        /memory_delete current      - 删除当前会话记忆
        /memory_delete private      - 删除我的私聊记忆
        /memory_delete group [scope] - 删除群聊记忆（管理员，群聊场景）
        /memory_delete all confirm   - 删除所有记忆（超管）

        Args:
            event: 消息事件对象
            delete_kv_func: 删除 KV 数据的函数

        Returns:
            str: 响应消息
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        parsed = CommandParser.parse_with_slash(event.message_str, "memory_delete")

        has_confirm = (
            len(parsed.args) >= 2 and
            parsed.args[-1].lower() == NumericDefaults.CONFIRM_VALUE
        )

        args_for_parser = parsed.args[:-1] if has_confirm else parsed.args
        result = UnifiedDeleteScopeParser.parse(args_for_parser, has_confirm)

        if not result.is_valid:
            return result.error_message

        handler_map = {
            DeleteMainScope.CURRENT: self._handle_delete_current,
            DeleteMainScope.PRIVATE: self._handle_delete_private,
            DeleteMainScope.GROUP: self._handle_delete_group,
            DeleteMainScope.ALL: self._handle_delete_all,
        }

        handler = handler_map.get(result.main_scope)
        if handler:
            return await handler(event, user_id, group_id, result, has_confirm, delete_kv_func)

        return ErrorMessages.DELETE_FAILED

    async def _handle_delete_current(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: Optional[str],
        result: Any,
        has_confirm: bool,
        delete_kv_func: Any
    ) -> str:
        """处理删除当前会话记忆"""
        success = await self._service.clear_memories(user_id, group_id)
        if success:
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await delete_kv_func(kv_key)
            return SuccessMessages.MEMORY_CLEARED
        return ErrorMessages.DELETE_FAILED

    async def _handle_delete_private(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: Optional[str],
        result: Any,
        has_confirm: bool,
        delete_kv_func: Any
    ) -> str:
        """处理删除私聊记忆"""
        success, count = await self._service.delete_private_memories(user_id)
        kv_key = SessionKeyBuilder.build_for_kv(user_id, None)
        await delete_kv_func(kv_key)
        if success:
            return SuccessMessages.PRIVATE_DELETED.format(count=count)
        return ErrorMessages.DELETE_FAILED

    async def _handle_delete_group(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: Optional[str],
        result: Any,
        has_confirm: bool,
        delete_kv_func: Any
    ) -> str:
        """处理删除群聊记忆"""
        if not group_id:
            return ErrorMessages.GROUP_ONLY

        if not self._perms.is_admin(event):
            return ErrorMessages.GROUP_ADMIN_REQUIRED

        success, count = await self._service.delete_group_memories(
            group_id=group_id,
            scope_filter=result.scope_filter,
            user_id=user_id
        )
        if success:
            return SuccessMessages.GROUP_DELETED.format(count=count, scope_desc=result.scope_desc)
        return ErrorMessages.DELETE_FAILED

    async def _handle_delete_all(
        self,
        event: AstrMessageEvent,
        user_id: str,
        group_id: Optional[str],
        result: Any,
        has_confirm: bool,
        delete_kv_func: Any
    ) -> str:
        """处理删除所有记忆"""
        if not self._perms.is_admin(event):
            return ErrorMessages.ADMIN_REQUIRED

        if not has_confirm:
            return ErrorMessages.DELETE_CONFIRM_REQUIRED

        success, count = await self._service.delete_all_memories()
        if success:
            return SuccessMessages.ALL_DELETED.format(count=count)
        return ErrorMessages.DELETE_FAILED

    async def handle_proactive_reply(
        self,
        event: AstrMessageEvent,
        put_kv_func: Any
    ) -> str:
        """
        处理 /proactive_reply 命令

        用法：
        /proactive_reply on      - 开启当前群的主动回复
        /proactive_reply off     - 关闭当前群的主动回复
        /proactive_reply status  - 查看当前群的主动回复状态
        /proactive_reply list    - 查看所有已开启主动回复的群聊

        Args:
            event: 消息事件对象
            put_kv_func: 保存 KV 数据的函数

        Returns:
            str: 响应消息
        """
        if not self._perms.is_admin(event):
            return ErrorMessages.ADMIN_REQUIRED

        parsed = CommandParser.parse_with_slash(event.message_str, "proactive_reply")
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else "status"

        proactive_mgr = self._service.proactive_manager
        if not proactive_mgr:
            return "主动回复功能未启用，请先在配置中开启 proactive_reply.enable"

        if not proactive_mgr.group_whitelist_mode:
            return "群聊白名单模式未开启，请先在配置中开启 proactive_reply.group_whitelist_mode"

        if sub_cmd == "list":
            whitelist = proactive_mgr.get_whitelist()
            if whitelist:
                group_list = "\n".join(f"- {gid}" for gid in whitelist)
                return f"已开启主动回复的群聊：\n{group_list}"
            return "当前没有群聊开启主动回复"

        group_id = self._perms.check_group_only(event)
        if not group_id:
            return ErrorMessages.GROUP_ONLY

        sub_handlers = {
            "on": self._handle_proactive_on,
            "off": self._handle_proactive_off,
            "status": self._handle_proactive_status,
        }

        handler = sub_handlers.get(sub_cmd)
        if handler:
            return await handler(group_id, proactive_mgr, put_kv_func)

        return (
            "用法：/proactive_reply <on|off|status|list>\n"
            "- on: 开启当前群的主动回复\n"
            "- off: 关闭当前群的主动回复\n"
            "- status: 查看当前群的状态\n"
            "- list: 查看所有已开启的群聊"
        )

    async def _handle_proactive_on(
        self,
        group_id: str,
        proactive_mgr: Any,
        put_kv_func: Any
    ) -> str:
        """处理开启主动回复"""
        added = proactive_mgr.add_group_to_whitelist(group_id)
        if added:
            await self._service.save_to_kv(put_kv_func)
            return "已开启当前群聊的主动回复功能"
        return "当前群聊已开启主动回复，无需重复操作"

    async def _handle_proactive_off(
        self,
        group_id: str,
        proactive_mgr: Any,
        put_kv_func: Any
    ) -> str:
        """处理关闭主动回复"""
        removed = proactive_mgr.remove_group_from_whitelist(group_id)
        if removed:
            await self._service.save_to_kv(put_kv_func)
            return "已关闭当前群聊的主动回复功能"
        return "当前群聊未开启主动回复，无需操作"

    async def _handle_proactive_status(
        self,
        group_id: str,
        proactive_mgr: Any,
        put_kv_func: Any
    ) -> str:
        """处理查看主动回复状态"""
        is_enabled = proactive_mgr.is_group_in_whitelist(group_id)
        status_text = "已开启" if is_enabled else "未开启"
        return f"当前群聊主动回复状态：{status_text}"

    async def handle_activity_status(self, event: AstrMessageEvent) -> str:
        """
        处理 /activity_status 命令

        用法：
        /activity_status          - 查看当前群的活跃度状态
        /activity_status all      - 查看所有群的活跃度概览（管理员）

        Args:
            event: 消息事件对象

        Returns:
            str: 响应消息
        """
        group_id = get_group_id(event)
        parsed = CommandParser.parse_with_slash(event.message_str, "activity_status")
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else ""

        provider = self._service.activity_provider
        if not provider or not provider.enabled:
            return "场景自适应系统未启用"

        level_labels = {
            "quiet": "🌙 安静",
            "moderate": "☀️ 中等",
            "active": "🔥 活跃",
            "intensive": "⚡ 超活跃",
        }

        if sub_cmd == "all":
            if not self._perms.is_admin(event):
                return ErrorMessages.ADMIN_REQUIRED

            summaries = provider.get_all_activity_summaries()
            if not summaries:
                return "暂无群活跃度数据"

            lines = ["📊 群活跃度概览：\n"]
            for s in summaries:
                label = level_labels.get(s["activity_level"], s["activity_level"])
                lines.append(
                    f"  群 {s['group_id']}: {label} "
                    f"({s['messages_per_hour']:.0f} 条/时)"
                )
            return "\n".join(lines)

        if not group_id:
            return "此指令仅限群聊使用"

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
        return "\n".join(result_lines)

    async def handle_iris_reset(
        self,
        event: AstrMessageEvent,
        delete_kv_func: Any
    ) -> str:
        """
        处理 /iris_reset 命令

        用法：/iris_reset confirm
        警告：这将删除所有记忆、用户画像、会话数据、聊天记录等，不可恢复！

        Args:
            event: 消息事件对象
            delete_kv_func: 删除 KV 数据的函数

        Returns:
            str: 响应消息
        """
        if not self._perms.is_admin(event):
            return ErrorMessages.ADMIN_REQUIRED

        parsed = CommandParser.parse_with_slash(event.message_str, "iris_reset")
        if "confirm" not in parsed.args:
            return (
                "⚠️ 警告：此操作将永久删除所有 Iris Memory 数据！\n"
                "包括：记忆、用户画像、会话记录、群成员信息、聊天记录等\n"
                "请使用 '/iris_reset confirm' 确认操作"
            )

        keys_to_delete = [
            KVStoreKeys.SESSIONS,
            KVStoreKeys.LIFECYCLE_STATE,
            KVStoreKeys.BATCH_QUEUES,
            KVStoreKeys.CHAT_HISTORY,
            KVStoreKeys.USER_PERSONAS,
            KVStoreKeys.MEMBER_IDENTITY,
            KVStoreKeys.GROUP_ACTIVITY,
            KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
            KVStoreKeys.PERSONA_BATCH_QUEUES,
        ]

        deleted_count = 0
        failed_keys = []
        for key in keys_to_delete:
            try:
                await delete_kv_func(key)
                deleted_count += 1
            except Exception as e:
                failed_keys.append(f"{key}: {e}")
                self._service.logger.warning(f"Failed to delete KV key {key}: {e}")

        self._service._user_personas.clear()
        self._service._user_emotional_states.clear()
        self._service._recently_injected.clear()

        db_deleted_count = 0
        try:
            success, db_deleted_count = await self._service.delete_all_memories()
            if not success:
                failed_keys.append("chroma_memories: delete_all_memories returned False")
        except Exception as e:
            failed_keys.append(f"chroma_memories: {e}")
            self._service.logger.warning(f"Failed to delete all memories: {e}")

        result_msg = f"✅ 已重置 Iris Memory 数据\n"
        result_msg += f"- 成功清理 {deleted_count}/{len(keys_to_delete)} 个存储键\n"
        result_msg += f"- 成功清理 {db_deleted_count} 条数据库记忆\n"

        if failed_keys:
            result_msg += f"- 失败 {len(failed_keys)} 项（查看日志了解详情）\n"

        result_msg += "\n📌 建议操作：\n"
        result_msg += "1. 重启 AstrBot 以确保所有缓存已清空\n"
        result_msg += "2. 重新初始化插件以开始使用"

        return result_msg
