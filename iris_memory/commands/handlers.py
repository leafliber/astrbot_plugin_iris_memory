"""
命令处理器模块

封装所有 AstrBot 插件命令的处理逻辑，与 main.py 解耦。
"""

from typing import Any, TYPE_CHECKING

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
    """命令处理器集合"""

    def __init__(self, service: "MemoryService") -> None:
        self._service = service
        self._perms = PermissionChecker()

    # ─────────────────────────────────────────────────────────────────────────────
    # Memory 命令
    # ─────────────────────────────────────────────────────────────────────────────

    async def handle_memory_command(
        self,
        event: AstrMessageEvent,
        delete_kv_func: Any = None,
        put_kv_func: Any = None,
    ) -> str:
        """处理 /memory 统一入口命令"""
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY)
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else ""

        if not sub_cmd:
            return (
                "用法：/memory <子命令> [参数]\n"
                "  save <内容>     - 保存记忆\n"
                "  search <查询>   - 搜索记忆\n"
                "  clear           - 清除当前会话记忆\n"
                "  stats           - 记忆统计\n"
                "  delete [scope]  - 删除记忆\n"
                "  review          - 查看待审核记忆\n"
                "  approve <id>    - 批准记忆\n"
                "  reject <id>     - 拒绝记忆"
            )

        handlers = {
            "save": self._handle_memory_save,
            "search": self._handle_memory_search,
            "clear": self._handle_memory_clear,
            "stats": self._handle_memory_stats,
            "review": self._handle_memory_review,
            "approve": self._handle_memory_approve,
            "reject": self._handle_memory_reject,
        }

        if sub_cmd == "delete":
            if delete_kv_func is None:
                return ErrorMessages.DELETE_FAILED
            return await self._handle_memory_delete(event, delete_kv_func)

        handler = handlers.get(sub_cmd)
        if handler:
            return await handler(event, parsed)

        return f"未知子命令：{sub_cmd}\n使用 /memory 查看可用子命令"

    async def _handle_memory_save(self, event: AstrMessageEvent, parsed: Any) -> str:
        """保存记忆"""
        content = " ".join(parsed.args[1:]) if len(parsed.args) > 1 else ""
        if not content:
            return ErrorMessages.EMPTY_CONTENT
        if len(content) > InputValidationConfig.MAX_SAVE_CONTENT_LENGTH:
            return f"内容过长（最大 {InputValidationConfig.MAX_SAVE_CONTENT_LENGTH} 字符）"

        content = InputValidationConfig.sanitize_input(content)
        if not content:
            return ErrorMessages.EMPTY_CONTENT

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        raw_persona_id = get_event_persona_id(event)
        store_persona = self._service.cfg.get("persona_isolation.default_persona_id", raw_persona_id)

        memory = await self._service.capture_and_store_memory(
            message=content,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=True,
            sender_name=get_sender_name(event),
            persona_id=store_persona,
        )

        if memory:
            return SuccessMessages.MEMORY_SAVED.format(
                memory_type=memory.type.value, confidence=memory.confidence
            )
        return ErrorMessages.CAPTURE_FAILED

    async def _handle_memory_search(self, event: AstrMessageEvent, parsed: Any) -> str:
        """搜索记忆"""
        content = " ".join(parsed.args[1:]) if len(parsed.args) > 1 else ""
        if not content:
            return ErrorMessages.EMPTY_QUERY
        if len(content) > InputValidationConfig.MAX_QUERY_LENGTH:
            return f"查询内容过长（最大 {InputValidationConfig.MAX_QUERY_LENGTH} 字符）"

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        raw_persona_id = get_event_persona_id(event)
        query_persona = self._service.cfg.get("persona_isolation.default_persona_id", raw_persona_id)

        memories = await self._service.search_memories(
            query=content,
            user_id=user_id,
            group_id=group_id,
            top_k=NumericDefaults.TOP_K_SEARCH,
            persona_id=query_persona,
        )
        return StatsFormatter.format_search_results(memories)

    async def _handle_memory_clear(self, event: AstrMessageEvent, parsed: Any) -> str:
        """清除当前会话记忆"""
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        success = await self._service.clear_memories(user_id, group_id)
        return SuccessMessages.MEMORY_CLEARED if success else ErrorMessages.DELETE_FAILED

    async def _handle_memory_stats(self, event: AstrMessageEvent, parsed: Any) -> str:
        """记忆统计"""
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        stats = await self._service.get_memory_stats(user_id, group_id)

        image_stats = ""
        if stats.get("image_analyzed", 0) > 0:
            image_stats = f"\n- 图片分析：{stats['image_analyzed']} 张\n- 缓存命中：{stats['cache_hits']} 次"

        return SuccessMessages.STATS_TEMPLATE.format(
            working_count=stats.get("working_count", 0),
            episodic_count=stats.get("episodic_count", 0),
            image_stats=image_stats,
        )

    async def _handle_memory_delete(self, event: AstrMessageEvent, delete_kv_func: Any) -> str:
        """删除记忆"""
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY)
        args = parsed.args[1:] if len(parsed.args) > 1 else []

        has_confirm = len(args) >= 2 and args[-1].lower() == NumericDefaults.CONFIRM_VALUE
        args_for_parser = args[:-1] if has_confirm else args
        result = UnifiedDeleteScopeParser.parse(args_for_parser, has_confirm)

        if not result.is_valid:
            return result.error_message

        if result.main_scope == DeleteMainScope.CURRENT:
            success = await self._service.clear_memories(user_id, group_id)
            if success:
                kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
                await delete_kv_func(kv_key)
                return SuccessMessages.MEMORY_CLEARED
            return ErrorMessages.DELETE_FAILED

        elif result.main_scope == DeleteMainScope.PRIVATE:
            success, count = await self._service.delete_private_memories(user_id)
            kv_key = SessionKeyBuilder.build_for_kv(user_id, None)
            await delete_kv_func(kv_key)
            return SuccessMessages.PRIVATE_DELETED.format(count=count) if success else ErrorMessages.DELETE_FAILED

        elif result.main_scope == DeleteMainScope.GROUP:
            if not group_id:
                return ErrorMessages.GROUP_ONLY
            if not self._perms.is_admin(event):
                return ErrorMessages.GROUP_ADMIN_REQUIRED
            success, count = await self._service.delete_group_memories(
                group_id=group_id, scope_filter=result.scope_filter, user_id=user_id
            )
            return SuccessMessages.GROUP_DELETED.format(count=count, scope_desc=result.scope_desc) if success else ErrorMessages.DELETE_FAILED

        elif result.main_scope == DeleteMainScope.ALL:
            if not self._perms.is_admin(event):
                return ErrorMessages.ADMIN_REQUIRED
            if not has_confirm:
                return ErrorMessages.DELETE_CONFIRM_REQUIRED
            success, count = await self._service.delete_all_memories()
            return SuccessMessages.ALL_DELETED.format(count=count) if success else ErrorMessages.DELETE_FAILED

        return ErrorMessages.DELETE_FAILED

    async def _handle_memory_review(self, event: AstrMessageEvent, parsed: Any) -> str:
        """查看待审核记忆"""
        chroma = self._get_chroma_manager()
        if not chroma:
            return "记忆存储未初始化"

        pending = await chroma.get_pending_review_memories(limit=20)
        if not pending:
            return "暂无待审核的语义记忆"

        lines = ["📝 待审核的语义记忆：\n"]
        for i, mem in enumerate(pending, 1):
            lines.append(f"{i}. [{mem.id[:8]}] {mem.content[:50]}...")
        lines.append("\n使用 /memory approve <id前缀> 或 /memory reject <id前缀> 处理")
        return "\n".join(lines)

    async def _handle_memory_approve(self, event: AstrMessageEvent, parsed: Any) -> str:
        """批准待审核记忆"""
        id_prefix = parsed.args[1] if len(parsed.args) > 1 else ""
        if not id_prefix:
            return "请提供记忆 ID 前缀：/memory approve <id前缀>"
        return await self._update_review_status(id_prefix, "approved")

    async def _handle_memory_reject(self, event: AstrMessageEvent, parsed: Any) -> str:
        """拒绝待审核记忆"""
        id_prefix = parsed.args[1] if len(parsed.args) > 1 else ""
        if not id_prefix:
            return "请提供记忆 ID 前缀：/memory reject <id前缀>"
        return await self._update_review_status(id_prefix, "rejected")

    async def _update_review_status(self, id_prefix: str, new_status: str) -> str:
        """更新审核状态"""
        chroma = self._get_chroma_manager()
        if not chroma:
            return "记忆存储未初始化"

        pending = await chroma.get_pending_review_memories(limit=100)
        target = next((m for m in pending if m.id.startswith(id_prefix)), None)

        if not target:
            return f"未找到以 '{id_prefix}' 开头的待审核记忆"

        target.review_status = new_status

        if new_status == "rejected":
            deleted = await chroma.delete_memory(target.id)
            return f"已拒绝并删除语义记忆 [{target.id[:8]}]" if deleted else f"拒绝失败：无法删除记忆 [{target.id[:8]}]"
        else:
            updated = await chroma.update_memory(target)
            return f"已批准语义记忆 [{target.id[:8]}]: {target.content[:40]}" if updated else f"批准失败：无法更新记忆 [{target.id[:8]}]"

    def _get_chroma_manager(self) -> Any:
        """获取 Chroma 管理器"""
        return getattr(self._service, "_chroma_manager", None)

    # ─────────────────────────────────────────────────────────────────────────────
    # Iris 命令
    # ─────────────────────────────────────────────────────────────────────────────

    async def handle_iris_command(
        self,
        event: AstrMessageEvent,
        delete_kv_func: Any = None,
        put_kv_func: Any = None,
    ) -> str:
        """处理 /iris 统一入口命令"""
        parsed = CommandParser.parse(event.message_str, CommandPrefix.IRIS)
        sub_cmd = parsed.first_arg.lower() if parsed.first_arg else ""

        if not sub_cmd:
            return (
                "用法：/iris <子命令> [参数]\n"
                "  proactive <on|off|status|list> - 主动回复控制\n"
                "  activity [all]                 - 活跃度状态\n"
                "  reset confirm                  - 重置所有数据\n"
                "  cooldown [action] [duration]   - 群冷却控制\n"
                "  persona [action]               - 用户画像管理"
            )

        if sub_cmd == "proactive":
            if put_kv_func is None:
                return "需要 KV 操作函数"
            return await self._handle_iris_proactive(event, put_kv_func)

        if sub_cmd == "reset":
            if delete_kv_func is None:
                return "需要 KV 操作函数"
            return await self._handle_iris_reset(event, delete_kv_func)

        if sub_cmd == "persona":
            return await self._handle_iris_persona(event, put_kv_func)

        handlers = {
            "activity": self._handle_iris_activity,
            "cooldown": self._handle_iris_cooldown,
        }

        handler = handlers.get(sub_cmd)
        if handler:
            return await handler(event, parsed)

        return f"未知子命令：{sub_cmd}\n使用 /iris 查看可用子命令"

    async def _handle_iris_proactive(self, event: AstrMessageEvent, put_kv_func: Any) -> str:
        """主动回复控制"""
        if not self._perms.is_admin(event):
            return ErrorMessages.ADMIN_REQUIRED

        parsed = CommandParser.parse(event.message_str, CommandPrefix.IRIS)
        action = parsed.args[1].lower() if len(parsed.args) > 1 else "status"

        proactive_mgr = self._service.proactive_manager
        if not proactive_mgr:
            return "主动回复功能未启用，请先在配置中开启 proactive_reply.enable"
        if not proactive_mgr.group_whitelist_mode:
            return "群聊白名单模式未开启，请先在配置中开启 proactive_reply.group_whitelist_mode"

        if action == "list":
            whitelist = proactive_mgr.get_whitelist()
            if whitelist:
                return "已开启主动回复的群聊：\n" + "\n".join(f"- {gid}" for gid in whitelist)
            return "当前没有群聊开启主动回复"

        group_id = self._perms.check_group_only(event)
        if not group_id:
            return ErrorMessages.GROUP_ONLY

        if action == "on":
            added = proactive_mgr.add_group_to_whitelist(group_id)
            if added:
                await self._service.save_to_kv(put_kv_func)
                return "已开启当前群聊的主动回复功能"
            return "当前群聊已开启主动回复，无需重复操作"

        elif action == "off":
            removed = proactive_mgr.remove_group_from_whitelist(group_id)
            if removed:
                await self._service.save_to_kv(put_kv_func)
                return "已关闭当前群聊的主动回复功能"
            return "当前群聊未开启主动回复，无需操作"

        elif action == "status":
            is_enabled = proactive_mgr.is_group_in_whitelist(group_id)
            return f"当前群聊主动回复状态：{'已开启' if is_enabled else '未开启'}"

        return "用法：/iris proactive <on|off|status|list>"

    async def _handle_iris_activity(self, event: AstrMessageEvent, parsed: Any) -> str:
        """活跃度状态"""
        group_id = get_group_id(event)
        sub_cmd = parsed.args[1].lower() if len(parsed.args) > 1 else ""

        provider = self._service.activity_provider
        if not provider or not provider.enabled:
            return "场景自适应系统未启用"

        level_labels = {
            "quiet": "🌙 安静", "moderate": "☀️ 中等",
            "active": "🔥 活跃", "intensive": "⚡ 超活跃",
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
                lines.append(f"  群 {s['group_id']}: {label} ({s['messages_per_hour']:.0f} 条/时)")
            return "\n".join(lines)

        if not group_id:
            return "此指令仅限群聊使用"

        summary = provider.get_group_activity_summary(group_id)
        level = summary["activity_level"]
        label = level_labels.get(level, level)
        cfg = summary["config"]

        return "\n".join([
            f"📊 当前群活跃度：{label}",
            f"消息频率：约 {summary['messages_per_hour']:.0f} 条/小时",
            "",
            "当前自适应配置：",
            f"  • 主动回复冷却：{cfg.get('proactive_reply.cooldown_seconds', '?')}秒",
            f"  • 每日回复上限：{cfg.get('proactive_reply.max_daily_replies', '?')}次",
            f"  • 批处理阈值：{cfg.get('message_processing.batch_threshold_count', '?')}条",
            f"  • 处理间隔：{cfg.get('message_processing.batch_threshold_interval', '?')}秒",
            f"  • 图片分析预算：{cfg.get('image_analysis.daily_budget', '?')}次/日",
            f"  • 上下文条数：{cfg.get('advanced.chat_context_count', '?')}条",
        ])

    async def _handle_iris_reset(self, event: AstrMessageEvent, delete_kv_func: Any) -> str:
        """重置所有数据"""
        if not self._perms.is_admin(event):
            return ErrorMessages.ADMIN_REQUIRED

        parsed = CommandParser.parse(event.message_str, CommandPrefix.IRIS)
        args = parsed.args[1:] if len(parsed.args) > 1 else []

        if "confirm" not in args:
            return (
                "⚠️ 警告：此操作将永久删除所有 Iris Memory 数据！\n"
                "包括：记忆、用户画像、会话记录、群成员信息、聊天记录等\n"
                "请使用 '/iris reset confirm' 确认操作"
            )

        keys_to_delete = [
            KVStoreKeys.SESSIONS, KVStoreKeys.LIFECYCLE_STATE, KVStoreKeys.BATCH_QUEUES,
            KVStoreKeys.CHAT_HISTORY, KVStoreKeys.USER_PERSONAS, KVStoreKeys.MEMBER_IDENTITY,
            KVStoreKeys.GROUP_ACTIVITY, KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
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

        result = f"✅ 已重置 Iris Memory 数据\n"
        result += f"- 成功清理 {deleted_count}/{len(keys_to_delete)} 个存储键\n"
        result += f"- 成功清理 {db_deleted_count} 条数据库记忆\n"
        if failed_keys:
            result += f"- 失败 {len(failed_keys)} 项（查看日志了解详情）\n"
        result += "\n📌 建议：重启 AstrBot 以确保所有缓存已清空"
        return result

    async def _handle_iris_cooldown(self, event: AstrMessageEvent, parsed: Any) -> str:
        """群冷却控制"""
        from iris_memory.core.constants import CooldownMessages
        from iris_memory.cooldown.cooldown_manager import parse_duration

        group_id = get_group_id(event)
        if not group_id:
            return CooldownMessages.GROUP_ONLY

        cooldown_mgr = getattr(getattr(self._service, 'cooldown', None), 'cooldown_manager', None)
        if cooldown_mgr is None:
            return "冷却模块未初始化"

        sub_cmd = parsed.args[1].lower() if len(parsed.args) > 1 else ""

        if sub_cmd == "status":
            info = cooldown_mgr.get_cooldown_info(group_id)
            if info["active"]:
                return f"冷却中，剩余 {info['remaining_minutes']:.0f} 分钟"
            return "当前群聊未在冷却中"

        if sub_cmd == "off":
            cooldown_mgr.clear_cooldown(group_id)
            return "已取消群冷却"

        duration_arg = parsed.args[1] if len(parsed.args) > 1 else ""
        if duration_arg in ("status", "off", ""):
            duration_arg = str(NumericDefaults.DEFAULT_COOLDOWN_MINUTES)

        duration = parse_duration(duration_arg)
        if duration is None:
            return CooldownMessages.INVALID_DURATION

        cooldown_mgr.set_cooldown(group_id, duration)
        return f"已开启群冷却 {duration} 分钟"

    async def _handle_iris_persona(self, event: AstrMessageEvent, put_kv_func: Any) -> str:
        """用户画像管理"""
        parsed = CommandParser.parse(event.message_str, CommandPrefix.IRIS)
        action = parsed.args[1].lower() if len(parsed.args) > 1 else ""

        if action == "" or action == "status":
            user_id = event.get_sender_id()
            persona = self._service._user_personas.get(user_id)
            if persona:
                update_count = getattr(persona, "update_count", 0)
                interests = getattr(persona, "interests", {}) or {}
                return f"📊 当前用户画像状态：\n- 用户ID: {user_id}\n- 更新次数: {update_count}\n- 兴趣数量: {len(interests)}"
            return f"用户 {user_id} 暂无画像数据"

        if action == "reset":
            target_user_id = parsed.args[2] if len(parsed.args) > 2 else None
            if target_user_id:
                if not self._perms.is_admin(event):
                    return ErrorMessages.ADMIN_REQUIRED
                user_id = target_user_id
            else:
                user_id = event.get_sender_id()

            deleted = self._service._shared_state.delete_user_persona(user_id)
            if deleted:
                if put_kv_func:
                    await self._service.save_to_kv(put_kv_func)
                return f"已重置用户 {user_id} 的画像"
            return f"用户 {user_id} 的画像不存在，无需重置"

        if action == "clear":
            target = parsed.args[2].lower() if len(parsed.args) > 2 else ""
            if target != "all":
                return "用法：/iris persona clear all\n此操作需要管理员权限"
            if not self._perms.is_admin(event):
                return ErrorMessages.ADMIN_REQUIRED

            count = self._service._shared_state.clear_all_user_personas()
            if put_kv_func:
                await self._service.save_to_kv(put_kv_func)
            return f"已清空所有用户画像，共 {count} 个"

        return (
            "用法：/iris persona <action>\n"
            "- status 或留空  - 查看当前用户画像状态\n"
            "- reset          - 重置当前用户的画像\n"
            "- reset <id>     - 重置指定用户的画像（管理员）\n"
            "- clear all      - 清空所有用户画像（超管）"
        )
