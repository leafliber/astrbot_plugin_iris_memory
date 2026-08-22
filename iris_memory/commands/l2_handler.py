"""
Iris Chat Memory - L2 指令处理器

处理 L2 记忆库的管理指令。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.platform import get_adapter
from .base import CommandHandler, CommandResult, ParsedArgs, DeleteScope

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.l2")


class L2CommandHandler(CommandHandler):
    """L2 记忆库指令处理器

    支持的子指令：
    - clear: 清空记忆库
    - stats: 查看统计信息
    - restore: 恢复归档记忆
    """

    @property
    def name(self) -> str:
        return "l2"

    @property
    def description(self) -> str:
        return "L2 记忆库管理"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "clear": "清空记忆库",
            "stats": "查看统计信息",
            "restore": "恢复归档记忆（梦境淘汰的记忆 30 天内可恢复）",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理 L2 指令"""

        if sub_command == "stats" or sub_command is None:
            return await self._handle_stats(event, args)
        elif sub_command == "clear":
            return await self._handle_clear(event, args)
        elif sub_command == "restore":
            return await self._handle_restore(event, args)
        elif sub_command == "help":
            return CommandResult(success=True, message=self.get_help_text())
        else:
            return CommandResult(
                success=False,
                message=f"未知的子指令: {sub_command}\n{self.get_help_text()}",
            )

    async def _handle_stats(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理统计查询"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
        if not l2_adapter or not l2_adapter.is_available:
            return CommandResult(success=False, message="L2 记忆库组件不可用")

        stats = await l2_adapter.get_stats()

        message = (
            "📊 L2 记忆库统计\n"
            f"记忆总数: {stats['total_count']}\n"
            f"群聊数量: {stats['group_count']}"
        )

        return CommandResult(success=True, message=message, details=stats)

    async def _handle_clear(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理清空操作"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
        if not l2_adapter or not l2_adapter.is_available:
            return CommandResult(success=False, message="L2 记忆库组件不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        current_user_id = adapter.get_user_id(event)
        # 清除操作需在当前 persona 命名空间内执行，否则非 default
        # persona 的记忆无法被命中（隔离未启用时 resolve 返回 default）
        from iris_memory.core.persona import resolve_persona

        persona_id = await resolve_persona(manager, event)

        scope = args.scope
        removed_count = 0

        if scope == DeleteScope.ALL:
            removed_count = await l2_adapter.delete_all()
            message = f"✅ 已清空所有 L2 记忆，共删除 {removed_count} 条"

        elif scope == DeleteScope.GROUP:
            removed_count = await l2_adapter.delete_by_group(
                group_id, persona_id=persona_id
            )
            message = f"✅ 已清空当前群聊的 L2 记忆，共删除 {removed_count} 条"

        elif scope == DeleteScope.SPECIFIED_USER:
            target_user_id = args.target_user_id
            if not target_user_id:
                return CommandResult(success=False, message="无法获取目标用户 ID")

            removed_count = await l2_adapter.delete_by_user(
                target_user_id, group_id, persona_id=persona_id
            )
            message = f"✅ 已清空用户 {args.target_user_name or target_user_id} 在当前群聊的 L2 记忆，共删除 {removed_count} 条"

        else:
            removed_count = await l2_adapter.delete_by_user(
                current_user_id, group_id, persona_id=persona_id
            )
            message = f"✅ 已清空你的 L2 记忆，共删除 {removed_count} 条"

        logger.info(f"L2 clear 操作: scope={scope.value}, removed={removed_count}")

        return CommandResult(
            success=True,
            message=message,
            details={"removed_count": removed_count, "scope": scope.value},
        )

    async def _handle_restore(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """恢复指定 ID 的归档记忆"""
        memory_id = args.raw_args[1] if len(args.raw_args) > 1 else None
        if not memory_id:
            return CommandResult(
                success=False,
                message="请提供要恢复的记忆 ID，例如：iris_mem l2 restore mem_xxxx",
            )

        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
        if not l2_adapter or not l2_adapter.is_available:
            return CommandResult(success=False, message="L2 记忆库组件不可用")

        restored = await l2_adapter.restore_archived_memory(memory_id)
        if restored:
            logger.info(f"L2 restore 操作: memory_id={memory_id}")
            return CommandResult(
                success=True,
                message=f"✅ 已恢复归档记忆 {memory_id}",
                details={"memory_id": memory_id},
            )
        return CommandResult(
            success=False,
            message=(
                f"❌ 恢复失败：归档中不存在 {memory_id}，"
                "或该 ID 已存在于正式记忆库"
            ),
        )
