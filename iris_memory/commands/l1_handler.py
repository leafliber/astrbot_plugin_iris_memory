"""
Iris Chat Memory - L1 指令处理器

处理 L1 消息缓冲的管理指令。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.platform import get_adapter
from .base import CommandHandler, CommandResult, ParsedArgs, DeleteScope

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.l1")


class L1CommandHandler(CommandHandler):
    """L1 消息缓冲指令处理器

    支持的子指令：
    - clear: 清空消息缓冲
    - stats: 查看统计信息
    """

    @property
    def name(self) -> str:
        return "l1"

    @property
    def description(self) -> str:
        return "L1 消息缓冲管理"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "clear": "清空消息缓冲",
            "stats": "查看统计信息",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理 L1 指令"""

        if sub_command == "stats" or sub_command is None:
            return await self._handle_stats(event, args)
        elif sub_command == "clear":
            return await self._handle_clear(event, args)
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

        l1_buffer = manager.get_component("l1_buffer", L1Buffer)
        if not l1_buffer or not l1_buffer.is_available:
            return CommandResult(success=False, message="L1 缓冲组件不可用")

        stats = l1_buffer.get_stats()

        message = (
            "📊 L1 消息缓冲统计\n"
            f"队列数量: {stats['queue_count']}\n"
            f"消息总数: {stats['total_messages']}\n"
            f"Token 总数: {stats['total_tokens']}"
        )

        return CommandResult(success=True, message=message, details=stats)

    async def _handle_clear(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理清空操作"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        l1_buffer = manager.get_component("l1_buffer", L1Buffer)
        if not l1_buffer or not l1_buffer.is_available:
            return CommandResult(success=False, message="L1 缓冲组件不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        # L1 队列键使用会话 ID：私聊为 private:{user_id}，
        # 使私聊的 /l1 clear 只清空当前私聊会话的队列
        session_id = adapter.get_session_id(event)
        current_user_id = adapter.get_user_id(event)

        scope = args.scope
        removed_count = 0

        if scope == DeleteScope.ALL:
            removed_count = l1_buffer.clear_all()
            message = f"✅ 已清空所有 L1 消息缓冲，共删除 {removed_count} 条消息"

        elif scope == DeleteScope.GROUP:
            removed_count = l1_buffer.clear_by_group(session_id)
            message = f"✅ 已清空当前会话的 L1 消息缓冲，共删除 {removed_count} 条消息"

        elif scope == DeleteScope.SPECIFIED_USER:
            target_user_id = args.target_user_id
            if not target_user_id:
                return CommandResult(success=False, message="无法获取目标用户 ID")

            removed_count = l1_buffer.clear_by_user(target_user_id, group_id)
            message = f"✅ 已清空用户 {args.target_user_name or target_user_id} 在当前群聊的 L1 消息，共删除 {removed_count} 条"

        else:
            removed_count = l1_buffer.clear_by_user(current_user_id, group_id)
            message = f"✅ 已清空你的 L1 消息缓冲，共删除 {removed_count} 条消息"

        logger.info(f"L1 clear 操作: scope={scope.value}, removed={removed_count}")

        return CommandResult(
            success=True,
            message=message,
            details={"removed_count": removed_count, "scope": scope.value},
        )
