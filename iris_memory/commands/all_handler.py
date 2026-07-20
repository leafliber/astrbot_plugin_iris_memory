"""
Iris Chat Memory - All 指令处理器

总删除开关，同时操作所有层级（L1、L2、L3、画像）。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.l1_buffer.buffer import L1Buffer
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.profile.storage import ProfileStorage
from iris_memory.platform import get_adapter
from .base import CommandHandler, CommandResult, ParsedArgs, DeleteScope

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.all")


class AllCommandHandler(CommandHandler):
    """总删除指令处理器

    同时操作所有层级：L1、L2、L3、画像

    支持的子指令：
    - clear: 清空所有层级记忆
    """

    @property
    def name(self) -> str:
        return "all"

    @property
    def description(self) -> str:
        return "总删除开关（同时操作 L1、L2、L3、画像）"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "clear": "清空所有层级记忆（含画像）",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理 all 指令"""

        if sub_command == "clear":
            return await self._handle_clear(event, args)
        elif sub_command == "help":
            return CommandResult(success=True, message=self.get_help_text())
        else:
            return CommandResult(
                success=False,
                message=f"未知的子指令: {sub_command}\n{self.get_help_text()}",
            )

    async def _handle_clear(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理清空操作"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        # L1 队列键使用会话 ID（私聊为 private:{user_id}）；
        # L2/L3/画像仍使用原始群 ID，保持既有归属行为
        session_id = adapter.get_session_id(event)
        current_user_id = adapter.get_user_id(event)

        scope = args.scope
        results = {
            "l1": 0,
            "l2": 0,
            "l3": 0,
            "profile_user": 0,
            "profile_group": False,
        }

        l1_buffer = manager.get_component("l1_buffer", L1Buffer)
        l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
        l3_adapter = manager.get_component("l3_kg", L3KGAdapter)
        profile_storage = manager.get_component("profile", ProfileStorage)

        if scope == DeleteScope.ALL:
            results["l1"] = (
                l1_buffer.clear_all() if l1_buffer and l1_buffer.is_available else 0
            )
            results["l2"] = (
                await l2_adapter.delete_all()
                if l2_adapter and l2_adapter.is_available
                else 0
            )
            results["l3"] = (
                await l3_adapter.delete_all()
                if l3_adapter and l3_adapter.is_available
                else 0
            )

            if profile_storage and profile_storage.is_available:
                profile_result = await profile_storage.delete_all_profiles()
                results["profile_user"] = profile_result.get("user_profiles", 0)
                results["profile_group_count"] = profile_result.get("group_profiles", 0)

            message = (
                "✅ 已清空所有记忆和画像\n"
                f"L1 消息缓冲: {results['l1']} 条\n"
                f"L2 记忆库: {results['l2']} 条\n"
                f"L3 知识图谱: {results['l3']} 个节点\n"
                f"用户画像: {results['profile_user']} 个\n"
                f"群聊画像: {results.get('profile_group_count', 0)} 个"
            )

        elif scope == DeleteScope.GROUP:
            results["l1"] = (
                l1_buffer.clear_by_group(session_id)
                if l1_buffer and l1_buffer.is_available
                else 0
            )
            results["l2"] = (
                await l2_adapter.delete_by_group(group_id)
                if l2_adapter and l2_adapter.is_available
                else 0
            )
            results["l3"] = (
                await l3_adapter.delete_by_group(group_id)
                if l3_adapter and l3_adapter.is_available
                else 0
            )

            if profile_storage and profile_storage.is_available:
                results[
                    "profile_user"
                ] = await profile_storage.delete_all_user_profiles_in_group(group_id)
                results["profile_group"] = await profile_storage.delete_group_profile(
                    group_id
                )

            message = (
                f"✅ 已清空当前群聊的所有记忆和画像\n"
                f"L1 消息缓冲: {results['l1']} 条\n"
                f"L2 记忆库: {results['l2']} 条\n"
                f"L3 知识图谱: {results['l3']} 个节点\n"
                f"用户画像: {results['profile_user']} 个\n"
                f"群聊画像: {'已删除' if results['profile_group'] else '无'}"
            )

        elif scope == DeleteScope.SPECIFIED_USER:
            target_user_id = args.target_user_id
            if not target_user_id:
                return CommandResult(success=False, message="无法获取目标用户 ID")

            results["l1"] = (
                l1_buffer.clear_by_user(target_user_id, group_id)
                if l1_buffer and l1_buffer.is_available
                else 0
            )
            results["l2"] = (
                await l2_adapter.delete_by_user(target_user_id, group_id)
                if l2_adapter and l2_adapter.is_available
                else 0
            )
            results["l3"] = (
                await l3_adapter.delete_by_user(target_user_id, group_id)
                if l3_adapter and l3_adapter.is_available
                else 0
            )

            if profile_storage and profile_storage.is_available:
                results["profile_group"] = await profile_storage.delete_user_profile(
                    target_user_id, group_id
                )

            user_name = args.target_user_name or target_user_id
            message = (
                f"✅ 已清空用户 {user_name} 的所有记忆和画像\n"
                f"L1 消息缓冲: {results['l1']} 条\n"
                f"L2 记忆库: {results['l2']} 条\n"
                f"L3 知识图谱: {results['l3']} 个节点\n"
                f"用户画像: {'已删除' if results['profile_group'] else '无'}"
            )

        else:
            results["l1"] = (
                l1_buffer.clear_by_user(current_user_id, group_id)
                if l1_buffer and l1_buffer.is_available
                else 0
            )
            results["l2"] = (
                await l2_adapter.delete_by_user(current_user_id, group_id)
                if l2_adapter and l2_adapter.is_available
                else 0
            )
            results["l3"] = (
                await l3_adapter.delete_by_user(current_user_id, group_id)
                if l3_adapter and l3_adapter.is_available
                else 0
            )

            if profile_storage and profile_storage.is_available:
                results["profile_group"] = await profile_storage.delete_user_profile(
                    current_user_id, group_id
                )

            message = (
                "✅ 已清空你的所有记忆和画像\n"
                f"L1 消息缓冲: {results['l1']} 条\n"
                f"L2 记忆库: {results['l2']} 条\n"
                f"L3 知识图谱: {results['l3']} 个节点\n"
                f"用户画像: {'已删除' if results['profile_group'] else '无'}"
            )

        logger.info(f"All clear 操作: scope={scope.value}, results={results}")

        return CommandResult(
            success=True,
            message=message,
            details={"scope": scope.value, "results": results},
        )
