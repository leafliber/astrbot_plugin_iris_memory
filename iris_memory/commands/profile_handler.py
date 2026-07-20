"""
Iris Chat Memory - Profile 指令处理器

处理画像的管理指令。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.profile.storage import ProfileStorage
from iris_memory.platform import get_adapter
from .base import CommandHandler, CommandResult, ParsedArgs, DeleteScope

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.profile")


class ProfileCommandHandler(CommandHandler):
    """画像指令处理器

    支持的子指令：
    - reset: 重置画像
    - show: 显示画像
    - group: 群聊画像操作
    """

    @property
    def name(self) -> str:
        return "profile"

    @property
    def description(self) -> str:
        return "画像管理"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "reset": "重置用户画像",
            "show": "显示用户画像",
            "group": "群聊画像操作",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理 profile 指令"""

        if sub_command == "show" or sub_command is None:
            return await self._handle_show(event, args)
        elif sub_command == "reset":
            return await self._handle_reset(event, args)
        elif sub_command == "group":
            return await self._handle_group(event, args)
        elif sub_command == "help":
            return CommandResult(success=True, message=self.get_help_text())
        else:
            return CommandResult(
                success=False,
                message=f"未知的子指令: {sub_command}\n{self.get_help_text()}",
            )

    async def _handle_show(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理显示画像"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        profile_storage = manager.get_component("profile", ProfileStorage)
        if not profile_storage or not profile_storage.is_available:
            return CommandResult(success=False, message="画像组件不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        current_user_id = adapter.get_user_id(event)

        target_user_id = args.target_user_id or current_user_id

        profile = await profile_storage.get_user_profile(target_user_id, group_id)

        if not profile:
            return CommandResult(
                success=True,
                message=f"用户 {args.target_user_name or target_user_id} 暂无画像数据",
            )

        lines = [
            f"📋 用户画像: {args.target_user_name or target_user_id}",
            f"群聊: {group_id}",
            "-" * 30,
        ]

        if profile.user_name:
            lines.append(f"昵称: {profile.user_name}")
        if profile.historical_names:
            lines.append(f"曾用名: {', '.join(profile.historical_names)}")
        if profile.personality_tags:
            lines.append(f"性格标签: {', '.join(profile.personality_tags)}")
        if profile.interests:
            lines.append(f"兴趣爱好: {', '.join(profile.interests)}")
        if profile.occupation:
            lines.append(f"职业: {profile.occupation}")
        if profile.language_style:
            lines.append(f"语言风格: {profile.language_style}")
        if profile.bot_relationship:
            lines.append(f"与Bot关系: {profile.bot_relationship}")
        if profile.taboo_topics:
            lines.append(f"禁忌话题: {', '.join(profile.taboo_topics)}")
        if profile.important_events:
            lines.append(f"重要事件: {'; '.join(profile.important_events[:3])}")
        if profile.custom_fields:
            lines.append(f"自定义字段: {list(profile.custom_fields.keys())}")

        lines.append(f"版本: {profile.version}")

        return CommandResult(success=True, message="\n".join(lines))

    async def _handle_reset(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理重置画像"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        profile_storage = manager.get_component("profile", ProfileStorage)
        if not profile_storage or not profile_storage.is_available:
            return CommandResult(success=False, message="画像组件不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)
        current_user_id = adapter.get_user_id(event)

        scope = args.scope

        if scope == DeleteScope.ALL:
            result = await profile_storage.delete_all_profiles()
            message = (
                f"✅ 已重置所有画像\n"
                f"用户画像: {result['user_profiles']} 个\n"
                f"群聊画像: {result['group_profiles']} 个"
            )
            logger.info(f"Profile reset 操作: scope=all, result={result}")

        elif scope == DeleteScope.GROUP:
            user_count = await profile_storage.delete_all_user_profiles_in_group(
                group_id
            )
            group_deleted = await profile_storage.delete_group_profile(group_id)
            message = (
                f"✅ 已重置当前群聊的所有画像\n"
                f"用户画像: {user_count} 个\n"
                f"群聊画像: {'已删除' if group_deleted else '无'}"
            )
            logger.info(f"Profile reset 操作: scope=group, group_id={group_id}")

        elif scope == DeleteScope.SPECIFIED_USER:
            target_user_id = args.target_user_id
            if not target_user_id:
                return CommandResult(success=False, message="无法获取目标用户 ID")

            success = await profile_storage.delete_user_profile(
                target_user_id, group_id
            )
            if success:
                message = (
                    f"✅ 已重置用户 {args.target_user_name or target_user_id} 的画像"
                )
            else:
                message = "❌ 重置用户画像失败"
            logger.info(f"Profile reset 操作: scope=user, user_id={target_user_id}")

        else:
            success = await profile_storage.delete_user_profile(
                current_user_id, group_id
            )
            if success:
                message = "✅ 已重置你的画像"
            else:
                message = "❌ 重置画像失败"
            logger.info(
                f"Profile reset 操作: scope=current_user, user_id={current_user_id}"
            )

        return CommandResult(success=True, message=message)

    async def _handle_group(
        self, event: "AstrMessageEvent", args: ParsedArgs
    ) -> CommandResult:
        """处理群聊画像操作"""
        manager = get_component_manager()
        if not manager:
            return CommandResult(success=False, message="组件管理器不可用")

        profile_storage = manager.get_component("profile", ProfileStorage)
        if not profile_storage or not profile_storage.is_available:
            return CommandResult(success=False, message="画像组件不可用")

        adapter = get_adapter(event)
        group_id = adapter.get_group_id(event)

        raw_args = args.raw_args
        group_sub_command = None

        for arg in raw_args:
            if arg.lower() not in [
                "group", "--group", "-g", "--all", "-a",
                "—group", "—all",  # em-dash 变体（parser.SCOPE_FLAGS 支持）
            ]:
                group_sub_command = arg.lower()
                break

        if group_sub_command == "reset":
            scope = args.scope

            if scope == DeleteScope.ALL:
                count = await profile_storage.delete_all_group_profiles()
                message = f"✅ 已重置所有群聊画像，共 {count} 个"
            else:
                success = await profile_storage.delete_group_profile(group_id)
                if success:
                    message = "✅ 已重置当前群聊画像"
                else:
                    message = "❌ 重置群聊画像失败"

            logger.info(
                f"Profile group reset 操作: group_id={group_id}, scope={scope.value}"
            )
            return CommandResult(success=True, message=message)

        elif group_sub_command == "show" or group_sub_command is None:
            profile = await profile_storage.get_group_profile(group_id)

            if not profile:
                return CommandResult(success=True, message="当前群聊暂无画像数据")

            lines = [
                f"📋 群聊画像: {group_id}",
                "-" * 30,
            ]

            if profile.group_name:
                lines.append(f"群名称: {profile.group_name}")
            if profile.interests:
                lines.append(f"兴趣点: {', '.join(profile.interests)}")
            if profile.atmosphere_tags:
                lines.append(f"氛围标签: {', '.join(profile.atmosphere_tags)}")
            if profile.long_term_tags:
                lines.append(f"长期标签: {', '.join(profile.long_term_tags)}")
            if profile.blacklist_topics:
                lines.append(f"禁忌话题: {', '.join(profile.blacklist_topics)}")
            if profile.custom_fields:
                lines.append(f"自定义字段: {list(profile.custom_fields.keys())}")

            lines.append(f"版本: {profile.version}")

            return CommandResult(success=True, message="\n".join(lines))

        else:
            return CommandResult(
                success=False,
                message=f"未知的群聊画像子指令: {group_sub_command}\n"
                "可用子指令: show, reset",
            )
