"""获取画像 LLM Tool"""

from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.config import get_config
from iris_memory.profile import UserProfileManager, GroupProfileManager
from iris_memory.profile.models import UserProfile, GroupProfile
from iris_memory.profile.storage import ProfileStorage

logger = get_logger("tools")


@dataclass
class GetProfileTool(FunctionTool[AstrAgentContext]):
    """获取画像的Tool

    通过 target_type 参数区分查询用户画像还是群聊画像。
    """

    name: str = "get_profile"
    description: str = (
        "获取用户或群聊的画像信息。"
        "用户画像包含性格、兴趣、禁忌话题等；群聊画像包含群聊兴趣、氛围标签、禁忌话题等。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "target_type": {
                    "type": "string",
                    "description": "查询类型：user（用户画像）或 group（群聊画像），默认user",
                    "default": "user",
                },
                "target_id": {
                    "type": "string",
                    "description": "用户ID或群聊ID（可选，不传则自动获取当前用户/群聊）",
                },
            },
            "required": [],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        try:
            target_type = kwargs.get("target_type", "user").strip().lower()
            target_id = kwargs.get("target_id", "").strip()

            event = context.context.event
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)

            if target_type == "group":
                return await self._get_group_profile(adapter, event, target_id)
            else:
                return await self._get_user_profile(adapter, event, target_id)

        except Exception as e:
            logger.error(f"获取画像失败: {e}", exc_info=True)
            return f"获取画像失败: {str(e)}"

    async def _get_user_profile(self, adapter, event, user_id: str) -> str:
        if not user_id:
            user_id = adapter.get_user_id(event)
            group_id = adapter.get_group_id(event)
        else:
            group_id = "default"

        if not user_id:
            return "无法获取用户ID，请手动指定target_id参数。"

        manager = get_component_manager()
        profile_storage = manager.get_component("profile", ProfileStorage)

        if not profile_storage or not profile_storage.is_available:
            return "画像系统未启用或不可用。"

        config = get_config()
        effective_group_id = (
            group_id
            if config.get("isolation_config.enable_group_isolation")
            else "default"
        )

        from iris_memory.core.persona import resolve_persona

        manager = get_component_manager()
        persona_id = await resolve_persona(manager, event)

        user_manager = UserProfileManager(profile_storage)
        profile = await user_manager.get_or_create(
            user_id, effective_group_id, persona_id
        )

        result = self._format_user_profile(profile)
        logger.info(f"获取用户画像: {user_id} (群聊: {effective_group_id})")
        return result

    async def _get_group_profile(self, adapter, event, group_id: str) -> str:
        if not group_id:
            group_id = adapter.get_group_id(event)

        if not group_id:
            return "无法获取群聊ID，请手动指定target_id参数。"

        manager = get_component_manager()
        profile_storage = manager.get_component("profile", ProfileStorage)

        if not profile_storage or not profile_storage.is_available:
            return "画像系统未启用或不可用。"

        from iris_memory.core.persona import resolve_persona

        manager = get_component_manager()
        persona_id = await resolve_persona(manager, event)

        group_manager = GroupProfileManager(profile_storage)
        profile = await group_manager.get_or_create(group_id, persona_id)

        result = self._format_group_profile(profile)
        logger.info(f"获取群聊画像: {group_id}")
        return result

    def _format_user_profile(self, profile: UserProfile) -> str:
        lines = [
            f"## 用户画像 - {profile.user_name or profile.user_id}",
            "",
            f"**用户昵称**: {profile.user_name or '未知'}",
        ]

        if profile.personality_tags:
            lines.append(f"**用户性格**: {', '.join(profile.personality_tags)}")

        if profile.interests:
            lines.append(f"**用户兴趣**: {', '.join(profile.interests)}")

        if profile.occupation:
            lines.append(f"**职业/身份**: {profile.occupation}")

        if profile.language_style:
            lines.append(f"**语言风格**: {profile.language_style}")

        if profile.communication_style:
            lines.append(f"**沟通偏好**: {profile.communication_style}")

        if profile.emotional_baseline:
            lines.append(f"**情感基线**: {profile.emotional_baseline}")

        if profile.bot_relationship:
            lines.append(f"**用户对你的称呼**: {profile.bot_relationship}")

        if profile.historical_names:
            lines.append(f"**历史曾用ID**: {', '.join(profile.historical_names)}")

        if profile.taboo_topics:
            lines.append(f"**⚠️ 用户禁忌话题**: {', '.join(profile.taboo_topics)}")

        if profile.important_dates:
            dates_str = ", ".join(
                [f"{d['date']}({d['description']})" for d in profile.important_dates]
            )
            lines.append(f"**重要日期**: {dates_str}")

        if profile.important_events:
            lines.append(f"**重要事件**: {', '.join(profile.important_events)}")

        return "\n".join(lines)

    def _format_group_profile(self, profile: GroupProfile) -> str:
        lines = [
            f"## 群聊画像 - {profile.group_name or profile.group_id}",
            "",
            f"**群聊兴趣**: {', '.join(profile.interests) or '暂无'}",
            f"**氛围标签**: {', '.join(profile.atmosphere_tags) or '暂无'}",
            f"**核心特征**: {', '.join(profile.long_term_tags) or '暂无'}",
            "",
            f"**禁忌话题**: {', '.join(profile.blacklist_topics) or '无'}",
        ]
        return "\n".join(lines)
