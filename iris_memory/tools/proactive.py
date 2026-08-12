"""主动回复 LLM Tool

将原 main.py 中 3 个 @filter.llm_tool 装饰器工具迁移为显式的
FunctionTool.call() 实现，与记忆侧工具走完全一致的 AstrBot 执行分支。
业务逻辑与装饰器版本保持一致，不在注册重构中改变行为。
"""

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from iris_memory.core import get_logger
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.tools import ToolContext

logger = get_logger("tools")


class _ProactiveTool(FunctionTool[AstrAgentContext]):
    """主动回复工具基类：只注入必要依赖，避免持有整个插件实例。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict,
        state: StateManager,
        tool_context: ToolContext,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._state = state
        self._tool_context = tool_context

    def _get_group_id(self, event) -> str | None:
        return self._tool_context.current_group_id or event.get_group_id()


class AddFollowUpTool(_ProactiveTool):
    """关注指定用户发言的 Tool（原 tool_add_follow_up）"""

    def __init__(self, state: StateManager, tool_context: ToolContext) -> None:
        super().__init__(
            name="add_follow_up",
            description=(
                "当你希望持续关注某些用户的发言时调用此工具。"
                "将在后续消息中匹配指定用户时自动触发回复。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user_ids": {
                        "type": "string",
                        "description": '逗号分隔的用户ID列表，如 "user1,user2"',
                    },
                },
            },
            state=state,
            tool_context=tool_context,
        )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        event = context.context.event
        user_ids = kwargs.get("user_ids", "")
        group_id = self._get_group_id(event)
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        if not uid_list:
            return "error: must provide at least one user_id"

        if len(uid_list) > 10:
            return "error: too many user_ids (max 10 per call)"

        async with self._state.get_lock(group_id):
            self._state.add_anchor_watch(group_id, users=uid_list)
        logger.debug(f"Iris Reply: add_follow_up for group {group_id}, users={uid_list}")
        return f"ok: following users={uid_list}"


class EndFollowUpTool(_ProactiveTool):
    """移除关注用户的 Tool（原 tool_end_follow_up）"""

    def __init__(self, state: StateManager, tool_context: ToolContext) -> None:
        super().__init__(
            name="end_follow_up",
            description=(
                "当你不再需要关注某些用户时调用此工具，移除对应的跟进记录。"
                "不提供参数则移除所有跟进记录。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "user_ids": {
                        "type": "string",
                        "description": '逗号分隔的用户ID列表，如 "user1,user2"',
                    },
                },
            },
            state=state,
            tool_context=tool_context,
        )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        event = context.context.event
        user_ids = kwargs.get("user_ids", "")
        group_id = self._get_group_id(event)
        if not group_id:
            return "error: no group context"

        uid_list = [u.strip() for u in user_ids.split(",") if u.strip()] if user_ids else None

        async with self._state.get_lock(group_id):
            self._state.remove_anchor_watch(group_id, user_ids=uid_list)
        logger.debug(f"Iris Reply: end_follow_up for group {group_id}, users={uid_list}")
        return f"ok: removed follow-up users={uid_list}"


class SetCooldownTool(_ProactiveTool):
    """设置主动回复冷却时间的 Tool（原 tool_set_cooldown）"""

    def __init__(self, state: StateManager, tool_context: ToolContext) -> None:
        super().__init__(
            name="set_cooldown",
            description=(
                "当你认为应该暂时停止主动回复时调用此工具。"
                "设置冷却时间，冷却期间不会主动触发任何回复。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "number",
                        "description": "冷却时间（分钟），范围 1-120，默认 5",
                    },
                },
            },
            state=state,
            tool_context=tool_context,
        )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        event = context.context.event
        minutes = kwargs.get("minutes", 5)
        group_id = self._get_group_id(event)
        if not group_id:
            return "error: no group context"

        async with self._state.get_lock(group_id):
            actual = self._state.set_cooldown(group_id, minutes)
        logger.debug(f"Iris Reply: set_cooldown for group {group_id}, {actual} min")
        return f"ok: cooldown set for {actual} minutes"
