"""主动回复工具行为测试：add_follow_up / end_follow_up / set_cooldown。

通过真实 ContextWrapper 调用 call()，与 6 个记忆工具的执行路径一致。
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from astrbot.core.agent.run_context import ContextWrapper

from iris_memory.proactive.config import ConfigManager
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.tools import ToolContext
from iris_memory.tools import AddFollowUpTool, EndFollowUpTool, SetCooldownTool


@pytest.fixture
def state():
    return StateManager(ConfigManager({}, hidden_get={}.get))


@pytest.fixture
def tool_context():
    ctx = ToolContext()
    yield ctx
    ctx.clear_context()


def make_run_context(group_id: str = "event_group") -> ContextWrapper:
    """构造与 AstrBot 执行分支一致的 ContextWrapper（context.context.event）。"""
    event = Mock()
    event.get_group_id = Mock(return_value=group_id)
    return ContextWrapper(context=SimpleNamespace(event=event))


# ── add_follow_up ──


@pytest.mark.asyncio
async def test_add_follow_up_without_group_context(state, tool_context):
    tool = AddFollowUpTool(state, tool_context)
    result = await tool.call(make_run_context(group_id=""), user_ids="u1")
    assert result == "error: no group context"


@pytest.mark.asyncio
async def test_add_follow_up_with_empty_user_ids(state, tool_context):
    tool = AddFollowUpTool(state, tool_context)
    assert (
        await tool.call(make_run_context(), user_ids="")
        == "error: must provide at least one user_id"
    )
    assert (
        await tool.call(make_run_context(), user_ids=" , ,")
        == "error: must provide at least one user_id"
    )
    # 缺省参数与空字符串行为一致
    assert (
        await tool.call(make_run_context())
        == "error: must provide at least one user_id"
    )


@pytest.mark.asyncio
async def test_add_follow_up_with_too_many_users(state, tool_context):
    tool = AddFollowUpTool(state, tool_context)
    user_ids = ",".join(f"u{i}" for i in range(11))
    result = await tool.call(make_run_context(), user_ids=user_ids)
    assert result == "error: too many user_ids (max 10 per call)"


@pytest.mark.asyncio
async def test_add_follow_up_success(state, tool_context):
    tool = AddFollowUpTool(state, tool_context)
    result = await tool.call(make_run_context(), user_ids="u1, u2")
    assert result == "ok: following users=['u1', 'u2']"
    assert state.get_anchor("event_group").participants == {"u1", "u2"}


# ── end_follow_up ──


@pytest.mark.asyncio
async def test_end_follow_up_without_group_context(state, tool_context):
    tool = EndFollowUpTool(state, tool_context)
    result = await tool.call(make_run_context(group_id=""), user_ids="u1")
    assert result == "error: no group context"


@pytest.mark.asyncio
async def test_end_follow_up_specific_users(state, tool_context):
    add_tool = AddFollowUpTool(state, tool_context)
    await add_tool.call(make_run_context(), user_ids="u1,u2,u3")

    end_tool = EndFollowUpTool(state, tool_context)
    result = await end_tool.call(make_run_context(), user_ids="u1, u2")
    assert result == "ok: removed follow-up users=['u1', 'u2']"
    assert state.get_anchor("event_group").participants == {"u3"}


@pytest.mark.asyncio
async def test_end_follow_up_empty_removes_all(state, tool_context):
    add_tool = AddFollowUpTool(state, tool_context)
    await add_tool.call(make_run_context(), user_ids="u1,u2")

    end_tool = EndFollowUpTool(state, tool_context)
    result = await end_tool.call(make_run_context(), user_ids="")
    assert result == "ok: removed follow-up users=None"
    assert not state.get_anchor("event_group").participants


# ── set_cooldown ──


@pytest.mark.asyncio
async def test_set_cooldown_without_group_context(state, tool_context):
    tool = SetCooldownTool(state, tool_context)
    result = await tool.call(make_run_context(group_id=""), minutes=5)
    assert result == "error: no group context"


@pytest.mark.asyncio
async def test_set_cooldown_returns_actual_minutes(state, tool_context):
    tool = SetCooldownTool(state, tool_context)
    result = await tool.call(make_run_context(), minutes=10)
    assert result == "ok: cooldown set for 10 minutes"


@pytest.mark.asyncio
async def test_set_cooldown_default_minutes(state, tool_context):
    tool = SetCooldownTool(state, tool_context)
    result = await tool.call(make_run_context())
    assert result == "ok: cooldown set for 5 minutes"


@pytest.mark.asyncio
async def test_set_cooldown_clamped_by_state_manager(state, tool_context):
    tool = SetCooldownTool(state, tool_context)
    result = await tool.call(make_run_context(), minutes=999)
    assert result == "ok: cooldown set for 120 minutes"


# ── ToolContext 群 ID 优先级 ──


@pytest.mark.asyncio
async def test_tool_context_group_id_takes_precedence(state, tool_context):
    tool_context.set_context("ctx_group")
    tool = AddFollowUpTool(state, tool_context)
    result = await tool.call(make_run_context(group_id="event_group"), user_ids="u1")
    assert result == "ok: following users=['u1']"
    assert state.get_anchor("ctx_group").participants == {"u1"}


@pytest.mark.asyncio
async def test_fallback_to_event_group_id(state, tool_context):
    tool = AddFollowUpTool(state, tool_context)
    assert tool_context.current_group_id is None
    result = await tool.call(make_run_context(group_id="event_group"), user_ids="u1")
    assert result == "ok: following users=['u1']"
    assert state.get_anchor("event_group").participants == {"u1"}
