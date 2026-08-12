"""Regression tests for AstrBot plugin ownership of all 9 LLM tools."""

from types import SimpleNamespace

import pytest
from astrbot.core.agent.tool import ToolSet
from astrbot.core.star.context import Context

from iris_memory.proactive.config import ConfigManager
from iris_memory.proactive.state import StateManager
from iris_memory.proactive.tools import ToolContext
from iris_memory.tools import EXPECTED_TOOL_NAMES
from main import IrisMemoryPlugin

OWNER_MODULE = IrisMemoryPlugin.__module__


def _make_plugin():
    context = object.__new__(Context)
    context.provider_manager = SimpleNamespace(llm_tools=ToolSet())

    plugin = object.__new__(IrisMemoryPlugin)
    plugin.context = context
    plugin._state = StateManager(ConfigManager({}, hidden_get={}.get))
    plugin._tool_ctx = ToolContext()
    return plugin, context


def test_all_tools_are_owned_by_plugin_module():
    plugin, context = _make_plugin()
    plugin._register_llm_tools()

    tools = context.provider_manager.llm_tools.func_list
    assert len(tools) == 9
    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES
    assert {tool.handler_module_path for tool in tools} == {OWNER_MODULE}
    assert not any(
        tool.handler_module_path.startswith("iris_memory.tools.") for tool in tools
    )


def test_plugin_keeps_registered_tool_references():
    plugin, context = _make_plugin()
    plugin._register_llm_tools()

    registered = plugin._registered_llm_tools
    assert len(registered) == 9
    tools = context.provider_manager.llm_tools.func_list
    assert all(tool in registered for tool in tools)


@pytest.fixture
def dashboard_service():
    """ToolsService 只需 tool_mgr.is_builtin_tool() 即可序列化归属。"""
    from astrbot.core.star import star_map
    from astrbot.dashboard.services.tools_service import ToolsService

    tool_mgr = SimpleNamespace(is_builtin_tool=lambda name: False)
    core_lifecycle = SimpleNamespace(
        provider_manager=SimpleNamespace(llm_tools=tool_mgr)
    )
    service = ToolsService(core_lifecycle)

    star_map[OWNER_MODULE] = SimpleNamespace(
        name="astrbot_plugin_iris_memory",
        display_name=None,
    )
    yield service
    star_map.pop(OWNER_MODULE, None)


def test_dashboard_serializes_plugin_origin(dashboard_service):
    """防止 Dashboard 归属回退为 unknown / unknown。"""
    plugin, context = _make_plugin()
    plugin._register_llm_tools()

    for tool in context.provider_manager.llm_tools.func_list:
        payload = dashboard_service._serialize_tool(tool, config_entries=[])
        assert payload["origin"] == "plugin"
        assert payload["origin_name"] == "astrbot_plugin_iris_memory"
