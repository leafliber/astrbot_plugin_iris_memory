"""Regression tests for AstrBot plugin ownership of memory tools."""

from types import SimpleNamespace

from astrbot.core.agent.tool import ToolSet
from astrbot.core.star.context import Context

from main import IrisMemoryPlugin


def test_memory_tools_are_registered_under_plugin_module():
    context = object.__new__(Context)
    context.provider_manager = SimpleNamespace(llm_tools=ToolSet())

    plugin = object.__new__(IrisMemoryPlugin)
    plugin.context = context
    plugin._register_llm_tools()

    tools = context.provider_manager.llm_tools.func_list
    assert len(tools) == 6
    assert {tool.handler_module_path for tool in tools} == {
        IrisMemoryPlugin.__module__
    }
    assert not tools[0].handler_module_path.startswith("iris_memory.tools.")
