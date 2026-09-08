import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def plugin_class(monkeypatch, tmp_path):
    def decorator(*args, **kwargs):
        return lambda target: target

    api = ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = logging.getLogger("hook-test")
    event = ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = SimpleNamespace(
        command=decorator,
        llm_tool=decorator,
        on_waiting_llm_request=decorator,
        on_llm_request=decorator,
        on_llm_response=decorator,
        event_message_type=decorator,
        EventMessageType=SimpleNamespace(ALL="all"),
    )
    star = ModuleType("astrbot.api.star")

    class Star:
        def __init__(self, context):
            self.context = context

    star.Star = Star
    star.Context = object
    star.register = decorator
    star.StarTools = SimpleNamespace(get_data_dir=lambda name: tmp_path)
    message = ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text):
            self.text = text
            self.temporary = False

        def mark_as_temp(self):
            self.temporary = True
            return self

    message.TextPart = TextPart
    for name, value in [
        ("astrbot", ModuleType("astrbot")),
        ("astrbot.api", api),
        ("astrbot.api.event", event),
        ("astrbot.api.star", star),
        ("astrbot.core", ModuleType("astrbot.core")),
        ("astrbot.core.agent", ModuleType("astrbot.core.agent")),
        ("astrbot.core.agent.message", message),
    ]:
        monkeypatch.setitem(sys.modules, name, value)
    prefix = ModuleType("iris_test_plugin")
    prefix.__path__ = [str(Path(__file__).parents[1])]
    monkeypatch.setitem(sys.modules, prefix.__name__, prefix)
    return importlib.import_module("iris_test_plugin.main").IrisMemoryPlugin


async def test_hook_preserves_host_prompt_history_tools_and_removes_stale_part(
    plugin_class, identity
):
    plugin = plugin_class(SimpleNamespace(), {})
    extras = {}
    event = SimpleNamespace(
        get_extra=lambda key: extras.get(key),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        message_str="current input",
    )
    plugin.identity = lambda event: identity

    class ToolSet:
        def __init__(self):
            self.tools = [
                SimpleNamespace(name="safe_host_tool"),
                SimpleNamespace(name="iris_task"),
            ]

        def remove_tool(self, name):
            self.tools = [t for t in self.tools if t.name != name]

        def openai_schema(self):
            return [{"name": t.name} for t in self.tools]

    shared = ToolSet()
    other_part = object()
    req = SimpleNamespace(
        func_tool=shared,
        extra_user_content_parts=[other_part],
        system_prompt="Host safety and persona",
        contexts=[{"role": "user", "content": "Host history"}],
        prompt="Current input",
    )
    plan = {"text": "Iris fact", "envelope": {"request_id": "one"}, "selected": ["one"]}
    calls = []

    async def run(*args, **kwargs):
        calls.append((args, kwargs))
        return plan

    control = SimpleNamespace(
        modules={"memory": True, "context": True},
        allowed=lambda i: True,
        run=run,
        logs=SimpleNamespace(emit=lambda *a, **k: None),
    )
    plugin.control = control
    await plugin.inject_context(event, req)
    assert req.system_prompt == "Host safety and persona"
    assert req.contexts == [{"role": "user", "content": "Host history"}]
    assert req.extra_user_content_parts[0] is other_part
    assert req.extra_user_content_parts[1].temporary
    assert [t.name for t in shared.tools] == ["safe_host_tool", "iris_task"]
    assert [t.name for t in req.func_tool.tools] == ["safe_host_tool"]
    await plugin.report_context(event, SimpleNamespace())
    assert calls[-1][0][:2] == ("context", "report") and calls[-1][1]["visible"]
    control.modules = {}
    await plugin.inject_context(event, req)
    assert req.extra_user_content_parts == [other_part]
