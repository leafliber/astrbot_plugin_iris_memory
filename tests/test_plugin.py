import base64
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from iris_memory.config import Config
from iris_memory.service import TOOL_NAMES
from tests.conftest import Context, HostEvent, Result
from tests.test_bridge import component


async def command(plugin, action):
    event = HostEvent(message_id="command-" + action)
    event.messages = [component("plain", text="/iris_mem " + action)]
    results = [result async for result in plugin.iris_mem(event)]
    return "".join(item.text for item in results[0].chain)


async def test_http_receive_prepare_send_feedback_image_reminder(plugin_factory, fake_iris):
    plugin = await plugin_factory(assume_injected_used=True)
    event = HostEvent(platform="aiocqhttp")
    event.messages += [
        component(
            "image", url="", file="base64://" + base64.b64encode(b"\x89PNG\r\n\x1a\nTEST").decode()
        )
    ]
    event.set_extra("iris_image_descriptions", ["窗边有猫"])
    fake_iris.notices = [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}]
    await plugin.bridge.poll()
    await plugin.receive(event)
    req = SimpleNamespace(
        system_prompt="AstrBot 原人格", contexts=[{"role": "assistant", "content": "旧历史"}]
    )
    await plugin.before_llm(event, req)
    await plugin.before_llm(event, req)
    assert req.contexts == []
    assert req.system_prompt.count("Iris 提供以下分区材料") == 1
    assert "窗边有猫" in req.system_prompt and "喝水" in req.system_prompt
    assert "DO_NOT_INJECT" not in req.system_prompt
    assert fake_iris.media[0]["entry_id"] == "astrbot:bot1:group:g1"
    assert fake_iris.messages["astrbot:bot1:group:g1"][0]["media_ids"] == ["media-1"]
    await event.send(Result("**一起喝茶**", llm=True))
    await plugin.after_message_sent(event)
    await plugin.bridge.queue.join()
    assert fake_iris.messages["astrbot:bot1:group:g1"][-1]["kind"] == "self_output"
    assert fake_iris.feedback == [{"recall_id": "recall-1", "memory_ids": [1]}]
    assert plugin.bridge.inbox.count() == 0


async def test_failed_send_and_disabled_feedback(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    event = HostEvent()
    await plugin.before_llm(event, SimpleNamespace(system_prompt="", contexts=[]))
    event.send_error = True
    with pytest.raises(RuntimeError):
        await event.send(Result("未发出"))
    await plugin.after_message_sent(event)
    await plugin.bridge.queue.join()
    assert len(fake_iris.messages["astrbot:bot1:group:g1"]) == 1
    assert not fake_iris.feedback
    event.send_error = False
    await event.send(Result("发送成功"))
    await plugin.bridge.queue.join()
    assert not fake_iris.feedback  # Default: injected does not mean used.


async def test_degraded_prepare_keeps_original_context(plugin_factory, fake_iris):
    plugin = await plugin_factory(request_timeout=0.2, prepare_timeout=0.3)
    fake_iris.fail = 503
    event = HostEvent()
    req = SimpleNamespace(system_prompt="原提示", contexts=[{"content": "原历史"}])
    await plugin.before_llm(event, req)
    assert req.system_prompt == "原提示" and req.contexts == [{"content": "原历史"}]
    assert not event.get_extra("iris_prepared")


@pytest.mark.parametrize("name", ["search_memory", "get_profile", "search_knowledge_graph"])
async def test_search_and_profile_are_entry_scoped(name, plugin_factory, fake_iris):
    plugin = await plugin_factory()
    event = HostEvent(group="g2")
    result = await plugin._tool(name, event, query="茶", person="小林")
    assert "小林喜欢茶" in result
    body = fake_iris.requests[-1][2]
    assert body["entry_id"] == "astrbot:bot1:group:g2"
    assert body["people"] == ["小林"]
    assert not fake_iris.messages  # Search is not a fabricated experience.


@pytest.mark.parametrize("name", ["save_memory", "correct_memory"])
async def test_learn_tools_use_received_dialogue(name, plugin_factory, fake_iris):
    plugin = await plugin_factory()
    event = HostEvent()
    result = await plugin._tool(name, event, content="NEVER_INSERT", correction="NEVER_INSERT")
    assert "不代表" in result and "暂停" in result
    assert "NEVER_INSERT" not in json.dumps(fake_iris.messages)
    paths = [r[1] for r in fake_iris.requests]
    assert paths[-2:] == [
        "/api/v1/entries/astrbot:bot1:group:g1/messages",
        "/api/v1/entries/astrbot:bot1:group:g1/learn",
    ]


async def test_follow_up_dedupe_revision_and_entry_guard(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    event = HostEvent()
    await plugin.add_follow_up(event, "周末问候", "2030-10-12", 30)
    await plugin.add_follow_up(event, "周末问候", "2030-10-12", 30)
    assert len(fake_iris.goals) == 1
    goal = fake_iris.goals[1]
    assert goal["entry_id"] == "astrbot:bot1:group:g1"
    fake_iris.goals[2] = {
        "id": 2,
        "state": "open",
        "entry_id": None,
        "revision": 1,
        "content": "全局",
    }
    fake_iris.goals[3] = {
        "id": 3,
        "state": "open",
        "entry_id": "another",
        "revision": 1,
        "content": "别处",
    }
    assert "未找到本入口" in await plugin.end_follow_up(event, 2)
    assert "未找到本入口" in await plugin.end_follow_up(event, 3)
    assert not any(r[0] == "PATCH" for r in fake_iris.requests)
    assert "完成" in await plugin.end_follow_up(event, 1)
    patch = fake_iris.requests[-1]
    assert patch[0] == "PATCH" and patch[2]["expected_revision"] == 1
    assert fake_iris.goals[1]["state"] == "completed"


async def test_tool_switches_do_not_mutate_shared_toolset(plugin_factory, fake_iris):
    plugin = await plugin_factory(tools_enabled=False)

    class ToolSet:
        def __init__(self):
            self.tools = list(TOOL_NAMES) + ["other_plugin"]

        def remove_tool(self, name):
            self.tools = [x for x in self.tools if x != name]

    original = ToolSet()
    req = SimpleNamespace(system_prompt="", contexts=[], func_tool=original)
    await plugin.before_llm(HostEvent(), req)
    assert req.func_tool.tools == ["other_plugin"] and original.tools == list(TOOL_NAMES) + [
        "other_plugin"
    ]
    assert "关闭" in await plugin.search_memory(HostEvent(), "茶")
    assert not any(r[1].endswith("/search") for r in fake_iris.requests)


@pytest.mark.parametrize("name", TOOL_NAMES)
async def test_individual_tool_switch(name, plugin_factory, fake_iris):
    plugin = await plugin_factory(**{name + "_enabled": False})
    assert "关闭" in await plugin._tool(name, HostEvent())
    assert not fake_iris.requests


async def test_commands_and_redacted_authenticated_page(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    assert "token" in await command(plugin, "status")
    assert "小林喜欢茶" in await command(plugin, "search 茶")
    assert "版本：1" in await command(plugin, "persona")
    assert "暂无" in await command(plugin, "goals")
    assert "已请求立即学习" in await command(plugin, "learn")
    for action in ["help", "delete", "edit", "export", "import", "l2 clear"]:
        assert fake_iris.url in await command(plugin, action)
    page = await plugin.page_status()
    assert page["connection"] == "connected" and page["token_configured"] is True
    assert page["entries"] and page["health"]["usage"]["today"]["tokens"] == 42
    assert page["allow_container_host"] is False
    assert "container_host_enabled" not in plugin.warnings
    assert "iris_test_token" not in json.dumps(page)
    plugin.context.config["provider_ltm_settings"]["group_message_max_cnt"] = 12
    plugin.check_host_config()
    assert "group_message_max_cnt" in await command(plugin, "status")
    plugin_module = __import__(plugin.__class__.__module__, fromlist=["request"])
    plugin_module.request.username = None
    assert (await plugin.page_status())["status_code"] == 401


async def test_http_active_reminder_is_not_goal_completion(plugin_factory, fake_iris):
    plugin = await plugin_factory(
        reminder_mode="active",
        reminder_target="bot1:FriendMessage:u1",
        reminder_entry_id="astrbot:bot1:private:u1",
        proactive_quiet_start="00:00",
        proactive_quiet_end="00:00",
    )
    event = HostEvent(group="")
    await plugin.receive(event)
    plugin.proactive.update(event.unified_msg_origin, {"enabled": True})
    fake_iris.notices = [
        {"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"},
        {"id": 2, "kind": "goal_dedup_result"},
        {"id": 3, "kind": "future_kind", "content": "不可发送"},
    ]
    await plugin.bridge.poll()
    await plugin.bridge.queue.join()
    assert len(plugin.context.sent) == 1
    assert plugin.bridge.inbox.count() == 1
    assert plugin.bridge.inbox.pending()[0]["id"] == 3
    assert not any(r[0] == "PATCH" for r in fake_iris.requests)
    assert fake_iris.messages["astrbot:bot1:private:u1"][-1]["kind"] == "self_output"


async def test_pure_mention_standard_pipeline(plugin_factory):
    plugin = await plugin_factory(pure_at_reply=True)
    event = HostEvent()
    event.messages = [component("at", qq="bot-self"), component("plain", text=" ")]
    await plugin.receive(event)
    req = event.get_extra("provider_request")
    assert req.prompt == " " and req.conversation.id == "conversation-1"
    await plugin.before_llm(event, req)
    assert "本入口近期消息" in req.system_prompt
    plugin.context.config["platform_settings"]["empty_mention_waiting"] = True
    other = HostEvent(message_id="m2")
    other.messages = event.messages
    await plugin.receive(other)
    assert other.get_extra("provider_request") is None
    assert "pure_at_waiting_conflict" in plugin.warnings


async def test_disabled_invalid_configuration_and_lifecycle(plugin_module, monkeypatch):
    module, logs = plugin_module
    monkeypatch.setenv("IRIS_HOST_TOKEN", "iris_test_token")
    for config in ({}, {"token": "iris_test_token", "base_url": "invalid"}, {"enabled": False}):
        context = Context()
        plugin = module.IrisMemoryPlugin(context, config)
        assert plugin.bridge is None
        await plugin.initialize()
        assert "未连接" in await command(plugin, "status")
        await plugin.receive(HostEvent())
        await plugin.before_llm(HostEvent(), SimpleNamespace())
        await plugin.terminate()
        assert not context.registered_web_apis
    assert "iris_test_token" not in str(logs)


def test_schema_covers_config_and_minimum_version():
    root = Path(__file__).parents[1]
    schema = json.loads((root / "_conf_schema.json").read_text())
    assert set(schema) == set(Config.__dataclass_fields__)
    assert schema["token"]["secret"] is True and schema["token"]["default"] == ""
    assert schema["assume_injected_used"]["default"] is False
    assert schema["allow_container_host"]["default"] is False
    assert 'astrbot_version: ">=4.28.2,<5"' in (root / "metadata.yaml").read_text()


async def test_proactive_lifecycle_hooks_and_close_order(plugin_module, fake_iris, monkeypatch):
    module, _ = plugin_module
    calls = []

    class Hooks:
        def __init__(self, plugin):
            self.plugin = plugin

        async def start(self):
            assert self.plugin.bridge.tasks
            calls.append("start")

        async def on_message(self, event):
            assert event.get_extra("iris_received") is not None
            calls.append("message")

        async def before_llm(self, event, request):
            assert event.get_extra("iris_prepared") is not None
            calls.append("prepare")

        async def after_sent(self, event):
            calls.append("after")

        async def close(self):
            assert not self.plugin.bridge.closed
            calls.append("close")

    monkeypatch.setattr(module, "create_proactive", Hooks)
    plugin = module.IrisMemoryPlugin(
        Context(),
        {"base_url": fake_iris.url, "token": "iris_test_token", "reminders_enabled": False},
    )
    await plugin.initialize()
    event = HostEvent()
    await plugin.receive(event)
    await plugin.before_llm(event, SimpleNamespace(system_prompt="", contexts=[]))
    await plugin.after_message_sent(event)
    await plugin.terminate()
    assert calls == ["start", "message", "prepare", "after", "close"]
    assert plugin.bridge.closed


async def test_container_http_receive_prepare_backflow_and_warnings(
    plugin_factory, plugin_module, fake_iris, monkeypatch
):
    original_resolver = socket.getaddrinfo

    def resolve(host, *args, **kwargs):
        if host in ("iris", b"iris"):
            host = "127.0.0.1"
        return original_resolver(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    url = f"http://iris:{urlsplit(fake_iris.url).port}"
    plugin = await plugin_factory(base_url=url, allow_container_host=True)
    event = HostEvent()
    await plugin.receive(event)
    req = SimpleNamespace(system_prompt="原提示", contexts=[])
    await plugin.before_llm(event, req)
    assert "本入口近期消息" in req.system_prompt
    await event.send(Result("已回复", llm=True))
    await plugin.bridge.queue.join()
    assert fake_iris.messages["astrbot:bot1:group:g1"][-1]["kind"] == "self_output"
    page = await plugin.page_status()
    assert page["connection"] == "connected" and page["allow_container_host"] is True
    assert page["management_url"] == url and page["token_configured"] is True
    warning = "请求在 Docker 网络中以明文传输令牌，只在可信网络使用"
    assert warning in "".join(page["warnings"])
    assert warning in await command(plugin, "status")
    plugin.check_host_config()
    logs = plugin_module[1]
    assert sum(warning in line for line in logs) == 1
    assert "iris_test_token" not in json.dumps(page) + str(logs)


async def test_container_hostname_rejected_before_client_creation(plugin_module, monkeypatch):
    module, logs = plugin_module

    def unexpected_client(*args, **kwargs):
        pytest.fail("Disabled container mode must not construct an HTTP client")

    monkeypatch.setattr(module, "AsyncIrisClient", unexpected_client)
    plugin = module.IrisMemoryPlugin(
        Context(), {"base_url": "http://iris:8080", "token": "iris_test_token"}
    )
    try:
        assert plugin.bridge is None
        assert "Iris: invalid_configuration" in logs
        assert (await plugin.page_status())["connection"] == "disabled"
        assert "iris_test_token" not in str(logs)
    finally:
        await plugin.terminate()


@pytest.mark.parametrize("url", ["http://iris:8080", "https://iris:8443"])
async def test_container_warning_without_token_and_unavailable_host_settings(
    url, plugin_module, monkeypatch
):
    module, logs = plugin_module
    context = Context()

    def unavailable_config(**kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(context, "get_config", unavailable_config)
    plugin = module.IrisMemoryPlugin(context, {"base_url": url, "allow_container_host": True})
    try:
        assert plugin.bridge is None
        assert "container_host_enabled" in plugin.warnings
        assert "host_settings_unavailable" in plugin.warnings
        page = await plugin.page_status()
        assert page["allow_container_host"] is True and not page["token_configured"]
        assert any("明文传输令牌" in warning for warning in page["warnings"])
        assert any("明文传输令牌" in line for line in logs)
    finally:
        await plugin.terminate()


async def test_container_origin_does_not_follow_redirects(plugin_factory, respx_mock):
    redirect = respx_mock.get("http://iris:8080/api/v1/status").respond(
        302, headers={"Location": "http://other-iris:8080/api/v1/status"}
    )
    other = respx_mock.get("http://other-iris:8080/api/v1/status").respond(200, json={})
    plugin = await plugin_factory(base_url="http://iris:8080", allow_container_host=True)
    assert (await plugin.page_status())["connection"] == "unavailable"
    assert redirect.call_count == 1 and other.call_count == 0
