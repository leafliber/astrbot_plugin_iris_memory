"""Output provenance and notice identity regressions (synthetic platform events)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_memory.config import Config
from iris_memory.mapping import (
    SenderNameCache,
    input_message,
    map_entry,
    other_output_message,
    output_message,
    stable_key,
)
from tests.conftest import Result
from tests.test_bridge import Event, component
from tests.test_bridge import bridge_factory as bridge_factory
from tests.test_proactive import enable, settle
from tests.test_proactive import setup as setup

PREFIX = "［机器人账号发送，非角色发言］"


def notice_event(kind="poke", *, name="", **kwargs):
    event = Event(user="10001", **kwargs)
    event.get_sender_name = lambda: name
    raw = {"post_type": "notice", "notice_type": kind, "user_id": 10001}
    if kind == "poke":
        raw.update(notice_type="notify", sub_type="poke", target_id=10002)
    elif kind == "group_request":
        raw = {"post_type": "request", "request_type": "group", "user_id": 10001}
    event.message_obj.raw_message = raw
    event.messages = []
    return event


def test_capture_config_default_schema_and_validation():
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert Config.from_mapping({}).other_output_capture == "event"
    assert schema["other_output_capture"]["default"] == "event"
    assert schema["other_output_capture"]["options"] == ["event", "off"]
    for value in ("event", "off"):
        assert Config.from_mapping({"other_output_capture": value}).other_output_capture == value
    for value in ("self_output", "EVENT", "", True, None, [], {}):
        with pytest.raises(ValueError, match="capture"):
            Config.from_mapping({"other_output_capture": value})


@pytest.mark.parametrize("mode", ["event", "off"])
async def test_other_output_then_llm_keeps_confirmation_feedback_and_reminders(
    bridge_factory, mode
):
    bridge, client, logs, _ = bridge_factory(other_output_capture=mode, assume_injected_used=True)
    bridge.inbox.save_page(
        {
            "items": [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}],
            "next_cursor": 1,
        }
    )
    event = Event()
    await bridge.prepare(event, SimpleNamespace(system_prompt=""))
    event.result = Result("其他插件的复读", llm=False)
    await event.send(event.result)
    await bridge.after_sent(event)
    await bridge.after_sent(event)
    await bridge.queue.join()
    assert event.get_extra("iris_confirmed_send") is True
    assert not logs and not client.feedbacks and bridge.inbox.count() == 1
    assert len(client.inputs) == (2 if mode == "event" else 1)
    if mode == "event":
        message = client.inputs[-1][1]
        assert message["kind"] == "event" and message["sender"] == "场景"
        assert message["content"] == PREFIX + "其他插件的复读"
        assert "account_id" not in message
    event.result = Result("记得喝水", llm=True)
    await event.send(event.result)
    await bridge.queue.join()
    message = client.inputs[-1][1]
    assert message["kind"] == "self_output" and message["sender"] == "我"
    assert message["content"] == "记得喝水"
    assert client.feedbacks == [("recall-1", [1])]
    assert bridge.inbox.count() == 0
    keys = [m["dedupe_key"] for _, m in client.inputs]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("source", ["none", "missing_marker", "bad_marker", "bad_getter", "truthy"])
@pytest.mark.parametrize("mode", ["event", "off"])
async def test_unknown_provenance_is_never_self_output(bridge_factory, source, mode):
    bridge, client, logs, _ = bridge_factory(other_output_capture=mode)
    event = Event()

    def broken():
        raise RuntimeError("classification unavailable")

    event.result = {
        "none": None,
        "missing_marker": SimpleNamespace(),
        "bad_marker": SimpleNamespace(is_llm_result=broken),
        "bad_getter": None,
        "truthy": SimpleNamespace(is_llm_result=lambda: "unknown"),
    }[source]
    if source == "bad_getter":
        event.get_result = broken
    await bridge.receive(event)
    await event.send(Result("命令输出"))
    await bridge.queue.join()
    assert not logs
    assert event.get_extra("iris_confirmed_send") is True
    assert len(client.inputs) == (2 if mode == "event" else 1)
    if mode == "event":
        assert client.inputs[-1][1]["kind"] == "event"
        assert client.inputs[-1][1]["content"] == PREFIX + "命令输出"


@pytest.mark.parametrize("llm", [False, True])
async def test_provenance_and_content_snapshot_precede_adapter_mutation(bridge_factory, llm):
    bridge, client, _, _ = bridge_factory()
    event = Event()
    event.result = Result("原始发送内容", llm=llm)

    async def send(chain):
        chain.chain.clear()
        event.result = Result(llm=not llm)

    event.send = send
    await bridge.receive(event)
    await event.send(event.result)
    await bridge.queue.join()
    output = client.inputs[-1][1]
    assert output["kind"] == ("self_output" if llm else "event")
    assert output["content"] == ("" if llm else PREFIX) + "原始发送内容"


@pytest.mark.parametrize("outcome", ["false", "exception", "cancelled"])
async def test_non_llm_failed_sends_do_not_submit(bridge_factory, outcome):
    bridge, client, _, _ = bridge_factory()
    event = Event()
    event.result = Result("复读")

    async def send(chain):
        if outcome == "exception":
            raise RuntimeError("test send failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return False

    event.send = send
    await bridge.receive(event)
    if outcome == "false":
        assert await event.send(event.result) is False
    else:
        with pytest.raises(RuntimeError if outcome == "exception" else asyncio.CancelledError):
            await event.send(event.result)
    await bridge.queue.join()
    assert len(client.inputs) == 1
    assert not event.get_extra("iris_confirmed_send")


async def test_event_output_preserves_key_format_and_bounded_body(bridge_factory):
    bridge, client, _, _ = bridge_factory()
    event = Event()
    event.result = Result("长" * 20000)
    await bridge.receive(event)
    entry = event.get_extra("iris_entry")
    assert (
        other_output_message(entry, "输出", "same-key")["dedupe_key"]
        == output_message(entry, "输出", "same-key")["dedupe_key"]
    )
    await event.send(event.result)
    await event.send(Result("[图片，未理解]"))
    await bridge.queue.join()
    outputs = [m for _, m in client.inputs[1:]]
    assert outputs[0]["content"].startswith(PREFIX)
    assert len(outputs[0]["content"].encode()) <= 32768
    assert outputs[0]["dedupe_key"] != outputs[1]["dedupe_key"]


@pytest.mark.parametrize(
    "kind,content",
    [
        ("poke", "有成员戳了戳某人"),
        ("group_increase", "有成员加入了群聊"),
        ("group_request", "收到入群申请或邀请"),
        ("group_recall", "群内有消息被撤回"),
        ("future_notice", "发生了一条平台通知"),
    ],
)
@pytest.mark.parametrize("name", ["", "10001", "   ", "N/A", "10002"])
def test_notice_without_usable_name_is_neutral_scene(kind, content, name):
    event = notice_event(kind, name=name)
    # Neither raw text nor a reply attachment may create an account-named person.
    event.message_obj.raw_message["message"] = "10001 戳了戳 10002"
    event.messages = [
        component("plain", text="10001"),
        component("reply", sender_id="10002", sender_nickname="10002", text="10002"),
    ]
    entry = map_entry(event, Config())
    message = input_message(event, entry)
    assert message["kind"] == "event" and message["sender"] == "场景"
    assert message["content"] == content
    assert "account_id" not in message
    assert not any(key.startswith("quote_") for key in message)
    assert "10001" not in json.dumps(message) and "10002" not in json.dumps(message)
    assert message["dedupe_key"] == "in:" + stable_key(entry.id, entry.platform, "m1")


async def test_notice_ingest_never_restores_wire_ids_or_uploads_attachments(
    bridge_factory, monkeypatch
):
    bridge, client, logs, _ = bridge_factory()

    async def unexpected_upload(*args):
        raise AssertionError("notice must not upload adapter-generated attachments")

    monkeypatch.setattr("iris_memory.bridge.attach_images", unexpected_upload)
    for index, components in enumerate([[], [component("image")]]):
        event = notice_event(message_id=f"notice-{index}")
        event.message_obj.raw_message["message"] = "10001 戳了戳 10002"
        event.messages = components
        await bridge.receive(event)
        assert await event.get_extra("iris_received")
    assert not logs
    assert all(m["content"] == "有成员戳了戳某人" for _, m in client.inputs)


def test_poke_component_and_wrapped_raw_data_are_not_chat():
    event = notice_event()
    event.message_obj.raw_message = SimpleNamespace(raw_data=event.message_obj.raw_message)
    assert input_message(event, map_entry(event, Config()))["kind"] == "event"
    event.message_obj.raw_message = {}
    event.messages = [component("poke", id="10002")]
    message = input_message(event, map_entry(event, Config()))
    assert message["kind"] == "event" and message["content"] == "有成员戳了戳某人"


@pytest.mark.parametrize("kind", ["poke", "group_increase", "group_request"])
async def test_notice_uses_latest_observed_nickname_across_same_platform_sessions(
    bridge_factory, kind
):
    bridge, client, _, _ = bridge_factory()
    for index, name in enumerate(["旧昵称", "新昵称"]):
        chat = Event(user="10001", group="other-group", message_id=f"chat-{index}")
        chat.get_sender_name = lambda name=name: name
        await bridge.receive(chat)
    event = notice_event(kind, name="10001")
    await bridge.receive(event)
    await bridge.queue.join()
    notice = client.inputs[-1][1]
    assert notice["kind"] == "message" and notice["sender"] == "新昵称"
    assert notice["account_id"] == "10001"
    assert "10001" not in notice["content"]
    assert not any(key.startswith("quote_") for key in notice)


def test_nickname_cache_is_bounded_and_does_not_cross_account_namespaces():
    cache = SenderNameCache(limit=2)
    config = Config()
    chat = Event(user="10001")
    chat.get_sender_name = lambda: "小林"
    cache.resolve(chat, map_entry(chat, config))
    for other in [notice_event(instance="bot2"), notice_event(platform="telegram")]:
        assert input_message(other, map_entry(other, config), cache)["kind"] == "event"
    for account in ("10002", "10003"):
        event = Event(user=account)
        cache.resolve(event, map_entry(event, config))
    assert len(cache.names) == 2
    event = notice_event()
    assert input_message(event, map_entry(event, config), cache)["kind"] == "event"
    event.get_sender_name = lambda: "刚见到的昵称"
    assert input_message(event, map_entry(event, config), cache)["sender"] == "刚见到的昵称"
    event.get_sender_name = lambda: "10001"
    assert input_message(event, map_entry(event, config), cache)["sender"] == "刚见到的昵称"


@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
def test_qq_official_fallback_names_and_cache_identity_spaces_are_preserved(platform):
    cache = SenderNameCache()
    event = notice_event(platform=platform)
    entry = map_entry(event, Config())
    expected = "成员_" + stable_key("10001")[:8]
    assert input_message(event, entry, cache)["sender"] == expected
    event.get_sender_name = lambda: "群内昵称"
    assert input_message(event, entry, cache)["sender"] == "群内昵称"
    private = notice_event(platform=platform, group="")
    assert input_message(private, map_entry(private, Config()), cache)["sender"] == (
        "用户_" + stable_key("10001")[:8]
    )
    guild = notice_event(platform=platform)
    guild.message_obj.raw_message.update(channel_id="channel", guild_id="guild")
    assert input_message(guild, map_entry(guild, Config()), cache)["sender"] == expected


@pytest.mark.parametrize("mode", ["event", "off"])
async def test_iris_reminder_and_proactive_content_stays_self_output(setup, fake_iris, mode):
    create, event = setup
    plugin, engine, _ = await create(
        other_output_capture=mode,
        reminder_mode="active",
        reminder_target="bot1:GroupMessage:g1",
        reminder_entry_id="astrbot:bot1:group:g1",
    )
    state = await enable(plugin, event())
    await plugin.receive(event("Iris 今天喝什么茶？"))
    await settle(plugin)
    assert len(plugin.context.sent) == 1
    engine.clock.advance(301)
    fake_iris.notices = [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}]
    await plugin.bridge.poll()
    await settle(plugin)
    assert len(plugin.context.sent) == 2
    outputs = [m for m in fake_iris.messages[state.entry_id] if m["kind"] == "self_output"]
    assert [m["content"] for m in outputs] == ["一起喝茶吧", "提醒：喝水"]
    assert all(m["sender"] == "我" for m in outputs)
