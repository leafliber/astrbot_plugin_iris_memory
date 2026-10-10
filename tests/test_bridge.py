import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from iris_memory.bridge import Bridge
from iris_memory.config import Config
from iris_memory.mapping import content_text, input_message, map_entry
from iris_memory.prompt import render, safe_hints
from iris_memory.store import Inbox


def component(kind, **kwargs):
    return SimpleNamespace(type=kind, **kwargs)


class Event:
    def __init__(
        self, *, platform="onebot", instance="bot1", group="g1", user="u1", message_id="m1"
    ):
        self.platform, self.instance, self.group, self.user = platform, instance, group, user
        self.unified_msg_origin = (
            f"{instance}:{'GroupMessage' if group else 'FriendMessage'}:{group or user}"
        )
        self.message_obj = SimpleNamespace(message_id=message_id, timestamp=1791580000)
        self.messages = [component("plain", text="你好")]
        self.extra, self.sent = {}, []
        self.send_error = False
        self.result = SimpleNamespace(is_llm_result=lambda: True)

    def get_platform_name(self):
        return self.platform

    def get_platform_id(self):
        return self.instance

    def get_group_id(self):
        return self.group

    def get_sender_id(self):
        return self.user

    def get_sender_name(self):
        return "小林"

    def get_session_id(self):
        return self.group or self.user

    def get_messages(self):
        return self.messages

    def get_result(self):
        return self.result

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    async def send(self, chain):
        if self.send_error:
            raise RuntimeError("iris_test_token")
        self.sent.append(chain)


def prepared():
    return {
        "recall_id": "recall-1",
        "persona": {"content": "我是 Iris", "version": 1},
        "memories": [
            {"id": 1, "content": "小林喜欢茶", "reason": "relevant", "belief": 80},
            {"id": 2, "content": "小林住在杭州", "reason": "person_highlight"},
        ],
        "recent_messages": [{"id": 3, "sender_name": "小林", "content": "你好", "unlearned": True}],
        "state": {"activity": "散步"},
        "goals": [{"id": 5, "content": "喝水", "kind": "normal"}],
        "hints": [
            {"code": "model_paused", "message": "DO_NOT_INJECT", "last_error": "DO_NOT_INJECT"}
        ],
    }


class Client:
    def __init__(self):
        self.inputs, self.feedbacks, self.prepares, self.pulls = [], [], [], []
        self.material = prepared()
        self.delay, self.error, self.closed = 0, False, False
        self.page = {"items": [], "next_cursor": 0, "has_more": False}

    async def receive_message(self, entry, message):
        await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError("iris_test_token")
        self.inputs.append((entry, message))
        return {"message_ids": [1], "pending_count": 1}

    async def prepare(self, entry, options):
        self.prepares.append((entry, options))
        if self.error:
            raise RuntimeError("iris_test_token")
        return self.material

    async def feedback(self, recall, ids):
        self.feedbacks.append((recall, list(ids)))

    async def pull_notifications(self, **kwargs):
        self.pulls.append(kwargs)
        return self.page

    async def aclose(self):
        self.closed = True


@pytest.fixture
async def bridge_factory(tmp_path):
    instances = []

    def factory(**kwargs):
        config = replace(Config(token="iris_test_token", reminders_enabled=False), **kwargs)
        client, logs, sends = Client(), [], []

        async def send_active(origin, text):
            sends.append((origin, text))
            return True

        bridge = Bridge(config, client, tmp_path / str(len(instances)), logs.append, send_active)
        bridge.start()
        instances.append(bridge)
        return bridge, client, logs, sends

    yield factory
    for bridge in instances:
        await bridge.close()


@pytest.mark.parametrize(
    "event,expected",
    [
        (Event(), "astrbot:bot1:group:g1"),
        (Event(group=""), "astrbot:bot1:private:u1"),
        (Event(instance="bot2"), "astrbot:bot2:group:g1"),
        (Event(group="群/a"), "astrbot:bot1:group:%E7%BE%A4%2Fa"),
    ],
)
def test_mapping(event, expected):
    assert map_entry(event, Config()).id == expected


def test_mapping_rules_and_group_identity():
    one, two = Event(), Event(user="u2")
    two.unified_msg_origin += "_per_user_session"
    assert map_entry(one, Config()).id == map_entry(two, Config()).id
    custom = Config(entry_prefix="test:", entry_template="$prefix$platform:$kind:$session_id")
    assert map_entry(one, custom).id == "test:onebot:group:g1"
    override = replace(custom, entry_overrides={one.unified_msg_origin: "manual-entry"})
    assert map_entry(one, override).id == "manual-entry"
    long = Event(group="a" * 400)
    assert len(map_entry(long, Config()).id) <= 200
    assert map_entry(long, Config()) == map_entry(long, Config())


def test_identity_quote_media_and_dedupe():
    event = Event()
    event.messages += [
        component("image", url="DO_NOT_TRANSMIT", file="DO_NOT_TRANSMIT"),
        component("reply", sender_nickname="小赵", sender_id="u2", message_str="喝茶吗"),
    ]
    entry = map_entry(event, Config())
    first = input_message(event, entry)
    assert first["content"] == "你好[图片，未理解]"
    assert first["quote_author_account_id"] == "u2"
    assert first["account_id"] == "u1"
    assert first["platform"] == "astrbot:onebot:bot1"
    assert first["dedupe_key"] == input_message(Event(), entry)["dedupe_key"]
    event.set_extra("iris_image_descriptions", ["窗边有猫"])
    assert "宿主理解：窗边有猫" in input_message(event, entry)["content"]
    assert "DO_NOT_TRANSMIT" not in str(input_message(event, entry))


def test_missing_id_does_not_dedupe_distinct_events():
    first, second = Event(message_id=""), Event(message_id="")
    entry = map_entry(first, Config())
    assert input_message(first, entry)["dedupe_key"] == input_message(first, entry)["dedupe_key"]
    assert input_message(first, entry)["dedupe_key"] != input_message(second, entry)["dedupe_key"]


def test_dedupe_survives_explicit_entry_merge():
    first, second = Event(instance="a"), Event(instance="b")
    config = Config(
        entry_overrides={first.unified_msg_origin: "shared", second.unified_msg_origin: "shared"}
    )
    assert (
        input_message(first, map_entry(first, config))["dedupe_key"]
        != input_message(second, map_entry(second, config))["dedupe_key"]
    )


def test_prompt_partitions_and_no_diagnostics():
    result = render(prepared(), [{"id": 4, "content": "喝水"}], Config())
    for text in ("我是 Iris", "小林喜欢茶", "散步", "喝水", "人物要点"):
        assert text in result.text
    assert "DO_NOT_INJECT" not in result.text
    assert result.memory_ids == (1,)
    assert result.reminder_ids == (4,)
    goals_only = render(prepared(), [], Config(prompt_template="$goals"))
    assert '"id":5' in goals_only.text
    assert safe_hints([{"code": "future_diagnostic", "message": "DO_NOT_INJECT"}]) == []


def test_prompt_limit_and_feedback_only_for_whole_visible_records():
    material = prepared()
    material["memories"][0]["content"] = "大" * 20000
    result = render(material, [{"id": 4, "content": "长" * 20000}], Config(max_prompt_chars=700))
    assert len(result.text) <= 700
    assert result.memory_ids == result.reminder_ids == ()
    result = render(prepared(), [], Config(prompt_template="$persona"))
    assert not result.memory_ids


async def test_full_flow_only_successful_output_and_opt_in_feedback(bridge_factory):
    bridge, client, logs, _ = bridge_factory(assume_injected_used=True)
    event = Event()
    req = SimpleNamespace(system_prompt="宿主人设")
    await bridge.receive(event)
    await bridge.prepare(event, req)
    await bridge.prepare(event, req)
    assert len(client.inputs) == len(client.prepares) == 1
    assert event.get_extra("enable_streaming") is False
    assert req.system_prompt.count("Iris 提供以下分区材料") == 1
    assert not client.feedbacks
    await event.send(SimpleNamespace(chain=[component("plain", text="一起喝茶吧")]))
    await bridge.after_sent(event)
    await bridge.queue.join()
    assert client.inputs[1][1]["kind"] == "self_output"
    assert client.inputs[1][1]["content"] == "一起喝茶吧"
    assert client.inputs[1][1]["sender"] == "我"
    assert client.feedbacks == [("recall-1", [1])]
    assert not logs


async def test_failed_send_hook_is_not_evidence(bridge_factory):
    bridge, client, logs, _ = bridge_factory(assume_injected_used=True)
    event = Event()
    await bridge.prepare(event, SimpleNamespace(system_prompt=""))
    event.send_error = True
    with pytest.raises(RuntimeError):
        await event.send(SimpleNamespace(chain=[component("plain", text="未发送")]))
    await bridge.after_sent(event)
    await bridge.queue.join()
    assert len(client.inputs) == 1
    assert not client.feedbacks
    assert logs == ["Iris: send_unconfirmed"]


async def test_default_no_usage_feedback_and_distinct_output_keys(bridge_factory):
    bridge, client, _, _ = bridge_factory()
    event = Event()
    await bridge.prepare(event, SimpleNamespace(system_prompt=""))
    chain = SimpleNamespace(chain=[component("plain", text="好")])
    await event.send(chain)
    await event.send(chain)
    await bridge.after_sent(event)
    await bridge.after_sent(event)
    await bridge.queue.join()
    assert len(client.inputs) == 3
    assert client.inputs[1][1]["dedupe_key"] != client.inputs[2][1]["dedupe_key"]
    assert not client.feedbacks


async def test_non_llm_send_never_claims_memory_usage(bridge_factory):
    bridge, client, _, _ = bridge_factory(assume_injected_used=True)
    event = Event()
    await bridge.prepare(event, SimpleNamespace(system_prompt=""))
    event.result = SimpleNamespace(is_llm_result=lambda: False)
    await event.send(SimpleNamespace(chain=[component("plain", text="命令已执行")]))
    await bridge.queue.join()
    assert len(client.inputs) == 2
    assert not client.feedbacks


async def test_per_request_host_extensions(bridge_factory):
    bridge, client, _, _ = bridge_factory()
    event = Event()
    event.set_extra("iris_participants", ["p1"])
    event.set_extra("iris_known_memory_ids", [3])
    event.set_extra("iris_text", "")
    await bridge.prepare(event, SimpleNamespace(system_prompt=""))
    assert client.prepares[0][1] == {
        "judge": True,
        "judge_budget_seconds": 5.0,
        "memory_limit": 8,
        "token_budget": 1500,
        "goal_limit": 10,
        "recent_limit": 20,
        "participants": ["p1"],
        "known_memory_ids": [3],
        "text": "",
    }


async def test_new_provider_request_gets_new_material_and_no_stale_usage(bridge_factory):
    bridge, client, _, _ = bridge_factory(assume_injected_used=True)
    event = Event()
    first, second = SimpleNamespace(system_prompt="first"), SimpleNamespace(system_prompt="second")
    await bridge.prepare(event, first)
    await bridge.prepare(event, second)
    assert len(client.prepares) == 2
    assert "Iris 提供以下分区材料" in second.system_prompt
    client.error = True
    await bridge.prepare(event, SimpleNamespace(system_prompt="third"))
    await event.send(SimpleNamespace(chain=[component("plain", text="降级回复")]))
    await bridge.queue.join()
    assert not client.feedbacks


async def test_failure_leaves_reply_available_and_logs_no_secrets(bridge_factory):
    bridge, client, logs, _ = bridge_factory()
    client.error = True
    event, req = Event(), SimpleNamespace(system_prompt="original")
    await bridge.prepare(event, req)
    assert req.system_prompt == "original"
    await event.send(SimpleNamespace(chain=[component("plain", text="正常回复")]))
    assert event.sent
    await bridge.queue.join()
    assert logs and "iris_test_token" not in str(logs)


async def test_prepare_budget_and_bounded_queue(bridge_factory):
    bridge, client, logs, _ = bridge_factory(
        prepare_timeout=0.02, request_timeout=0.05, queue_size=1
    )
    client.delay = 10
    req = SimpleNamespace(system_prompt="original")
    await asyncio.wait_for(bridge.prepare(Event(), req), 0.2)
    assert req.system_prompt == "original"
    for i in range(10):
        await bridge.receive(Event(message_id=str(i)))
    assert bridge.queue.qsize() <= 1
    assert "Iris: queue_full" in logs


async def test_context_reminder_persists_until_confirmed_reply(bridge_factory):
    bridge, client, _, _ = bridge_factory()
    client.page = {
        "items": [
            {"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"},
            {"id": 2, "kind": "goal_reminder", "goal_id": 999, "content": "别的目标"},
        ],
        "next_cursor": 2,
        "has_more": False,
    }
    await bridge.poll()
    assert bridge.inbox.cursor == 2
    event, req = Event(), SimpleNamespace(system_prompt="")
    await bridge.prepare(event, req)
    assert "别的目标" not in req.system_prompt
    assert bridge.inbox.count() == 2
    await event.send(SimpleNamespace(chain=[component("plain", text="记得喝水")]))
    assert [n["id"] for n in bridge.inbox.pending()] == [2]
    await bridge.poll()
    assert client.pulls[-1]["after"] == 2
    assert [n["id"] for n in bridge.inbox.pending()] == [2]


async def test_active_reminder_failure_retries_and_success_is_not_goal_completion(bridge_factory):
    bridge, client, _, sends = bridge_factory(
        reminder_mode="active",
        reminder_target="bot1:GroupMessage:g1",
        reminder_entry_id="astrbot:bot1:group:g1",
    )
    client.page = {
        "items": [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}],
        "next_cursor": 1,
        "has_more": False,
    }
    success_sender = bridge.send_active

    async def fail(*args):
        return False

    bridge.send_active = fail
    await bridge.poll()
    assert bridge.inbox.count() == 1
    bridge.send_active = success_sender
    await bridge.poll()
    await bridge.queue.join()
    assert bridge.inbox.count() == 0
    assert sends == [("bot1:GroupMessage:g1", "提醒：喝水")]
    assert client.inputs[0][1]["kind"] == "self_output"
    await bridge.poll()
    assert len(sends) == 1


async def test_prepare_failure_does_not_prevent_later_success(bridge_factory):
    bridge, client, logs, _ = bridge_factory()
    first = Event()
    await bridge.receive(first)
    await bridge.queue.join()
    client.error = True
    request = SimpleNamespace(system_prompt="original")
    await bridge.prepare(first, request)
    assert request.system_prompt == "original"
    client.error = False
    await bridge.prepare(Event(message_id="m2"), request)
    assert "Iris 提供以下分区材料" in request.system_prompt
    assert logs == ["Iris: prepare_degraded"]


async def test_repeated_initialization_does_not_duplicate_workers(bridge_factory):
    bridge, _, _, _ = bridge_factory()
    bridge.start()
    assert len(bridge.tasks) == 1


async def test_checkpoint_namespaces_isolate_token_scopes(bridge_factory):
    bridge, _, _, _ = bridge_factory()
    bridge.inbox.save_page({"items": [], "next_cursor": 7})
    path = bridge.inbox.db.execute("PRAGMA database_list").fetchone()[2]
    assert "iris_test_token" not in path


async def test_active_private_backflow_has_private_entry_kind(bridge_factory):
    bridge, client, _, _ = bridge_factory(
        reminder_mode="active",
        reminder_target="bot1:FriendMessage:u1",
        reminder_entry_id="astrbot:bot1:private:u1",
    )
    client.page = {
        "items": [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}],
        "next_cursor": 1,
        "has_more": False,
    }
    await bridge.poll()
    await bridge.queue.join()
    assert client.inputs[0][1]["entry_kind"] == "private"


def test_inbox_transaction_restart_and_replay(tmp_path):
    path = tmp_path / "inbox.db"
    inbox = Inbox(path)
    page = {"items": [{"id": 1, "kind": "goal_reminder"}], "next_cursor": 1, "has_more": False}
    inbox.save_page(page)
    inbox.close()
    inbox = Inbox(path)
    assert inbox.cursor == 1 and inbox.count() == 1
    inbox.consume([1])
    inbox.save_page(page)
    assert inbox.count() == 0
    with pytest.raises(ValueError):
        inbox.save_page({"items": [{"id": 100}], "next_cursor": 2})
    assert inbox.cursor == 1
    inbox.close()


@pytest.mark.parametrize(
    "values",
    [
        {"entry_prefix": "bad/"},
        {"prompt_template": "$unknown"},
        {"entry_template": "$unknown"},
        {"max_prompt_chars": 1},
        {"reminder_mode": "active"},
        {"entry_overrides": "invalid"},
        {"request_timeout": float("nan")},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValueError):
        Config.from_mapping(values)


def test_token_source_and_repr(monkeypatch):
    monkeypatch.setenv("IRIS_HOST_TOKEN", "iris_test_token")
    config = Config.from_mapping({})
    assert config.token == ""
    assert "iris_test_token" not in repr(config)
    assert (
        Config.from_mapping({"token": "iris_test_config_token"}).token == "iris_test_config_token"
    )


def test_attachment_placeholders():
    assert (
        content_text([component("record"), component("video"), component("file")])
        == "[音频，未理解][视频，未理解][附件]"
    )


async def test_plain_receive_keeps_independent_background_budget(bridge_factory):
    bridge, client, _, _ = bridge_factory(request_timeout=0.02, image_timeout=2)
    client.delay = 0.1
    event = Event()
    await bridge.receive(event)
    assert await asyncio.wait_for(event.get_extra("iris_received"), 0.2) is False
    assert not client.inputs


def test_source_metadata_whitelist_preserves_lifecycle():
    data = prepared()
    data["memories"][0].update(
        lifecycle="active",
        sources=[
            {
                "message_id": 1,
                "entry_id": "astrbot:group",
                "occurred_at": "2026-10-10T00:00:00+00:00",
                "last_error": "DO_NOT_INJECT_SOURCE",
                "content": "DO_NOT_INJECT_SOURCE",
            }
        ],
    )
    rendered = render(data, [], Config())
    assert '"lifecycle":"active"' in rendered.text
    assert '"message_id":1' in rendered.text
    assert "DO_NOT_INJECT_SOURCE" not in rendered.text


async def test_active_markdown_backflow_matches_actual_text(bridge_factory):
    bridge, client, _, sends = bridge_factory(
        reminder_mode="active",
        reminder_target="bot1:GroupMessage:g1",
        reminder_entry_id="astrbot:g1",
    )
    client.page = {
        "items": [{"id": 1, "kind": "goal_reminder", "content": "**喝水**"}],
        "next_cursor": 1,
        "has_more": False,
    }
    await bridge.poll()
    await bridge.queue.join()
    assert sends[0][1] == client.inputs[0][1]["content"] == "提醒：喝水"
