import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from iris_memory.config import Config
from iris_memory.proactive.policy import muted
from iris_memory.proactive.state import Session
from iris_memory.prompt import render
from tests.conftest import HostEvent
from tests.test_bridge import component


class Clock:
    def __init__(self):
        self.now = datetime(2030, 1, 1, 12).timestamp()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Random:
    value = 0

    def random(self):
        return self.value

    def uniform(self, low, high):
        return low + (high - low) * self.value


def answer(**changes):
    return {
        "speak": True,
        "content": "**一起喝茶吧**",
        "topic": "茶",
        "why_now": "最近的话题留下了茶的线索",
        "topic_source": "recent_context",
        "cooldown_minutes": 0,
        "watch_goal_ids": [],
        **changes,
    }


class Provider:
    def __init__(self):
        self.calls = []
        self.selections = []
        self.response = answer()
        self.error = False
        self.started = asyncio.Event()
        self.release = None

    async def select(self, origin):
        self.selections.append(origin)
        return "session-provider"

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.release:
            await self.release.wait()
        if self.error:
            raise RuntimeError("iris_test_token")
        raw = (
            self.response
            if isinstance(self.response, str)
            else json.dumps(self.response, ensure_ascii=False)
        )
        return SimpleNamespace(completion_text=raw)


@pytest.fixture
async def setup(plugin_factory):
    sequence = 0

    async def create(**settings):
        plugin = await plugin_factory(
            proactive_quiet_start="00:00",
            proactive_quiet_end="00:00",
            **settings,
        )
        engine = plugin.proactive
        engine.scheduler.cancel()
        await asyncio.gather(engine.scheduler, return_exceptions=True)
        engine.scheduler = None
        engine.clock, engine.rng = Clock(), Random()
        provider = Provider()
        plugin.context.get_current_chat_provider_id = provider.select
        plugin.context.llm_generate = provider.generate
        return plugin, engine, provider

    def event(text="你好", **kwargs):
        nonlocal sequence
        sequence += 1
        result = HostEvent(message_id=str(sequence), **kwargs)
        result.messages = [component("plain", text=text)]
        return result

    return create, event


async def settle(plugin):
    await asyncio.gather(*list(plugin.proactive.tasks))
    await plugin.bridge.queue.join()


async def enable(plugin, event):
    await plugin.receive(event)
    plugin.proactive.update(event.unified_msg_origin, {"enabled": True})
    return plugin.proactive.sessions[event.unified_msg_origin]


async def command(plugin, event):
    results = [r async for r in plugin.iris_reply(event)]
    return "".join(c.text for c in results[0].chain)


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "global_disabled",
        "ordinary",
        "cooldown",
        "quiet",
        "limit",
        "low_question",
        "wake",
        "pure_at",
        "command",
        "self",
        "private",
    ],
)
async def test_local_gate_makes_zero_provider_and_prepare_calls(setup, fake_iris, case):
    create, event = setup
    plugin, engine, provider = await create(proactive_enabled=case != "global_disabled")
    kwargs = {"group": ""} if case == "private" else {}
    seed = event(**kwargs)
    await plugin.receive(seed)
    state = engine.sessions[seed.unified_msg_origin]
    if case != "disabled":
        engine.update(state.origin, {"enabled": True})
    message = event("Iris 你怎么看？", **kwargs)
    if case == "ordinary":
        message.messages = [component("plain", text="路过看看")]
    if case == "cooldown":
        engine.update(state.origin, {"cooldown_minutes": 10})
    if case == "quiet":
        engine.update(state.origin, {"quiet_start": "11:00", "quiet_end": "13:00"})
    if case == "limit":
        state.roll(engine.clock())
        state.calls_today = 20
    if case == "low_question":
        engine.update(state.origin, {"willingness": "低"})
        message.messages = [component("plain", text="晚饭吃什么？")]
    if case == "wake":
        message.is_at_or_wake_command = True
    if case == "pure_at":
        message.set_extra("provider_request", SimpleNamespace())
    if case == "command":
        message.messages = [component("plain", text="/iris_reply status")]
    if case == "self":
        message.user = "bot-self"
    await plugin.receive(message)
    await settle(plugin)
    assert not provider.calls and not plugin.context.sent
    assert not any(r[1].endswith("/prepare") for r in fake_iris.requests)


@pytest.mark.parametrize("signal", ["alias", "question", "mention", "high"])
async def test_chime_single_call_final_text_and_entry_scoped_backflow(setup, fake_iris, signal):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    text = {
        "alias": "Iris 我想到一件事",
        "question": "今天喝什么茶？",
        "mention": "来聊聊",
        "high": "求推荐一杯茶",
    }[signal]
    if signal == "alias":
        engine.update(state.origin, {"willingness": "低"})
    if signal == "high":
        engine.update(state.origin, {"willingness": "高"})
    message = event(text)
    if signal == "mention":
        message.messages.append(component("at", qq="bot-self"))
    await plugin.receive(message)
    await settle(plugin)
    assert len(provider.calls) == len(plugin.context.sent) == 1
    call = provider.calls[0]
    assert call["chat_provider_id"] == "session-provider"
    assert provider.selections == [state.origin]
    assert call["contexts"] == [] and call["tools"] is None
    assert json.loads(call["prompt"])["motive"] == "chime_in"
    assert all(word in call["system_prompt"] for word in ("我是 Iris", "小林喜欢茶", "喝水", text))
    assert "DO_NOT_INJECT" not in call["system_prompt"]
    assert plugin.context.sent[0][1].chain[0].text == "一起喝茶吧"
    output = fake_iris.messages[state.entry_id][-1]
    assert output["kind"] == "self_output" and output["sender"] == "我"
    assert output["dedupe_key"].startswith("out:")
    assert state.calls == state.sends == 1
    assert state.sends_by_motive == {"chime_in": 1}
    assert not fake_iris.feedback  # Same default as PL-1: injection is not usage evidence.
    await plugin.receive(event("Iris 再问一次"))
    await settle(plugin)
    assert len(provider.calls) == 1  # Success cooldown blocks another call.


async def test_provider_override_feedback_and_model_cooldown(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_provider_id="chosen-provider", assume_injected_used=True
    )
    state = await enable(plugin, event())
    provider.response = answer(cooldown_minutes=12)
    await plugin.receive(event("Iris 看看这里"))
    await settle(plugin)
    assert provider.calls[0]["chat_provider_id"] == "chosen-provider" and not provider.selections
    assert fake_iris.feedback == [{"recall_id": "recall-1", "memory_ids": [1]}]
    assert state.cooldown_until == engine.clock() + 12 * 60


@pytest.mark.parametrize(
    "profile",
    [
        "missing",
        None,
        {
            "text": "大家爱聊园艺",
            "version": 2,
            "generated_at": "2030-01-01T00:00:00Z",
            "possibly_stale": True,
            "diagnostic": "DO_NOT_INJECT",
        },
    ],
)
async def test_entry_profile_optional_in_normal_and_proactive_prepare(setup, fake_iris, profile):
    create, event = setup
    plugin, engine, provider = await create()
    if profile != "missing":
        fake_iris.prepared["entry_profile"] = profile
    state = await enable(plugin, event())
    message = event("Iris 你有什么想法")
    await plugin.receive(message)
    await settle(plugin)
    req = SimpleNamespace(system_prompt="原提示", contexts=[])
    await plugin.before_llm(event(), req)
    proactive_prompt = provider.calls[0]["system_prompt"]
    for prompt in (req.system_prompt, proactive_prompt):
        assert "DO_NOT_INJECT" not in prompt
        if isinstance(profile, dict):
            assert '"text":"大家爱聊园艺"' in prompt and '"possibly_stale":true' in prompt
        else:
            assert "入口画像（可能过时）：\n" in prompt
    assert state.sends == 1


@pytest.mark.parametrize("profile", [None, {}])
def test_entry_profile_custom_template_empty_or_projected(profile):
    cfg = Config.from_mapping({"prompt_template": "[$entry_profile]"})
    data = {} if profile == {} else {"entry_profile": None}
    assert render(data, [], cfg).text == "[]"


async def test_follow_up_goals_dedupe_matching_end_and_scope(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create()
    seed = event()
    state = await enable(plugin, seed)
    await plugin.add_follow_up(seed, "关注小林的期末考试进度")
    await plugin.add_follow_up(seed, "关注小林的期末考试进度")
    assert len(fake_iris.goals) == 1
    provider.response = answer(speak=False, watch_goal_ids=[1, 999])
    await plugin.receive(event("我来汇报进度了"))
    await settle(plugin)
    assert json.loads(provider.calls[-1]["prompt"])["motive"] == "follow_up"
    assert [a["goal_id"] for a in state.anchors] == [1]
    assert not plugin.context.sent and fake_iris.goals[1]["state"] == "open"
    await plugin.end_follow_up(seed, 1)
    engine.clock.advance(31)
    await plugin.receive(event("我的考试进度更新了"))
    await settle(plugin)
    assert len(provider.calls) == 1 and not state.anchors
    fake_iris.goals[2] = {"id": 2, "content": "小林考试", "entry_id": None, "state": "open"}
    fake_iris.goals[3] = {
        "id": 3,
        "content": "小林考试",
        "entry_id": "different-entry",
        "state": "open",
    }
    engine.clock.advance(301)
    await plugin.receive(event("考试的事情"))
    await settle(plugin)
    assert len(provider.calls) == 1


async def test_topic_matching_and_watch_anchor_expires(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(proactive_watch_enabled=True)
    seed = event()
    state = await enable(plugin, seed)
    await plugin.add_follow_up(seed, "留意摄影展的安排")
    provider.response = answer(watch_goal_ids=[1])  # Even speak=true cannot escape watch mode.
    await plugin.receive(event("这是大家正在聊的一段普通闲话"))
    await settle(plugin)
    assert json.loads(provider.calls[-1]["prompt"])["motive"] == "watch"
    assert state.last_result == "watch" and not plugin.context.sent
    assert state.anchors[0]["users"] == ["u1"]
    engine.clock.advance(31)
    provider.response = answer(speak=False, watch_goal_ids=[1])
    await plugin.receive(event("又来啦"))
    await settle(plugin)
    assert json.loads(provider.calls[-1]["prompt"])["motive"] == "follow_up"
    engine.clock.advance(601)
    await plugin.receive(event("又来啦"))
    await settle(plugin)
    assert len(provider.calls) == 2
    await plugin.receive(event("摄影展开始了", user="different-user"))
    await settle(plugin)
    assert json.loads(provider.calls[-1]["prompt"])["motive"] == "follow_up"
    assert len(provider.calls) == 3


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON iris_test_token",
        answer(speak="yes"),
        answer(why_now=""),
        answer(topic_source="invented"),
        answer(cooldown_minutes=True),
        answer(watch_goal_ids=[True]),
        answer(content="字" * 301),
    ],
)
async def test_invalid_decision_counts_call_never_sends(setup, raw):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    provider.response = raw
    await plugin.receive(event("Iris 看看"))
    await settle(plugin)
    assert state.calls_today == 1 and state.last_result == "invalid_decision"
    assert not plugin.context.sent and not state.sends


@pytest.mark.parametrize("failure", ["provider", "timeout", "send_false", "send_exception"])
async def test_failure_no_output_feedback_or_send_statistics(setup, fake_iris, failure):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_decision_timeout=0.05, assume_injected_used=True
    )
    state = await enable(plugin, event())
    if failure == "provider":
        provider.error = True
    if failure == "timeout":
        provider.release = asyncio.Event()
    if failure.startswith("send"):

        async def send(*args):
            if failure == "send_exception":
                raise RuntimeError("iris_test_token")
            return False

        plugin.context.send_message = send
    await plugin.receive(event("Iris 听听看"))
    await settle(plugin)
    assert state.calls == 1 and state.sends == 0
    assert not any(m["kind"] == "self_output" for m in fake_iris.messages[state.entry_id])
    assert not fake_iris.feedback
    await plugin.receive(event("Iris 第二次"))
    await settle(plugin)
    assert len(provider.calls) == 1  # Failure backoff still applies.


@pytest.mark.parametrize(
    "change", ["message", "disable", "quiet", "cooldown", "normal_reply", "reset"]
)
async def test_send_recheck_cancels_stale_candidates(setup, fake_iris, change):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    provider.release = asyncio.Event()
    await plugin.receive(event("Iris 这件事呢"))
    await asyncio.wait_for(provider.started.wait(), 2)
    if change == "message":
        await plugin.receive(event("另一条消息，没有邀请信号"))
    elif change == "normal_reply":
        await plugin.before_llm(event(), SimpleNamespace(system_prompt="", contexts=[]))
    else:
        patch = {
            "disable": {"enabled": False},
            "quiet": {"quiet_start": "11:00", "quiet_end": "13:00"},
            "cooldown": {"cooldown_minutes": 10},
            "reset": {"reset": True},
        }[change]
        engine.update(state.origin, patch)
    provider.release.set()
    await settle(plugin)
    assert not plugin.context.sent and not state.sends and not state.anchors
    assert all(m["kind"] == "message" for m in fake_iris.messages[state.entry_id])


async def test_concurrency_dedup_and_session_isolation(setup):
    create, event = setup
    plugin, engine, provider = await create()
    first = await enable(plugin, event())
    second = await enable(plugin, event(group="g2"))
    provider.release = asyncio.Event()
    message = event("Iris 是这样的吗")
    await plugin.receive(message)
    await plugin.receive(message)
    await asyncio.wait_for(provider.started.wait(), 2)
    await plugin.receive(event("Iris 又有新消息"))
    await plugin.receive(event("Iris 另一个群", group="g2"))
    provider.release.set()
    await settle(plugin)
    assert len(provider.calls) == 2
    assert first.sends == 0 and second.sends == 1
    assert plugin.context.sent[0][0] == second.origin


async def test_daily_call_cap_rollover_and_admin_reset(setup):
    create, event = setup
    plugin, engine, provider = await create(proactive_daily_calls=2)
    state = await enable(plugin, event())
    provider.response = answer(speak=False)
    for _ in range(3):
        await plugin.receive(event("今天喝什么？"))
        await settle(plugin)
        engine.clock.advance(31)
    assert len(provider.calls) == state.calls_today == 2
    assert state.outcomes["watch"] == 2
    engine.clock.advance(86400)
    await plugin.receive(event("今天呢？"))
    await settle(plugin)
    assert state.calls_today == 1 and state.calls == 3
    text = await command(plugin, event("/iris_reply reset"))
    assert "累计调用：0" in text and not state.calls_by_motive
    assert state.enabled  # reset doesn't silently change administrator opt-in.


async def test_initiate_probability_reservation_and_rolling_limit(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_initiate_enabled=True,
        proactive_silence_minutes=1,
        proactive_check_interval_seconds=1,
        proactive_initiate_probability=1,
        proactive_delay_min_minutes=1,
        proactive_delay_max_minutes=1,
        proactive_initiate_interval_minutes=1,
    )
    state = await enable(plugin, event())
    engine.clock.advance(59)
    await engine.tick()
    assert not state.pending_at and not provider.calls
    engine.clock.advance(1)
    await engine.tick()
    assert state.pending_at == engine.clock() + 60 and not provider.calls
    # A human message cancels the outstanding appointment before any prepare/model call.
    await plugin.receive(event("路过"))
    assert not state.pending_at
    for _ in range(2):
        engine.clock.advance(360)
        await engine.tick()
        engine.clock.advance(60)
        await engine.tick()
        await settle(plugin)
        await plugin.receive(event("收到"))
    assert state.sends_by_motive == {"initiate": 2}
    engine.clock.advance(3600)
    await engine.tick()
    assert len(provider.calls) == 2 and not state.pending_at
    assert len([m for m in fake_iris.messages[state.entry_id] if m["kind"] == "self_output"]) == 2
    engine.clock.advance(86401)
    await engine.tick()
    assert state.pending_at > engine.clock()  # Rolling quota expires independently of date.


async def test_zero_probability_and_muted_resume_rerandomize(setup):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_initiate_enabled=True,
        proactive_silence_minutes=1,
        proactive_initiate_probability=0,
        proactive_delay_min_minutes=1,
        proactive_delay_max_minutes=2,
    )
    state = await enable(plugin, event())
    engine.clock.advance(600)
    await engine.tick()
    assert not state.pending_at and not provider.calls
    engine.update(state.origin, {"quiet_start": "13:00", "quiet_end": "14:00"})
    state.pending_at = engine.clock() + 100
    engine.clock.now = datetime(2030, 1, 1, 13, 30).timestamp()
    await engine.tick()
    assert not state.pending_at and state.was_muted
    engine.rng.value = 0.5
    engine.clock.now = datetime(2030, 1, 1, 14).timestamp()
    await engine.tick()
    assert not state.was_muted and state.next_check == engine.clock() + 90
    assert not provider.calls and not state.pending_at


async def test_manual_initiate_and_private_care_switches(setup):
    create, event = setup
    plugin, engine, provider = await create()
    seed = event(group="")
    state = await enable(plugin, seed)
    result = await command(plugin, event("/iris_reply initiate", group=""))
    assert "private_care_disabled" in result and not provider.calls
    other = await enable(plugin, event())
    result = await command(plugin, event("/iris_reply initiate"))
    await settle(plugin)
    assert "sent" in result and other.sends == 1
    result = await command(plugin, event("/iris_reply initiate"))
    assert "cooldown" in result and len(provider.calls) == 1
    assert state.sends == 0


async def test_private_care_uses_same_limits_and_backflow(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_private_care_enabled=True,
        proactive_silence_minutes=1,
        proactive_initiate_probability=1,
        proactive_delay_min_minutes=1,
        proactive_delay_max_minutes=1,
    )
    state = await enable(plugin, event(group=""))
    engine.clock.advance(120)
    await engine.tick()
    engine.clock.advance(60)
    await engine.tick()
    await settle(plugin)
    assert state.sends_by_motive == {"initiate": 1}
    assert json.loads(provider.calls[0]["prompt"])["kind"] == "private"
    assert fake_iris.messages[state.entry_id][-1]["kind"] == "self_output"


async def test_midnight_does_not_reset_rolling_initiate_quota(setup):
    create, event = setup
    plugin, engine, provider = await create(proactive_initiate_limit=1)
    engine.clock.now = datetime(2030, 1, 1, 23, 50).timestamp()
    state = await enable(plugin, event())
    assert "sent" in await command(plugin, event("/iris_reply initiate"))
    engine.clock.advance(1200)
    assert "rolling_limit" in await command(plugin, event("/iris_reply initiate"))
    assert state.calls_today == 0 and len(provider.calls) == 1


@pytest.mark.parametrize(
    "hour,minute,expected", [(0, 59, False), (1, 0, True), (6, 59, True), (7, 0, False)]
)
def test_quiet_time_boundaries(hour, minute, expected):
    state = Session("test", "test", "group")
    assert muted(state, datetime(2030, 1, 1, hour, minute).timestamp()) == expected
    state.quiet_start, state.quiet_end = "23:00", "06:00"
    assert muted(state, datetime(2030, 1, 1, 23, 59).timestamp())
    assert muted(state, datetime(2030, 1, 1, 0, 1).timestamp())
    assert not muted(state, datetime(2030, 1, 1, 6).timestamp())


async def test_commands_tool_admin_validation_and_switch(setup):
    create, event = setup
    plugin, engine, _ = await create()
    unauthorized = event("/iris_reply enable")
    unauthorized.admin = False
    assert "管理员" in await command(plugin, unauthorized)
    assert "管理员" in await plugin.set_cooldown(unauthorized, 3)
    assert not engine.sessions
    seed = event("/iris_reply enable")
    assert "主动回复：开启" in await command(plugin, seed)
    state = engine.sessions[seed.unified_msg_origin]
    assert "意愿：低" in await command(plugin, event("/iris_reply willingness 低"))
    assert "意愿：低" in await command(plugin, event("/iris_reply willingness"))
    assert "冷却：5 分钟" in await command(plugin, event("/iris_reply cooldown"))
    assert "冷却：0 分钟" in await command(plugin, event("/iris_reply cooldown 0"))
    assert "设置为 12" in await plugin.set_cooldown(seed, 12)
    for invalid in (True, 0, 121, -1, 1.5, "3"):
        assert "整数" in await plugin.set_cooldown(seed, invalid)
    assert state.cooldown_until == engine.clock() + 720
    for suffix in ("cooldown -1", "cooldown nope", "willingness maybe", "enable wrong"):
        assert "参数" in await command(plugin, event("/iris_reply " + suffix))
    assert "enable|disable" in await command(plugin, event("/iris_reply help"))
    assert "主动回复：关闭" in await command(plugin, event("/iris_reply disable"))
    assert not state.enabled


@pytest.mark.parametrize("switch", ["tools_enabled", "set_cooldown_enabled"])
async def test_cooldown_tool_can_be_disabled(setup, switch):
    create, event = setup
    plugin, engine, _ = await create(**{switch: False})
    assert "关闭" in await plugin.set_cooldown(event(), 5)
    assert not engine.sessions

    class Tools:
        def __init__(self):
            self.names = ["set_cooldown", "external"]

        def remove_tool(self, name):
            self.names = [n for n in self.names if n != name]

    original = Tools()
    req = SimpleNamespace(system_prompt="", contexts=[], func_tool=original)
    await plugin.before_llm(event(), req)
    assert req.func_tool.names == ["external"] and original.names[0] == "set_cooldown"


async def test_pages_auth_atomic_update_and_statistics(setup):
    create, event = setup
    plugin, engine, provider = await create()
    seed = event()
    state = await enable(plugin, seed)
    await plugin.receive(event("Iris 来聊聊"))
    await settle(plugin)
    module = __import__(plugin.__class__.__module__, fromlist=["request"])
    page = await plugin.page_status()
    assert page["proactive"]["sessions"][0]["calls"] == 1
    assert "iris_test_token" not in json.dumps(page)
    payload = {
        "origin": state.origin,
        "changes": {
            "enabled": False,
            "willingness": "high",
            "quiet_start": "23:30",
            "quiet_end": "08:15",
            "cooldown_minutes": 12,
        },
    }

    async def body():
        return payload

    module.request.json = body
    changed = await plugin.page_proactive()
    assert (
        not changed["enabled"]
        and changed["quiet_start"] == "23:30"
        and changed["cooldown_minutes"] == 12
    )
    for invalid in (
        {"enabled": True, "quiet_start": "24:00"},
        {"reset": True, "willingness": []},
        {"token": "iris_test_token"},
        {"cooldown_minutes": True},
    ):
        payload["changes"] = invalid
        assert (await plugin.page_proactive())["status_code"] == 400
        assert not state.enabled and state.calls == 1
    payload["changes"] = {"reset": True}
    assert (await plugin.page_proactive())["calls"] == 0
    payload["origin"] = "unknown"
    assert (await plugin.page_proactive())["status_code"] == 400
    module.request.username = None
    assert (await plugin.page_proactive())["status_code"] == 401
    assert (await plugin.page_status())["status_code"] == 401


async def test_reminders_share_enable_quiet_cooldown_and_confirmed_backflow(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(
        reminder_mode="active",
        reminder_target="bot1:GroupMessage:g1",
        reminder_entry_id="astrbot:bot1:group:g1",
    )
    message = event()
    await plugin.receive(message)
    state = engine.sessions[message.unified_msg_origin]
    fake_iris.notices = [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}]
    await plugin.bridge.poll()
    assert not plugin.context.sent and plugin.bridge.inbox.count() == 1
    engine.update(state.origin, {"enabled": True, "quiet_start": "11:00", "quiet_end": "13:00"})
    await plugin.bridge.poll()
    assert not plugin.context.sent
    engine.update(
        state.origin, {"quiet_start": "00:00", "quiet_end": "00:00", "cooldown_minutes": 5}
    )
    await plugin.bridge.poll()
    assert not plugin.context.sent
    engine.clock.advance(301)
    await plugin.bridge.poll()
    await settle(plugin)
    assert len(plugin.context.sent) == 1 and plugin.bridge.inbox.count() == 0
    assert not provider.calls and state.sends_by_motive == {"reminder": 1}
    assert fake_iris.messages[state.entry_id][-1]["kind"] == "self_output"
    assert not any(r[0] == "PATCH" for r in fake_iris.requests)


async def test_restart_retains_opt_in_stats_limits_but_not_appointments(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    await plugin.receive(event("Iris 你好"))
    await settle(plugin)
    engine.update(state.origin, {"willingness": "high"})
    state.pending_at = engine.clock() + 300
    await engine.close()
    replacement = type(engine)(plugin, clock=engine.clock, rng=Random())
    plugin.proactive = replacement
    restored = replacement.sessions[state.origin]
    assert restored.enabled and restored.willingness == "high"
    assert restored.calls == restored.sends == 1
    assert not restored.pending_at and not replacement.live
    assert replacement.snapshot()["sessions"][0]["live"] is False
    await replacement.tick()
    assert len(provider.calls) == 1
    await plugin.receive(event("重启后的消息"))
    assert replacement.snapshot()["sessions"][0]["live"] is True


@pytest.mark.parametrize(
    "values",
    [
        {"proactive_quiet_start": "25:00"},
        {"proactive_quiet_end": "1:00"},
        {"proactive_initiate_probability": float("nan")},
        {"proactive_initiate_probability": True},
        {"proactive_daily_calls": -1},
        {"proactive_enabled": "yes"},
        {"proactive_delay_min_minutes": 31},
        {"proactive_bot_aliases": [""]},
        {"proactive_willingness": "unknown"},
        {"set_cooldown_enabled": 1},
    ],
)
def test_invalid_proactive_configuration_rejected(values):
    with pytest.raises(ValueError):
        Config.from_mapping(values)


async def test_initiate_send_recheck_after_new_message(setup):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    provider.release = asyncio.Event()
    task = asyncio.create_task(command(plugin, event("/iris_reply initiate")))
    await asyncio.wait_for(provider.started.wait(), 2)
    await plugin.receive(event("有人回来了"))
    provider.release.set()
    assert "new_activity" in await task
    assert not state.initiate_times and not plugin.context.sent


async def test_prepare_failure_never_calls_provider(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create(prepare_timeout=0.05, request_timeout=0.05)
    state = await enable(plugin, event())
    await plugin.bridge.queue.join()
    fake_iris.fail = 503
    await plugin.receive(event("Iris 来看看"))
    await settle(plugin)
    assert not provider.calls and state.sends == 0 and state.calls == 0


async def test_suspend_across_quiet_window_discards_old_appointment(setup):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_initiate_enabled=True, proactive_silence_minutes=1
    )
    state = await enable(plugin, event())
    engine.update(state.origin, {"quiet_start": "13:00", "quiet_end": "14:00"})
    engine.clock.now = datetime(2030, 1, 1, 12, 59).timestamp()
    await engine.tick()
    state.pending_at = engine.clock() + 120
    # No ticks during quiet hours: e.g. suspend/resume.
    engine.clock.now = datetime(2030, 1, 1, 14, 1).timestamp()
    await engine.tick()
    assert not provider.calls and not state.pending_at
    assert state.next_check > engine.clock()


async def test_only_confirmed_passive_send_imposes_cooldown(setup):
    from tests.conftest import Result

    create, event = setup
    plugin, engine, _ = await create()
    message = event()
    state = await enable(plugin, message)
    message.send_error = True
    with pytest.raises(RuntimeError):
        await message.send(Result("失败"))
    await plugin.after_message_sent(message)
    assert not state.cooldown_until
    message.send_error = False
    await message.send(Result("成功"))
    await plugin.after_message_sent(message)
    assert state.cooldown_until == engine.clock() + 300
    assert state.sends == 0  # Ordinary replies aren't proactive statistics.


async def test_shutdown_cancels_manual_decision_before_closing_store(setup):
    create, event = setup
    plugin, engine, provider = await create()
    await enable(plugin, event())
    provider.release = asyncio.Event()
    task = asyncio.create_task(command(plugin, event("/iris_reply initiate")))
    await asyncio.wait_for(provider.started.wait(), 2)
    await engine.close()
    assert task.cancelled() and not plugin.context.sent
    assert not engine.inflight


async def test_global_off_also_blocks_active_reminders(setup, fake_iris):
    create, event = setup
    plugin, engine, _ = await create(
        proactive_enabled=False,
        reminder_mode="active",
        reminder_target="bot1:GroupMessage:g1",
        reminder_entry_id="astrbot:bot1:group:g1",
    )
    await enable(plugin, event())
    fake_iris.notices = [{"id": 1, "kind": "goal_reminder", "goal_id": 5, "content": "喝水"}]
    await plugin.bridge.poll()
    assert not plugin.context.sent and plugin.bridge.inbox.count() == 1


async def test_command_ack_does_not_undo_explicit_cooldown_clear(setup):
    create, event = setup
    plugin, engine, _ = await create()
    state = await enable(plugin, event())
    engine.update(state.origin, {"cooldown_minutes": 10})
    message = event("/iris_reply cooldown 0")
    await plugin.receive(message)
    results = [r async for r in plugin.iris_reply(message)]
    message.result = results[0]
    await message.send(message.result)
    await plugin.after_message_sent(message)
    assert state.cooldown_until <= engine.clock()


async def test_disabled_follow_tool_does_not_invalidate_active_candidate(setup):
    create, event = setup
    plugin, engine, _ = await create(add_follow_up_enabled=False)
    message = event()
    state = await enable(plugin, message)
    generation = state.generation
    assert "关闭" in await plugin.add_follow_up(message, "关注考试")
    assert state.generation == generation
