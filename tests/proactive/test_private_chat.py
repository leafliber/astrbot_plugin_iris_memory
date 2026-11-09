"""私聊主动消息门控与会话键的单元测试。"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_iris_memory.iris_memory.proactive.decision import DecisionCore
from astrbot_plugin_iris_memory.iris_memory.proactive.parser import parse_decision
from astrbot_plugin_iris_memory.iris_memory.proactive.perception import (
    ContextPackager,
    Gatekeeper,
    SlidingWindow,
    WindowMessage,
)
from astrbot_plugin_iris_memory.iris_memory.proactive.proactive import ProactiveEngine
from astrbot_plugin_iris_memory.iris_memory.proactive.signals import SignalGate
from astrbot_plugin_iris_memory.iris_memory.proactive.state import StateManager
from astrbot_plugin_iris_memory.iris_memory.proactive.tickets import DecisionTicketRegistry
from astrbot_plugin_iris_memory.main import IrisMemoryPlugin

PRIVATE_UID = "943379519"
SESSION_KEY = f"private:{PRIVATE_UID}"


def _event(*, private: bool, sender_id: str = PRIVATE_UID, text: str = "在吗"):
    return SimpleNamespace(
        message_str=text,
        is_private_chat=lambda: private,
        get_group_id=lambda: None if private else "123456",
        get_sender_id=lambda: sender_id,
        set_result=lambda msg: None,
    )


def _gate(config) -> Gatekeeper:
    state = SimpleNamespace(is_muted=lambda: False, is_whitelisted=lambda key: False)
    return Gatekeeper(config, state)


def test_private_chat_rejected_when_switch_off():
    cfg = SimpleNamespace(private_enabled=False, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_rejected_when_not_whitelisted():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=["111"])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_rejected_when_whitelist_empty():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[])
    assert _gate(cfg).should_process(_event(private=True)) is False


def test_private_chat_allowed_when_enabled_and_listed():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=True)) is True


def test_group_path_still_requires_whitelist():
    cfg = SimpleNamespace(private_enabled=True, private_whitelist=[PRIVATE_UID])
    assert _gate(cfg).should_process(_event(private=False)) is False


def test_session_key_for_private_chat():
    plugin = object.__new__(IrisMemoryPlugin)
    assert plugin._get_session_key(_event(private=True)) == f"private:{PRIVATE_UID}"


def test_session_key_for_group_chat():
    plugin = object.__new__(IrisMemoryPlugin)
    assert plugin._get_session_key(_event(private=False)) == "123456"


def _private_config(nm_config):
    return nm_config(
        cfg={
            "proactive": {
                "enabled": True,
                "proactive_enabled": True,
                "private_enabled": True,
                "private_whitelist": [PRIVATE_UID],
            }
        }
    )


def _hook_event(extras: dict, reply: str = "好的") -> SimpleNamespace:
    return SimpleNamespace(
        is_private_chat=lambda: True,
        get_group_id=lambda: "",
        get_sender_id=lambda: PRIVATE_UID,
        get_self_id=lambda: "bot",
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=extras.__setitem__,
        get_result=lambda: SimpleNamespace(get_plain_text=lambda: reply),
        stop_event=Mock(),
    )


def _plugin(config) -> IrisMemoryPlugin:
    plugin = object.__new__(IrisMemoryPlugin)
    plugin._reply_config = config
    plugin._state = StateManager(config)
    plugin._state.add_to_whitelist(SESSION_KEY)
    plugin._sliding_window = SlidingWindow(config)
    plugin._stats = Mock()
    plugin._tool_ctx = Mock()
    plugin._passive_active = {}
    plugin._reply_in_progress = {}
    plugin._triggering = {}
    plugin._decision_tickets = DecisionTicketRegistry(120)
    plugin._decision_core = Mock()
    plugin._llm_manager = Mock()
    plugin._kv_save = AsyncMock()
    plugin._get_provider_id = AsyncMock(return_value=None)
    return plugin


def _engine(config):
    state = StateManager(config)
    window = SlidingWindow(config)
    packager = ContextPackager(config)
    context = SimpleNamespace(send_message=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        context,
        config,
        state,
        window,
        SignalGate(config, state),
        DecisionCore(config, state, window, packager),
        Mock(),
        llm_manager=SimpleNamespace(generate_direct=AsyncMock()),
        packager=packager,
        umo_get=lambda _key: "umo",
        is_busy=lambda _key: False,
        self_id_get=lambda: "bot",
        save_fn=AsyncMock(),
    )
    now = time.time()
    state.add_to_whitelist(SESSION_KEY)
    state.record_human_message(SESSION_KEY, now - 7200)
    window.append(SESSION_KEY, WindowMessage("u", "U", "在吗", now - 7200))
    state.set_next_initiate_check(SESSION_KEY, now - 1)
    engine._signals.evaluate_timer = Mock(return_value=None)
    return engine, state


@pytest.mark.asyncio
async def test_passive_private_reply_releases_session_and_records_bot_message(nm_config):
    """默认配置下私聊都走被动回复，回复后会话不能一直处于忙碌状态"""
    plugin = _plugin(_private_config(nm_config))
    plugin._passive_active[SESSION_KEY] = time.time()
    event = _hook_event({"iris_mode": "passive"})

    await plugin._reply_on_llm_response(event, SimpleNamespace(completion_text="好的"))
    await plugin.on_message_sent(event)

    assert not plugin._is_busy(SESSION_KEY)
    window = plugin._sliding_window.get_messages(SESSION_KEY)
    assert [message.content for message in window] == ["好的"]
    assert plugin._state.get_anchor(SESSION_KEY).bot_message == "好的"


@pytest.mark.asyncio
async def test_private_decision_runs_and_releases_claim(nm_config):
    """需要唤醒词的私聊消息由决策层拦下时，认领要立即释放"""
    plugin = _plugin(_private_config(nm_config))
    ticket = plugin._decision_tickets.claim(SESSION_KEY, "event-1")
    plugin._triggering[SESSION_KEY] = time.time()
    plugin._decision_core.decide = AsyncMock(
        return_value=SimpleNamespace(
            error=None,
            error_kind="",
            decision=parse_decision('{"action": "skip"}', mode="chime_in"),
            system_prompt="",
            user_prompt="",
            raw_text="{}",
            duration_ms=1,
        )
    )
    event = _hook_event(
        {
            "iris_decision": {
                "motive": "chime_in",
                "provider_id": "provider",
                "event_id": "event-1",
                "ticket_id": ticket.ticket_id,
            }
        }
    )

    assert await plugin._handle_reply_decision(event) is True

    event.stop_event.assert_called_once()
    assert not plugin._is_busy(SESSION_KEY)


@pytest.mark.asyncio
async def test_unclaimed_private_event_passes_through(nm_config):
    plugin = _plugin(_private_config(nm_config))
    event = _hook_event({})

    assert await plugin._handle_reply_decision(event) is False

    event.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_private_session_is_scheduled(nm_config):
    engine, _state = _engine(_private_config(nm_config))

    await engine._scan_due()

    engine._signals.evaluate_timer.assert_called_once()


@pytest.mark.parametrize(
    "revocation",
    [{"private_enabled": False}, {"private_whitelist": ["111"]}],
)
@pytest.mark.asyncio
async def test_revoked_private_session_stops_immediately(nm_config, revocation):
    """关掉开关或移出私聊白名单后，已登记的会话不能再被主动发起"""
    config = _private_config(nm_config)
    engine, state = _engine(config)

    config._cfg["proactive"].update(revocation)
    await engine._scan_due()
    result = await engine.attempt_initiate(SESSION_KEY)

    engine._signals.evaluate_timer.assert_not_called()
    engine._context.send_message.assert_not_awaited()
    assert result == "该群未启用"
    assert SESSION_KEY not in state.get_whitelist()


def test_revocation_during_generation_cancels_send(nm_config):
    config = _private_config(nm_config)
    engine, _state = _engine(config)

    config._cfg["proactive"]["private_enabled"] = False

    assert engine._initiate_guard_reason(SESSION_KEY, time.time(), require_quiet=True)


@pytest.mark.asyncio
async def test_revoked_session_is_skipped_by_reply_hooks(nm_config):
    config = _private_config(nm_config)
    plugin = _plugin(config)
    event = _hook_event({})

    config._cfg["proactive"]["private_enabled"] = False
    await plugin.on_message_sent(event)

    assert plugin._sliding_window.get_messages(SESSION_KEY) == []


def test_reauthorized_session_resumes_without_reregistering(nm_config):
    config = _private_config(nm_config)
    state = StateManager(config)
    state.add_to_whitelist(SESSION_KEY)

    config._cfg["proactive"]["private_enabled"] = False
    assert not state.is_whitelisted(SESSION_KEY)

    config._cfg["proactive"]["private_enabled"] = True
    assert state.is_whitelisted(SESSION_KEY)
