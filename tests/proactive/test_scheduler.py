"""每群独立预约与主动发送现场二次校验。"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from iris_memory.proactive.decision import DecisionCore
from iris_memory.proactive.perception import ContextPackager, SlidingWindow, WindowMessage
from iris_memory.proactive.proactive import ProactiveEngine
from iris_memory.proactive.signals import SignalGate
from iris_memory.proactive.state import StateManager


GID = "g1"


def _engine(nm_config, *, llm_manager=None):
    config = nm_config(
        cfg={
            "proactive": {
                "enabled": True,
                "proactive_enabled": True,
                "provider_id": "provider",
            }
        }
    )
    state = StateManager(config)
    window = SlidingWindow(config)
    packager = ContextPackager(config)
    core = DecisionCore(config, state, window, packager)
    context = SimpleNamespace(
        persona_manager=SimpleNamespace(get_default_persona_v3=AsyncMock(return_value={})),
        send_message=AsyncMock(return_value=True),
        get_config=lambda _umo=None: {},
    )
    manager = llm_manager or SimpleNamespace(generate_direct=AsyncMock())
    engine = ProactiveEngine(
        context,
        config,
        state,
        window,
        SignalGate(config, state),
        core,
        Mock(),
        llm_manager=manager,
        packager=packager,
        umo_get=lambda _gid: "umo",
        is_busy=lambda _gid: False,
        self_id_get=lambda: "bot",
        save_fn=AsyncMock(),
    )
    return engine, state, window, context


class TestIndependentScheduling:
    def test_human_message_replaces_old_candidate_with_independent_jitter(self, nm_config):
        engine, state, _, _ = _engine(nm_config)
        now = time.time()
        with patch("iris_memory.proactive.proactive.random.uniform", return_value=123.4):
            engine.notify_human_message(GID, now)

        data = state.get_state(GID)
        expected = now + state.minimum_quiet_seconds() + 123.4
        assert data.last_human_message_at == now
        assert data.initiate_next_check_at == pytest.approx(expected)

    def test_mute_exit_adds_thirty_to_ninety_minute_resample(self, nm_config):
        engine, state, _, _ = _engine(nm_config)
        now = time.time()
        state.is_muted = lambda: True
        state.seconds_until_unmuted = lambda _now=None: 3600.0
        with patch("iris_memory.proactive.proactive.random.uniform", return_value=1800.0):
            when = engine._schedule_group(GID, now=now)
        assert when == pytest.approx(now + 3600 + 1800)

    @pytest.mark.asyncio
    async def test_only_due_group_is_evaluated(self, nm_config):
        engine, state, window, _ = _engine(nm_config)
        now = time.time()
        for gid in ("g1", "g2"):
            state.add_to_whitelist(gid)
            state.record_human_message(gid, now - 7200)
            window.append(gid, WindowMessage("u", "U", "旧消息", now - 7200))
        state.set_next_initiate_check("g1", now - 1)
        state.set_next_initiate_check("g2", now + 3600)
        engine._signals.evaluate_timer = Mock(return_value=None)

        await engine._scan_due()

        engine._signals.evaluate_timer.assert_called_once()
        assert engine._signals.evaluate_timer.call_args.args[0] == "g1"
        assert state.get_next_initiate_check("g2") == pytest.approx(now + 3600)


class TestSendGuard:
    def test_new_message_after_snapshot_cancels(self, nm_config):
        engine, state, window, _ = _engine(nm_config)
        old = time.time() - 7200
        window.append(GID, WindowMessage("u", "U", "旧消息", old))
        state.record_human_message(GID, old)
        window.append(GID, WindowMessage("u2", "U2", "新消息", old + 1))
        state.record_human_message(GID, old + 1)

        assert "新消息" in engine._initiate_guard_reason(
            GID, old, require_quiet=True
        )

    @pytest.mark.asyncio
    async def test_message_arriving_during_generation_prevents_send(self, nm_config):
        calls = 0
        holder = {}

        async def generate_direct(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return (
                    '{"action":"speak","topic":"接着问上次的计划",'
                    '"why_now":"现在是之前约好的回访时间",'
                    '"topic_source":"time_context","obs":"在聊计划",'
                    '"watch":[],"watch_keywords":[],"why":"回访",'
                    '"drifted":false,"cooldown":0}'
                )
            now = time.time()
            holder["window"].append(GID, WindowMessage("u2", "U2", "刚来一条", now))
            holder["state"].record_human_message(GID, now)
            return "那我们接着聊聊？"

        manager = SimpleNamespace(generate_direct=generate_direct)
        engine, state, window, context = _engine(nm_config, llm_manager=manager)
        holder.update(state=state, window=window)
        state.add_to_whitelist(GID)
        old = time.time() - 7200
        window.append(GID, WindowMessage("u1", "U1", "之前说周末出门", old))
        state.record_human_message(GID, old)

        result = await engine.attempt_initiate(GID)

        assert result == "已取消发起：决策期间出现了新消息"
        context.send_message.assert_not_awaited()
