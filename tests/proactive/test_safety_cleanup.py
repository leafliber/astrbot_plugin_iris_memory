"""主动回复 Provider 输入安全拒绝后的上下文隔离测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.proactive.admin import AdminCommands
from iris_memory.proactive.api import register_web_apis
from iris_memory.proactive.config import ConfigManager
from iris_memory.proactive.decision import (
    INPUT_SAFETY_COOLDOWN_MINUTES,
    DecisionCore,
    DecisionRequest,
)
from iris_memory.proactive.perception import (
    ContextPackager,
    SlidingWindow,
    WindowMessage,
)
from iris_memory.proactive.proactive import ProactiveEngine
from iris_memory.proactive.state import GroupState, StateManager

GID = "g1"


def _message(content: str, timestamp: float) -> WindowMessage:
    return WindowMessage(
        sender_id="u1",
        sender_name="User1",
        content=content,
        timestamp=timestamp,
    )


def _components(nm_config, *, cfg=None):
    config = nm_config(cfg=cfg)
    state = StateManager(config)
    window = SlidingWindow(config)
    packager = ContextPackager(config)
    core = DecisionCore(config, state, window, packager)
    return config, state, window, packager, core


async def _rejected_outcome(core: DecisionCore, *, motive: str = "chime_in"):
    return await core.decide(
        DecisionRequest(group_id=GID, wake="message", motive=motive),
        SimpleNamespace(
            generate_direct=AsyncMock(
                side_effect=RuntimeError("input new_sensitive (1026)")
            )
        ),
        "provider",
    )


class TestCoreSafetyCleanup:
    @pytest.mark.asyncio
    async def test_removes_rejected_snapshot_but_preserves_new_messages(self, nm_config):
        _, state, window, _, core = _components(nm_config)
        state.set_willingness(GID, "high")
        state.set_observation(GID, "疑似源头观察")
        state.write_anchor(
            GID,
            kind="follow_up",
            topic="旧话题",
            bot_message="疑似源头回复",
            users=["u1"],
            keywords=["疑似关键词"],
            reason="疑似原因",
        )
        window.append(GID, _message("旧消息一", 10.0))
        window.append(GID, _message("旧消息二", 20.0))

        outcome = await _rejected_outcome(core)
        # 模拟 Provider 调用期间到达的新消息；它没有出现在 rejected prompt 中。
        window.append(GID, _message("请求期间的新消息", 30.0))

        cleanup = core.clear_rejected_dynamic_context(
            GID,
            outcome.dynamic_context_sources,
        )

        assert cleanup.window_removed == 2
        assert cleanup.observation_cleared is True
        assert cleanup.anchor_cleared is True
        assert [msg.content for msg in window.get_messages(GID)] == ["请求期间的新消息"]
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        assert state.get_willingness(GID) == "high"

    def test_discard_through_preserves_configured_capacity(self, nm_config):
        config, _, window, _, _ = _components(nm_config)
        window.append(GID, _message("旧消息", 10.0))
        window.append(GID, _message("新消息", 20.0))

        assert window.discard_through(GID, 10.0) == 1
        assert [msg.content for msg in window.get_messages(GID)] == ["新消息"]
        assert window._windows[GID].maxlen == config.window_size


class TestRuntimeSafetyPaths:
    @pytest.mark.asyncio
    async def test_message_decision_1026_purges_and_persists(self, nm_config):
        from main import IrisMemoryPlugin

        _, state, window, _, core = _components(nm_config)
        state.set_observation(GID, "疑似观察")
        state.write_anchor(GID, kind="follow_up", bot_message="疑似回复", users=["u1"])
        window.append(GID, _message("疑似消息", 10.0))

        plugin = object.__new__(IrisMemoryPlugin)
        plugin.context = SimpleNamespace(
            llm_generate=AsyncMock(side_effect=RuntimeError("input new_sensitive (1026)"))
        )
        plugin._llm_manager = SimpleNamespace(
            generate_direct=plugin.context.llm_generate
        )
        plugin._state = state
        plugin._sliding_window = window
        plugin._decision_core = core
        plugin._stats = Mock()
        plugin._triggering = {GID: 10.0}
        plugin._kv_save = AsyncMock()

        event = Mock()
        event.get_group_id.return_value = GID
        event.get_extra.return_value = {
            "motive": "chime_in",
            "provider_id": "provider",
        }

        stopped = await plugin._handle_reply_decision(event)

        assert stopped is True
        assert window.get_messages(GID) == []
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        assert state.get_state(GID).state == GroupState.COOLDOWN
        assert GID not in plugin._triggering
        event.stop_event.assert_called_once()
        plugin._kv_save.assert_awaited()

    @pytest.mark.asyncio
    async def test_other_provider_error_keeps_context(self, nm_config):
        from main import IrisMemoryPlugin

        _, state, window, _, core = _components(nm_config)
        state.set_observation(GID, "应保留的观察")
        state.write_anchor(GID, kind="follow_up", bot_message="应保留的回复", users=["u1"])
        window.append(GID, _message("应保留的消息", 10.0))

        plugin = object.__new__(IrisMemoryPlugin)
        plugin.context = SimpleNamespace(
            llm_generate=AsyncMock(side_effect=RuntimeError("temporary timeout"))
        )
        plugin._llm_manager = SimpleNamespace(
            generate_direct=plugin.context.llm_generate
        )
        plugin._state = state
        plugin._sliding_window = window
        plugin._decision_core = core
        plugin._stats = Mock()
        plugin._triggering = {GID: 10.0}
        plugin._kv_save = AsyncMock()

        event = Mock()
        event.get_group_id.return_value = GID
        event.get_extra.return_value = {
            "motive": "chime_in",
            "provider_id": "provider",
        }

        assert await plugin._handle_reply_decision(event) is True
        assert [msg.content for msg in window.get_messages(GID)] == ["应保留的消息"]
        assert state.get_observation(GID) == "应保留的观察"
        assert state.get_anchor(GID).bot_message == "应保留的回复"
        assert state.get_state(GID).state == GroupState.IDLE

    @pytest.mark.asyncio
    async def test_passive_watch_1026_does_not_restore_fallback_anchor(self, nm_config):
        from main import IrisMemoryPlugin

        _, state, window, _, core = _components(nm_config)
        state.set_observation(GID, "疑似观察")
        state.write_anchor(GID, kind="passive", bot_message="旧回复", users=["u1"])
        window.append(GID, _message("用户消息", 10.0))
        window.append(GID, _message("机器人刚刚的回复", 20.0))

        plugin = object.__new__(IrisMemoryPlugin)
        plugin.context = SimpleNamespace(
            llm_generate=AsyncMock(side_effect=RuntimeError("input new_sensitive (1026)"))
        )
        plugin._llm_manager = SimpleNamespace(
            generate_direct=plugin.context.llm_generate
        )
        plugin._state = state
        plugin._sliding_window = window
        plugin._decision_core = core
        plugin._stats = Mock()
        plugin._kv_save = AsyncMock()

        await plugin._passive_watch_eval(
            GID,
            "provider",
            "u1",
            "机器人刚刚的回复",
        )

        assert window.get_messages(GID) == []
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        assert state.get_anchor(GID).bot_message == ""
        assert state.get_state(GID).state == GroupState.COOLDOWN

    @pytest.mark.asyncio
    async def test_timer_initiate_1026_uses_same_cleanup(self, nm_config):
        config, state, window, packager, core = _components(
            nm_config,
            cfg={"proactive": {"provider_id": "provider"}},
        )
        state.set_observation(GID, "疑似观察")
        state.write_anchor(GID, kind="initiate", bot_message="旧主动消息", users=["u1"])
        window.append(GID, _message("疑似群消息", 10.0))

        context = SimpleNamespace(
            llm_generate=AsyncMock(side_effect=RuntimeError("input new_sensitive (1026)"))
        )
        save_fn = AsyncMock()
        engine = ProactiveEngine(
            context,
            config,
            state,
            window,
            Mock(),
            core,
            Mock(),
            llm_manager=SimpleNamespace(generate_direct=context.llm_generate),
            packager=packager,
            umo_get=lambda _gid: "umo",
            is_busy=lambda _gid: False,
            self_id_get=lambda: "bot",
            save_fn=save_fn,
        )

        result = await engine.attempt_initiate(GID, force=True)

        assert result.startswith("决策调用失败:")
        assert window.get_messages(GID) == []
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        data = state.get_state(GID)
        assert data.state == GroupState.COOLDOWN
        assert data.cooldown_until > 0
        assert engine._skip_retry_after[GID] > 0
        save_fn.assert_awaited_once()
        assert INPUT_SAFETY_COOLDOWN_MINUTES == 30


class TestResetPaths:
    @pytest.mark.asyncio
    async def test_admin_reset_also_clears_sliding_window(self, nm_config):
        from main import IrisMemoryPlugin

        _, state, window, _, _ = _components(nm_config)
        state.set_observation(GID, "旧观察")
        state.write_anchor(GID, kind="follow_up", bot_message="旧回复", users=["u1"])
        window.append(GID, _message("旧窗口消息", 10.0))

        plugin = object.__new__(IrisMemoryPlugin)
        plugin._state = state
        plugin._sliding_window = window
        plugin._admin = AdminCommands(state)
        plugin._kv_save = AsyncMock()
        event = Mock()
        event.get_group_id.return_value = GID

        await plugin.cmd_reset(event)

        assert window.get_messages(GID) == []
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        event.set_result.assert_called_once_with(f"群 {GID} 状态已重置")

    @pytest.mark.asyncio
    async def test_web_reset_also_clears_sliding_window(self, nm_config):
        from quart import Quart

        _, state, window, _, _ = _components(nm_config)
        state.set_observation(GID, "旧观察")
        state.write_anchor(GID, kind="follow_up", bot_message="旧回复", users=["u1"])
        window.append(GID, _message("旧窗口消息", 10.0))

        handlers = {}
        context = Mock()
        context.register_web_api.side_effect = (
            lambda path, handler, _methods, _description: handlers.__setitem__(path, handler)
        )
        kv_save = AsyncMock()
        register_web_apis(
            context=context,
            plugin_name="test_plugin",
            state=state,
            stats=Mock(),
            window=window,
            kv_save=kv_save,
        )

        app = Quart(__name__)
        async with app.test_request_context(
            "/test_plugin/reply/group/reset",
            method="POST",
            json={"group_id": GID},
        ):
            response = await handlers["/test_plugin/reply/group/reset"]()

        assert (await response.get_json())["ok"] is True
        assert window.get_messages(GID) == []
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        kv_save.assert_awaited()
