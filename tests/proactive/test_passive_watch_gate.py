from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from astrbot_plugin_iris_memory.main import IrisMemoryPlugin


@pytest.mark.asyncio
async def test_passive_watch_burst_only_calls_decision_once():
    plugin = object.__new__(IrisMemoryPlugin)
    plugin._passive_watch_last = {}
    plugin._passive_watch_hash = {}
    plugin._reply_config = SimpleNamespace(
        passive_watch_min_interval=600, passive_watch_enabled=True
    )
    plugin._sliding_window = Mock()
    plugin._sliding_window.get_messages.return_value = [
        SimpleNamespace(content="用户：我晚点把结果发你")
    ]
    plugin._llm_manager = Mock()
    plugin._llm_manager.get_governor_metrics.return_value = {
        "queue_depth": 0,
        "queue_limit": 500,
    }
    plugin._decision_core = Mock()
    plugin._decision_core.decide = AsyncMock(
        return_value=SimpleNamespace(
            error="temporary",
            error_kind="",
            decision=None,
        )
    )
    plugin._stats = Mock()
    plugin._kv_save = AsyncMock()
    plugin._state = Mock()
    lock = MagicMock()
    lock.__aenter__ = AsyncMock(return_value=None)
    lock.__aexit__ = AsyncMock(return_value=None)
    plugin._state.get_lock.return_value = lock
    plugin._state.save_dirty = AsyncMock()

    for _ in range(100):
        await plugin._passive_watch_eval(
            "group-1", "provider-1", "user-1", "我会在有结果后再告诉你。"
        )

    assert plugin._decision_core.decide.await_count == 1


@pytest.mark.asyncio
async def test_passive_watch_without_followup_signal_is_local_only():
    plugin = object.__new__(IrisMemoryPlugin)
    plugin._passive_watch_last = {}
    plugin._passive_watch_hash = {}
    plugin._reply_config = SimpleNamespace(
        passive_watch_min_interval=600, passive_watch_enabled=True
    )
    plugin._sliding_window = Mock()
    plugin._sliding_window.get_messages.return_value = [
        SimpleNamespace(content="普通聊天")
    ]
    plugin._llm_manager = Mock()
    plugin._decision_core = Mock()
    plugin._decision_core.decide = AsyncMock()

    for _ in range(100):
        await plugin._passive_watch_eval(
            "group-1", "provider-1", "user-1", "好的，收到。"
        )

    plugin._decision_core.decide.assert_not_awaited()
