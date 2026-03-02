"""
ProactiveManager (新 Facade) 单元测试
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.proactive.manager import ProactiveManager


@pytest.fixture
def tmp_data_path(tmp_path):
    return tmp_path / "plugin_data"


class TestNewProactiveManagerInit:
    def test_create(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path)
        assert mgr is not None
        assert not mgr._initialized

    def test_enabled_property(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path, enabled=False)
        assert not mgr.enabled
        mgr.enabled = True
        assert mgr.enabled


class TestNewProactiveManagerProcess:
    @pytest.fixture
    def manager(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path)
        # Mock internal components as initialized
        mgr._initialized = True
        mgr._decision_engine = AsyncMock()
        mgr._context_engine = AsyncMock()
        mgr._strategy_router = MagicMock()
        mgr._feedback_tracker = AsyncMock()
        mgr._cold_start = MagicMock()
        return mgr

    @pytest.mark.asyncio
    async def test_process_not_initialized(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path)
        result = await mgr.process_message(
            messages=[], user_id="u1", session_key="u1:g1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_process_not_enabled(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path, enabled=False)
        mgr._initialized = True
        result = await mgr.process_message(
            messages=[], user_id="u1", session_key="u1:g1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_process_user_response(self, manager):
        manager._feedback_tracker.process_user_response = AsyncMock()
        await manager.process_user_response("u1:g1", user_replied_directly=True)
        manager._feedback_tracker.process_user_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats_no_store(self, tmp_data_path):
        mgr = ProactiveManager(plugin_data_path=tmp_data_path)
        stats = await mgr.get_stats()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_close(self, manager):
        manager._decision_engine.close = AsyncMock()
        manager._feedback_store = AsyncMock()
        manager._feedback_store.close = AsyncMock()
        manager._scene_store = AsyncMock()
        manager._scene_store.close = AsyncMock()
        await manager.close()
        assert not manager._initialized
