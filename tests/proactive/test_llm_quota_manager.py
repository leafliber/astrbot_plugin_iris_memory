"""
LLMQuotaManager 单元测试
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from iris_memory.proactive.core.llm_quota_manager import LLMQuotaManager


class TestLLMQuotaManager:
    @pytest.fixture
    def manager(self):
        m = LLMQuotaManager(feedback_store=None)
        m._initialized = True
        return m

    @pytest.mark.asyncio
    async def test_acquire_success(self, manager):
        result = await manager.acquire("s1", max_per_hour=5)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self, manager):
        # Fill quota
        for _ in range(5):
            await manager.acquire("s1", max_per_hour=5)
        result = await manager.acquire("s1", max_per_hour=5)
        assert result is False

    @pytest.mark.asyncio
    async def test_sessions_independent(self, manager):
        for _ in range(5):
            await manager.acquire("s1", max_per_hour=5)
        # s2 should still have quota
        result = await manager.acquire("s2", max_per_hour=5)
        assert result is True

    def test_get_remaining(self, manager):
        assert manager.get_remaining("s1", max_per_hour=5) == 5

    @pytest.mark.asyncio
    async def test_get_remaining_after_acquire(self, manager):
        await manager.acquire("s1", max_per_hour=5)
        await manager.acquire("s1", max_per_hour=5)
        assert manager.get_remaining("s1", max_per_hour=5) == 3

    @pytest.mark.asyncio
    async def test_hourly_reset(self, manager):
        await manager.acquire("s1", max_per_hour=2)
        await manager.acquire("s1", max_per_hour=2)
        assert await manager.acquire("s1", max_per_hour=2) is False

        # Simulate hour change
        manager._current_hour = (manager._current_hour - 1) % 24
        assert await manager.acquire("s1", max_per_hour=2) is True

    @pytest.mark.asyncio
    async def test_initialize_from_store(self):
        store = AsyncMock()
        store.get_llm_quotas_for_hour = AsyncMock(
            return_value=[{"session_key": "s1", "count": 3}]
        )
        mgr = LLMQuotaManager(feedback_store=store)
        await mgr.initialize()
        assert mgr._initialized is True
        assert mgr.get_remaining("s1", max_per_hour=5) == 2

    @pytest.mark.asyncio
    async def test_initialize_store_failure(self):
        store = AsyncMock()
        store.get_llm_quotas_for_hour = AsyncMock(side_effect=Exception("db error"))
        mgr = LLMQuotaManager(feedback_store=store)
        await mgr.initialize()
        # Should still be initialized, just empty counts
        assert mgr._initialized is True
        assert mgr.get_remaining("s1") == 5
