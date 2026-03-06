"""宽限期管理器测试"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from iris_memory.storage.grace_period import GracePeriodManager
from iris_memory.models.memory import Memory
from iris_memory.models.protection import ProtectionFlag
from iris_memory.core.types import MemoryType, QualityLevel, StorageLayer


@pytest.fixture
def mock_chroma():
    m = AsyncMock()
    m.update_memory = AsyncMock(return_value=True)
    m.delete_memory = AsyncMock(return_value=True)
    m.query_memories = AsyncMock(return_value=[])
    return m


@pytest.fixture
def manager(mock_chroma):
    return GracePeriodManager(chroma_manager=mock_chroma, grace_days=7)


@pytest.fixture
def protected_memory():
    mem = Memory(
        id="mem_protected",
        type=MemoryType.FACT,
        content="我叫张三",
        user_id="user_1",
        confidence=0.9,
        storage_layer=StorageLayer.EPISODIC,
    )
    mem.add_protection(ProtectionFlag.CORE_IDENTITY)
    return mem


@pytest.fixture
def valuable_memory():
    return Memory(
        id="mem_valuable",
        type=MemoryType.FACT,
        content="明天有个重要面试",
        user_id="user_1",
        confidence=0.6,
        access_count=2,
        emotional_weight=0.5,
        storage_layer=StorageLayer.EPISODIC,
    )


@pytest.fixture
def low_value_memory():
    return Memory(
        id="mem_low",
        type=MemoryType.INTERACTION,
        content="嗯",
        user_id="user_1",
        confidence=0.2,
        access_count=0,
        emotional_weight=0.1,
        storage_layer=StorageLayer.EPISODIC,
    )


class TestEvaluateAndApply:
    @pytest.mark.asyncio
    async def test_protected_memory_not_deleted(self, manager, protected_memory):
        result = await manager.evaluate_and_apply(protected_memory)
        assert result == "protected"

    @pytest.mark.asyncio
    async def test_user_requested_is_protected(self, manager):
        mem = Memory(
            id="m1", type=MemoryType.FACT, content="记住",
            user_id="u1", is_user_requested=True,
            storage_layer=StorageLayer.EPISODIC,
        )
        result = await manager.evaluate_and_apply(mem)
        assert result == "protected"

    @pytest.mark.asyncio
    async def test_confirmed_is_protected(self, manager):
        mem = Memory(
            id="m2", type=MemoryType.FACT, content="已确认",
            user_id="u1", quality_level=QualityLevel.CONFIRMED,
            storage_layer=StorageLayer.EPISODIC,
        )
        result = await manager.evaluate_and_apply(mem)
        assert result == "protected"

    @pytest.mark.asyncio
    async def test_valuable_memory_enters_grace_period(self, manager, valuable_memory):
        result = await manager.evaluate_and_apply(valuable_memory)
        assert result == "grace_period"
        assert valuable_memory.grace_period_expires_at is not None
        assert valuable_memory.review_status == "pending_review"

    @pytest.mark.asyncio
    async def test_low_value_memory_silent_delete(self, manager, low_value_memory):
        result = await manager.evaluate_and_apply(low_value_memory)
        assert result == "silent_delete"

    @pytest.mark.asyncio
    async def test_already_pending_not_expired(self, manager, valuable_memory):
        valuable_memory.grace_period_expires_at = datetime.now() + timedelta(days=3)
        result = await manager.evaluate_and_apply(valuable_memory)
        assert result == "already_pending"

    @pytest.mark.asyncio
    async def test_expired_grace_period(self, manager, valuable_memory):
        valuable_memory.grace_period_expires_at = datetime.now() - timedelta(hours=1)
        result = await manager.evaluate_and_apply(valuable_memory)
        assert result == "expired"

    @pytest.mark.asyncio
    async def test_grace_period_persisted(self, manager, mock_chroma, valuable_memory):
        await manager.evaluate_and_apply(valuable_memory)
        mock_chroma.update_memory.assert_called()


class TestResolveGracePeriod:
    @pytest.mark.asyncio
    async def test_keep_clears_grace(self, manager, valuable_memory):
        valuable_memory.grace_period_expires_at = datetime.now() + timedelta(days=3)
        valuable_memory.review_status = "pending_review"
        result = await manager.resolve_grace_period(valuable_memory, "keep")
        assert result.grace_period_expires_at is None
        assert result.review_status is None

    @pytest.mark.asyncio
    async def test_keep_refreshes_access(self, manager, valuable_memory):
        old_time = datetime.now() - timedelta(days=10)
        valuable_memory.last_access_time = old_time
        valuable_memory.grace_period_expires_at = datetime.now() + timedelta(days=3)
        result = await manager.resolve_grace_period(valuable_memory, "keep")
        assert result.last_access_time > old_time

    @pytest.mark.asyncio
    async def test_archive_sets_rejected(self, manager, valuable_memory):
        result = await manager.resolve_grace_period(valuable_memory, "archive")
        assert result.review_status == "rejected"

    @pytest.mark.asyncio
    async def test_upgrade_changes_layer(self, manager, valuable_memory):
        valuable_memory.storage_layer = StorageLayer.EPISODIC
        result = await manager.resolve_grace_period(valuable_memory, "upgrade")
        assert result.storage_layer == StorageLayer.SEMANTIC
        assert result.review_status == "approved"
