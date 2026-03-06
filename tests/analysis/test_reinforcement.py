"""记忆回顾强化引擎测试"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from iris_memory.analysis.reinforcement import (
    MemoryReinforcementEngine,
    ReviewPromptGenerator,
)
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def mock_chroma():
    m = AsyncMock()
    m.get_memories_by_storage_layer = AsyncMock(return_value=[])
    m.get_memory = AsyncMock(return_value=None)
    m.update_memory = AsyncMock(return_value=True)
    return m


@pytest.fixture
def engine(mock_chroma):
    return MemoryReinforcementEngine(chroma_manager=mock_chroma)


@pytest.fixture
def sample_memories():
    """创建一批候选记忆"""
    now = datetime.now()
    return [
        Memory(
            id="mem_low_rif",
            type=MemoryType.FACT,
            content="用户喜欢Python",
            user_id="user_1",
            rif_score=0.2,
            importance_score=0.6,
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=now - timedelta(days=20),
        ),
        Memory(
            id="mem_mid_rif",
            type=MemoryType.EMOTION,
            content="用户对学习很开心",
            user_id="user_1",
            rif_score=0.4,
            importance_score=0.7,
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=now - timedelta(days=10),
        ),
        Memory(
            id="mem_high_rif",
            type=MemoryType.FACT,
            content="用户住在北京",
            user_id="user_1",
            rif_score=0.8,
            importance_score=0.5,
            storage_layer=StorageLayer.SEMANTIC,
            last_access_time=now - timedelta(days=2),
        ),
    ]


class TestReviewPromptGenerator:
    def test_generate_fact_prompt(self):
        mem = Memory(
            id="m1", type=MemoryType.FACT, content="用户喜欢编程",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
        )
        prompt = ReviewPromptGenerator.generate(mem, style="caring")
        assert "用户喜欢编程" in prompt

    def test_generate_emotion_prompt(self):
        mem = Memory(
            id="m2", type=MemoryType.EMOTION, content="今天很开心",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
        )
        prompt = ReviewPromptGenerator.generate(mem, style="casual")
        assert "今天很开心" in prompt

    def test_generate_relationship_prompt(self):
        mem = Memory(
            id="m3", type=MemoryType.RELATIONSHIP, content="和小明是好朋友",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
        )
        prompt = ReviewPromptGenerator.generate(mem, style="curious")
        assert "和小明是好朋友" in prompt

    def test_long_content_truncated(self):
        long_content = "这是一段很长的文本" * 20
        mem = Memory(
            id="m4", type=MemoryType.FACT, content=long_content,
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
        )
        prompt = ReviewPromptGenerator.generate(mem)
        assert "..." in prompt

    def test_unknown_type_falls_back_to_fact(self):
        mem = Memory(
            id="m5", type=MemoryType.INTERACTION, content="你好",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
        )
        prompt = ReviewPromptGenerator.generate(mem)
        assert isinstance(prompt, str) and len(prompt) > 0


class TestGetReviewCandidates:
    @pytest.mark.asyncio
    async def test_low_rif_prioritized(self, engine, mock_chroma, sample_memories):
        mock_chroma.get_memories_by_storage_layer = AsyncMock(
            side_effect=lambda layer: [m for m in sample_memories if m.storage_layer == layer]
        )
        candidates = await engine.get_review_candidates("user_1", max_count=3)
        if len(candidates) >= 2:
            assert candidates[0].rif_score <= candidates[1].rif_score

    @pytest.mark.asyncio
    async def test_daily_limit(self, engine, mock_chroma, sample_memories):
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=sample_memories)
        # Record 3 reviews today
        for i in range(3):
            engine.record_review(f"some_id_{i}", "user_1")
        candidates = await engine.get_review_candidates("user_1", max_count=3)
        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_no_chroma_returns_empty(self):
        engine = MemoryReinforcementEngine(chroma_manager=None)
        candidates = await engine.get_review_candidates("user_1")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_pending_review_excluded(self, engine, mock_chroma):
        mem = Memory(
            id="m_pending", type=MemoryType.FACT, content="test",
            user_id="user_1", importance_score=0.8, rif_score=0.2,
            storage_layer=StorageLayer.EPISODIC,
            review_status="pending_review",
        )
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=[mem])
        candidates = await engine.get_review_candidates("user_1")
        assert len(candidates) == 0


class TestProcessReviewResponse:
    @pytest.mark.asyncio
    async def test_positive_response_refreshes_access(self, engine, mock_chroma):
        old_time = datetime.now() - timedelta(days=10)
        mem = Memory(
            id="m_review", type=MemoryType.FACT, content="用户喜欢画画",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            last_access_time=old_time, importance_score=0.5,
        )
        mock_chroma.get_memory = AsyncMock(return_value=mem)
        await engine.process_review_response("m_review", "u1", "还好啊")
        assert mem.last_access_time > old_time
        mock_chroma.update_memory.assert_called_once_with(mem)

    @pytest.mark.asyncio
    async def test_negative_response_marks_rejected(self, engine, mock_chroma):
        mem = Memory(
            id="m_neg", type=MemoryType.FACT, content="test",
            user_id="u1", storage_layer=StorageLayer.EPISODIC,
            importance_score=0.6,
        )
        mock_chroma.get_memory = AsyncMock(return_value=mem)
        await engine.process_review_response("m_neg", "u1", "不用记了")
        assert mem.review_status == "rejected"
        assert mem.importance_score < 0.6


class TestRecordReview:
    def test_record_increments_count(self, engine):
        engine.record_review("m1", "u1")
        engine.record_review("m2", "u1")
        assert engine._get_today_review_count("u1") == 2

    def test_separate_user_counts(self, engine):
        engine.record_review("m1", "u1")
        engine.record_review("m2", "u2")
        assert engine._get_today_review_count("u1") == 1
        assert engine._get_today_review_count("u2") == 1


class TestLifecycle:
    """start() / stop() 生命周期管理测试"""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        assert engine._is_running is True
        assert engine._task is not None
        assert not engine._task.done()
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        await engine.stop()
        assert engine._is_running is False
        assert engine._task is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.start()
        first_task = engine._task
        await engine.start()  # second start should be no-op
        assert engine._task is first_task
        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, mock_chroma):
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine.stop()  # should not raise
        assert engine._is_running is False


class TestReviewLoop:
    """_review_loop / _run_review_cycle 后台循环测试"""

    @pytest.mark.asyncio
    async def test_run_review_cycle_notifies(self, mock_chroma, sample_memories):
        notify = AsyncMock()
        mock_chroma.get_active_user_ids = AsyncMock(return_value=["user_1"])
        mock_chroma.get_memories_by_storage_layer = AsyncMock(
            side_effect=lambda layer: [m for m in sample_memories if m.storage_layer == layer]
        )

        engine = MemoryReinforcementEngine(
            chroma_manager=mock_chroma,
            notify_callback=notify,
        )
        await engine._run_review_cycle()
        assert notify.call_count > 0

    @pytest.mark.asyncio
    async def test_run_review_cycle_no_notify_callback(self, mock_chroma):
        """No notify_callback → cycle does nothing"""
        engine = MemoryReinforcementEngine(chroma_manager=mock_chroma)
        await engine._run_review_cycle()  # should not raise

    @pytest.mark.asyncio
    async def test_run_review_cycle_no_chroma(self):
        """No chroma → cycle does nothing"""
        notify = AsyncMock()
        engine = MemoryReinforcementEngine(notify_callback=notify)
        await engine._run_review_cycle()
        notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_review_cycle_handles_error(self, mock_chroma):
        """Active user lookup failure is handled gracefully"""
        mock_chroma.get_active_user_ids = AsyncMock(side_effect=RuntimeError("db error"))
        notify = AsyncMock()
        engine = MemoryReinforcementEngine(
            chroma_manager=mock_chroma, notify_callback=notify,
        )
        await engine._run_review_cycle()  # should not raise
        notify.assert_not_called()


class TestConfigIntegration:
    """配置联动测试"""

    def test_constructor_overrides_config(self):
        engine = MemoryReinforcementEngine(
            review_interval_hours=12,
            max_daily_reviews=5,
        )
        assert engine._review_interval_hours == 12
        assert engine._max_daily_reviews == 5

    def test_default_config_fallback(self):
        """Without ConfigStore, defaults are used"""
        engine = MemoryReinforcementEngine()
        assert engine._max_daily_reviews == MemoryReinforcementEngine.DEFAULT_MAX_DAILY_REVIEWS
        assert engine._review_interval_hours == MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS

    @pytest.mark.asyncio
    async def test_max_daily_from_config_applies(self, mock_chroma, sample_memories):
        """max_daily_reviews 控制每日回顾上限"""
        mock_chroma.get_memories_by_storage_layer = AsyncMock(return_value=sample_memories)
        engine = MemoryReinforcementEngine(
            chroma_manager=mock_chroma,
            max_daily_reviews=1,
        )
        candidates = await engine.get_review_candidates("user_1")
        assert len(candidates) <= 1
