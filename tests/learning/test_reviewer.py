"""攒批 LLM 审查器测试"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from iris_memory.learning.reviewer import LearningReviewer


def _seed_pending(storage, pairs=0, patterns=0):
    """插入 pending 数据，返回 (pair_ids, pattern_ids)"""
    pair_ids = [
        storage.insert_pair("g1", "u1", f"问{i}", f"答{i}") for i in range(pairs)
    ]
    pattern_ids = [
        storage.insert_pattern("g1", "chat", f"表达{i}") for i in range(patterns)
    ]
    return pair_ids, pattern_ids


class TestEnqueue:
    """待审队列"""

    def test_enqueue_and_queue_size(self, config, storage):
        reviewer = LearningReviewer(storage)
        reviewer.enqueue(1)
        reviewer.enqueue(2)
        reviewer.enqueue_pattern(10)
        assert reviewer.queue_size == 3

    def test_enqueue_dedupe(self, config, storage):
        reviewer = LearningReviewer(storage)
        reviewer.enqueue(1)
        reviewer.enqueue(1)
        assert reviewer.queue_size == 1

    def test_batch_full_threshold(self, config, storage):
        # 默认 review_batch_size=10
        reviewer = LearningReviewer(storage)
        for i in range(9):
            reviewer.enqueue(i + 1)
        assert not reviewer.is_batch_full()
        reviewer.enqueue(10)
        assert reviewer.is_batch_full()


class TestRunReview:
    """批量审查执行"""

    @pytest.mark.asyncio
    async def test_review_applies_verdicts(self, config, storage):
        pair_ids, pattern_ids = _seed_pending(storage, pairs=2, patterns=1)
        reviewer = LearningReviewer(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value=(
                f'[{{"id": {pair_ids[0]}, "type": "pair", "pass": true, "reason": "ok"}},'
                f'{{"id": {pair_ids[1]}, "type": "pair", "pass": false, "reason": "复读"}},'
                f'{{"id": {pattern_ids[0]}, "type": "pattern", "pass": true, "reason": "ok"}}]'
            )
        )
        result = await reviewer.run_review(llm)
        assert result is True
        approved_shots = storage.get_approved_few_shots("g1", 10)
        assert [s["id"] for s in approved_shots] == [pair_ids[0]]
        approved_patterns = storage.get_approved_patterns("g1", 10)
        assert [p["id"] for p in approved_patterns] == [pattern_ids[0]]
        # 拒绝的不再 pending
        assert storage.get_pending_pairs(10) == []

    @pytest.mark.asyncio
    async def test_review_malformed_json_keeps_pending(self, config, storage):
        _seed_pending(storage, pairs=1, patterns=1)
        reviewer = LearningReviewer(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(return_value="无法解析的输出")
        result = await reviewer.run_review(llm)
        assert result is False
        assert len(storage.get_pending_pairs(10)) == 1
        assert len(storage.get_pending_patterns(10)) == 1

    @pytest.mark.asyncio
    async def test_review_llm_exception_retry_once(self, config, storage):
        _seed_pending(storage, pairs=1)
        reviewer = LearningReviewer(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(side_effect=RuntimeError("LLM 挂了"))
        result = await reviewer.run_review(llm)
        assert result is False
        # 重试一次后放弃：共调用 2 次
        assert llm.generate_direct.call_count == 2
        assert len(storage.get_pending_pairs(10)) == 1

    @pytest.mark.asyncio
    async def test_review_uncovered_items_stay_pending(self, config, storage):
        """LLM 未覆盖的条目保持 pending 下轮再审"""
        pair_ids, _ = _seed_pending(storage, pairs=2)
        reviewer = LearningReviewer(storage)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value=(
                f'[{{"id": {pair_ids[0]}, "type": "pair", "pass": true, "reason": "ok"}}]'
            )
        )
        result = await reviewer.run_review(llm)
        assert result is True
        pending = storage.get_pending_pairs(10)
        assert [p["id"] for p in pending] == [pair_ids[1]]

    @pytest.mark.asyncio
    async def test_review_queue_synced_after_judgement(self, config, storage):
        """内存队列移除已裁决条目"""
        pair_ids, _ = _seed_pending(storage, pairs=2)
        reviewer = LearningReviewer(storage)
        for pid in pair_ids:
            reviewer.enqueue(pid)
        llm = MagicMock()
        llm.generate_direct = AsyncMock(
            return_value=(
                f'[{{"id": {pair_ids[0]}, "type": "pair", "pass": true, "reason": "ok"}},'
                f'{{"id": {pair_ids[1]}, "type": "pair", "pass": false, "reason": "bad"}}]'
            )
        )
        await reviewer.run_review(llm)
        assert reviewer.queue_size == 0

    @pytest.mark.asyncio
    async def test_review_empty_pending_noop(self, config, storage):
        reviewer = LearningReviewer(storage)
        reviewer.enqueue(999)  # 队列有残留但库里没有
        llm = MagicMock()
        llm.generate_direct = AsyncMock()
        result = await reviewer.run_review(llm)
        assert result is False
        llm.generate_direct.assert_not_called()
        # 队列滞后于库（重启场景），以库为准清空
        assert reviewer.queue_size == 0
