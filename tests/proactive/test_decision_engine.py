"""
DecisionEngine 和 FollowUpDetector 单元测试
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from iris_memory.proactive.core.decision_engine import (
    DecisionEngine,
    FollowUpDetector,
    _cosine_similarity,
)
from iris_memory.proactive.core.models import (
    ConversationContext,
    DecisionType,
    ProactiveContext,
    ProactiveDecision,
    ReplyRecord,
    ReplyType,
    RuleResult,
    SceneMatch,
    TemporalContext,
    UrgencyLevel,
    VectorResult,
)


def _make_context(
    session_type: str = "group",
    session_key: str = "u1:g1",
    has_new_msg: bool = True,
    is_quiet: bool = False,
    topic_vector: list = None,
    new_participants: int = 0,
) -> ProactiveContext:
    return ProactiveContext(
        session_type=session_type,
        session_key=session_key,
        conversation=ConversationContext(
            recent_messages=[{"text": "测试消息", "sender_id": "u1"}],
            recent_text="测试消息",
            current_topic_vector=topic_vector,
        ),
        temporal=TemporalContext(
            hour=12,
            is_quiet_hours=is_quiet,
        ),
        has_new_user_message=has_new_msg,
        new_participant_count=new_participants,
    )


# ========== FollowUpDetector ==========


class TestFollowUpDetector:
    @pytest.fixture
    def detector(self):
        return FollowUpDetector(
            feedback_store=None,
            max_count=2,
            time_window=120,
            similarity_threshold=0.6,
        )

    def test_no_last_record(self, detector):
        ctx = _make_context()
        result = detector.check_followup(ctx, None)
        assert result is None

    def test_time_window_expired(self, detector):
        ctx = _make_context()
        old_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=300),
        )
        result = detector.check_followup(ctx, old_record)
        assert result is None

    def test_no_new_user_message(self, detector):
        ctx = _make_context(has_new_msg=False)
        recent_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=30),
        )
        result = detector.check_followup(ctx, recent_record)
        assert result is None

    def test_semantically_unrelated(self, detector):
        ctx = _make_context(topic_vector=[1.0, 0.0, 0.0])
        recent_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=30),
            topic_vector=[0.0, 1.0, 0.0],  # orthogonal
        )
        result = detector.check_followup(ctx, recent_record)
        assert result is None

    def test_followup_succeeds(self, detector):
        ctx = _make_context(topic_vector=[1.0, 0.0, 0.0])
        recent_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=30),
            topic_vector=[0.9, 0.1, 0.0],  # similar
        )
        result = detector.check_followup(ctx, recent_record)
        assert result is not None
        assert result.followup_count == 1

    def test_followup_max_count(self, detector):
        ctx = _make_context()
        recent_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=30),
        )
        # Fill up followup count
        detector._followup_states[ctx.session_key] = MagicMock(count=2)
        result = detector.check_followup(ctx, recent_record)
        assert result is None

    def test_group_other_participant(self, detector):
        ctx = _make_context(new_participants=1)
        recent_record = ReplyRecord(
            record_id="r1",
            sent_at=datetime.now() - timedelta(seconds=30),
        )
        result = detector.check_followup(ctx, recent_record)
        assert result is None


# ========== DecisionEngine ==========


class TestDecisionEngine:
    @pytest.fixture
    def mock_rule(self):
        d = AsyncMock()
        d.detect = AsyncMock(return_value=RuleResult())
        return d

    @pytest.fixture
    def mock_vector(self):
        d = AsyncMock()
        d.detect = AsyncMock(return_value=VectorResult())
        return d

    @pytest.fixture
    def mock_llm(self):
        d = AsyncMock()
        d.detect = AsyncMock()
        return d

    @pytest.fixture
    def engine(self, mock_rule, mock_vector, mock_llm):
        return DecisionEngine(
            rule_detector=mock_rule,
            vector_detector=mock_vector,
            llm_detector=mock_llm,
            personality="balanced",
        )

    @pytest.mark.asyncio
    async def test_quiet_hours_rejected(self, engine):
        ctx = _make_context(is_quiet=True)
        result = await engine.decide(ctx)
        assert not result.should_reply
        assert result.reason == "quiet_hours"

    @pytest.mark.asyncio
    async def test_l1_direct_reply(self, engine, mock_rule):
        mock_rule.detect.return_value = RuleResult(
            score=0.8,
            should_reply=True,
            confidence=0.8,
            matched_rules=["question", "mention_bot"],
            reply_type=ReplyType.QUESTION,
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert result.should_reply
        assert result.urgency == UrgencyLevel.HIGH
        assert result.decision_type == DecisionType.RULE

    @pytest.mark.asyncio
    async def test_l1_fast_reject(self, engine, mock_rule):
        mock_rule.detect.return_value = RuleResult(
            score=0.05,
            should_reply=False,
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert not result.should_reply
        assert "L1_fast_reject" in result.reason

    @pytest.mark.asyncio
    async def test_l2_high_confidence(self, engine, mock_rule, mock_vector):
        mock_rule.detect.return_value = RuleResult(
            score=0.4,
            should_reply=False,
        )
        mock_vector.detect.return_value = VectorResult(
            final_score=0.9,
            should_reply=True,
            confidence=0.9,
            reply_type=ReplyType.QUESTION,
            matches=[SceneMatch(scene_id="s1")],
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert result.should_reply
        assert result.decision_type == DecisionType.VECTOR
        assert "L2_high_confidence" in result.reason

    @pytest.mark.asyncio
    async def test_l2_low_confidence_rejected(self, engine, mock_rule, mock_vector):
        mock_rule.detect.return_value = RuleResult(score=0.4, should_reply=False)
        mock_vector.detect.return_value = VectorResult(
            final_score=0.3,
            should_reply=False,
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert not result.should_reply
        assert "L2_low_confidence" in result.reason

    @pytest.mark.asyncio
    async def test_l3_confirm(self, engine, mock_rule, mock_vector, mock_llm):
        from iris_memory.proactive.core.models import LLMResult

        mock_rule.detect.return_value = RuleResult(score=0.4, should_reply=False)
        mock_vector.detect.return_value = VectorResult(
            final_score=0.7,  # between mid(0.6) and high(0.85) for balanced/group
            should_reply=False,
            matches=[SceneMatch(scene_id="s1")],
        )
        mock_llm.detect.return_value = LLMResult(
            should_reply=True,
            urgency=UrgencyLevel.MEDIUM,
            confidence=0.8,
            reason="user needs help",
            reply_type=ReplyType.QUESTION,
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert result.should_reply
        assert result.decision_type == DecisionType.LLM
        assert result.llm_used

    @pytest.mark.asyncio
    async def test_l3_reject(self, engine, mock_rule, mock_vector, mock_llm):
        from iris_memory.proactive.core.models import LLMResult

        mock_rule.detect.return_value = RuleResult(score=0.4, should_reply=False)
        mock_vector.detect.return_value = VectorResult(
            final_score=0.7,
            should_reply=False,
        )
        mock_llm.detect.return_value = LLMResult(
            should_reply=False,
            reason="not relevant",
        )
        ctx = _make_context()
        result = await engine.decide(ctx)
        assert not result.should_reply
        assert "L3_rejected" in result.reason


# ========== Cosine Similarity ==========


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0, 0], [0, 1, 0])) < 0.001

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1, 2], [1, 2, 3]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
