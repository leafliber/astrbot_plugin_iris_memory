"""
FeedbackTracker 单元测试
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.proactive.core.feedback_tracker import FeedbackTracker
from iris_memory.proactive.core.models import (
    ConversationContext,
    DecisionType,
    ProactiveContext,
    ProactiveDecision,
    ReplyRecord,
    ReplyType,
    SceneMatch,
    SceneWeight,
    UrgencyLevel,
)


@pytest.fixture
def mock_feedback_store():
    store = AsyncMock()
    store.record_reply = AsyncMock()
    store.record_feedback = AsyncMock()
    store.get_recent_replies = AsyncMock(return_value=[])
    store.get_scene_weight = AsyncMock(return_value=None)
    store.upsert_scene_weight = AsyncMock()
    store.record_weight_change = AsyncMock()
    store.increment_daily_stats = AsyncMock()
    return store


@pytest.fixture
def mock_scene_store():
    store = AsyncMock()
    store.deactivate = AsyncMock()
    return store


@pytest.fixture
def tracker(mock_feedback_store, mock_scene_store):
    return FeedbackTracker(
        feedback_store=mock_feedback_store,
        scene_store=mock_scene_store,
        tracking_window=300,
        ema_alpha=0.2,
    )


def _make_context(session_key: str = "u1:g1") -> ProactiveContext:
    return ProactiveContext(
        session_type="group",
        session_key=session_key,
        conversation=ConversationContext(
            recent_messages=[{"text": "test", "sender_id": "u1"}],
            current_topic_vector=[0.1, 0.2, 0.3],
        ),
    )


def _make_decision(
    reply_type: ReplyType = ReplyType.QUESTION,
    matched_rules: list = None,
    matched_scenes: list = None,
) -> ProactiveDecision:
    return ProactiveDecision(
        should_reply=True,
        urgency=UrgencyLevel.MEDIUM,
        reply_type=reply_type,
        decision_type=DecisionType.RULE,
        confidence=0.8,
        matched_rules=matched_rules or ["question"],
        matched_scenes=matched_scenes or [],
    )


class TestFeedbackTracker:
    @pytest.mark.asyncio
    async def test_record_reply(self, tracker, mock_feedback_store):
        ctx = _make_context()
        decision = _make_decision()
        record_id = await tracker.record_reply(ctx, decision, "test content")
        assert record_id is not None
        assert record_id.startswith("reply_")
        mock_feedback_store.record_reply.assert_called_once()
        assert "u1:g1" in tracker._pending_feedback

    @pytest.mark.asyncio
    async def test_record_reply_no_store(self):
        tracker = FeedbackTracker(feedback_store=None)
        result = await tracker.record_reply(
            _make_context(), _make_decision()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_record_reply_with_scene_ids(self, tracker, mock_feedback_store):
        ctx = _make_context()
        decision = _make_decision(
            matched_scenes=[SceneMatch(scene_id="scene_1")],
            matched_rules=[],
        )
        record_id = await tracker.record_reply(ctx, decision)
        assert record_id is not None
        call_args = mock_feedback_store.record_reply.call_args
        record = call_args[0][0]
        assert "scene_1" in record.scene_ids

    @pytest.mark.asyncio
    async def test_record_reply_virtual_scene_ids(self, tracker, mock_feedback_store):
        ctx = _make_context()
        decision = _make_decision(matched_rules=["question", "mention_bot"])
        record_id = await tracker.record_reply(ctx, decision)
        assert record_id is not None
        call_args = mock_feedback_store.record_reply.call_args
        record = call_args[0][0]
        assert "rule:question" in record.scene_ids

    @pytest.mark.asyncio
    async def test_process_user_response(self, tracker, mock_feedback_store):
        # Set up pending feedback
        tracker._pending_feedback["u1:g1"] = "reply_abc"
        await tracker.process_user_response("u1:g1", user_replied_directly=True)
        mock_feedback_store.record_feedback.assert_called_once()
        assert "u1:g1" not in tracker._pending_feedback

    @pytest.mark.asyncio
    async def test_process_user_response_no_pending(self, tracker, mock_feedback_store):
        await tracker.process_user_response("u1:g1", user_replied_directly=True)
        mock_feedback_store.record_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_no_response(self, tracker, mock_feedback_store):
        tracker._pending_feedback["u1:g1"] = "reply_abc"
        await tracker.process_no_response("u1:g1")
        mock_feedback_store.record_feedback.assert_called_once()
        call_args = mock_feedback_store.record_feedback.call_args
        feedback = call_args[0][0]
        assert feedback.user_replied is False
        assert feedback.engagement_score == 0.0

    @pytest.mark.asyncio
    async def test_weight_update_ema(self, tracker, mock_feedback_store):
        """Test EMA weight update for a scene"""
        mock_feedback_store.get_scene_weight.return_value = SceneWeight(
            scene_id="s1",
            success_rate=0.5,
            usage_count=10,
        )
        await tracker._update_single_scene_weight("s1", engagement=1.0)
        mock_feedback_store.upsert_scene_weight.assert_called_once()
        call_kwargs = mock_feedback_store.upsert_scene_weight.call_args[1]
        # EMA: 0.8 * 0.5 + 0.2 * 1.0 = 0.6
        assert abs(call_kwargs["success_rate"] - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_auto_deactivation(self, tracker, mock_feedback_store, mock_scene_store):
        """Low success_rate + many uses → auto-deactivate"""
        mock_feedback_store.get_scene_weight.return_value = SceneWeight(
            scene_id="s1",
            success_rate=0.15,  # low
            usage_count=10,     # enough uses
        )
        # EMA: 0.8 * 0.15 + 0.2 * 0.0 = 0.12 → below 0.2
        await tracker._update_single_scene_weight("s1", engagement=0.0)
        mock_scene_store.deactivate.assert_called_with("s1")

    def test_get_pending_sessions(self, tracker):
        tracker._pending_feedback["a"] = "r1"
        tracker._pending_feedback["b"] = "r2"
        assert set(tracker.get_pending_sessions()) == {"a", "b"}
