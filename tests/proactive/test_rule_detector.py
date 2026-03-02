"""
RuleDetector 单元测试
"""

from __future__ import annotations

import pytest

from iris_memory.proactive.core.models import (
    ConversationContext,
    ProactiveContext,
    ReplyType,
    UrgencyLevel,
)
from iris_memory.proactive.detectors.rule_detector import RuleDetector


@pytest.fixture
def detector():
    return RuleDetector(personality="balanced")


def _make_context(text: str, session_type: str = "group") -> ProactiveContext:
    return ProactiveContext(
        session_type=session_type,
        session_key="user1:group1" if session_type == "group" else "user1:private",
        conversation=ConversationContext(
            recent_messages=[{"text": text, "sender_id": "user1", "sender_name": "Alice"}],
        ),
    )


class TestRuleDetector:
    @pytest.mark.asyncio
    async def test_question_detected(self, detector):
        ctx = _make_context("这个怎么用？有教程吗")
        result = await detector.detect(ctx)
        assert result.score > 0
        assert "question" in result.matched_rules

    @pytest.mark.asyncio
    async def test_mention_detected(self, detector):
        ctx = _make_context("你怎么看这个问题")
        result = await detector.detect(ctx)
        assert "mention_bot" in result.matched_rules
        assert result.score >= 0.4

    @pytest.mark.asyncio
    async def test_emotion_negative(self, detector):
        ctx = _make_context("好烦啊工作压力太大了")
        result = await detector.detect(ctx)
        assert any("emotion" in r for r in result.matched_rules)

    @pytest.mark.asyncio
    async def test_short_confirm_rejected(self, detector):
        ctx = _make_context("嗯")
        result = await detector.detect(ctx)
        assert not result.should_reply
        assert result.score < 0

    @pytest.mark.asyncio
    async def test_empty_message(self, detector):
        ctx = _make_context("")
        result = await detector.detect(ctx)
        assert not result.should_reply

    @pytest.mark.asyncio
    async def test_high_score_direct_reply(self, detector):
        # mention(0.4) + question(0.3) = 0.7 → direct reply for group
        ctx = _make_context("你怎么看这个问题？怎么解决", "group")
        result = await detector.detect(ctx)
        assert result.should_reply
        assert result.urgency == UrgencyLevel.HIGH

    @pytest.mark.asyncio
    async def test_private_lower_threshold(self, detector):
        # In private, threshold is 0.6 (lower)
        ctx = _make_context("你怎么看这个问题？怎么解决", "private")
        result = await detector.detect(ctx)
        assert result.should_reply

    @pytest.mark.asyncio
    async def test_low_score_fast_reject(self, detector):
        ctx = _make_context("今天天气不错")
        result = await detector.detect(ctx)
        # No strong signals → low score → fast reject or pass to L2
        assert result.score <= 0.2 or not result.should_reply

    @pytest.mark.asyncio
    async def test_attention_seeking(self, detector):
        ctx = _make_context("有人吗，好无聊")
        result = await detector.detect(ctx)
        assert "attention" in result.matched_rules
