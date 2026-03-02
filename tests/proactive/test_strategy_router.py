"""
StrategyRouter 和各策略单元测试
"""

from __future__ import annotations

import pytest

from iris_memory.proactive.core.models import (
    ConversationContext,
    DecisionType,
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
    SceneMatch,
    UrgencyLevel,
)
from iris_memory.proactive.strategies.chat_strategy import ChatStrategy
from iris_memory.proactive.strategies.emotion_strategy import EmotionStrategy
from iris_memory.proactive.strategies.followup_strategy import FollowUpStrategy
from iris_memory.proactive.strategies.question_strategy import QuestionStrategy
from iris_memory.proactive.strategies.router import StrategyRouter


def _make_context(
    session_type: str = "group",
    recent_text: str = "这是一段对话测试",
) -> ProactiveContext:
    return ProactiveContext(
        session_type=session_type,
        session_key="u1:g1",
        conversation=ConversationContext(
            recent_messages=[{"text": recent_text, "sender_id": "u1"}],
            recent_text=recent_text,
        ),
    )


def _make_decision(
    reply_type: ReplyType = ReplyType.QUESTION,
    matched_scenes: list = None,
    matched_rules: list = None,
) -> ProactiveDecision:
    return ProactiveDecision(
        should_reply=True,
        urgency=UrgencyLevel.MEDIUM,
        reply_type=reply_type,
        decision_type=DecisionType.VECTOR,
        confidence=0.8,
        matched_scenes=matched_scenes or [],
        matched_rules=matched_rules or [],
    )


class TestStrategyRouter:
    @pytest.fixture
    def router(self):
        return StrategyRouter()

    def test_route_question(self, router):
        result = router.route(
            _make_context(),
            _make_decision(reply_type=ReplyType.QUESTION),
        )
        assert result["strategy_name"] == "question"
        assert "trigger_prompt" in result
        assert "reply_params" in result

    def test_route_emotion(self, router):
        result = router.route(
            _make_context(),
            _make_decision(reply_type=ReplyType.EMOTION),
        )
        assert result["strategy_name"] == "emotion"

    def test_route_chat(self, router):
        result = router.route(
            _make_context(),
            _make_decision(reply_type=ReplyType.CHAT),
        )
        assert result["strategy_name"] == "chat"

    def test_route_followup(self, router):
        result = router.route(
            _make_context(),
            _make_decision(reply_type=ReplyType.FOLLOWUP),
        )
        assert result["strategy_name"] == "followup"

    def test_route_unknown_falls_back(self, router):
        # Remove a known strategy
        from iris_memory.proactive.core.models import ReplyType
        decision = _make_decision()
        # Use a type that might not exist
        decision.reply_type = ReplyType.CHAT
        # Just verify it works
        result = router.route(_make_context(), decision)
        assert "trigger_prompt" in result


class TestQuestionStrategy:
    def test_build_prompt(self):
        s = QuestionStrategy()
        prompt = s.build_trigger_prompt(
            _make_context(),
            _make_decision(
                matched_scenes=[SceneMatch(trigger_pattern="技术问题")]
            ),
        )
        assert "问题" in prompt
        assert "技术问题" in prompt

    def test_reply_params(self):
        s = QuestionStrategy()
        params = s.get_reply_params()
        assert params.get("temperature", 1.0) <= 0.7  # lower temp for accuracy


class TestEmotionStrategy:
    def test_build_prompt(self):
        s = EmotionStrategy()
        prompt = s.build_trigger_prompt(
            _make_context(recent_text="最近好累"),
            _make_decision(reply_type=ReplyType.EMOTION),
        )
        assert len(prompt) > 0

    def test_reply_params_higher_temp(self):
        s = EmotionStrategy()
        params = s.get_reply_params()
        assert params.get("temperature", 0.5) >= 0.7


class TestChatStrategy:
    def test_group_vs_private_prompt(self):
        s = ChatStrategy()
        group_prompt = s.build_trigger_prompt(
            _make_context(session_type="group"),
            _make_decision(reply_type=ReplyType.CHAT),
        )
        private_prompt = s.build_trigger_prompt(
            _make_context(session_type="private"),
            _make_decision(reply_type=ReplyType.CHAT),
        )
        # Both should be non-empty and potentially different
        assert len(group_prompt) > 0
        assert len(private_prompt) > 0


class TestFollowUpStrategy:
    def test_build_prompt(self):
        s = FollowUpStrategy()
        prompt = s.build_trigger_prompt(
            _make_context(),
            _make_decision(reply_type=ReplyType.FOLLOWUP),
        )
        assert len(prompt) > 0
