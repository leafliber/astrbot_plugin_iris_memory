"""
情感支持策略

处理 reply_type=EMOTION 的场景，共情优先，语气适配情绪。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.proactive.strategies.base import BaseStrategy


class EmotionStrategy(BaseStrategy):
    """情感支持策略

    特点：
    - 共情优先
    - 语气适配情绪（正面情绪积极回应，负面情绪温柔安慰）
    - 避免说教
    """

    def __init__(self) -> None:
        super().__init__(ReplyType.EMOTION)

    def build_trigger_prompt(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        recent = context.conversation.recent_text[:200]

        emotion = "unknown"
        if context.user.emotional_state:
            emotion = getattr(
                context.user.emotional_state,
                "primary",
                getattr(context.user.emotional_state, "emotion", "unknown"),
            )

        prompt = (
            f"用户表达了{emotion}的情绪，请给予适当的情感回应。\n"
            f"对话摘要：{recent}\n"
            "要求：\n"
            "- 先表达理解和共情\n"
            "- 不要说教或给建议（除非用户主动请求）\n"
            "- 语气温暖自然\n"
            "- 回复简短温馨（不超过80字）"
        )
        return prompt

    def get_reply_params(self) -> Dict[str, Any]:
        return {
            "temperature": 0.8,
            "max_tokens": 150,
        }
