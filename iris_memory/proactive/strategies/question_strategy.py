"""
问题回复策略

处理 reply_type=QUESTION 的场景，优先准确回答，可引用记忆。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.proactive.strategies.base import BaseStrategy


class QuestionStrategy(BaseStrategy):
    """问题回复策略

    特点：
    - 优先准确回答
    - 可引用记忆
    - 语气正式但友好
    """

    def __init__(self) -> None:
        super().__init__(ReplyType.QUESTION)

    def build_trigger_prompt(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        recent = context.conversation.recent_text[:200]
        scene_desc = ""
        if decision.matched_scenes:
            scene_desc = decision.matched_scenes[0].trigger_pattern

        prompt = (
            f"用户似乎在询问一个问题，请基于上下文给出有帮助的回答。\n"
            f"对话摘要：{recent}\n"
        )

        if scene_desc:
            prompt += f"匹配场景：{scene_desc}\n"

        prompt += (
            "要求：\n"
            "- 直接回答问题，不绕弯子\n"
            "- 如果不确定答案，诚实表示并提供参考方向\n"
            "- 语气友好自然\n"
            "- 回复简短精炼（不超过100字）"
        )
        return prompt

    def get_reply_params(self) -> Dict[str, Any]:
        return {
            "temperature": 0.6,
            "max_tokens": 200,
        }
