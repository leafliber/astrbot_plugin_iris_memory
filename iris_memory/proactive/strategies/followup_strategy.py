"""
多轮跟进策略

处理 reply_type=FOLLOWUP 的场景，保持上下文连贯，不重复。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.proactive.strategies.base import BaseStrategy


class FollowUpStrategy(BaseStrategy):
    """多轮跟进策略

    特点：
    - 保持上下文连贯
    - 不重复之前说过的内容
    - 推进对话而非重复
    """

    def __init__(self) -> None:
        super().__init__(ReplyType.FOLLOWUP)

    def build_trigger_prompt(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        recent = context.conversation.recent_text[:200]

        followup_hint = ""
        if decision.followup_count > 0:
            followup_hint = f"这是第{decision.followup_count}次跟进回复。"

        prompt = (
            f"用户延续了之前的话题，请自然地跟进对话。\n"
            f"对话摘要：{recent}\n"
            f"{followup_hint}\n"
            "要求：\n"
            "- 延续之前的话题方向\n"
            "- 不重复之前说过的内容\n"
            "- 可以提出新的角度或问题\n"
            "- 回复简短自然（不超过80字）"
        )
        return prompt

    def get_reply_params(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "max_tokens": 150,
        }
