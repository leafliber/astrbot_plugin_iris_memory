"""
闲聊参与策略

处理 reply_type=CHAT 的场景，轻松参与，不抢话题。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.proactive.strategies.base import BaseStrategy


class ChatStrategy(BaseStrategy):
    """闲聊参与策略

    特点：
    - 轻松参与对话
    - 不抢话题
    - 适时分享相关经验或看法
    """

    def __init__(self) -> None:
        super().__init__(ReplyType.CHAT)

    def build_trigger_prompt(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        recent = context.conversation.recent_text[:200]

        session_hint = ""
        if context.session_type == "group":
            session_hint = "这是群聊场景，注意不要喧宾夺主。"
        else:
            session_hint = "这是私聊场景，可以更加自然亲切。"

        prompt = (
            f"用户在闲聊，请自然地参与对话。\n"
            f"对话摘要：{recent}\n"
            f"{session_hint}\n"
            "要求：\n"
            "- 自然融入对话，不刻意\n"
            "- 可以分享相关的看法或经验\n"
            "- 不要转移话题\n"
            "- 回复轻松简短（不超过60字）"
        )
        return prompt

    def get_reply_params(self) -> Dict[str, Any]:
        return {
            "temperature": 0.8,
            "max_tokens": 120,
        }
