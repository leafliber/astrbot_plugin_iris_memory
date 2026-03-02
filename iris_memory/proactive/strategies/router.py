"""
策略路由器

根据 ProactiveDecision 的 reply_type 分发到对应的回复策略。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.proactive.strategies.base import BaseStrategy
from iris_memory.proactive.strategies.chat_strategy import ChatStrategy
from iris_memory.proactive.strategies.emotion_strategy import EmotionStrategy
from iris_memory.proactive.strategies.followup_strategy import FollowUpStrategy
from iris_memory.proactive.strategies.question_strategy import QuestionStrategy
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.strategy_router")


class StrategyRouter:
    """策略路由器

    维护 reply_type → Strategy 的映射，根据决策选择合适策略。
    """

    def __init__(self) -> None:
        self._strategies: Dict[ReplyType, BaseStrategy] = {
            ReplyType.QUESTION: QuestionStrategy(),
            ReplyType.EMOTION: EmotionStrategy(),
            ReplyType.CHAT: ChatStrategy(),
            ReplyType.FOLLOWUP: FollowUpStrategy(),
        }

    def register(self, reply_type: ReplyType, strategy: BaseStrategy) -> None:
        """注册/替换策略"""
        self._strategies[reply_type] = strategy

    def route(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """路由到对应策略，生成触发 prompt 和回复参数

        Args:
            context: 主动回复上下文
            decision: 最终决策
            extra: 额外参数

        Returns:
            dict with "trigger_prompt" and "reply_params"
        """
        strategy = self._strategies.get(decision.reply_type)
        if not strategy:
            logger.warning(
                f"No strategy found for reply_type={decision.reply_type}, "
                f"falling back to CHAT"
            )
            strategy = self._strategies[ReplyType.CHAT]

        trigger_prompt = strategy.build_trigger_prompt(
            context, decision, extra
        )
        reply_params = strategy.get_reply_params()

        return {
            "trigger_prompt": trigger_prompt,
            "reply_params": reply_params,
            "strategy_name": strategy.reply_type.value,
        }
