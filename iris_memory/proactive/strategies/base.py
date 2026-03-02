"""
回复策略基类

所有回复策略（Question / Emotion / Chat / FollowUp）继承自此基类，
统一提示词构建和回复行为。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyType,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.strategy.base")


class BaseStrategy(ABC):
    """回复策略抽象基类

    每个策略负责：
    1. 根据决策和上下文构建触发提示词（trigger prompt）
    2. 提供该策略的默认回复参数（温度、最大 token 等）

    策略不直接调用 LLM 或发送消息 —— 它只生成提示词，
    由 ProactiveManager 统一走 AstrBot Pipeline。
    """

    def __init__(self, reply_type: ReplyType) -> None:
        self._reply_type = reply_type

    @property
    def reply_type(self) -> ReplyType:
        """此策略对应的回复类型"""
        return self._reply_type

    @abstractmethod
    def build_trigger_prompt(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建注入 LLM 的触发提示词

        Args:
            context: 主动回复上下文
            decision: 最终决策
            extra: 策略特定的额外参数

        Returns:
            触发提示词字符串
        """
        ...

    def get_reply_params(self) -> Dict[str, Any]:
        """获取回复参数（温度、token上限等）

        Returns:
            参数字典，可被具体策略覆盖
        """
        return {
            "temperature": 0.7,
            "max_tokens": 150,
        }
