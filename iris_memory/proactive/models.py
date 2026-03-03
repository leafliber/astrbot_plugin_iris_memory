"""
主动回复 v3 数据模型

定义 Signal、FollowUpExpectation 等核心数据结构。
所有模型均使用 dataclass，字段不可变或半不可变。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ========== 枚举定义 ==========


class SignalType(str, Enum):
    """信号类型"""
    EMOTION_HIGH = "emotion_high"   # 高情感信号
    RULE_MATCH = "rule_match"       # 规则匹配信号


class FollowUpReplyType(str, Enum):
    """跟进回复类型"""
    ACKNOWLEDGE = "acknowledge"         # 确认回应
    CONTINUE_TOPIC = "continue_topic"   # 继续话题
    EMOTION_SUPPORT = "emotion_support" # 情感支持
    QUESTION = "question"               # 提问互动


# ========== Signal 模型 ==========


@dataclass
class Signal:
    """信号对象

    由 SignalGenerator 生成，存入 SignalQueue。

    Attributes:
        signal_id: 唯一标识
        signal_type: 信号类型
        session_key: 会话标识（user_id:group_id）
        group_id: 群组 ID
        user_id: 用户 ID
        weight: 信号权重（0.0 - 1.0）
        created_at: 创建时间
        expires_at: 过期时间
        metadata: 附加元数据（如匹配的规则名称、情感强度等）
    """
    signal_type: SignalType
    session_key: str
    group_id: str
    user_id: str
    weight: float
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_expired(self) -> bool:
        """判断信号是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


# ========== FollowUpExpectation 模型 ==========


@dataclass
class FollowUpExpectation:
    """跟进期待对象

    Bot 主动回复后创建，用于监控触发者的后续消息。

    Attributes:
        expectation_id: 唯一标识
        session_key: 会话标识
        group_id: 群组 ID
        trigger_user_id: 触发主动回复的用户 ID
        trigger_message: 触发时的用户消息快照
        bot_reply_summary: Bot 回复的摘要
        followup_window_end: FollowUp 窗口截止时间
        short_window_end: 短期窗口截止时间（等待用户发言后设置）
        aggregated_messages: 聚合的用户消息
        followup_count: 当前连续跟进次数
        created_at: 创建时间
        recent_context: 近期对话上下文（用于 LLM 判断）
    """
    session_key: str
    group_id: str
    trigger_user_id: str
    trigger_message: str
    bot_reply_summary: str
    followup_window_end: datetime
    short_window_end: Optional[datetime] = None
    aggregated_messages: List[Dict[str, Any]] = field(default_factory=list)
    followup_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    recent_context: List[Dict[str, Any]] = field(default_factory=list)
    expectation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_window_expired(self) -> bool:
        """FollowUp 窗口是否已过期"""
        return datetime.now() >= self.followup_window_end

    @property
    def is_short_window_expired(self) -> bool:
        """短期窗口是否已过期"""
        if self.short_window_end is None:
            return False
        return datetime.now() >= self.short_window_end

    @property
    def has_aggregated_messages(self) -> bool:
        """是否有聚合的用户消息"""
        return len(self.aggregated_messages) > 0


# ========== 聚合决策结果 ==========


@dataclass
class AggregatedDecision:
    """聚合决策结果

    由 GroupScheduler 在聚合信号后产生。

    Attributes:
        should_reply: 是否应回复
        session_key: 目标会话
        group_id: 目标群组
        target_user_id: 目标用户
        aggregated_weight: 聚合后的信号权重
        signals: 聚合的信号列表
        reason: 决策原因
        recent_messages: 近期消息上下文
        llm_confirmed: 是否经过 LLM 确认
    """
    should_reply: bool
    session_key: str
    group_id: str
    target_user_id: str = ""
    aggregated_weight: float = 0.0
    signals: List[Signal] = field(default_factory=list)
    reason: str = ""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    llm_confirmed: bool = False


@dataclass
class FollowUpDecision:
    """FollowUp LLM 判断结果

    Attributes:
        should_reply: 是否需要跟进回复
        reason: 判断原因
        reply_type: 回复类型
        suggested_direction: 建议的回复方向
    """
    should_reply: bool
    reason: str = ""
    reply_type: FollowUpReplyType = FollowUpReplyType.ACKNOWLEDGE
    suggested_direction: str = ""


@dataclass
class ProactiveReplyResult:
    """主动回复最终结果

    由 ProactiveManager 返回给调用方。

    Attributes:
        trigger_prompt: 系统指令 prompt（注入到 LLM 上下文）
        reply_params: LLM 生成参数
        reason: 触发原因
        group_id: 目标群组 ID
        session_key: 会话标识
        target_user: 目标用户
        recent_messages: 近期消息上下文
        emotion_summary: 用户情绪摘要
        source: 触发来源（signal_queue / followup）
    """
    trigger_prompt: str
    reply_params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    group_id: str = ""
    session_key: str = ""
    target_user: str = ""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    emotion_summary: str = ""
    source: str = "signal_queue"
