"""
主动回复核心数据模型

定义主动回复模块中所有共享的数据结构，包括上下文、检测结果、
决策、反馈等。所有模型均为不可变或半不可变 dataclass。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ========== 枚举定义 ==========


class UrgencyLevel(str, Enum):
    """回复紧急程度"""
    HIGH = "high"          # 高紧急（L1直接回复、强信号）
    MEDIUM = "medium"      # 中紧急（L2/L3确认后回复）
    LOW = "low"            # 低紧急（保守场景）


class ReplyType(str, Enum):
    """回复类型（决定使用哪个策略）"""
    QUESTION = "question"      # 问题回复
    EMOTION = "emotion"        # 情感支持
    CHAT = "chat"              # 闲聊参与
    FOLLOWUP = "followup"      # 多轮跟进


class DecisionType(str, Enum):
    """决策类型（标识哪级检测器做出的决策）"""
    RULE = "rule"              # L1: 规则检测器
    VECTOR = "vector"          # L2: 向量检测器
    LLM = "llm"                # L3: LLM确认检测器


class SceneType(str, Enum):
    """场景类型"""
    QUESTION = "question"      # 技术/知识问题
    EMOTION = "emotion"        # 情感支持
    CHAT = "chat"              # 日常闲聊
    FOLLOWUP = "followup"      # 话题延续


class PersonalityType(str, Enum):
    """回复人格类型"""
    RESERVED = "reserved"      # 内敛
    BALANCED = "balanced"      # 均衡
    PROACTIVE = "proactive"    # 主动


# ========== 上下文数据结构 ==========


@dataclass
class ConversationContext:
    """对话上下文"""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    recent_text: str = ""                  # 拼接后的文本（用于向量化）
    time_span: float = 0.0                 # 时间跨度（秒）
    silence_duration: float = 0.0          # 当前沉默时长（秒）
    topic_continuity: float = 0.0          # 话题连续性评分（0-1）
    current_topic_vector: Optional[List[float]] = None  # 当前话题向量
    base_query_vector: Optional[List[float]] = None     # 基础查询向量


@dataclass
class UserContext:
    """用户维度上下文"""
    user_id: str = ""
    persona: Optional[Any] = None              # UserPersona
    emotional_state: Optional[Any] = None      # EmotionalState
    proactive_preference: float = 0.5          # 用户偏好（0-1）
    reply_history: List[ReplyRecord] = field(default_factory=list)


@dataclass
class GroupContext:
    """群聊维度上下文（仅群聊场景）"""
    group_id: str = ""
    activity_level: float = 0.5            # 群活跃度 (0-1)
    topic_heat: Dict[str, float] = field(default_factory=dict)
    participant_count: int = 0             # 参与人数
    last_bot_reply_ago: float = 0.0        # 上次Bot回复距今秒数


@dataclass
class TemporalContext:
    """时间维度上下文"""
    hour: int = 0                       # 当前小时（服务器本地时间）
    is_weekend: bool = False            # 是否周末
    is_holiday: bool = False            # 是否节假日
    is_quiet_hours: bool = False        # 是否在静音时段


@dataclass
class ProactiveContext:
    """主动回复完整上下文

    由 ContextEngine 聚合产生，传递给 DecisionEngine 进行决策。
    """
    # 会话标识
    session_type: str = "group"          # "group" | "private"
    session_key: str = ""                # user_id:group_id 或 user_id

    # 各维度上下文
    conversation: ConversationContext = field(default_factory=ConversationContext)
    user: UserContext = field(default_factory=UserContext)
    group: Optional[GroupContext] = None  # 私聊时为 None
    temporal: TemporalContext = field(default_factory=TemporalContext)

    # 检测状态
    has_new_user_message: bool = False   # 自上次Bot回复后是否有新的用户消息
    new_participant_count: int = 0       # 自上次Bot回复后新参与发言的人数（仅群聊）


# ========== 检测器结果 ==========


@dataclass
class RuleResult:
    """L1 规则检测器结果"""
    score: float = 0.0                   # 综合得分
    signals: Dict[str, float] = field(default_factory=dict)  # 各信号及其权重
    should_reply: bool = False           # 是否应该回复
    urgency: UrgencyLevel = UrgencyLevel.LOW
    confidence: float = 0.0              # 置信度 (0-1)
    matched_rules: List[str] = field(default_factory=list)  # 匹配的规则列表
    reply_type: ReplyType = ReplyType.CHAT  # 推荐的回复类型


@dataclass
class SceneMatch:
    """场景匹配结果"""
    scene_id: str = ""
    scene_type: SceneType = SceneType.CHAT
    similarity: float = 0.0              # 原始向量相似度
    weighted_similarity: float = 0.0     # 加权后相似度
    final_score: float = 0.0             # 最终分数
    success_rate: float = 0.5            # 历史成功率
    usage_count: int = 0                 # 使用次数
    exploration_mode: bool = False       # 是否处于探索期
    trigger_pattern: str = ""            # 触发场景描述


@dataclass
class VectorResult:
    """L2 向量检测器结果"""
    matches: List[SceneMatch] = field(default_factory=list)
    best_match: Optional[SceneMatch] = None
    final_score: float = 0.0             # 最佳匹配最终分数
    confidence: float = 0.0
    should_reply: bool = False
    reply_type: ReplyType = ReplyType.CHAT


@dataclass
class LLMResult:
    """L3 LLM确认检测器结果"""
    should_reply: bool = False
    urgency: UrgencyLevel = UrgencyLevel.LOW
    reason: str = ""
    confidence: float = 0.0
    reply_type: ReplyType = ReplyType.CHAT


# ========== 跟进检测 ==========


@dataclass
class FollowUpState:
    """跟进状态"""
    count: int = 0


@dataclass
class FollowUpDecision:
    """跟进决策"""
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    reply_type: str = "followup"
    followup_count: int = 0
    original_decision_id: str = ""


# ========== 最终决策 ==========


@dataclass
class ProactiveDecision:
    """最终的主动回复决策

    由 DecisionEngine 产出，决定是否回复、如何回复。
    """
    should_reply: bool = False
    urgency: UrgencyLevel = UrgencyLevel.LOW
    reply_type: ReplyType = ReplyType.CHAT
    decision_type: DecisionType = DecisionType.RULE
    confidence: float = 0.0
    reason: str = ""

    # 匹配的场景信息（L2命中时）
    matched_scenes: List[SceneMatch] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)

    # 跟进信息（FOLLOWUP时）
    followup_count: int = 0
    original_decision_id: str = ""

    # 元数据
    detection_latency_ms: float = 0.0    # 检测耗时
    llm_used: bool = False               # 是否调用了LLM


# ========== 回复记录与反馈 ==========


@dataclass
class ReplyRecord:
    """回复记录

    每次主动回复后存储，用于反馈追踪和跟进检测。
    """
    record_id: str = ""
    session_key: str = ""
    session_type: str = ""               # group / private
    scene_ids: List[str] = field(default_factory=list)  # 匹配的场景ID列表
    decision_type: str = ""              # rule / vector / llm
    urgency: str = ""
    reply_type: str = ""
    confidence: float = 0.0
    sent_at: datetime = field(default_factory=datetime.now)
    content_summary: str = ""
    topic_vector: Optional[List[float]] = None  # 话题向量（用于跟进检测）


@dataclass
class ReplyFeedback:
    """回复反馈

    追踪回复效果，用于更新场景权重。
    """
    feedback_id: str = ""
    record_id: str = ""
    user_replied: bool = False           # 用户是否直接回复Bot
    reply_within_window: bool = False    # 窗口内是否有消息
    engagement_score: float = 0.0        # 参与度评分
    recorded_at: datetime = field(default_factory=datetime.now)


# ========== 场景模型 ==========


@dataclass
class ProactiveScene:
    """预定义的主动回复场景

    存储在 ChromaDB 中的场景元数据。
    success_rate 和 usage_count 存储在 SQLite 中。
    """
    scene_id: str = ""
    description: str = ""                # 场景描述（用于向量化）
    keywords: List[str] = field(default_factory=list)
    scene_type: str = "chat"             # question / emotion / chat / followup
    target_emotion: Optional[str] = None
    time_pattern: Optional[str] = None   # any / morning / afternoon / evening
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    # 来自 SQLite 的动态字段（非持久化在 ChromaDB）
    success_rate: float = 0.5
    usage_count: int = 0
    exploration_mode: bool = False


@dataclass
class SceneWeight:
    """场景权重（存储在 SQLite scene_weights 表）"""
    scene_id: str = ""
    success_rate: float = 0.5
    usage_count: int = 0
    last_used: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)


# ========== 配置相关 ==========


@dataclass
class PersonalityConfig:
    """人格配置（根据 personality 类型和 session_type 生成）"""
    rule_direct_reply: float = 0.7
    vector_threshold_high: float = 0.85
    vector_threshold_mid: float = 0.6
    cooldown_multiplier: float = 1.0
    followup_max_count: int = 2
    llm_quota_per_hour: int = 5


def get_personality_config(
    personality: str,
    session_type: str,
) -> PersonalityConfig:
    """根据性格类型和会话类型获取配置

    Args:
        personality: 性格类型 (reserved / balanced / proactive)
        session_type: 会话类型 (group / private)

    Returns:
        PersonalityConfig 实例
    """
    configs = {
        "reserved": PersonalityConfig(
            rule_direct_reply=0.8 if session_type == "group" else 0.7,
            vector_threshold_high=0.9,
            vector_threshold_mid=0.7,
            cooldown_multiplier=1.5,
            followup_max_count=1,
            llm_quota_per_hour=3,
        ),
        "balanced": PersonalityConfig(
            rule_direct_reply=0.7 if session_type == "group" else 0.6,
            vector_threshold_high=0.85,
            vector_threshold_mid=0.6,
            cooldown_multiplier=1.0,
            followup_max_count=2,
            llm_quota_per_hour=5,
        ),
        "proactive": PersonalityConfig(
            rule_direct_reply=0.6 if session_type == "group" else 0.5,
            vector_threshold_high=0.8,
            vector_threshold_mid=0.5,
            cooldown_multiplier=0.7,
            followup_max_count=3,
            llm_quota_per_hour=10,
        ),
    }
    return configs.get(personality, configs["balanced"])


# ========== 工具函数 ==========


def calculate_engagement(feedback: ReplyFeedback) -> float:
    """计算简化的参与度评分

    Args:
        feedback: 回复反馈

    Returns:
        参与度评分 (0.0 - 1.0)
    """
    if feedback.user_replied:
        return 1.0
    elif feedback.reply_within_window:
        return 0.3
    else:
        return 0.0


def is_quiet_hours(hour: int, quiet_hours: List[int]) -> bool:
    """判断当前是否在静音时段

    Args:
        hour: 当前小时 (0-23)
        quiet_hours: [start, end]，支持跨午夜

    Returns:
        是否在静音时段

    Examples:
        >>> is_quiet_hours(1, [23, 7])   # 凌晨1点，在23:00-07:00内
        True
        >>> is_quiet_hours(12, [23, 7])  # 中午12点，不在23:00-07:00内
        False
    """
    if len(quiet_hours) != 2:
        return False
    start, end = quiet_hours
    if start <= end:
        return start <= hour < end
    else:  # 跨午夜
        return hour >= start or hour < end


def count_tokens(text: str) -> int:
    """计算 token 数（保守估算）

    使用保守估算策略，避免引入额外 tokenizer 依赖。
    中文字符按 1 token 计，其他字符按 0.5 token 估算。

    Args:
        text: 输入文本

    Returns:
        估算的 token 数
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars // 2) + 1
