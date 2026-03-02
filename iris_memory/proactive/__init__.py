"""
主动回复模块（v2）

架构概览：
- manager.py          : ProactiveManager (Facade)
- core/               : 上下文引擎、决策引擎、反馈追踪、数据模型
- detectors/          : 三级漏斗检测器 (Rule → Vector → LLM)
- strategies/         : 回复策略 (Question / Emotion / Chat / FollowUp)
- storage/            : 场景向量存储 (ChromaDB) + 反馈数据存储 (SQLite)
- data/               : 预定义场景 + 场景初始化器
- web/                : Web 管理界面 (可选)
"""

from iris_memory.proactive.core.models import (
    UrgencyLevel,
    ReplyType,
    DecisionType,
    SceneType,
    PersonalityType,
    ProactiveContext,
    ProactiveDecision,
    ReplyRecord,
    ReplyFeedback,
    ProactiveScene,
    PersonalityConfig,
    get_personality_config,
)

from iris_memory.proactive.manager import ProactiveManager

__all__ = [
    # Facade
    "ProactiveManager",
    # 数据模型
    "UrgencyLevel",
    "ReplyType",
    "DecisionType",
    "SceneType",
    "PersonalityType",
    "ProactiveContext",
    "ProactiveDecision",
    "ReplyRecord",
    "ReplyFeedback",
    "ProactiveScene",
    "PersonalityConfig",
    "get_personality_config",
]
