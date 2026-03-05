"""
主动回复模块（v3）

架构概览：
- manager.py            : ProactiveManager (Facade)
- models.py             : 数据模型（Signal, FollowUpExpectation 等）
- signal_queue.py       : 信号队列（按群隔离）
- signal_generator.py   : 信号生成器（改造自 v2 RuleDetector）
- group_scheduler.py    : 群定时器调度器
- followup_planner.py   : FollowUp 跟进规划器
- storage/              : ExpectationStore（内存存储）

配置说明：
所有配置直接从全局 ConfigStore 读取（使用 get_store()），
无需单独的 ProactiveConfig 类。
"""

from iris_memory.proactive.manager import ProactiveManager
from iris_memory.proactive.models import (
    AggregatedDecision,
    FollowUpDecision,
    FollowUpExpectation,
    FollowUpReplyType,
    ProactiveReplyResult,
    Signal,
    SignalType,
)

__all__ = [
    # Facade
    "ProactiveManager",
    # 数据模型
    "Signal",
    "SignalType",
    "FollowUpExpectation",
    "FollowUpReplyType",
    "FollowUpDecision",
    "AggregatedDecision",
    "ProactiveReplyResult",
]
