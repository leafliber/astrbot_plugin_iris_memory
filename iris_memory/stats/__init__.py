"""
LLM 统计模块

提供全局 LLM 调用统计，支持：
- 自动来源推断
- 按 provider_id 和 source 分类统计
- 多维度查询
- 持久化到 AstrBot KV 存储

Usage:
    from iris_memory.stats import get_stats_registry, StatsQuery
    
    # 获取统计注册表
    registry = get_stats_registry()
    
    # 查询
    query = StatsQuery(provider_id="openai-gpt-4", limit=50)
    records = registry.query(query)
    
    # 获取聚合统计
    aggregated = registry.get_aggregated()
    
    # 获取摘要
    summary = registry.get_summary()
"""

from iris_memory.stats.registry import (
    LLMStatsRegistry,
    get_stats_registry,
    SOURCE_ALIASES,
)
from iris_memory.stats.models import (
    LLMCallRecord,
    LLMAggregatedStats,
    StatsQuery,
    StatsSummary,
)
from iris_memory.stats.store import StatsKVStore

__all__ = [
    "get_stats_registry",
    "LLMStatsRegistry",
    "LLMCallRecord",
    "LLMAggregatedStats",
    "StatsQuery",
    "StatsSummary",
    "StatsKVStore",
    "SOURCE_ALIASES",
]
