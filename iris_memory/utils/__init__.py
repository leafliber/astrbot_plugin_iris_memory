"""
Iris Chat Memory - 公共工具模块

提供通用工具函数，供各模块使用。
"""

from .token_counter import count_tokens, get_encoder
from .forgetting import (
    calculate_recency,
    calculate_frequency,
    calculate_confidence,
    calculate_isolation_degree,
    calculate_forgetting_score,
    should_evict,
)
from .input_sanitizer import sanitize_input, is_injection_attempt
from .persistence import atomic_write_text, atomic_write_json

__all__ = [
    # Token 计数
    "count_tokens",
    "get_encoder",
    # 遗忘权重算法
    "calculate_recency",
    "calculate_frequency",
    "calculate_confidence",
    "calculate_isolation_degree",
    "calculate_forgetting_score",
    "should_evict",
    # 输入清理
    "sanitize_input",
    "is_injection_attempt",
    # 原子持久化
    "atomic_write_text",
    "atomic_write_json",
]
