"""Web 模块通用工具函数

提供参数解析、格式化等通用工具。
"""

from __future__ import annotations

from typing import Optional


def safe_int(
    value: Optional[str],
    default: int,
    min_val: int = 1,
    max_val: int = 10000,
) -> int:
    """安全解析整数参数，并限制范围

    Args:
        value: 字符串值
        default: 默认值
        min_val: 最小值
        max_val: 最大值

    Returns:
        解析后的整数，在 [min_val, max_val] 范围内
    """
    try:
        n = int(value) if value else default
    except (ValueError, TypeError):
        n = default
    return max(min_val, min(n, max_val))
