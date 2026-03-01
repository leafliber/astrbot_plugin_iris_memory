"""
masking.py - 敏感信息脱敏工具

从 sensitivity_detector 提取的独立脱敏逻辑。
"""

from __future__ import annotations


def mask_sensitive(value: str) -> str:
    """对敏感信息进行脱敏处理

    规则：
    - 长度 <= 4: 全部替换为 *
    - 长度 <= 8: 保留首尾各1字符
    - 长度 > 8: 保留首尾各4字符

    Args:
        value: 需要脱敏的字符串

    Returns:
        脱敏后的字符串
    """
    length = len(value)
    if length <= 4:
        return "*" * length
    if length <= 8:
        return value[0] + "*" * (length - 2) + value[-1]
    return value[:4] + "*" * (length - 8) + value[-4:]
