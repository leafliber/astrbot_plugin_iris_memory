"""
消息指纹计算工具

提供统一的消息指纹生成函数，用于去重和缓存。
避免 ``MessageMerger`` 和 ``MessageClassifier`` 各自维护独立实现。
"""

from __future__ import annotations

import hashlib


def compute_message_fingerprint(
    content: str,
    *,
    max_length: int = 80,
    hash_length: int = 12,
) -> str:
    """计算文本的简短指纹。

    算法：移除非字母数字 → 转小写 → 截断 → MD5 前缀。

    Args:
        content: 原始文本
        max_length: 简化后保留的最大字符数
        hash_length: MD5 hex 截取长度

    Returns:
        str: 长度为 *hash_length* 的十六进制指纹
    """
    simplified = "".join(c.lower() for c in content if c.isalnum())
    simplified = simplified[:max_length]
    return hashlib.md5(simplified.encode()).hexdigest()[:hash_length]
