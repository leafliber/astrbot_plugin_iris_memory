"""
Iris Chat Memory - 输入清理模块

对所有进入插件的外部输入统一执行 Prompt 注入过滤，
过滤逻辑集中在输入网关层。

覆盖范围：
- 用户消息
- TOOL 调用参数
- 插件间接口入参
"""

import re

from iris_memory.core import get_logger

logger = get_logger("utils.input_sanitizer")

_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)", re.IGNORECASE
    ),
    re.compile(
        r"forget\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)", re.IGNORECASE
    ),
    re.compile(
        r"disregard\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|jailbreak|unrestricted)", re.IGNORECASE
    ),
    re.compile(r"^system\s*:\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"###\s*(?:system|instruction|human|assistant)\s*:", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(
        r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?:a\s+)?(?:unrestricted|DAN|jailbreak)",
        re.IGNORECASE,
    ),
    re.compile(
        r"bypass\s+(?:your|the)\s+(?:restrictions?|rules?|guidelines?)", re.IGNORECASE
    ),
    re.compile(
        r"override\s+(?:your|the)\s+(?:restrictions?|rules?|guidelines?)", re.IGNORECASE
    ),
    re.compile(
        r"do\s+not\s+(?:follow|obey|comply\s+with)\s+(?:your|the)\s+rules",
        re.IGNORECASE,
    ),
    re.compile(r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"show\s+me\s+(?:your|the)\s+(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(
        r"what\s+(?:are|is)\s+(?:your|the)\s+(?:system\s+)?prompt", re.IGNORECASE
    ),
]


def sanitize_input(text: str, source: str = "unknown") -> str:
    """对外部输入执行 Prompt 注入过滤

    检测并清理潜在的 Prompt 注入攻击内容。
    不修改原文，仅记录警告日志，由调用方决定是否拒绝。

    Args:
        text: 待检查的输入文本
        source: 输入来源标识（用于日志）

    Returns:
        清理后的文本（移除注入模式部分）
    """
    if not text:
        return text

    from iris_memory.config import get_config

    config = get_config()

    if not config.get("input_sanitizer_enable"):
        return text

    max_length = config.get("input_sanitizer_max_length")

    if len(text) > max_length:
        logger.warning(f"输入过长（{len(text)} 字符），来源：{source}，已截断")
        text = text[:max_length]

    detected_patterns = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            detected_patterns.append(match.group())

    if detected_patterns:
        logger.warning(
            f"检测到潜在 Prompt 注入，来源：{source}，匹配模式：{detected_patterns[:3]}"
        )
        text = _strip_injection_patterns(text)

    return text


def is_injection_attempt(text: str) -> bool:
    """检测输入是否为 Prompt 注入攻击

    Args:
        text: 待检查的输入文本

    Returns:
        是否检测到注入模式
    """
    if not text:
        return False

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True

    return False


def _strip_injection_patterns(text: str) -> str:
    """移除文本中的注入模式

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("", text)

    return text.strip()
