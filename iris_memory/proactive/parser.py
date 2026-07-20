from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger


@dataclass
class Decision:
    """LLM 统一决策的结构化结果。"""

    action: str = "none"  # "speak" | "none"
    mode: str = ""  # 确认后的发言模式（跟随请求动机）
    message: str = ""  # initiate 模式下的直发内容
    observation: str = ""
    watch: list[str] = field(default_factory=list)
    watch_keywords: list[str] = field(default_factory=list)
    why: str = ""
    drifted: bool = False
    cooldown_minutes: int = 0
    parse_failed: bool = False

    @property
    def should_speak(self) -> bool:
        return self.action == "speak"


def extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取第一个合法 JSON 对象。

    依次尝试：markdown 代码块剥离 -> True/False 修正 -> 整体解析 ->
    括号配平扫描（跳过字符串内部的花括号）。
    """
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    text = re.sub(r'"\s*:\s*True', '": true', text)
    text = re.sub(r'"\s*:\s*False', '": false', text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    in_string = False
    escape = False
    brace_count = 0
    start = -1
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if brace_count == 0:
                start = i
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = -1

    return None


def parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1")
    if isinstance(val, (int, float)):
        return bool(val)
    return bool(val)


def parse_string_list(raw: Any, max_len: int = 10) -> list[str]:
    if not isinstance(raw, list):
        return []
    result = [str(u).strip() for u in raw if str(u).strip()]
    return result[:max_len]


def _parse_action(raw: Any, fallback_reply: Any) -> str:
    """解析 action 字段；缺失时兼容旧的 reply 布尔字段。"""
    if raw is not None:
        text = str(raw).strip().lower()
        if text in ("speak", "reply", "yes", "true"):
            return "speak"
        if text in ("none", "skip", "no", "false", ""):
            return "none"
    return "speak" if parse_bool(fallback_reply) else "none"


def _parse_cooldown(raw: Any) -> int:
    try:
        minutes = int(float(raw))
    except (TypeError, ValueError):
        return 0
    if minutes <= 0:
        return 0
    return min(minutes, 120)


def parse_decision(text: str, mode: str = "") -> Decision:
    """解析 LLM 统一决策输出，兼容多种字段命名。"""
    obj = extract_json(text)
    if not obj:
        logger.warning("Iris Reply: decision JSON parse failed, raw text: %.300s", text)
        return Decision(mode=mode, parse_failed=True)

    action = _parse_action(obj.get("action"), obj.get("reply"))
    message = str(obj.get("message", "") or "").strip()
    observation = str(obj.get("obs", obj.get("observation", "")))
    watch = parse_string_list(obj.get("watch", obj.get("follow_up_users", [])))
    watch_keywords = parse_string_list(
        obj.get("watch_keywords", obj.get("follow_up_keywords", [])),
        max_len=10,
    )
    why = str(obj.get("why", obj.get("interest_reason", "")))
    drifted = parse_bool(obj.get("drifted", obj.get("topic_drifted", False)))
    cooldown = _parse_cooldown(obj.get("cooldown", 0))

    # 防御：不要求发言时忽略 message；drifted 时不应发言
    if action != "speak":
        message = ""
    if drifted and action == "speak":
        action = "none"
        message = ""

    return Decision(
        action=action,
        mode=mode,
        message=message,
        observation=observation,
        watch=watch,
        watch_keywords=watch_keywords,
        why=why,
        drifted=drifted,
        cooldown_minutes=cooldown,
    )
