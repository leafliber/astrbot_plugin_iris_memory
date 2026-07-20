from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import tiktoken

from astrbot.api import logger

from .config import ConfigManager
from .state import StateManager

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

_RE_DIGITS_ONLY = re.compile(r"[\d\s]+")
_RE_NONALNUM_ONLY = re.compile(r"[\W_]+")


@dataclass
class WindowMessage:
    sender_id: str
    sender_name: str
    content: str
    timestamp: float


class Gatekeeper:
    def __init__(self, config: ConfigManager, state: StateManager) -> None:
        self._config = config
        self._state = state

    def should_process(self, event: AstrMessageEvent) -> bool:
        message_str = event.message_str
        if not message_str:
            return False
        if message_str.startswith("/"):
            return False
        if event.is_private_chat():
            return False
        if self._state.is_muted():
            return False
        group_id = event.get_group_id()
        if not self._state.is_whitelisted(group_id):
            return False
        return True

    def quality_score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        stripped = text.strip()
        if _RE_DIGITS_ONLY.fullmatch(stripped):
            return 0.1
        if len(stripped) <= 2:
            return 0.2
        if _RE_NONALNUM_ONLY.fullmatch(stripped):
            return 0.1
        # str.isalnum() 对 CJK 字符同样返回 True，无需额外的区间判断
        alpha_count = sum(1 for c in stripped if c.isalnum())
        ratio = alpha_count / len(stripped) if stripped else 0
        return min(1.0, ratio)


class SlidingWindow:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._windows: dict[str, deque[WindowMessage]] = {}

    def _ensure_window(self, group_id: str) -> deque[WindowMessage]:
        if group_id not in self._windows:
            self._windows[group_id] = deque(maxlen=self._config.window_size)
        return self._windows[group_id]

    def append(self, group_id: str, msg: WindowMessage) -> None:
        window = self._ensure_window(group_id)
        window.append(msg)

    def get_messages(self, group_id: str) -> list[WindowMessage]:
        window = self._ensure_window(group_id)
        return list(window)

    def remove_group(self, group_id: str) -> None:
        self._windows.pop(group_id, None)

    def cleanup(self, active_group_ids: set[str]) -> None:
        stale = [gid for gid in self._windows if gid not in active_group_ids]
        for gid in stale:
            self._windows.pop(gid, None)


class ContextPackager:
    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning("Iris Reply: tiktoken init failed, falling back to char estimate: %s", e)
            self._encoding = None

    def _count_tokens(self, text: str) -> int:
        if self._encoding:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 2)

    def package(
        self,
        group_id: str,
        messages: list[WindowMessage],
        motive: str,
    ) -> str:
        lines: list[str] = []
        token_counts: list[int] = []
        for msg in messages:
            line = f"[{msg.sender_name}({msg.sender_id})] {msg.content}"
            tc = self._count_tokens(line)
            lines.append(line)
            token_counts.append(tc)

        total_tokens = sum(token_counts)
        max_tokens = self._config.max_token

        start = 0
        while total_tokens > max_tokens and start < len(lines):
            total_tokens -= token_counts[start]
            start += 1

        context_text = "\n".join(lines[start:])

        escaped_motive = quoteattr(motive)
        header = f"<iris:reply-context motive={escaped_motive}>\n"
        footer = "\n</iris:reply-context>"
        return header + context_text + footer
