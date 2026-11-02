"""会话收尾状态：continue / waiting / ended 与结束点（DESIGN.md §8）。

Portions adapted from SurfCore (commit 59ec3066, MIT, © 2026 Derpyu520 and
qq-bridge / QSH contributors): the continue / linger / close state idea, the
closure record (summary + next opening) and the "suppress proactive wake
during closure cooldown" rule from ``src/conversation-closure.js``.
See THIRD_PARTY_NOTICES.md in this package for the full MIT license text.

与 SurfCore 的差异：
- 不做打分；状态只由主人消息分类（rules.classify）与 Frisk 回复驱动。
- linger → waiting 仅表示“等我说完”的显式等待。
- 收尾只读事项（open_questions / next_opening），绝不创建或完成事项。

持久化：唯一写入口是 ContinuityStore.closures 表。
``last_bot_kind`` 语义为“主人最近一条消息之后 Frisk 回复的类别”，
主人每发一条新消息就清空，用来判断主人的问题是否已被回答。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from astrbot.api import logger

from . import models as m
from . import rules

__all__ = ["ClosureTracker", "STATE_CONTINUE", "STATE_WAITING", "STATE_ENDED", "ENDING_HINT"]

STATE_CONTINUE = "continue"
STATE_WAITING = "waiting"
STATE_ENDED = "ended"

ENDING_HINT = "她在收尾，简短回应即可，不必追加问题或再补结束语"

SUMMARY_MAX_CHARS = 80
OPEN_QUESTIONS_MAX = 5

# 视为“实质内容”的主人消息类别（用于结束点 summary）
_CONTENT_KINDS = frozenset({rules.SUBSTANTIVE, rules.QUESTION, rules.REQUEST})
# 需要 Frisk 回答的主人消息类别
_ASKING_KINDS = frozenset({rules.QUESTION, rules.REQUEST, rules.WAIT_DONE})


def _default_row(scope_key: str) -> dict[str, Any]:
    return {
        "scope_key": scope_key,
        "state": STATE_CONTINUE,
        "explicit_wait": 0,
        "wait_since": None,
        "ended_at": None,
        "summary": "",
        "open_questions": [],
        "next_opening": "",
        "last_owner_kind": "",
        "last_bot_kind": "",
        "updated_at": None,
    }


class ClosureTracker:
    """按 scope 维护收尾状态。同步方法（service 层 to_thread 调用），线程安全。

    所有公共方法在边界处捕获异常并降级（记录日志，返回安全默认值），
    收尾功能故障不得影响正常私聊。
    """

    def __init__(
        self,
        store: Any,
        clock: Callable[[], float] = time.time,
        *,
        explicit_wait_timeout_s: float = 180.0,
    ) -> None:
        self._store = store
        self._clock = clock
        self._wait_timeout = max(1.0, float(explicit_wait_timeout_s))
        self._lock = threading.RLock()
        # 最近一条主人实质消息（仅内存；重启后从空开始，已结束的 summary 已落盘）
        self._last_content: dict[str, str] = {}

    # ------------------------------------------------------------------ read

    def current(self, scope_key: str) -> dict[str, Any]:
        try:
            row = self._store.get_closure(scope_key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] closure read failed scope={scope_key}: {type(e).__name__}")
            row = None
        return dict(row) if row else _default_row(scope_key)

    def hint_for_reply(self, scope_key: str) -> str | None:
        """主人最新一条是收尾语且本会话已 ended 时，返回本轮回复提示。"""
        cur = self.current(scope_key)
        if cur.get("state") == STATE_ENDED and cur.get("last_owner_kind") == rules.ENDING:
            return ENDING_HINT
        return None

    def suppress_proactive(
        self,
        scope_key: str,
        now: float,
        cooldown_minutes: float,
        has_due_reminder: bool,
    ) -> bool:
        """ended 且仍在冷却期内、且没有到期的提醒项 → True（跳过主动）。"""
        if has_due_reminder:
            return False
        cur = self.current(scope_key)
        if cur.get("state") != STATE_ENDED:
            return False
        ended_at = cur.get("ended_at")
        if ended_at is None:
            return False
        try:
            return float(now) - float(ended_at) < max(0.0, float(cooldown_minutes)) * 60.0
        except (TypeError, ValueError):
            return False

    # ----------------------------------------------------------------- write

    def on_owner_message(self, scope_key: str, text: str | None, *, has_media: bool = False) -> str:
        """记录一条主人新消息，返回更新后的 state。"""
        try:
            kind = rules.classify(text, has_media=has_media)
            if kind == rules.SILENCE:
                return self.current(scope_key)["state"]
            with self._lock:
                return self._apply_owner(scope_key, kind, text or "")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] closure owner update failed scope={scope_key}: {type(e).__name__}")
            return STATE_CONTINUE

    def on_bot_sent(self, scope_key: str, text: str | None) -> str:
        """记录 Frisk 一条已发送的回复，返回更新后的 state。"""
        try:
            kind = rules.classify(text)
            if kind == rules.SILENCE:
                return self.current(scope_key)["state"]
            with self._lock:
                cur = self.current(scope_key)
                fields: dict[str, Any] = {"last_bot_kind": kind}
                if (
                    cur["state"] != STATE_ENDED
                    and cur.get("last_owner_kind") == rules.ACK
                    and rules.is_ending(text)
                ):
                    fields.update(self._end_point(scope_key))
                    fields["state"] = STATE_ENDED
                    fields["explicit_wait"] = 0
                    fields["wait_since"] = None
                row = self._store.put_closure(scope_key, **fields)
                if fields.get("state") == STATE_ENDED:
                    logger.info(f"[continuity] closure scope={scope_key} state=ended by=bot_ending")
                return row["state"]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] closure bot update failed scope={scope_key}: {type(e).__name__}")
            return STATE_CONTINUE

    # -------------------------------------------------------------- internals

    def _apply_owner(self, scope_key: str, kind: str, text: str) -> str:
        now = self._clock()
        cur = self.current(scope_key)
        prev_state = cur["state"]
        if kind in _CONTENT_KINDS and text.strip():
            self._last_content[scope_key] = m.clip(text, SUMMARY_MAX_CHARS)

        # 主人每条新消息都清空 last_bot_kind（尚未回复）
        fields: dict[str, Any] = {"last_owner_kind": kind, "last_bot_kind": ""}

        explicit = bool(cur.get("explicit_wait")) and prev_state == STATE_WAITING
        if explicit:
            since = cur.get("wait_since")
            if since is None or now - float(since) > self._wait_timeout:
                explicit = False  # 显式等待已超时，按普通消息处理

        if kind == rules.WAIT_START:
            fields.update(state=STATE_WAITING, explicit_wait=1, wait_since=now)
        elif explicit and kind not in _ASKING_KINDS:
            # 显式等待期间的片段：保持 waiting（刷新等待起点）
            fields.update(state=STATE_WAITING, explicit_wait=1, wait_since=now)
        elif kind == rules.ENDING and not self._has_unanswered(cur):
            fields.update(self._end_point(scope_key))
            fields.update(state=STATE_ENDED, explicit_wait=0, wait_since=None)
        else:
            fields.update(state=STATE_CONTINUE, explicit_wait=0, wait_since=None)

        row = self._store.put_closure(scope_key, **fields)
        if row["state"] != prev_state:
            logger.info(
                f"[continuity] closure scope={scope_key} state={prev_state}->{row['state']} owner_kind={kind}"
            )
        return row["state"]

    @staticmethod
    def _has_unanswered(cur: dict[str, Any]) -> bool:
        """主人上一条是问题/请求，而 Frisk 尚未回复。"""
        return cur.get("last_owner_kind") in _ASKING_KINDS and not cur.get("last_bot_kind")

    def _end_point(self, scope_key: str) -> dict[str, Any]:
        """结束点：summary / open_questions / next_opening（只读事项）。"""
        topics: list[str] = []
        openable: list[str] = []
        try:
            items = self._store.list_items(scope_key=scope_key, statuses=(m.STATUS_ACTIVE,), limit=OPEN_QUESTIONS_MAX)
            topics = [str(it.get("topic") or "") for it in items if it.get("topic")]
            # next_opening 只取可主动接续的事项（“不用提醒”/已退出主动候选的不算）
            openable = [
                str(it.get("topic") or "")
                for it in items
                if it.get("topic") and it.get("proactive_allowed")
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[continuity] closure end-point items failed scope={scope_key}: {type(e).__name__}")
        fields: dict[str, Any] = {
            "ended_at": self._clock(),
            "open_questions": topics,
            "next_opening": openable[0] if openable else "",
        }
        # 取走最近实质消息：下一段对话的结束点不复用旧内容
        fields["summary"] = self._last_content.pop(scope_key, "")
        return fields
