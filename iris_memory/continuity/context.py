"""
Iris Chat Memory - 统一上下文组装（DESIGN.md §7）

普通回复与主动消息共用同一组装逻辑：

1. 权限过滤（最先）：audience = scope_key；私聊专属层只在
   ``scope_key == f"private:{owner}"`` 时注入；事项只取 allowed_audiences 含
   scope_key 的 active 项；非主人私聊再额外剔除私聊来源 / 频道观察类事项。
2. 排序：焦点事项 → remind_requested → importance → 已到时 → 最近更新。
3. 截断：最多 ``inject_max_items`` 项、事项总字数 ≤ ``inject_max_chars``；
   proactive 只放焦点事项（+ 最多 1 项背景，背景须同样满足主动候选资格：
   proactive_allowed、not_before 已到、未过期、ask_count<2、6h 提及冷却）；
   收尾结束点的 next_opening 在 proactive 中只在它正是焦点事项时保留。

叙述约束（§6）：草稿 / 失败 / 未确认的出站绝不出现；已发出表达标“你说过”；
bot 承诺标“你答应过（未完成）”。

任一子源失败只跳过该子源（日志只记来源名与异常类型，不记正文）。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import (
    Any,
)

from astrbot.api import logger

from . import models as m

__all__ = ["ContextAssembler", "PURPOSE_REPLY", "PURPOSE_PROACTIVE", "OPEN_TAG", "CLOSE_TAG"]

PURPOSE_REPLY = "reply"
PURPOSE_PROACTIVE = "proactive"

OPEN_TAG = "<iris:continuity>"
CLOSE_TAG = "</iris:continuity>"

OUTBOUND_LIMIT = 3
OUTBOUND_MAX_CHARS = 80
L1_LIMIT = 12
L1_LINE_MAX_CHARS = 200
PRIVATE_SECTION_MAX_CHARS = 2000
CLOSURE_SUMMARY_MAX_CHARS = 80
PROACTIVE_BACKGROUND_MAX = 1
# 从库中多取一些再排序截断，保证焦点 / 高优先项不因 SQL 排序被挤掉
_ITEM_FETCH_LIMIT = 50

_DEFAULTS = {"inject_max_items": 3, "inject_max_chars": 600}
_BOUNDS = {"inject_max_items": (1, 10), "inject_max_chars": (100, 4000)}

# private_context 段名 → 标题
_SECTION_TITLES = {
    "identity": "身份",
    "relationship": "相处方式",
    "rapport": "相处方式",
    "memory": "记忆",
    "memories": "记忆",
    "recent_dynamics": "近期动态",
}

_KIND_LABELS = {
    m.KIND_PLAN: "计划",
    m.KIND_PENDING_RESULT: "等结果",
    m.KIND_UNFINISHED_TOPIC: "没聊完",
    m.KIND_BOT_PROMISE: "你的承诺",
    m.KIND_INTEREST_QUESTION: "好奇",
    m.KIND_NOTE: "记着",
}


def _cfg_int(config: Any, key: str) -> int:
    default = _DEFAULTS[key]
    if config is None:
        val = default
    elif isinstance(config, Mapping):
        val = config.get(key, default)
    else:
        val = getattr(config, key, default)
    try:
        val = int(val)
    except (TypeError, ValueError):
        val = default
    lo, hi = _BOUNDS[key]
    return min(hi, max(lo, val))


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"


def _fmt_day(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), m.TZ_SHANGHAI).strftime("%m-%d")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ContextAssembler:
    """统一上下文组装器（只读 store）"""

    def __init__(
        self,
        store: Any,
        owner_ids: Callable[[], Iterable[str]],
        config: Any = None,
        clock: Callable[[], float] = time.time,
        profile_text: Callable[[str], Awaitable[str | None]] | None = None,
        l1_recent: Callable[[str, int], Any] | None = None,
    ) -> None:
        self._store = store
        self._owner_ids = owner_ids
        self._config = config
        self._clock = clock
        self._profile_text = profile_text
        self._l1_recent = l1_recent

    # ------------------------------------------------------------------ config

    @property
    def max_items(self) -> int:
        return _cfg_int(self._config, "inject_max_items")

    @property
    def max_chars(self) -> int:
        return _cfg_int(self._config, "inject_max_chars")

    # ------------------------------------------------------------------ permission

    def _owners(self) -> set[str]:
        try:
            return {str(o).strip() for o in (self._owner_ids() or ()) if str(o).strip()}
        except Exception as e:
            logger.warning(f"[continuity] context owner_ids failed: {type(e).__name__}")
            return set()

    def is_owner_private(self, scope_key: str, sender_id: str | None = None) -> bool:
        """scope_key == private:<owner>；sender_id 给出时还必须是该主人"""
        scope_key = str(scope_key or "")
        if not scope_key.startswith("private:"):
            return False
        uid = scope_key.split(":", 1)[1]
        if uid not in self._owners():
            return False
        if sender_id not in (None, "") and str(sender_id) != uid:
            return False
        return True

    @staticmethod
    def _item_permitted(item: dict[str, Any], scope_key: str, owner_private: bool) -> bool:
        if item.get("status") != m.STATUS_ACTIVE:
            return False
        audiences = item.get("allowed_audiences") or []
        if scope_key not in [str(a) for a in audiences]:
            return False
        if not owner_private:
            # 纵深防御：私聊来源、频道观察类事项只进主人私聊
            if str(item.get("scope_key") or "").startswith("private:"):
                return False
            if item.get("source_kind") == m.SOURCE_CHANNEL_OBSERVATION:
                return False
            if any(str(a).startswith("private:") for a in audiences):
                return False
        return True

    # ------------------------------------------------------------------ public

    async def assemble(
        self,
        scope_key: str,
        *,
        purpose: str = PURPOSE_REPLY,
        sender_id: str | None = None,
        focus_item_id: str | None = None,
        now: float | None = None,
    ) -> str:
        """返回 ``<iris:continuity>…</iris:continuity>``；无内容时返回 ''"""
        try:
            scope_key = str(scope_key or "")
            if not scope_key:
                return ""
            now = float(self._clock() if now is None else now)
            proactive = purpose == PURPOSE_PROACTIVE
            owner_private = self.is_owner_private(scope_key, None if proactive else sender_id)
        except Exception as e:
            logger.warning(f"[continuity] context setup failed: {type(e).__name__}")
            return ""

        sections: list[str] = []

        async def run(name: str, coro: Awaitable[str | None]) -> None:
            try:
                text = await coro
            except Exception as e:
                logger.warning(
                    f"[continuity] context source={name} scope={scope_key} failed: {type(e).__name__}"
                )
                return
            if text:
                sections.append(text)

        if owner_private:
            await run("private_context", self._private_section(scope_key))
        await run(
            "items",
            self._items_section(scope_key, owner_private, proactive, focus_item_id, now),
        )
        await run("outbound", self._outbound_section(scope_key))
        await run("closure", self._closure_section(scope_key, proactive, focus_item_id))
        if proactive:
            if owner_private:
                await run("profile", self._profile_section(scope_key))
            await run("l1", self._l1_section(scope_key, owner_private))

        if not sections:
            return ""
        header = (
            "以下是连续性参考，只用于自然接续，不要逐条复述。"
            "标注“你说过”的是你真实发出过的话；“你答应过（未完成）”的事还没做，"
            "只有确有执行证据的事才能说已经做了。"
        )
        if proactive:
            header += "这次主动开口只围绕一个话题。"
        return "\n".join([OPEN_TAG, header, *sections, CLOSE_TAG])

    # ------------------------------------------------------------------ sources

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _private_section(self, scope_key: str) -> str | None:
        rows = await self._call(self._store.get_private_context, scope_key)
        lines: list[str] = []
        for row in rows or []:
            if str(row.get("audience")) != scope_key:
                continue
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            section = str(row.get("section") or "")
            title = _SECTION_TITLES.get(section, section or "私聊")
            if len(content) > PRIVATE_SECTION_MAX_CHARS:
                content = content[: PRIVATE_SECTION_MAX_CHARS - 1] + "…"
            lines.append(f"【{title}】\n{content}")
        return "\n".join(lines) if lines else None

    async def _items_section(
        self,
        scope_key: str,
        owner_private: bool,
        proactive: bool,
        focus_item_id: str | None,
        now: float,
    ) -> str | None:
        rows = await self._call(
            self._store.list_items,
            statuses=(m.STATUS_ACTIVE,),
            audience=scope_key,
            limit=_ITEM_FETCH_LIMIT,
        )
        permitted = [r for r in rows or [] if self._item_permitted(r, scope_key, owner_private)]

        focus: dict[str, Any] | None = None
        if focus_item_id:
            focus = next((r for r in permitted if r.get("item_id") == focus_item_id), None)
            if focus is None:
                row = await self._call(self._store.get_item, focus_item_id)
                if row is not None and self._item_permitted(row, scope_key, owner_private):
                    focus = row
        others = [r for r in permitted if focus is None or r.get("item_id") != focus.get("item_id")]
        others.sort(key=lambda r: self._rank_key(r, now))

        if proactive:
            # 背景项必须与 bridge 候选同资格，否则“不用提醒”/超问次数/冷却中的事项会混进主动话题
            others = [r for r in others if m.proactive_eligible(r, scope_key, now)]
            chosen = ([focus] if focus else []) + others[:PROACTIVE_BACKGROUND_MAX]
        else:
            chosen = ([focus] if focus else []) + others
        chosen = chosen[: self.max_items]

        budget = self.max_chars
        lines: list[str] = []
        used = 0
        for item in chosen:
            is_focus = focus is not None and item is focus
            line = self._render_item(item, now, is_focus)
            if used + len(line) > budget:
                if not lines:  # 至少保留首项（截到预算内）
                    lines.append(_clip(line, budget))
                break
            lines.append(line)
            used += len(line)
        if not lines:
            return None
        return "【待续事项】\n" + "\n".join(lines)

    @staticmethod
    def _rank_key(item: dict[str, Any], now: float) -> tuple[Any, ...]:
        nb = item.get("not_before")
        due = nb is None or nb <= now
        return (
            -int(item.get("remind_requested") or 0),
            -int(item.get("importance") or 1),
            0 if due else 1,
            -float(item.get("updated_at") or 0.0),
            str(item.get("item_id") or ""),
        )

    @staticmethod
    def _attribution(item: dict[str, Any]) -> str:
        source = item.get("source_kind")
        if source == m.SOURCE_SELF_EXPRESSION:
            return "你答应过（未完成）" if item.get("kind") == m.KIND_BOT_PROMISE else "你说过"
        if source == m.SOURCE_CHANNEL_OBSERVATION:
            return "她在频道发过"
        if source == m.SOURCE_THIRD_PARTY:
            who = ""
            for ref in item.get("source_refs") or []:
                if isinstance(ref, Mapping):
                    who = str(ref.get("source") or ref.get("from") or ref.get("name") or "").strip()
                    if who:
                        break
            return f"据{_clip(who, 20) or '别人'}说"
        if source == m.SOURCE_HISTORY_IMPORT:
            return "（旧记录，非共同经历）"
        return "你们在私聊里聊过"

    def _render_item(self, item: dict[str, Any], now: float, is_focus: bool) -> str:
        label = _KIND_LABELS.get(item.get("kind"), "事项")
        topic = str(item.get("topic") or "").strip()
        summary = str(item.get("summary") or "").strip()
        body = topic if not summary or summary == topic else f"{topic}：{summary}"
        parts = [f"- [{label}] {self._attribution(item)}：{body}"]
        basis = str(item.get("time_basis") or "").strip()
        start = item.get("event_at_start")
        if basis or start is not None:
            when = basis or ""
            if start is not None:
                when = f"{when}（{_fmt_day(start)}）" if when else _fmt_day(start)
            parts.append(f"时间：{when}")
        nb = item.get("not_before")
        if nb is not None and now < nb:
            parts.append("还没到时候，先别问结果")
        if item.get("remind_requested"):
            parts.append("她要你到时提醒")
        if is_focus:
            parts.append("本次焦点")
        return "；".join(parts)

    async def _outbound_section(self, scope_key: str) -> str | None:
        rows = await self._call(
            self._store.recent_outbound,
            scope_key,
            statuses=(m.OUT_SENT,),
            authors=("bot",),
            limit=OUTBOUND_LIMIT,
        )
        lines: list[str] = []
        for row in rows or []:
            # 纵深防御：草稿 / 失败 / 未确认 / 手动发言绝不当作“你说过”
            if row.get("status") != m.OUT_SENT or row.get("author") != "bot":
                continue
            text = _clip(row.get("text"), OUTBOUND_MAX_CHARS)
            if text:
                lines.append(f"- 你说过：{text}")
            if len(lines) >= OUTBOUND_LIMIT:
                break
        return "【你最近发出的话】\n" + "\n".join(lines) if lines else None

    async def _closure_section(
        self, scope_key: str, proactive: bool = False, focus_item_id: str | None = None
    ) -> str | None:
        row = await self._call(self._store.get_closure, scope_key)
        if not row or row.get("state") != "ended":
            return None
        summary = _clip(row.get("summary"), CLOSURE_SUMMARY_MAX_CHARS)
        opening = _clip(row.get("next_opening"), m.TOPIC_MAX_CHARS)
        if proactive and opening:
            # 主动场景只允许接焦点事项本身
            focus = await self._call(self._store.get_item, focus_item_id) if focus_item_id else None
            if not focus or _clip(focus.get("topic"), m.TOPIC_MAX_CHARS) != opening:
                opening = ""
        parts: list[str] = []
        if summary:
            parts.append(f"上次聊到：{summary}")
        if opening:
            parts.append(f"下次可以从“{opening}”接上")
        return "【上次结束点】\n" + "；".join(parts) if parts else None

    async def _profile_section(self, scope_key: str) -> str | None:
        if self._profile_text is None:
            return None
        text = await _maybe_await(self._profile_text(scope_key))
        text = str(text or "").strip()
        return f"【画像摘要】\n{text}" if text else None

    async def _l1_section(self, scope_key: str, owner_private: bool) -> str | None:
        if self._l1_recent is None:
            return None
        rows: Sequence[Any] = await _maybe_await(self._l1_recent(scope_key, L1_LIMIT)) or []
        lines: list[str] = []
        for entry in list(rows)[-L1_LIMIT:]:
            try:
                role, text = entry
            except (TypeError, ValueError):
                continue
            text = _clip(text, L1_LINE_MAX_CHARS)
            if not text:
                continue
            if str(role) == "assistant":
                who = "你"
            elif str(role) == "user":
                who = "她" if owner_private else "对方"
            else:
                who = str(role)
            lines.append(f"{who}：{text}")
        return "【最近对话】\n" + "\n".join(lines) if lines else None
