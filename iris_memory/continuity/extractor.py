"""
Iris Chat Memory - 入站事项抽取（DESIGN.md §4、§5.12）

流程：持久化 inbox（store.inbound）→ 按 scope 合批 → 规则预筛（rules.py）
→ LLM 只输出 JSON → 严格校验 → 单事务应用到 store → 标记入站行状态。

- 纯应答/笑声/表情（“哈哈”“嗯”）→ skipped，不调 LLM、不改事项；
- 无事项线索、reply_to 未指向关联事项、也没点名 active 事项 → skipped；
- 校验失败的 op 一律丢弃（整批都不合法 → failed + last_error），绝不写库；
- 确定性覆盖：“不用提醒/只记着” → remind_requested=0, proactive_allowed=0；
  “提醒我/到时候叫我” → remind_requested=1；
- 关闭类 op（complete/cancel/decline/reschedule）：evidence 所在消息必须含对应
  措辞（完成/结果已出、取消、拒绝、改期）；批内有 reply_to 关联事项时只能作用于
  关联事项；无关联时每条 evidence 消息至多一个关闭类 op（decline 全批至多一个）；
- 时间：timeparse.parse(when_text, base=原始消息时间)；pending_result 或 evidence
  含过去标记时取最近一次已过去的日期；
- 暂时性失败（LLM 超时/网络、claim 失败）：attempts<3 的行回到 pending，worker
  退避重试；校验失败保持 failed；
- RevisionConflict：重读事项，仍 active 则在新 revision 上重放一次；已关闭则
  入站行保持 pending（重新抽取），不标 done。

ExtractionWorker：单并发、按 scope 去抖、信号队列有上限（超限丢弃最旧的
scope 信号，inbox 行仍在库中，启动时恢复）。

record_bot_promise：Frisk 已确认发出的文本中的承诺 → bot_promise 事项
（只能在拿到 Telegram 回执后调用）。承诺挂靠到同 scope 相关的 active
plan/pending_result（reply_to 关联 > 24h 内话题重叠 > 最近一项），复制其时间窗，
并在用户事项 source_refs 记一条 bot_promise；承诺本身 proactive_allowed=0
（只作“你答应过（未完成）”上下文，主动候选仍是用户事项），用户事项关闭时
store 在同一事务内关闭关联承诺。

日志只记录 id / 计数 / 状态，不记录 prompt 与消息正文。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from astrbot.api import logger

from . import models as m
from . import rules, timeparse
from .store import ItemNotFound, RevisionConflict

LLMGenerate = Callable[[str, float], Awaitable[str]]

HOUR = 3600.0
DAY = 86400.0

# LLM 可新建/改写的事项种类（bot_promise / interest_question 不由抽取产生）
LLM_KINDS = frozenset({m.KIND_PLAN, m.KIND_PENDING_RESULT, m.KIND_UNFINISHED_TOPIC, m.KIND_NOTE})
OPS = frozenset({"create", "update", "complete", "cancel", "decline", "reschedule"})
ITEM_OPS = OPS - {"create"}

DEFAULT_DEBOUNCE_S = 6.0
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_QUEUE_MAX = 50
BATCH_LIMIT = 20
STALE_AFTER_S = 86400.0
RECENT_FOCUS_WINDOW_S = 30 * 60.0
MAX_OPS = 8
MAX_CANDIDATES = 12
MSG_MAX_CHARS = 500
EVIDENCE_MIN_CHARS = 2
LLM_RETRIES = 1  # 超时/网络错误最多重试 1 次
MAX_ATTEMPTS = 3  # 暂时性失败：attempts < 3 的入站行回到 pending
RETRY_BACKOFF_S = 30.0  # worker 重试退避基数（×2^n）
TRANSIENT_ERRORS = frozenset({"llm_timeout", "llm_network", "claim_error"})
CLOSING_OPS = frozenset({"complete", "cancel", "decline", "reschedule"})
PROMISE_LINK_WINDOW_S = 24 * 3600.0
PROMISE_ADOPT_WINDOW_S = 3600.0  # 用户事项晚于承诺建出时，可收养此窗口内的未挂靠承诺
PAST_FALLBACK_S = 31 * 86400.0  # pending_result 无过去标记时，过去日期最多回溯 31 天
PROMISE_REF_TEXT_MAX = 80

# 标签：候选事项的来源
TAG_ACTIVE = "active"
TAG_REPLY_LINKED = "reply_linked"
TAG_RECENT_FOCUS = "recent_proactive_focus"

_RETRYABLE = (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)


def _cfg(config: Any, name: str, default: Any) -> Any:
    """从 ContinuityConfig（属性）或 dict 读取配置，缺省回退 default"""
    if config is None:
        return default
    if isinstance(config, dict):
        val = config.get(name, default)
    else:
        val = getattr(config, name, default)
    return default if val is None else val


def _owner_from_scope(scope_key: str) -> str | None:
    if isinstance(scope_key, str) and scope_key.startswith("private:"):
        uid = scope_key.split(":", 1)[1]
        return uid or None
    return None


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    scope_key: str
    status: str = "empty"  # empty | skipped | done | failed
    claimed: int = 0
    msg_ids: list[int] = field(default_factory=list)
    llm_called: bool = False
    llm_attempts: int = 0
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    invalid_ops: int = 0
    error: str = ""
    retry: bool = False  # 暂时性失败，行已回到 pending，worker 应退避重试


@dataclass
class _Candidate:
    item: dict[str, Any]
    tags: list[str]


@dataclass
class _Op:
    op: str
    item_id: str | None
    kind: str | None
    topic: str
    summary: str
    when_text: str
    remind_requested: bool | None
    evidence: str
    msg: dict[str, Any]  # evidence 所在入站行
    when_msg: dict[str, Any] | None  # when_text 所在入站行（优先 evidence 行）


class _InvalidOp(ValueError):
    pass


# ---------------------------------------------------------------------------
# 时间计算
# ---------------------------------------------------------------------------


def _parse_when(kind: str, when_text: str, base_ts: float, context_text: str) -> Any:
    """过去标记 / pending_result → 优先取最近一次已过去的日期"""
    if not when_text:
        return None
    marked = timeparse.has_past_marker(context_text)
    prefer_past = marked or (
        kind == m.KIND_PENDING_RESULT and not timeparse.has_future_marker(context_text)
    )
    parsed = timeparse.parse(when_text, base_ts, prefer_past=prefer_past)
    if (
        parsed is not None
        and prefer_past
        and not marked
        and parsed.end_ts < base_ts - PAST_FALLBACK_S
    ):
        # pending_result 只是“在等结果”，太久以前的同名日期多半指将来
        parsed = timeparse.parse(when_text, base_ts)
    return parsed


def compute_time_fields(
    kind: str, when_text: str, base_ts: float, context_text: str = ""
) -> dict[str, Any]:
    """when_text + 原始消息时间 → event_*/not_before/expires_at/time_* 字段（DESIGN §4）

    context_text：when_text 所在的原消息，用于判断过去标记。
    """
    parsed = _parse_when(kind, when_text, base_ts, context_text or when_text)
    out: dict[str, Any] = {
        "event_at_start": parsed.start_ts if parsed else None,
        "event_at_end": parsed.end_ts if parsed else None,
        "time_basis": (parsed.basis if parsed else (when_text or ""))[: m.TOPIC_MAX_CHARS],
        "time_precision": parsed.precision if parsed else "none",
        "time_zone": m.DEFAULT_TIME_ZONE,
    }
    if kind in (m.KIND_PLAN, m.KIND_PENDING_RESULT, m.KIND_BOT_PROMISE):
        out["not_before"] = parsed.end_ts if parsed else base_ts + 12 * HOUR
        # 事件已过去（等结果）时，过期从消息时间起算，避免刚建就过期
        out["expires_at"] = (max(parsed.end_ts, base_ts) + 3 * DAY) if parsed else base_ts + 7 * DAY
    elif kind == m.KIND_UNFINISHED_TOPIC:
        out["not_before"] = None
        out["expires_at"] = base_ts + 2 * DAY
    else:  # note / interest_question：不过期
        out["not_before"] = None
        out["expires_at"] = None
    return out


def _remind_overrides(text: str, llm_value: bool | None) -> dict[str, int]:
    """确定性提醒覆盖：不用提醒 > 提醒我 > LLM 给出的值"""
    if rules.wants_no_reminder(text):
        return {"remind_requested": 0, "proactive_allowed": 0}
    if rules.wants_reminder(text):
        return {"remind_requested": 1, "proactive_allowed": 1}
    if llm_value is not None:
        return {"remind_requested": 1 if llm_value else 0}
    return {}


# ---------------------------------------------------------------------------
# LLM 输出解析与校验
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_llm_json(raw: str) -> list[Any] | None:
    """解析 LLM 输出为 ops 列表；不是 {"ops": [...]} → None"""
    if not isinstance(raw, str):
        return None
    text = _FENCE_RE.sub("", raw.strip())
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("ops"), list):
        return None
    return data["ops"]


def _opt_str(raw: dict[str, Any], key: str, limit: int) -> str:
    val = raw.get(key)
    if val is None:
        return ""
    if not isinstance(val, str):
        raise _InvalidOp(f"{key}_type")
    val = val.strip()
    if len(val) > limit:
        raise _InvalidOp(f"{key}_length")
    return val


def _find_msg(fragment: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for r in rows:
        if fragment and fragment in r["_text"]:
            return r
    return None


def validate_op(
    raw: Any, rows: Sequence[dict[str, Any]], candidates: dict[str, _Candidate]
) -> _Op:
    """严格校验单个 op；不合法抛 _InvalidOp（原因码，不含正文）"""
    if not isinstance(raw, dict):
        raise _InvalidOp("not_object")
    op = raw.get("op")
    if op not in OPS:
        raise _InvalidOp("op")

    item_id = raw.get("item_id")
    if item_id in ("", None):
        item_id = None
    elif not isinstance(item_id, str):
        raise _InvalidOp("item_id_type")
    if op == "create":
        if item_id is not None:
            raise _InvalidOp("create_with_item_id")
    else:
        if item_id is None or item_id not in candidates:
            raise _InvalidOp("item_id_unknown")

    kind = raw.get("kind")
    if kind in ("", None):
        kind = None
    elif not isinstance(kind, str):
        raise _InvalidOp("kind_type")
    if op == "create":
        if kind not in LLM_KINDS:
            raise _InvalidOp("kind")
    elif kind is not None:
        current = candidates[item_id].item["kind"]  # type: ignore[index]
        if kind == current:
            kind = None  # 原样回显 → 不改
        elif kind not in LLM_KINDS or current not in LLM_KINDS:
            raise _InvalidOp("kind")

    topic = _opt_str(raw, "topic", m.TOPIC_MAX_CHARS)
    summary = _opt_str(raw, "summary", m.SUMMARY_MAX_CHARS)
    if op == "create" and not topic:
        raise _InvalidOp("topic_missing")
    when_text = _opt_str(raw, "when_text", MSG_MAX_CHARS)
    evidence = _opt_str(raw, "evidence", MSG_MAX_CHARS)

    remind = raw.get("remind_requested")
    if remind is not None and not isinstance(remind, bool):
        raise _InvalidOp("remind_type")

    if len(evidence) < EVIDENCE_MIN_CHARS:
        raise _InvalidOp("evidence_short")
    ev_msg = _find_msg(evidence, rows)
    if ev_msg is None:
        raise _InvalidOp("evidence_not_substring")
    when_msg = None
    if when_text:
        when_msg = ev_msg if when_text in ev_msg["_text"] else _find_msg(when_text, rows)
        if when_msg is None:
            raise _InvalidOp("when_not_substring")

    return _Op(
        op=op,
        item_id=item_id,
        kind=kind,
        topic=topic,
        summary=summary,
        when_text=when_text,
        remind_requested=remind,
        evidence=evidence,
        msg=ev_msg,
        when_msg=when_msg,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_HEAD = """你是对话事项记录器。根据主人发来的最新消息，判断是否需要新建或更新“待续事项”。
事项种类：plan（她打算做/要去做的事，有时间更好）、pending_result（在等结果的事）、
unfinished_topic（没聊完、之后值得接着聊的话题）、note（她要求记下的信息）。
操作：create 新建；update 修改已有事项的话题/摘要/种类/提醒；reschedule 改时间；
complete 已完成；cancel 取消/不做了；decline 她明确不想再聊该事项。
规则：
- 只依据消息原文，不要猜测；闲聊、情绪表达、纯应答不建事项，返回 {"ops":[]}。
- item_id 只能取自下方候选列表；create 不带 item_id。
- evidence 必须是消息原文中连续出现的片段（至少 2 个字），不要改写。
- when_text 必须是原文中的时间表达原样片段，没有就留空字符串；不要自己换算日期。
- remind_requested 只有她明确要你提醒时才为 true。
- topic 不超过 40 字，summary 不超过 160 字，用第三人称“她”。
只输出 JSON，不要任何解释：
{"ops":[{"op":"create|update|complete|cancel|decline|reschedule","item_id":"","kind":"plan|pending_result|unfinished_topic|note","topic":"","summary":"","when_text":"","remind_requested":false,"evidence":""}]}
"""


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "未知"
    return datetime.fromtimestamp(float(ts), m.TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M")


def build_prompt(rows: Sequence[dict[str, Any]], candidates: dict[str, _Candidate]) -> str:
    lines = [_PROMPT_HEAD, "<candidates>"]
    if not candidates:
        lines.append("（无）")
    for cid, c in candidates.items():
        it = c.item
        when = it.get("time_basis") or ""
        if it.get("event_at_start") is not None:
            when = f"{when}（{_fmt_ts(it['event_at_start'])} 起）"
        lines.append(
            json.dumps(
                {
                    "item_id": cid,
                    "kind": it["kind"],
                    "topic": it["topic"],
                    "summary": it["summary"],
                    "when": when,
                    "tags": c.tags,
                },
                ensure_ascii=False,
            )
        )
    lines.append("</candidates>")
    lines.append("<messages>")
    for r in rows:
        prefix = f"[{_fmt_ts(r['msg_ts'])}]"
        if r.get("_linked"):
            prefix += "（回复 Frisk 关于候选事项的消息）"
        lines.append(f"{prefix} {r['_text']}")
    lines.append("</messages>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------


class ItemExtractor:
    """单 scope 一次抽取（process_scope）；store 调用经 asyncio.to_thread"""

    def __init__(
        self,
        store: Any,
        llm_generate: LLMGenerate,
        config: Any = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._llm = llm_generate
        self._config = config
        self._clock = clock or time.time
        self.batch_limit = BATCH_LIMIT

    @property
    def store(self) -> Any:
        return self._store

    @property
    def timeout_s(self) -> float:
        return max(1.0, float(_cfg(self._config, "extract_timeout_s", DEFAULT_TIMEOUT_S)))

    # ------------------------------------------------------------ main
    async def process_scope(self, scope_key: str, owner_id: str) -> ExtractResult:
        res = ExtractResult(scope_key=scope_key)
        try:
            rows = await asyncio.to_thread(
                self._store.claim_pending_inbound, scope_key, self.batch_limit, STALE_AFTER_S
            )
        except Exception as e:
            # claim 事务已回滚，行仍是 pending；交给 worker 退避重试
            logger.warning(f"[continuity] extract claim failed scope={scope_key}: {type(e).__name__}")
            res.status, res.error, res.retry = "failed", "claim_error", True
            return res
        if not rows:
            return res
        res.claimed = len(rows)
        res.msg_ids = [int(r["msg_id"]) for r in rows]
        try:
            await self._process_rows(scope_key, str(owner_id), rows, res)
        except Exception as e:  # 边界兜底：行保持 pending 之外的确定状态
            logger.warning(
                f"[continuity] extract failed scope={scope_key} msgs={res.msg_ids}: {type(e).__name__}"
            )
            res.status, res.error = "failed", f"internal:{type(e).__name__}"
            await self._mark_safe(scope_key, res.msg_ids, m.INBOUND_FAILED, res.error)
        logger.info(
            f"[continuity] extract scope={scope_key} msgs={len(res.msg_ids)} status={res.status}"
            f" llm={res.llm_attempts} created={len(res.created)} updated={len(res.updated)}"
            f" closed={len(res.closed)} invalid={res.invalid_ops}"
            + (f" error={res.error}" if res.error else "")
        )
        return res

    async def _process_rows(
        self, scope_key: str, owner_id: str, rows: list[dict[str, Any]], res: ExtractResult
    ) -> None:
        now = float(self._clock())
        ignored: list[int] = []  # 纯应答 / 非主人
        work: list[dict[str, Any]] = []
        for r in rows:
            text = str(r.get("text") or "")
            if str(r.get("sender_id")) != owner_id or not text.strip() or rules.is_ack_only(text):
                ignored.append(int(r["msg_id"]))
                continue
            r = dict(r)
            r["_text"] = text[:MSG_MAX_CHARS]
            work.append(r)

        candidates = await asyncio.to_thread(self._collect_candidates, scope_key, work, now)

        relevant = any(r.get("_linked") or rules.has_item_cue(r["_text"]) for r in work) or any(
            self._names_item(r["_text"], c.item) for r in work for c in candidates.values()
        )
        if not relevant:
            res.status = "skipped"
            await asyncio.to_thread(
                self._store.mark_inbound, scope_key, res.msg_ids, m.INBOUND_SKIPPED, ""
            )
            return

        res.llm_called = True
        raw, err = await self._call_llm(build_prompt(work, candidates), res)
        work_ids = [int(r["msg_id"]) for r in work]
        if raw is None:
            res.status, res.error = "failed", err
            if err in TRANSIENT_ERRORS:
                retry_ids = [int(r["msg_id"]) for r in work if int(r.get("attempts") or 0) < MAX_ATTEMPTS]
                final_ids = [i for i in work_ids if i not in retry_ids]
                res.retry = bool(retry_ids)
                await asyncio.to_thread(
                    self._mark_split, scope_key, ignored, retry_ids, final_ids, err
                )
                return
            await asyncio.to_thread(self._mark_all, scope_key, ignored, work_ids, m.INBOUND_FAILED, err)
            return

        raw_ops = parse_llm_json(raw)
        if raw_ops is None:
            res.status, res.error = "failed", "invalid_json"
            await asyncio.to_thread(
                self._mark_all, scope_key, ignored, work_ids, m.INBOUND_FAILED, "invalid_json"
            )
            return

        ops: list[_Op] = []
        reasons: list[str] = []
        for raw_op in raw_ops[:MAX_OPS]:
            try:
                ops.append(validate_op(raw_op, work, candidates))
            except _InvalidOp as e:
                reasons.append(str(e))
        res.invalid_ops = len(reasons) + max(0, len(raw_ops) - MAX_OPS)
        ops = self._guard_closing(ops, candidates, work, res)

        if raw_ops and not ops:
            res.status = "failed"
            res.error = "invalid_ops:" + ",".join(sorted(set(reasons)) or ["guarded"])
            await asyncio.to_thread(
                self._mark_all, scope_key, ignored, work_ids, m.INBOUND_FAILED, res.error
            )
            return

        await asyncio.to_thread(
            self._apply, scope_key, owner_id, ops, candidates, work, ignored, work_ids, res
        )
        res.status = "failed" if res.retry else "done"

    # ------------------------------------------------------------ candidates
    def _collect_candidates(
        self, scope_key: str, work: list[dict[str, Any]], now: float
    ) -> dict[str, _Candidate]:
        store = self._store
        cands: dict[str, _Candidate] = {}

        def add(item: dict[str, Any] | None, tag: str) -> None:
            if not item or item.get("status") != m.STATUS_ACTIVE or item.get("scope_key") != scope_key:
                return
            c = cands.get(item["item_id"])
            if c is None:
                cands[item["item_id"]] = _Candidate(item=item, tags=[tag])
            elif tag not in c.tags:
                c.tags.append(tag)

        # 1) reply_to 指向的出站消息所关联事项（优先）
        any_reply = False
        for r in work:
            rt = r.get("reply_to_msg_id")
            if rt is None:
                continue
            any_reply = True
            try:
                ob = store.outbound_by_tg_id(scope_key, int(rt))
            except Exception as e:
                logger.warning(f"[continuity] extract outbound lookup failed: {type(e).__name__}")
                ob = None
            linked = []
            for iid in (ob or {}).get("item_ids") or []:
                item = store.get_item(str(iid))
                if item and item.get("status") == m.STATUS_ACTIVE and item.get("scope_key") == scope_key:
                    add(item, TAG_REPLY_LINKED)
                    linked.append(item["item_id"])
            r["_linked"] = linked

        # 2) 无 reply_to 时：30 分钟内最近一次主动回合的焦点事项
        if not any_reply:
            turns = store.turns_in_state((m.TURN_SENT, m.TURN_PARTIAL), scope_key)
            recent = [
                t
                for t in turns
                if t.get("item_id") and now - float(t.get("updated_at") or 0) <= RECENT_FOCUS_WINDOW_S
            ]
            if recent:
                latest = max(recent, key=lambda t: float(t.get("updated_at") or 0))
                add(store.get_item(latest["item_id"]), TAG_RECENT_FOCUS)

        # 3) 该 scope 的其它 active 事项
        for item in store.list_items(scope_key=scope_key, statuses=(m.STATUS_ACTIVE,), limit=MAX_CANDIDATES):
            if len(cands) >= MAX_CANDIDATES:
                break
            add(item, TAG_ACTIVE)
        return cands

    @staticmethod
    def _names_item(text: str, item: dict[str, Any]) -> bool:
        topic = m.norm_topic(item.get("topic") or "")
        return len(topic) >= 2 and topic in m.norm_topic(text)

    @staticmethod
    def _closing_cue(op: _Op, candidates: dict[str, _Candidate]) -> bool:
        """关闭类 op 的 evidence 所在消息必须含对应措辞（“哈哈”/应答永远关不掉事项）"""
        text = op.msg["_text"]
        if rules.is_ack_only(text):
            return False
        if op.op == "decline":
            return rules.declines_topic(text)
        if op.op == "cancel":
            return rules.signals_cancel(text)
        if op.op == "reschedule":
            return rules.signals_reschedule(text)
        if op.op == "complete":
            if rules.signals_done(text):
                return True
            cand = candidates.get(op.item_id or "")
            return bool(
                cand and cand.item.get("kind") == m.KIND_PENDING_RESULT and rules.signals_result(text)
            )
        return True

    @classmethod
    def _guard_closing(
        cls,
        ops: list[_Op],
        candidates: dict[str, _Candidate],
        work: list[dict[str, Any]],
        res: ExtractResult,
    ) -> list[_Op]:
        """关闭类 op 限制（complete/cancel/decline/reschedule）：

        - evidence 所在消息必须含对应措辞；
        - 批内有 reply_to 关联事项 → 只能作用于关联事项（decline 只能作用于
          自身 evidence 消息的关联事项）；
        - 无关联 → 每条 evidence 消息至多一个关闭类 op；decline 全批至多一个。
        """
        linked_all: set[str] = set()
        for r in work:
            linked_all.update(r.get("_linked") or [])
        any_linked = bool(linked_all) or any(TAG_REPLY_LINKED in c.tags for c in candidates.values())
        out: list[_Op] = []
        used_msgs: set[int] = set()
        declined_unlinked = False
        for op in ops:
            if op.op not in CLOSING_OPS:
                out.append(op)
                continue
            if not cls._closing_cue(op, candidates):
                res.invalid_ops += 1
                continue
            if any_linked:
                allowed = (op.msg.get("_linked") or []) if op.op == "decline" else linked_all
                if op.item_id in allowed:
                    out.append(op)
                else:
                    res.invalid_ops += 1
                continue
            msg_id = int(op.msg["msg_id"])
            if msg_id in used_msgs or (op.op == "decline" and declined_unlinked):
                res.invalid_ops += 1
                continue
            used_msgs.add(msg_id)
            if op.op == "decline":
                declined_unlinked = True
            out.append(op)
        return out

    # ------------------------------------------------------------ LLM
    async def _call_llm(self, prompt: str, res: ExtractResult) -> tuple[str | None, str]:
        timeout = self.timeout_s
        last_err = ""
        for attempt in range(1 + LLM_RETRIES):
            res.llm_attempts = attempt + 1
            try:
                raw = await asyncio.wait_for(self._llm(prompt, timeout), timeout + 5.0)
                return str(raw or ""), ""
            except _RETRYABLE as e:
                last_err = "llm_timeout" if isinstance(e, (asyncio.TimeoutError, TimeoutError)) else "llm_network"
                logger.warning(
                    f"[continuity] extract llm attempt={attempt + 1} failed scope={res.scope_key}: {last_err}"
                )
            except Exception as e:
                logger.warning(
                    f"[continuity] extract llm error scope={res.scope_key}: {type(e).__name__}"
                )
                return None, f"llm_error:{type(e).__name__}"
        return None, last_err

    # ------------------------------------------------------------ apply (worker thread)
    def _mark_all(
        self, scope_key: str, ignored: list[int], work_ids: list[int], status: str, error: str
    ) -> None:
        with self._tx():
            if ignored:
                self._store.mark_inbound(scope_key, ignored, m.INBOUND_SKIPPED, "")
            if work_ids:
                self._store.mark_inbound(scope_key, work_ids, status, error)

    def _mark_split(
        self, scope_key: str, ignored: list[int], retry_ids: list[int], final_ids: list[int], error: str
    ) -> None:
        """暂时性失败：attempts 未用尽的行回到 pending（记 last_error），其余 failed"""
        with self._tx():
            if ignored:
                self._store.mark_inbound(scope_key, ignored, m.INBOUND_SKIPPED, "")
            if retry_ids:
                self._store.mark_inbound(scope_key, retry_ids, m.INBOUND_PENDING, error)
            if final_ids:
                self._store.mark_inbound(scope_key, final_ids, m.INBOUND_FAILED, error)

    async def _mark_safe(self, scope_key: str, ids: list[int], status: str, error: str) -> None:
        try:
            await asyncio.to_thread(self._store.mark_inbound, scope_key, ids, status, error)
        except Exception as e:
            logger.warning(f"[continuity] extract mark_inbound failed: {type(e).__name__}")

    def _tx(self):
        """复用 store 的可重入写事务，使事项变更与入站状态原子落盘"""
        tx = getattr(self._store, "_tx", None)
        return tx() if callable(tx) else contextlib.nullcontext()

    def _apply(
        self,
        scope_key: str,
        owner_id: str,
        ops: list[_Op],
        candidates: dict[str, _Candidate],
        work: list[dict[str, Any]],
        ignored: list[int],
        work_ids: list[int],
        res: ExtractResult,
    ) -> None:
        store = self._store
        creates = sum(1 for o in ops if o.op == "create")
        batch_text = "\n".join(r["_text"] for r in work)
        conflict_closed = False  # LLM 期间有事项被关闭 → 本批行保持 pending 重新抽取
        with self._tx():
            revisions = {cid: int(c.item["revision"]) for cid, c in candidates.items()}
            for op in ops:
                # 提醒覆盖看 evidence 所在消息；只有一个 create 时看整批
                remind_text = batch_text if (op.op == "create" and creates == 1) else op.msg["_text"]
                base_msg = op.when_msg or op.msg
                base_ts = float(base_msg["msg_ts"])
                src_id = str(op.msg["msg_id"])
                try:
                    if op.op == "create":
                        item = self._create(scope_key, owner_id, op, remind_text, base_ts)
                        if item is not None:
                            res.created.append(item["item_id"])
                            self._adopt_promises(scope_key, item, float(op.msg["msg_ts"]))
                        continue
                    assert op.item_id is not None
                    cur = store.get_item(op.item_id)
                    if cur is None or cur["status"] != m.STATUS_ACTIVE:
                        res.invalid_ops += 1
                        if cur is not None and int(cur["revision"]) != revisions.get(op.item_id):
                            conflict_closed = True  # 抽取期间被别处关闭
                        continue
                    changes = self._item_changes(op, cur, remind_text, base_ts)
                    if not changes:
                        continue
                    try:
                        new = store.update_item(
                            op.item_id,
                            expected_revision=revisions.get(op.item_id),
                            reason=f"extract_{op.op}",
                            source_msg_id=src_id,
                            **changes,
                        )
                    except RevisionConflict:
                        # 抽取期间事项被别处修改：重读，仍 active 则在新 revision 上重放一次
                        fresh = store.get_item(op.item_id)
                        if fresh is None or fresh["status"] != m.STATUS_ACTIVE:
                            conflict_closed = True
                            raise
                        changes = self._item_changes(op, fresh, remind_text, base_ts)
                        revisions[op.item_id] = int(fresh["revision"])
                        if not changes:
                            continue
                        new = store.update_item(
                            op.item_id,
                            expected_revision=int(fresh["revision"]),
                            reason=f"extract_{op.op}",
                            source_msg_id=src_id,
                            **changes,
                        )
                        logger.info(
                            f"[continuity] extract op={op.op} item={op.item_id} reapplied after conflict"
                        )
                    revisions[op.item_id] = int(new["revision"])
                    if new["status"] != m.STATUS_ACTIVE:
                        res.closed.append(op.item_id)
                    else:
                        res.updated.append(op.item_id)
                except (RevisionConflict, ItemNotFound) as e:
                    res.invalid_ops += 1
                    logger.info(
                        f"[continuity] extract op={op.op} item={op.item_id} skipped: {type(e).__name__}"
                    )
            if ignored:
                store.mark_inbound(scope_key, ignored, m.INBOUND_SKIPPED, "")
            if work_ids and conflict_closed:
                attempts = {int(r["msg_id"]): int(r.get("attempts") or 0) for r in work}
                retry_ids = [i for i in work_ids if attempts.get(i, 0) < MAX_ATTEMPTS]
                final_ids = [i for i in work_ids if i not in retry_ids]
                if retry_ids:
                    store.mark_inbound(scope_key, retry_ids, m.INBOUND_PENDING, "revision_conflict")
                    res.retry = True
                    res.error = "revision_conflict"
                if final_ids:
                    store.mark_inbound(scope_key, final_ids, m.INBOUND_DONE, "")
            elif work_ids:
                store.mark_inbound(scope_key, work_ids, m.INBOUND_DONE, "")

    def _adopt_promises(self, scope_key: str, item: dict[str, Any], msg_ts: float) -> None:
        """新建的 plan/pending_result 收养稍早记录、尚未挂靠（或只是弱挂靠）的承诺"""
        if item.get("kind") not in PROMISE_LINK_KINDS:
            return
        try:
            promises = self._store.list_items(
                scope_key=scope_key, statuses=(m.STATUS_ACTIVE,), limit=MAX_CANDIDATES * 4
            )
        except Exception as e:
            logger.warning(f"[continuity] extract adopt promises failed: {type(e).__name__}")
            return
        for p in promises:
            if p.get("kind") != m.KIND_BOT_PROMISE:
                continue
            created = float(p.get("created_at") or 0)
            if created < msg_ts - 60.0 or created > msg_ts + PROMISE_ADOPT_WINDOW_S:
                continue
            link = _promise_link(p)
            if link and link.get("link") != LINK_RECENT:
                continue
            if link and link.get("linked_item_id") == item["item_id"]:
                continue
            _link_promise(self._store, p, item, LINK_ADOPTED)

    def _create(
        self, scope_key: str, owner_id: str, op: _Op, remind_text: str, base_ts: float
    ) -> dict[str, Any] | None:
        assert op.kind is not None
        fields: dict[str, Any] = {
            "subject_id": owner_id,
            "subject_type": "user",
            "scope_key": scope_key,
            "topic": op.topic,
            "summary": op.summary or op.topic,
            "kind": op.kind,
            "source_kind": m.SOURCE_SHARED_DIALOGUE,
            "source_refs": [
                {
                    "adapter": "telegram",
                    "scope_key": scope_key,
                    "msg_id": int(op.msg["msg_id"]),
                    "at": float(op.msg["msg_ts"]),
                }
            ],
            "allowed_audiences": [scope_key],
            "reason": "extract_create",
            "source_msg_id": str(op.msg["msg_id"]),
        }
        when_ctx = (op.when_msg or op.msg)["_text"]
        fields.update(compute_time_fields(op.kind, op.when_text, base_ts, when_ctx))
        fields.update(_remind_overrides(remind_text, op.remind_requested))
        if fields.get("remind_requested"):
            fields["importance"] = 2
        return self._store.create_item(**fields)

    @staticmethod
    def _item_changes(op: _Op, cur: dict[str, Any], remind_text: str, base_ts: float) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        if op.op == "complete":
            changes = {"status": m.STATUS_DONE, "closed_reason": "completed"}
        elif op.op == "cancel":
            changes = {"status": m.STATUS_CANCELLED, "closed_reason": "cancelled"}
        elif op.op == "decline":
            changes = {"status": m.STATUS_DECLINED, "closed_reason": "declined", "proactive_allowed": 0}
        else:  # update / reschedule
            if op.topic:
                changes["topic"] = op.topic
            if op.summary:
                changes["summary"] = op.summary
            kind = op.kind or cur["kind"]
            if op.kind:
                changes["kind"] = op.kind
            if op.op == "reschedule" or op.when_text:
                when_ctx = (op.when_msg or op.msg)["_text"]
                changes.update(compute_time_fields(kind, op.when_text, base_ts, when_ctx))
            changes.update(_remind_overrides(remind_text, op.remind_requested))
        # 只保留真实变化，避免无意义的 revision
        return {k: v for k, v in changes.items() if cur.get(k) != v}


# ---------------------------------------------------------------------------
# 后台 worker
# ---------------------------------------------------------------------------


class ExtractionWorker:
    """单并发抽取 worker：按 scope 去抖，信号队列有上限，启动时恢复 pending 行"""

    def __init__(
        self,
        extractor: ItemExtractor,
        *,
        config: Any = None,
        owner_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        self._extractor = extractor
        cfg = config if config is not None else extractor._config
        self._debounce = max(0.0, float(_cfg(cfg, "extract_debounce_s", DEFAULT_DEBOUNCE_S)))
        self._queue_max = max(1, int(_cfg(cfg, "extract_queue_max", DEFAULT_QUEUE_MAX)))
        self._owner_resolver = owner_resolver or _owner_from_scope
        self._due: dict[str, float] = {}  # scope → 到期（loop.time）
        self._first: dict[str, float] = {}  # scope → 首次信号时间（插入顺序 = 新旧）
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._busy = False
        self._retries: dict[str, int] = {}  # scope → 连续暂时性失败次数
        self.dropped = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def is_idle(self) -> bool:
        return not self._due and not self._busy

    async def start(self) -> None:
        if self.running:
            return
        self._wake = asyncio.Event()
        store = self._extractor.store
        try:
            # limit=0：只把超过 24h 的 pending 行标 stale，不领取
            await asyncio.to_thread(store.claim_pending_inbound, None, 0, STALE_AFTER_S)
            scopes = await asyncio.to_thread(store.pending_inbound_scopes)
        except Exception as e:
            logger.warning(f"[continuity] extract recovery failed: {type(e).__name__}")
            scopes = []
        for sk in scopes:
            self.notify(sk)
        if scopes:
            logger.info(f"[continuity] extract recovery scopes={len(scopes)}")
        self._task = asyncio.create_task(self._run(), name="iris-continuity-extract")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def notify(self, scope_key: str, *, delay: float | None = None) -> None:
        """登记 scope 待抽取信号（去抖：重复信号推迟到期，最长 4×debounce）"""
        try:
            loop_now = asyncio.get_running_loop().time()
        except RuntimeError:
            return
        d = self._debounce if delay is None else max(0.0, float(delay))
        if scope_key not in self._due:
            while len(self._due) >= self._queue_max:
                oldest = next(iter(self._first))
                self._first.pop(oldest, None)
                self._due.pop(oldest, None)
                self.dropped += 1
                logger.warning(f"[continuity] extract queue full; dropped signal scope={oldest}")
            self._first[scope_key] = loop_now
        first = self._first[scope_key]
        # 显式 delay（如退避重试）不受去抖上限约束
        cap = first + 4 * self._debounce if delay is None else loop_now + d
        self._due[scope_key] = min(loop_now + d, max(cap, loop_now))
        if self._wake is not None:
            self._wake.set()

    async def _run(self) -> None:
        assert self._wake is not None
        loop = asyncio.get_running_loop()
        while True:
            if not self._due:
                self._wake.clear()
                await self._wake.wait()
                continue
            scope, due = min(self._due.items(), key=lambda kv: kv[1])
            wait = due - loop.time()
            if wait > 0:
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), wait)
                continue
            self._due.pop(scope, None)
            self._first.pop(scope, None)
            self._busy = True
            try:
                await self._process(scope)
            finally:
                self._busy = False

    async def _process(self, scope_key: str) -> None:
        owner = self._owner_resolver(scope_key)
        if not owner:
            logger.warning(f"[continuity] extract no owner for scope={scope_key}; skipped")
            return
        try:
            res = await self._extractor.process_scope(scope_key, owner)
        except Exception as e:  # process_scope 自身已兜底，这里防御
            logger.warning(f"[continuity] extract worker error scope={scope_key}: {type(e).__name__}")
            return
        if res.retry:
            n = self._retries.get(scope_key, 0)
            if n < MAX_ATTEMPTS - 1:
                self._retries[scope_key] = n + 1
                self.notify(scope_key, delay=RETRY_BACKOFF_S * (2**n))
            else:
                self._retries.pop(scope_key, None)
            return
        self._retries.pop(scope_key, None)
        if res.claimed >= self._extractor.batch_limit:
            self.notify(scope_key, delay=0)


# ---------------------------------------------------------------------------
# Frisk 承诺
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?\n]")


def _sentence_with(text: str, evidence: str) -> str:
    for seg in _SENTENCE_SPLIT_RE.split(text):
        if evidence in seg:
            return seg.strip()
    return evidence


PROMISE_LINK_KINDS = frozenset({m.KIND_PLAN, m.KIND_PENDING_RESULT})
LINK_REPLY = "reply"
LINK_TOPIC = "topic"
LINK_RECENT = "recent"
LINK_ADOPTED = "adopted"


def _promise_link(promise: dict[str, Any]) -> dict[str, Any] | None:
    for ref in promise.get("source_refs") or []:
        if isinstance(ref, dict) and ref.get("linked_item_id"):
            return ref
    return None


def _bigrams(text: str) -> set[str]:
    t = m.norm_topic(text)
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _topic_overlap(topic: str, sentence: str) -> bool:
    return bool(_bigrams(topic) & _bigrams(sentence))


def _find_promise_target(
    store: Any, scope_key: str, owner_id: str, sentence: str, now: float, reply_to_msg_id: int | None
) -> tuple[dict[str, Any] | None, str]:
    """承诺挂靠目标：reply 关联 > 24h 内创建/提及且话题重叠 > scope 内最近一项"""

    def usable(item: dict[str, Any] | None) -> bool:
        return bool(
            item
            and item.get("status") == m.STATUS_ACTIVE
            and item.get("scope_key") == scope_key
            and item.get("kind") in PROMISE_LINK_KINDS
        )

    # 1) Frisk 所回复的入站消息若 reply_to 了带事项的出站消息 → 该事项
    inbound = None
    if reply_to_msg_id is not None and hasattr(store, "get_inbound"):
        inbound = store.get_inbound(scope_key, int(reply_to_msg_id))
    if inbound is None and hasattr(store, "latest_inbound"):
        inbound = store.latest_inbound(scope_key, str(owner_id), now - PROMISE_LINK_WINDOW_S)
    if inbound and inbound.get("reply_to_msg_id") is not None:
        ob = store.outbound_by_tg_id(scope_key, int(inbound["reply_to_msg_id"]))
        for iid in (ob or {}).get("item_ids") or []:
            item = store.get_item(str(iid))
            if usable(item):
                return item, LINK_REPLY

    items = [
        it
        for it in store.list_items(scope_key=scope_key, statuses=(m.STATUS_ACTIVE,), limit=50)
        if usable(it)
    ]
    if not items:
        return None, ""

    def touched(it: dict[str, Any]) -> float:
        return max(
            float(it.get("created_at") or 0),
            float(it.get("last_mentioned_at") or 0),
            float(it.get("updated_at") or 0),
        )

    items.sort(key=lambda it: (touched(it), str(it.get("item_id"))), reverse=True)
    # 2) 24h 内创建或提及、话题重叠
    for it in items:
        if now - touched(it) <= PROMISE_LINK_WINDOW_S and _topic_overlap(
            str(it.get("topic") or "") + str(it.get("summary") or ""), sentence
        ):
            return it, LINK_TOPIC
    # 3) scope 内最近一项
    return items[0], LINK_RECENT


def _link_promise(store: Any, promise: dict[str, Any], target: dict[str, Any], link: str) -> None:
    """承诺挂靠到用户事项：复制时间窗 + linked_item_id；用户事项记一条 bot_promise 引用"""
    refs = [
        r for r in (promise.get("source_refs") or []) if not (isinstance(r, dict) and r.get("linked_item_id"))
    ]
    refs.append({"linked_item_id": target["item_id"], "link": link})
    store.update_item(
        promise["item_id"],
        expected_revision=promise["revision"],
        reason="bot_promise_link",
        source_refs=refs,
        proactive_allowed=0,
        event_at_start=target.get("event_at_start"),
        event_at_end=target.get("event_at_end"),
        not_before=target.get("not_before"),
        expires_at=target.get("expires_at"),
        time_basis=target.get("time_basis") or "",
        time_precision=target.get("time_precision") or "none",
    )
    _mark_carrier(store, target["item_id"], promise)


def _mark_carrier(store: Any, item_id: str, promise: dict[str, Any]) -> None:
    item = store.get_item(item_id)
    if item is None:
        return
    refs = list(item.get("source_refs") or [])
    if any(isinstance(r, dict) and r.get("promise_item_id") == promise["item_id"] for r in refs):
        return
    refs.append(
        {
            "kind": "bot_promise",
            "promise_item_id": promise["item_id"],
            "tg_ids": list(promise.get("last_outbound_msg_ids") or []),
            "text": m.clip(promise.get("topic"), PROMISE_REF_TEXT_MAX),
        }
    )
    store.update_item(item_id, expected_revision=item["revision"], reason="bot_promise_carrier", source_refs=refs)


def record_bot_promise(
    store: Any,
    scope_key: str,
    owner_id: str,
    text: str,
    tg_ids: Sequence[int],
    now: float,
    reply_to_msg_id: int | None = None,
) -> dict[str, Any] | None:
    """已确认发出的 Frisk 文本含承诺 → bot_promise 事项；无承诺/失败返回 None（同步）

    只能在拿到 Telegram 回执（tg_ids 非空）后调用。承诺永远 proactive_allowed=0：
    - 能挂靠到同 scope 相关的 active plan/pending_result → 复制其时间窗并记
      linked_item_id，用户事项 source_refs 记一条 bot_promise（主动候选只有用户事项；
      用户事项关闭时 store 同事务关闭承诺）；
    - 无可挂靠事项 → 只作上下文（“你答应过（未完成）”），never due：
      not_before=None，expires_at=now+7d。
    """
    try:
        ids = [int(i) for i in (tg_ids or ())]
        if not ids:
            return None
        evidence = rules.detect_bot_promise(text)
        if not evidence:
            return None
        sentence = _sentence_with(str(text), evidence)
        now = float(now)
        tx = getattr(store, "_tx", None)
        with (tx() if callable(tx) else contextlib.nullcontext()):
            item = store.create_item(
                subject_id=str(owner_id),
                subject_type="self",
                scope_key=scope_key,
                topic=m.clip(sentence, m.TOPIC_MAX_CHARS),
                summary=m.clip(sentence, m.SUMMARY_MAX_CHARS),
                kind=m.KIND_BOT_PROMISE,
                source_kind=m.SOURCE_SELF_EXPRESSION,
                source_refs=[
                    {"adapter": "telegram", "scope_key": scope_key, "msg_id": i, "at": now} for i in ids
                ],
                allowed_audiences=[scope_key],
                last_outbound_msg_ids=ids,
                proactive_allowed=0,
                time_basis="",
                time_precision="none",
                not_before=None,
                expires_at=now + 7 * DAY,
                reason="bot_promise",
                source_msg_id=str(ids[0]),
            )
            link = ""
            if not _promise_link(item):
                target, link = _find_promise_target(
                    store, scope_key, str(owner_id), sentence, now, reply_to_msg_id
                )
                if target is not None:
                    _link_promise(store, item, target, link)
                    item = store.get_item(item["item_id"]) or item
        logger.info(
            f"[continuity] bot_promise item={item['item_id']} scope={scope_key} tg={ids}"
            f" link={link or 'none'}"
        )
        return item
    except Exception as e:
        logger.warning(f"[continuity] record_bot_promise failed scope={scope_key}: {type(e).__name__}")
        return None
