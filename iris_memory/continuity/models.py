"""
Iris Chat Memory - 连续性层常量与小工具

集中定义 continuity.db 中各枚举列的合法取值（DESIGN.md §3.1），
供 store / extractor / bridge 等模块共享，避免魔法字符串散落。
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

try:  # pragma: no cover - 容器内 tzdata 可用；兜底固定 UTC+8
    from zoneinfo import ZoneInfo

    TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    TZ_SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")

DEFAULT_TIME_ZONE = "Asia/Shanghai"

# ---------------------------------------------------------------- items
STATUS_ACTIVE = "active"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_DECLINED = "declined"
STATUS_EXPIRED = "expired"
STATUS_SUPERSEDED = "superseded"
ITEM_STATUSES = frozenset(
    {
        STATUS_ACTIVE,
        STATUS_DONE,
        STATUS_CANCELLED,
        STATUS_DECLINED,
        STATUS_EXPIRED,
        STATUS_SUPERSEDED,
    }
)

KIND_PLAN = "plan"
KIND_PENDING_RESULT = "pending_result"
KIND_UNFINISHED_TOPIC = "unfinished_topic"
KIND_BOT_PROMISE = "bot_promise"
KIND_INTEREST_QUESTION = "interest_question"
KIND_NOTE = "note"
ITEM_KINDS = frozenset(
    {
        KIND_PLAN,
        KIND_PENDING_RESULT,
        KIND_UNFINISHED_TOPIC,
        KIND_BOT_PROMISE,
        KIND_INTEREST_QUESTION,
        KIND_NOTE,
    }
)
# 到期可被自动置 expired 的“临时关注”类事项
TEMPORARY_KINDS = frozenset({KIND_UNFINISHED_TOPIC, KIND_INTEREST_QUESTION, KIND_NOTE})
# 受保护事项：到期只退出主动候选（proactive_allowed=0），绝不自动改状态
PROTECTED_KINDS = frozenset({KIND_PLAN, KIND_PENDING_RESULT, KIND_BOT_PROMISE})

SOURCE_SHARED_DIALOGUE = "shared_dialogue"
SOURCE_CHANNEL_OBSERVATION = "channel_observation"
SOURCE_HISTORY_IMPORT = "history_import"
SOURCE_THIRD_PARTY = "third_party"
SOURCE_SELF_EXPRESSION = "self_expression"
SOURCE_KINDS = frozenset(
    {
        SOURCE_SHARED_DIALOGUE,
        SOURCE_CHANNEL_OBSERVATION,
        SOURCE_HISTORY_IMPORT,
        SOURCE_THIRD_PARTY,
        SOURCE_SELF_EXPRESSION,
    }
)

SUBJECT_TYPES = frozenset({"user", "self"})
TIME_PRECISIONS = frozenset({"none", "day", "part_of_day", "range", "fuzzy"})

TOPIC_MAX_CHARS = 40
SUMMARY_MAX_CHARS = 160
IMPORTANCE_MIN = 1
IMPORTANCE_MAX = 3

# remind_requested 受保护事项到期后仍保留主动资格的宽限期
REMIND_GRACE_SECONDS = 7 * 86400

# 主动候选资格（与 ProactiveBridge.candidates 保持一致，DESIGN.md §5.3）
PROACTIVE_MENTION_COOLDOWN_S = 6 * 3600
PROACTIVE_MAX_ASK_COUNT = 2


def proactive_eligible(item: dict, audience: str, now: float) -> bool:
    """事项此刻能否作为主动话题：active、proactive_allowed、audience 允许、
    not_before 已到、未过期、ask_count<2、距上次提及 ≥6h。"""
    if item.get("status") != STATUS_ACTIVE or not item.get("proactive_allowed"):
        return False
    if audience not in [str(a) for a in (item.get("allowed_audiences") or [])]:
        return False
    nb = item.get("not_before")
    if nb is not None and now < float(nb):
        return False
    exp = item.get("expires_at")
    if exp is not None and now >= float(exp):
        return False
    last = item.get("last_mentioned_at")
    if last is not None and now - float(last) < PROACTIVE_MENTION_COOLDOWN_S:
        return False
    if int(item.get("ask_count") or 0) >= PROACTIVE_MAX_ASK_COUNT:
        return False
    return True


# ---------------------------------------------------------------- inbound
INBOUND_PENDING = "pending"
INBOUND_SKIPPED = "skipped"
INBOUND_DONE = "done"
INBOUND_FAILED = "failed"
INBOUND_STALE = "stale"
INBOUND_STATUSES = frozenset(
    {INBOUND_PENDING, INBOUND_SKIPPED, INBOUND_DONE, INBOUND_FAILED, INBOUND_STALE}
)

# ---------------------------------------------------------------- outbound
OUT_DRAFT = "draft"
OUT_SENDING = "sending"
OUT_SENT = "sent"
OUT_FAILED = "failed"
OUT_UNCERTAIN = "uncertain"
OUT_CANCELLED = "cancelled"
OUTBOUND_STATUSES = frozenset(
    {OUT_DRAFT, OUT_SENDING, OUT_SENT, OUT_FAILED, OUT_UNCERTAIN, OUT_CANCELLED}
)
OUTBOUND_ORIGINS = frozenset({"proactive", "reply", "manual", "probe"})  # probe = 运维验收探针
OUTBOUND_AUTHORS = frozenset({"bot", "manual", "unknown"})

# ---------------------------------------------------------------- turns
TURN_DECIDING = "deciding"
TURN_GENERATING = "generating"
TURN_SENDING = "sending"
TURN_SENT = "sent"
TURN_PARTIAL = "partial"
TURN_FAILED = "failed"
TURN_UNCERTAIN = "uncertain"
TURN_SKIPPED = "skipped"
TURN_CANCELLED = "cancelled"
TURN_STATES = frozenset(
    {
        TURN_DECIDING,
        TURN_GENERATING,
        TURN_SENDING,
        TURN_SENT,
        TURN_PARTIAL,
        TURN_FAILED,
        TURN_UNCERTAIN,
        TURN_SKIPPED,
        TURN_CANCELLED,
    }
)
QUOTA_STATES = frozenset({"none", "charged", "reserved", "released"})

# ---------------------------------------------------------------- closures / posts
CLOSURE_STATES = frozenset({"continue", "waiting", "ended"})
AUTHOR_KINDS = frozenset({"owner", "self", "other", "channel", "unknown"})


# ---------------------------------------------------------------- helpers
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def norm_topic(topic: str) -> str:
    """规范化话题用于 dedupe_key：NFKC、小写、去空白与标点"""
    text = unicodedata.normalize("NFKC", str(topic or "")).lower()
    return _PUNCT_RE.sub("", text)


def make_dedupe_key(scope_key: str, kind: str, topic: str) -> str:
    """dedupe_key = scope|kind|norm(topic)（DESIGN.md §3.1）"""
    return f"{scope_key}|{kind}|{norm_topic(topic)}"


def clip(text: Optional[str], limit: int) -> str:
    """按字符截断（不追加省略号）"""
    s = str(text or "").strip()
    return s if len(s) <= limit else s[:limit]


def local_day(ts: float) -> str:
    """时间戳 → Asia/Shanghai 本地日期 YYYY-MM-DD（计数器分日）"""
    return datetime.fromtimestamp(float(ts), TZ_SHANGHAI).strftime("%Y-%m-%d")
