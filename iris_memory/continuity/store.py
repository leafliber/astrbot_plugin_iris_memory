"""
Iris Chat Memory - 连续性层存储（continuity.db）

DESIGN.md §3：独立 SQLite 库，连续性层的**唯一写入入口**。

- 同步 sqlite3，check_same_thread=False，threading.RLock 串行化；
- WAL、busy_timeout=5000、foreign_keys=ON、row_factory=Row；
- PRAGMA user_version 迁移（_MIGRATIONS 逐版单事务），对已有版本（>0）
  的库迁移前 wal_checkpoint(TRUNCATE) 并复制 continuity.backup_v{N}.db；
- 连接为 autocommit 模式，多语句写操作显式 BEGIN IMMEDIATE 事务；
- 返回的 dict 中 JSON 列已解码为 Python 对象。

service 层以 ``await asyncio.to_thread(store.method, ...)`` 调用。
日志只记录 id / 状态，不记录消息正文。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from astrbot.api import logger

from . import models as m

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS items(
  item_id TEXT PRIMARY KEY,
  persona_id TEXT NOT NULL DEFAULT 'default',
  subject_id TEXT NOT NULL,
  subject_type TEXT NOT NULL,
  scope_key TEXT NOT NULL,
  topic TEXT NOT NULL,
  summary TEXT NOT NULL,
  kind TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  source_refs TEXT NOT NULL,
  allowed_audiences TEXT NOT NULL,
  status TEXT NOT NULL,
  remind_requested INTEGER NOT NULL DEFAULT 0,
  proactive_allowed INTEGER NOT NULL DEFAULT 1,
  importance INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL, updated_at REAL NOT NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  event_at_start REAL, event_at_end REAL,
  not_before REAL, expires_at REAL,
  time_basis TEXT NOT NULL DEFAULT '',
  time_zone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
  time_precision TEXT NOT NULL DEFAULT 'none',
  last_mentioned_at REAL,
  last_outbound_msg_ids TEXT NOT NULL DEFAULT '[]',
  ask_count INTEGER NOT NULL DEFAULT 0,
  closed_reason TEXT NOT NULL DEFAULT '',
  dedupe_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_scope_status ON items(scope_key,status);
CREATE INDEX IF NOT EXISTS idx_items_dedupe ON items(dedupe_key,status);
CREATE TABLE IF NOT EXISTS item_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL, revision INTEGER NOT NULL,
  snapshot TEXT NOT NULL, reason TEXT NOT NULL, source_msg_id TEXT, changed_at REAL NOT NULL,
  UNIQUE(item_id,revision));
CREATE TABLE IF NOT EXISTS inbound(
  scope_key TEXT NOT NULL, msg_id INTEGER NOT NULL, sender_id TEXT NOT NULL,
  text TEXT NOT NULL, msg_ts REAL NOT NULL, reply_to_msg_id INTEGER,
  received_at REAL NOT NULL,
  extract_status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(scope_key,msg_id));
CREATE TABLE IF NOT EXISTS watermarks(
  scope_key TEXT PRIMARY KEY, last_msg_id INTEGER NOT NULL, last_msg_ts REAL NOT NULL,
  session_version INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS outbound(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope_key TEXT NOT NULL, turn_id TEXT,
  origin TEXT NOT NULL,
  part_index INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL,
  status TEXT NOT NULL,
  author TEXT NOT NULL,
  tg_msg_ids TEXT NOT NULL DEFAULT '[]',
  reply_to_msg_id INTEGER, item_ids TEXT NOT NULL DEFAULT '[]',
  error TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL, updated_at REAL NOT NULL, sent_at REAL);
CREATE INDEX IF NOT EXISTS idx_outbound_scope ON outbound(scope_key,status,created_at);
CREATE TABLE IF NOT EXISTS outbound_ids(
  scope_key TEXT NOT NULL, tg_msg_id INTEGER NOT NULL, outbound_id INTEGER NOT NULL,
  PRIMARY KEY(scope_key,tg_msg_id));
CREATE TABLE IF NOT EXISTS proactive_turns(
  turn_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL, item_id TEXT,
  input_msg_id INTEGER NOT NULL DEFAULT 0, session_version INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '', forced INTEGER NOT NULL DEFAULT 0,
  quota_state TEXT NOT NULL DEFAULT 'none',
  llm_calls INTEGER NOT NULL DEFAULT 0, parts_sent INTEGER NOT NULL DEFAULT 0,
  parts_total INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_turns_scope ON proactive_turns(scope_key,state,created_at);
CREATE TABLE IF NOT EXISTS closures(
  scope_key TEXT PRIMARY KEY, state TEXT NOT NULL,
  explicit_wait INTEGER NOT NULL DEFAULT 0, wait_since REAL, ended_at REAL,
  summary TEXT NOT NULL DEFAULT '', open_questions TEXT NOT NULL DEFAULT '[]',
  next_opening TEXT NOT NULL DEFAULT '',
  last_owner_kind TEXT NOT NULL DEFAULT '', last_bot_kind TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS windows(
  scope_key TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS private_context(
  section TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
  updated_at REAL NOT NULL, audience TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observed_posts(
  chat_id INTEGER NOT NULL, msg_id INTEGER NOT NULL, posted_at REAL NOT NULL,
  sender_id TEXT NOT NULL DEFAULT '', author_kind TEXT NOT NULL,
  is_forward INTEGER NOT NULL DEFAULT 0, fwd_from TEXT NOT NULL DEFAULT '', text TEXT NOT NULL,
  via TEXT NOT NULL, ingested_at REAL NOT NULL, PRIMARY KEY(chat_id,msg_id));
CREATE TABLE IF NOT EXISTS cursors(name TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS counters(
  day TEXT NOT NULL, name TEXT NOT NULL, value INTEGER NOT NULL, PRIMARY KEY(day,name));
"""

# user_version=N 的库执行 _MIGRATIONS[N] 后升级到 N+1。只增不删。
_MIGRATIONS: Dict[int, str] = {
    0: _SCHEMA_V1,
}

# 各表 JSON 列（读出时解码）
_JSON_COLUMNS = {
    "items": ("source_refs", "allowed_audiences", "last_outbound_msg_ids"),
    "outbound": ("tg_msg_ids", "item_ids"),
    "closures": ("open_questions",),
    "item_history": ("snapshot",),
}

_ITEM_UPDATABLE = (
    "persona_id",
    "subject_id",
    "subject_type",
    "topic",
    "summary",
    "kind",
    "source_kind",
    "source_refs",
    "allowed_audiences",
    "status",
    "remind_requested",
    "proactive_allowed",
    "importance",
    "event_at_start",
    "event_at_end",
    "not_before",
    "expires_at",
    "time_basis",
    "time_zone",
    "time_precision",
    "last_mentioned_at",
    "last_outbound_msg_ids",
    "ask_count",
    "closed_reason",
)
_ITEM_CREATE_OPTIONAL = tuple(
    f for f in _ITEM_UPDATABLE if f not in ("subject_id", "subject_type", "topic", "summary", "kind", "source_kind")
)

_TURN_UPDATABLE = (
    "item_id",
    "input_msg_id",
    "session_version",
    "state",
    "reason",
    "forced",
    "quota_state",
    "llm_calls",
    "parts_sent",
    "parts_total",
)

_CLOSURE_FIELDS = (
    "state",
    "explicit_wait",
    "wait_since",
    "ended_at",
    "summary",
    "open_questions",
    "next_opening",
    "last_owner_kind",
    "last_bot_kind",
)

_POST_FIELDS = (
    "chat_id",
    "msg_id",
    "posted_at",
    "sender_id",
    "author_kind",
    "is_forward",
    "fwd_from",
    "text",
    "via",
)


class RevisionConflict(Exception):
    """update_item 的 expected_revision 与库中 revision 不一致"""

    def __init__(self, item_id: str, expected: int, actual: int):
        super().__init__(f"item {item_id}: expected revision {expected}, actual {actual}")
        self.item_id = item_id
        self.expected = expected
        self.actual = actual


class ItemNotFound(KeyError):
    """事项不存在"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _row_to_dict(row: Optional[sqlite3.Row], table: str) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    for col in _JSON_COLUMNS.get(table, ()):
        if col in d:
            d[col] = _loads(d[col], {} if col == "snapshot" else [])
    return d


def _flag(value: Any) -> int:
    return 1 if value else 0


def _require(value: Any, allowed: frozenset, field: str) -> str:
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")
    return value


def _int_list(values: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for v in values or ():
        iv = int(v)
        if iv not in out:
            out.append(iv)
    return out


class ContinuityStore:
    """continuity.db 读写（同步；线程安全）"""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 5000,
    ):
        self._db_path = Path(db_path)
        self._clock = clock
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # 生命周期与迁移
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def is_open(self) -> bool:
        return self._db is not None

    def open(self) -> None:
        """打开连接并执行迁移（幂等）；迁移失败关闭连接并抛出"""
        with self._lock:
            if self._db is not None:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(
                str(self._db_path), check_same_thread=False, isolation_level=None
            )
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
                db.execute("PRAGMA foreign_keys=ON")
                self._db = db
                self._migrate()
            except Exception:
                self._db = None
                db.close()
                raise

    def close(self) -> None:
        with self._lock:
            if self._db is None:
                return
            try:
                self._db.close()
            finally:
                self._db = None

    def __enter__(self) -> "ContinuityStore":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def get_schema_version(self) -> int:
        with self._lock:
            return int(self._conn().execute("PRAGMA user_version").fetchone()[0])

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("ContinuityStore is not open")
        return self._db

    def _migrate(self) -> None:
        current = self.get_schema_version()
        if current >= SCHEMA_VERSION:
            return
        if current > 0:
            self._backup_before_migration(current)
        db = self._conn()
        for version in range(current, SCHEMA_VERSION):
            script = _MIGRATIONS.get(version)
            if script is None:
                raise RuntimeError(f"missing continuity migration v{version} -> v{version + 1}")
            # 逐句执行保证单事务（executescript 会隐式 COMMIT）
            db.execute("BEGIN IMMEDIATE")
            try:
                for stmt in script.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        db.execute(stmt)
                db.execute(f"PRAGMA user_version={version + 1}")
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
            logger.info(f"[continuity] continuity.db schema migrated to v{version + 1}")

    def _backup_before_migration(self, current_version: int) -> None:
        if not self._db_path.exists() or self._db_path.stat().st_size == 0:
            return
        backup = self._db_path.with_name(f"{self._db_path.stem}.backup_v{current_version}.db")
        try:
            self._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
            shutil.copy2(self._db_path, backup)
            logger.info(f"[continuity] backup before migration: {backup.name}")
        except Exception as e:
            logger.warning(f"[continuity] backup before migration failed (continuing): {e}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """加锁 + BEGIN IMMEDIATE 写事务（可重入：嵌套时复用外层事务）"""
        with self._lock:
            db = self._conn()
            if db.in_transaction:
                yield db
                return
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")

    def _now(self) -> float:
        return float(self._clock())

    # ------------------------------------------------------------------
    # items
    # ------------------------------------------------------------------

    def create_item(self, **fields: Any) -> Dict[str, Any]:
        """新建事项（revision=1 + history）

        必填：subject_id, subject_type, scope_key, topic, summary, kind, source_kind。
        dedupe_key 缺省为 scope|kind|norm(topic)；同 dedupe_key 已有 active 项时
        返回该项，不新建、不改动。allowed_audiences 缺省为 [scope_key]。
        可选 reason / source_msg_id 写入 history。
        """
        reason = str(fields.pop("reason", "create") or "create")
        source_msg_id = fields.pop("source_msg_id", None)
        missing = [
            k
            for k in ("subject_id", "subject_type", "scope_key", "topic", "summary", "kind", "source_kind")
            if fields.get(k) in (None, "")
        ]
        if missing:
            raise ValueError(f"create_item missing fields: {missing}")
        unknown = set(fields) - set(_ITEM_UPDATABLE) - {"scope_key", "dedupe_key", "item_id"}
        if unknown:
            raise ValueError(f"create_item unknown fields: {sorted(unknown)}")

        scope_key = str(fields["scope_key"])
        kind = _require(fields["kind"], m.ITEM_KINDS, "kind")
        topic = m.clip(fields["topic"], m.TOPIC_MAX_CHARS)
        if not topic:
            raise ValueError("create_item: empty topic")
        dedupe_key = str(fields.get("dedupe_key") or m.make_dedupe_key(scope_key, kind, topic))
        now = self._now()

        row: Dict[str, Any] = {
            "item_id": str(fields.get("item_id") or uuid.uuid4().hex),
            "persona_id": "default",
            "subject_id": str(fields["subject_id"]),
            "subject_type": _require(fields["subject_type"], m.SUBJECT_TYPES, "subject_type"),
            "scope_key": scope_key,
            "topic": topic,
            "summary": m.clip(fields["summary"], m.SUMMARY_MAX_CHARS),
            "kind": kind,
            "source_kind": _require(fields["source_kind"], m.SOURCE_KINDS, "source_kind"),
            "source_refs": [],
            "allowed_audiences": [scope_key],
            "status": m.STATUS_ACTIVE,
            "remind_requested": 0,
            "proactive_allowed": 1,
            "importance": 1,
            "created_at": now,
            "updated_at": now,
            "revision": 1,
            "event_at_start": None,
            "event_at_end": None,
            "not_before": None,
            "expires_at": None,
            "time_basis": "",
            "time_zone": m.DEFAULT_TIME_ZONE,
            "time_precision": "none",
            "last_mentioned_at": None,
            "last_outbound_msg_ids": [],
            "ask_count": 0,
            "closed_reason": "",
            "dedupe_key": dedupe_key,
        }
        for k in _ITEM_CREATE_OPTIONAL:
            if k in fields and fields[k] is not None:
                row[k] = fields[k]
        row = self._normalize_item(row)

        with self._tx() as db:
            if row["status"] == m.STATUS_ACTIVE:
                existing = db.execute(
                    "SELECT * FROM items WHERE dedupe_key=? AND status='active'"
                    " ORDER BY created_at LIMIT 1",
                    (dedupe_key,),
                ).fetchone()
                if existing is not None:
                    return _row_to_dict(existing, "items")
            cols = list(row)
            db.execute(
                f"INSERT INTO items({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [self._encode_item_value(c, row[c]) for c in cols],
            )
            self._write_history(db, row, reason, source_msg_id, now)
        return self.get_item(row["item_id"])  # type: ignore[return-value]

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
        return _row_to_dict(row, "items")

    def find_active_by_dedupe(self, dedupe_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM items WHERE dedupe_key=? AND status='active' ORDER BY created_at LIMIT 1",
                (dedupe_key,),
            ).fetchone()
        return _row_to_dict(row, "items")

    def update_item(
        self,
        item_id: str,
        *,
        expected_revision: Optional[int],
        reason: str,
        source_msg_id: Optional[str] = None,
        **changes: Any,
    ) -> Dict[str, Any]:
        """修改事项：revision+1 并写 history（快照为新版本全量）

        expected_revision 非 None 且与库中不符 → RevisionConflict；
        事项不存在 → ItemNotFound。topic/kind 变化时重算 dedupe_key。
        """
        unknown = set(changes) - set(_ITEM_UPDATABLE)
        if unknown:
            raise ValueError(f"update_item unknown fields: {sorted(unknown)}")
        with self._tx() as db:
            row = db.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
            if row is None:
                raise ItemNotFound(item_id)
            cur = _row_to_dict(row, "items")
            assert cur is not None
            if expected_revision is not None and int(expected_revision) != int(cur["revision"]):
                raise RevisionConflict(item_id, int(expected_revision), int(cur["revision"]))
            new = dict(cur)
            new.update(changes)
            if "topic" in changes:
                new["topic"] = m.clip(new["topic"], m.TOPIC_MAX_CHARS)
                if not new["topic"]:
                    raise ValueError("update_item: empty topic")
            if "summary" in changes:
                new["summary"] = m.clip(new["summary"], m.SUMMARY_MAX_CHARS)
            new = self._normalize_item(new)
            if new["topic"] != cur["topic"] or new["kind"] != cur["kind"]:
                new["dedupe_key"] = m.make_dedupe_key(new["scope_key"], new["kind"], new["topic"])
            now = self._now()
            new["revision"] = int(cur["revision"]) + 1
            new["updated_at"] = now
            cols = [c for c in new if c not in ("item_id", "created_at", "scope_key")]
            res = db.execute(
                f"UPDATE items SET {','.join(f'{c}=?' for c in cols)} WHERE item_id=? AND revision=?",
                [self._encode_item_value(c, new[c]) for c in cols] + [item_id, cur["revision"]],
            )
            if res.rowcount != 1:  # pragma: no cover - 锁内不应发生
                raise RevisionConflict(item_id, int(cur["revision"]), -1)
            self._write_history(db, new, reason, source_msg_id, now)
            if new["kind"] != m.KIND_BOT_PROMISE:
                self._sync_linked_promises(cur, new, source_msg_id)
        return self.get_item(item_id)  # type: ignore[return-value]

    # 关联承诺随用户事项同步的字段（时间窗口）
    _PROMISE_TIME_FIELDS = (
        "event_at_start",
        "event_at_end",
        "not_before",
        "expires_at",
        "time_basis",
        "time_precision",
    )
    # 用户事项进入这些状态时，关联的 bot_promise 同事务关闭
    _PROMISE_CLOSING = (m.STATUS_DONE, m.STATUS_CANCELLED, m.STATUS_DECLINED)

    def linked_promises(self, item_id: str) -> List[Dict[str, Any]]:
        """source_refs 中 linked_item_id == item_id 的 active bot_promise 事项"""
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM items WHERE kind=? AND status='active' AND EXISTS ("
                "SELECT 1 FROM json_each(items.source_refs)"
                " WHERE json_type(value)='object' AND json_extract(value,'$.linked_item_id')=?)"
                " ORDER BY created_at, item_id",
                (m.KIND_BOT_PROMISE, str(item_id)),
            ).fetchall()
        return [_row_to_dict(r, "items") for r in rows]  # type: ignore[misc]

    def _sync_linked_promises(
        self, cur: Dict[str, Any], new: Dict[str, Any], source_msg_id: Optional[str]
    ) -> None:
        """在 update_item 的同一事务内：用户事项关闭 → 关联承诺一并关闭；改时间 → 承诺跟随"""
        closing = cur["status"] == m.STATUS_ACTIVE and new["status"] in self._PROMISE_CLOSING
        time_changed = new["status"] == m.STATUS_ACTIVE and any(
            cur.get(f) != new.get(f) for f in self._PROMISE_TIME_FIELDS
        )
        if not closing and not time_changed:
            return
        for p in self.linked_promises(new["item_id"]):
            if p["scope_key"] != new["scope_key"]:
                continue
            if closing:
                changes: Dict[str, Any] = {
                    "status": new["status"],
                    "closed_reason": f"linked_{new.get('closed_reason') or new['status']}",
                    "proactive_allowed": 0,
                }
                reason = "linked_close"
            else:
                changes = {f: new.get(f) for f in self._PROMISE_TIME_FIELDS}
                reason = "linked_reschedule"
            self.update_item(
                p["item_id"],
                expected_revision=p["revision"],
                reason=reason,
                source_msg_id=source_msg_id,
                **changes,
            )

    def item_history(self, item_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM item_history WHERE item_id=? ORDER BY revision", (item_id,)
            ).fetchall()
        return [_row_to_dict(r, "item_history") for r in rows]  # type: ignore[misc]

    def list_items(
        self,
        scope_key: Optional[str] = None,
        statuses: Optional[Sequence[str]] = (m.STATUS_ACTIVE,),
        audience: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出事项；statuses=None 表示不限状态；audience 非空时只返回 allowed_audiences 含它的

        排序：remind_requested desc, importance desc, updated_at desc。
        """
        where: List[str] = []
        args: List[Any] = []
        if scope_key is not None:
            where.append("scope_key=?")
            args.append(scope_key)
        if statuses is not None:
            sts = list(statuses)
            if not sts:
                return []
            where.append(f"status IN ({','.join('?' * len(sts))})")
            args.extend(sts)
        if audience:
            where.append("EXISTS (SELECT 1 FROM json_each(items.allowed_audiences) WHERE value=?)")
            args.append(audience)
        sql = "SELECT * FROM items"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY remind_requested DESC, importance DESC, updated_at DESC, item_id LIMIT ?"
        args.append(max(0, int(limit)))
        with self._lock:
            rows = self._conn().execute(sql, args).fetchall()
        return [_row_to_dict(r, "items") for r in rows]  # type: ignore[misc]

    def expire_items(self, now: float) -> int:
        """到期清理（DESIGN.md §3.2）

        - 临时类（unfinished_topic / interest_question / note）且 remind_requested=0、
          expires_at<now 的 active 项 → status=expired；
        - 其余 active 项（plan / pending_result / bot_promise，或 remind_requested=1）
          绝不改状态：remind_requested=0 时 expires_at<now、remind_requested=1 时
          expires_at+7d<now，仅把 proactive_allowed 置 0（退出主动候选）。
        每次变更走 update_item（revision+1 + history）。返回变更的事项数。
        """
        now = float(now)
        changed = 0
        with self._tx() as db:
            rows = db.execute(
                "SELECT item_id, kind, remind_requested, proactive_allowed, expires_at, revision"
                " FROM items WHERE status='active' AND expires_at IS NOT NULL AND expires_at<?",
                (now,),
            ).fetchall()
            for r in rows:
                temporary = r["kind"] in m.TEMPORARY_KINDS and not r["remind_requested"]
                if temporary:
                    self.update_item(
                        r["item_id"],
                        expected_revision=r["revision"],
                        reason="expire",
                        status=m.STATUS_EXPIRED,
                        closed_reason="expired",
                    )
                    changed += 1
                    continue
                threshold = r["expires_at"] + (m.REMIND_GRACE_SECONDS if r["remind_requested"] else 0)
                if r["proactive_allowed"] and threshold < now:
                    self.update_item(
                        r["item_id"],
                        expected_revision=r["revision"],
                        reason="expire_proactive",
                        proactive_allowed=0,
                    )
                    changed += 1
        if changed:
            logger.info(f"[continuity] expire_items changed={changed}")
        return changed

    def _normalize_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        _require(row["kind"], m.ITEM_KINDS, "kind")
        _require(row["source_kind"], m.SOURCE_KINDS, "source_kind")
        _require(row["subject_type"], m.SUBJECT_TYPES, "subject_type")
        _require(row["status"], m.ITEM_STATUSES, "status")
        _require(row["time_precision"], m.TIME_PRECISIONS, "time_precision")
        row["remind_requested"] = _flag(row["remind_requested"])
        row["proactive_allowed"] = _flag(row["proactive_allowed"])
        row["importance"] = min(m.IMPORTANCE_MAX, max(m.IMPORTANCE_MIN, int(row["importance"])))
        row["ask_count"] = max(0, int(row["ask_count"]))
        for k in ("source_refs", "allowed_audiences"):
            if not isinstance(row[k], (list, tuple)):
                raise ValueError(f"{k} must be a list")
            row[k] = list(row[k])
        row["allowed_audiences"] = [str(a) for a in row["allowed_audiences"]]
        row["last_outbound_msg_ids"] = _int_list(row["last_outbound_msg_ids"])
        for k in ("event_at_start", "event_at_end", "not_before", "expires_at", "last_mentioned_at"):
            if row[k] is not None:
                row[k] = float(row[k])
        return row

    @staticmethod
    def _encode_item_value(col: str, value: Any) -> Any:
        if col in _JSON_COLUMNS["items"]:
            return _dumps(value)
        return value

    @staticmethod
    def _write_history(
        db: sqlite3.Connection,
        row: Dict[str, Any],
        reason: str,
        source_msg_id: Optional[str],
        now: float,
    ) -> None:
        db.execute(
            "INSERT INTO item_history(item_id,revision,snapshot,reason,source_msg_id,changed_at)"
            " VALUES (?,?,?,?,?,?)",
            (
                row["item_id"],
                int(row["revision"]),
                _dumps(row),
                str(reason or ""),
                None if source_msg_id is None else str(source_msg_id),
                now,
            ),
        )

    # ------------------------------------------------------------------
    # inbound / watermark
    # ------------------------------------------------------------------

    def record_inbound(
        self,
        scope_key: str,
        msg_id: int,
        sender_id: str,
        text: str,
        msg_ts: float,
        reply_to_msg_id: Optional[int] = None,
    ) -> bool:
        """持久化入站消息；True=新消息，重复返回 False 且不改任何状态"""
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO inbound(scope_key,msg_id,sender_id,text,msg_ts,"
                "reply_to_msg_id,received_at) VALUES (?,?,?,?,?,?,?)",
                (
                    scope_key,
                    int(msg_id),
                    str(sender_id),
                    str(text or ""),
                    float(msg_ts),
                    None if reply_to_msg_id is None else int(reply_to_msg_id),
                    self._now(),
                ),
            )
            return cur.rowcount == 1

    def get_inbound(self, scope_key: str, msg_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM inbound WHERE scope_key=? AND msg_id=?", (scope_key, int(msg_id))
            ).fetchone()
        return _row_to_dict(row, "inbound")

    def advance_watermark(self, scope_key: str, msg_id: int, msg_ts: float) -> Tuple[bool, int]:
        """只有 msg_id > last_msg_id 才推进并 session_version+1；返回 (advanced, session_version)"""
        msg_id = int(msg_id)
        with self._tx() as db:
            row = db.execute(
                "SELECT last_msg_id, session_version FROM watermarks WHERE scope_key=?", (scope_key,)
            ).fetchone()
            now = self._now()
            if row is None:
                db.execute(
                    "INSERT INTO watermarks(scope_key,last_msg_id,last_msg_ts,session_version,updated_at)"
                    " VALUES (?,?,?,1,?)",
                    (scope_key, msg_id, float(msg_ts), now),
                )
                return True, 1
            if msg_id <= int(row["last_msg_id"]):
                return False, int(row["session_version"])
            version = int(row["session_version"]) + 1
            db.execute(
                "UPDATE watermarks SET last_msg_id=?, last_msg_ts=?, session_version=?, updated_at=?"
                " WHERE scope_key=?",
                (msg_id, float(msg_ts), version, now, scope_key),
            )
            return True, version

    def latest_inbound(
        self, scope_key: str, sender_id: Optional[str] = None, since_ts: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """该 scope 最近一条入站消息（可限定发送者与 msg_ts 下限）"""
        sql = "SELECT * FROM inbound WHERE scope_key=?"
        args: List[Any] = [scope_key]
        if sender_id is not None:
            sql += " AND sender_id=?"
            args.append(str(sender_id))
        if since_ts is not None:
            sql += " AND msg_ts>=?"
            args.append(float(since_ts))
        sql += " ORDER BY msg_ts DESC, msg_id DESC LIMIT 1"
        with self._lock:
            row = self._conn().execute(sql, args).fetchone()
        return _row_to_dict(row, "inbound")

    def get_watermark(self, scope_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM watermarks WHERE scope_key=?", (scope_key,)
            ).fetchone()
        return _row_to_dict(row, "watermarks")

    def claim_pending_inbound(
        self,
        scope_key: Optional[str] = None,
        limit: int = 20,
        stale_after: float = 86400,
    ) -> List[Dict[str, Any]]:
        """取 pending 入站行（按 scope, msg_id 升序），attempts+1

        msg_ts 早于 now-stale_after 的 pending 行先标 stale，不返回。
        """
        now = self._now()
        scope_sql = " AND scope_key=?" if scope_key is not None else ""
        scope_args: List[Any] = [scope_key] if scope_key is not None else []
        with self._tx() as db:
            db.execute(
                "UPDATE inbound SET extract_status='stale' WHERE extract_status='pending' AND msg_ts<?"
                + scope_sql,
                [now - float(stale_after)] + scope_args,
            )
            rows = db.execute(
                "SELECT scope_key, msg_id FROM inbound WHERE extract_status='pending'"
                + scope_sql
                + " ORDER BY scope_key, msg_id LIMIT ?",
                scope_args + [max(0, int(limit))],
            ).fetchall()
            keys = [(r["scope_key"], r["msg_id"]) for r in rows]
            for sk, mid in keys:
                db.execute(
                    "UPDATE inbound SET attempts=attempts+1 WHERE scope_key=? AND msg_id=?", (sk, mid)
                )
            out = [
                _row_to_dict(
                    db.execute(
                        "SELECT * FROM inbound WHERE scope_key=? AND msg_id=?", (sk, mid)
                    ).fetchone(),
                    "inbound",
                )
                for sk, mid in keys
            ]
        return out  # type: ignore[return-value]

    def pending_inbound_scopes(self) -> List[str]:
        """有 pending 入站行的 scope（启动时恢复抽取队列用）"""
        with self._lock:
            rows = self._conn().execute(
                "SELECT DISTINCT scope_key FROM inbound WHERE extract_status='pending' ORDER BY scope_key"
            ).fetchall()
        return [r[0] for r in rows]

    def mark_inbound(
        self, scope_key: str, msg_ids: Iterable[int], status: str, error: str = ""
    ) -> int:
        _require(status, m.INBOUND_STATUSES, "extract_status")
        ids = _int_list(msg_ids)
        if not ids:
            return 0
        with self._tx() as db:
            cur = db.execute(
                f"UPDATE inbound SET extract_status=?, last_error=? WHERE scope_key=?"
                f" AND msg_id IN ({','.join('?' * len(ids))})",
                [status, str(error or "")[:500], scope_key] + ids,
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # outbound
    # ------------------------------------------------------------------

    def add_outbound(
        self,
        scope_key: str,
        origin: str,
        text: str,
        status: str,
        author: str,
        turn_id: Optional[str] = None,
        part_index: int = 0,
        reply_to_msg_id: Optional[int] = None,
        item_ids: Sequence[str] = (),
    ) -> int:
        _require(origin, m.OUTBOUND_ORIGINS, "origin")
        _require(status, m.OUTBOUND_STATUSES, "status")
        _require(author, m.OUTBOUND_AUTHORS, "author")
        now = self._now()
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO outbound(scope_key,turn_id,origin,part_index,text,status,author,"
                "reply_to_msg_id,item_ids,created_at,updated_at,sent_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope_key,
                    turn_id,
                    origin,
                    int(part_index),
                    str(text or ""),
                    status,
                    author,
                    None if reply_to_msg_id is None else int(reply_to_msg_id),
                    _dumps([str(i) for i in item_ids]),
                    now,
                    now,
                    now if status == m.OUT_SENT else None,
                ),
            )
            return int(cur.lastrowid)

    def get_outbound(self, outbound_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM outbound WHERE id=?", (int(outbound_id),)
            ).fetchone()
        return _row_to_dict(row, "outbound")

    def mark_outbound(
        self,
        outbound_id: int,
        status: str,
        tg_msg_ids: Iterable[int] = (),
        error: str = "",
    ) -> Dict[str, Any]:
        """更新出站状态并登记 Telegram id（回执重放幂等）

        - tg id 已登记给本行 → 不重复；已登记给其它行 → 忽略并告警；
        - 已 sent 的行不会被降级（发送证据优先），仅合并新 id；
        - 首次进入 sent 时写 sent_at。
        """
        _require(status, m.OUTBOUND_STATUSES, "status")
        ids = _int_list(tg_msg_ids)
        with self._tx() as db:
            row = db.execute("SELECT * FROM outbound WHERE id=?", (int(outbound_id),)).fetchone()
            if row is None:
                raise KeyError(f"outbound {outbound_id} not found")
            cur = _row_to_dict(row, "outbound")
            assert cur is not None
            scope_key = cur["scope_key"]
            merged = list(cur["tg_msg_ids"])
            for tid in ids:
                db.execute(
                    "INSERT OR IGNORE INTO outbound_ids(scope_key,tg_msg_id,outbound_id) VALUES (?,?,?)",
                    (scope_key, tid, cur["id"]),
                )
                owner = db.execute(
                    "SELECT outbound_id FROM outbound_ids WHERE scope_key=? AND tg_msg_id=?",
                    (scope_key, tid),
                ).fetchone()[0]
                if owner != cur["id"]:
                    logger.warning(
                        f"[continuity] tg id {tid} already bound to outbound {owner}; "
                        f"ignored for outbound {cur['id']}"
                    )
                    continue
                if tid not in merged:
                    merged.append(tid)
            new_status = status
            if cur["status"] == m.OUT_SENT and status != m.OUT_SENT:
                logger.warning(
                    f"[continuity] outbound {cur['id']} already sent; ignore downgrade to {status}"
                )
                new_status = m.OUT_SENT
            now = self._now()
            sent_at = cur["sent_at"]
            if new_status == m.OUT_SENT and sent_at is None:
                sent_at = now
            new_error = str(error or "")[:500] if error else cur["error"]
            db.execute(
                "UPDATE outbound SET status=?, tg_msg_ids=?, error=?, sent_at=?, updated_at=? WHERE id=?",
                (new_status, _dumps(merged), new_error, sent_at, now, cur["id"]),
            )
        return self.get_outbound(outbound_id)  # type: ignore[return-value]

    def outbound_by_tg_id(self, scope_key: str, tg_msg_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT o.* FROM outbound_ids i JOIN outbound o ON o.id=i.outbound_id"
                " WHERE i.scope_key=? AND i.tg_msg_id=?",
                (scope_key, int(tg_msg_id)),
            ).fetchone()
        return _row_to_dict(row, "outbound")

    def recent_outbound(
        self,
        scope_key: str,
        statuses: Sequence[str] = (m.OUT_SENT,),
        authors: Sequence[str] = ("bot",),
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """最近出站（新→旧，按 COALESCE(sent_at, created_at)）"""
        sts, aus = list(statuses), list(authors)
        if not sts or not aus:
            return []
        with self._lock:
            rows = self._conn().execute(
                f"SELECT * FROM outbound WHERE scope_key=? AND status IN ({','.join('?' * len(sts))})"
                f" AND author IN ({','.join('?' * len(aus))})"
                " ORDER BY COALESCE(sent_at, created_at) DESC, id DESC LIMIT ?",
                [scope_key] + sts + aus + [max(0, int(limit))],
            ).fetchall()
        return [_row_to_dict(r, "outbound") for r in rows]  # type: ignore[misc]

    def outbound_in_state(
        self, statuses: Sequence[str], scope_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按状态列出出站（对账用，旧→新）"""
        sts = list(statuses)
        if not sts:
            return []
        sql = f"SELECT * FROM outbound WHERE status IN ({','.join('?' * len(sts))})"
        args: List[Any] = list(sts)
        if scope_key is not None:
            sql += " AND scope_key=?"
            args.append(scope_key)
        with self._lock:
            rows = self._conn().execute(sql + " ORDER BY id", args).fetchall()
        return [_row_to_dict(r, "outbound") for r in rows]  # type: ignore[misc]

    def outbound_for_turn(self, turn_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM outbound WHERE turn_id=? ORDER BY part_index, id", (turn_id,)
            ).fetchall()
        return [_row_to_dict(r, "outbound") for r in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # proactive turns
    # ------------------------------------------------------------------

    def create_turn(
        self,
        scope_key: str,
        *,
        input_msg_id: int,
        session_version: int,
        forced: bool,
        item_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        turn_id = uuid.uuid4().hex
        now = self._now()
        with self._tx() as db:
            db.execute(
                "INSERT INTO proactive_turns(turn_id,scope_key,item_id,input_msg_id,session_version,"
                "state,forced,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    turn_id,
                    scope_key,
                    item_id,
                    int(input_msg_id or 0),
                    int(session_version or 0),
                    m.TURN_DECIDING,
                    _flag(forced),
                    now,
                    now,
                ),
            )
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM proactive_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
        return _row_to_dict(row, "proactive_turns")

    def update_turn(self, turn_id: str, **changes: Any) -> Dict[str, Any]:
        unknown = set(changes) - set(_TURN_UPDATABLE)
        if unknown:
            raise ValueError(f"update_turn unknown fields: {sorted(unknown)}")
        if "state" in changes:
            _require(changes["state"], m.TURN_STATES, "state")
        if "quota_state" in changes:
            _require(changes["quota_state"], m.QUOTA_STATES, "quota_state")
        if "forced" in changes:
            changes["forced"] = _flag(changes["forced"])
        if "reason" in changes:
            changes["reason"] = str(changes["reason"] or "")
        with self._tx() as db:
            cols = list(changes)
            res = db.execute(
                f"UPDATE proactive_turns SET {''.join(f'{c}=?,' for c in cols)}updated_at=?"
                " WHERE turn_id=?",
                [changes[c] for c in cols] + [self._now(), turn_id],
            )
            if res.rowcount != 1:
                raise KeyError(f"turn {turn_id} not found")
        return self.get_turn(turn_id)  # type: ignore[return-value]

    def turns_in_state(
        self, states: Sequence[str], scope_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sts = list(states)
        if not sts:
            return []
        sql = f"SELECT * FROM proactive_turns WHERE state IN ({','.join('?' * len(sts))})"
        args: List[Any] = list(sts)
        if scope_key is not None:
            sql += " AND scope_key=?"
            args.append(scope_key)
        with self._lock:
            rows = self._conn().execute(sql + " ORDER BY created_at, turn_id", args).fetchall()
        return [_row_to_dict(r, "proactive_turns") for r in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # closures
    # ------------------------------------------------------------------

    def get_closure(self, scope_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn().execute(
                "SELECT * FROM closures WHERE scope_key=?", (scope_key,)
            ).fetchone()
        return _row_to_dict(row, "closures")

    def put_closure(self, scope_key: str, **fields: Any) -> Dict[str, Any]:
        """合并写入 closure（未给出的字段保留旧值；新行 state 缺省 continue）"""
        unknown = set(fields) - set(_CLOSURE_FIELDS)
        if unknown:
            raise ValueError(f"put_closure unknown fields: {sorted(unknown)}")
        with self._tx() as db:
            cur = _row_to_dict(
                db.execute("SELECT * FROM closures WHERE scope_key=?", (scope_key,)).fetchone(),
                "closures",
            ) or {
                "state": "continue",
                "explicit_wait": 0,
                "wait_since": None,
                "ended_at": None,
                "summary": "",
                "open_questions": [],
                "next_opening": "",
                "last_owner_kind": "",
                "last_bot_kind": "",
            }
            cur.update(fields)
            _require(cur["state"], m.CLOSURE_STATES, "closure state")
            if not isinstance(cur["open_questions"], (list, tuple)):
                raise ValueError("open_questions must be a list")
            values = [
                cur["state"],
                _flag(cur["explicit_wait"]),
                cur["wait_since"],
                cur["ended_at"],
                str(cur["summary"] or ""),
                _dumps(list(cur["open_questions"])),
                str(cur["next_opening"] or ""),
                str(cur["last_owner_kind"] or ""),
                str(cur["last_bot_kind"] or ""),
            ]
            db.execute(
                f"INSERT OR REPLACE INTO closures(scope_key,{','.join(_CLOSURE_FIELDS)},updated_at)"
                f" VALUES (?,{','.join('?' * len(_CLOSURE_FIELDS))},?)",
                [scope_key] + values + [self._now()],
            )
        return self.get_closure(scope_key)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # windows
    # ------------------------------------------------------------------

    def save_window(self, scope_key: str, payload: List[Dict[str, Any]]) -> None:
        if not isinstance(payload, list):
            raise ValueError("window payload must be a list")
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO windows(scope_key,payload,updated_at) VALUES (?,?,?)",
                (scope_key, _dumps(payload), self._now()),
            )

    def load_windows(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            rows = self._conn().execute("SELECT scope_key, payload FROM windows").fetchall()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            payload = _loads(r["payload"], [])
            if isinstance(payload, list):
                out[r["scope_key"]] = payload
        return out

    def delete_window(self, scope_key: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM windows WHERE scope_key=?", (scope_key,))

    # ------------------------------------------------------------------
    # private context
    # ------------------------------------------------------------------

    def get_private_context(self, audience: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM private_context WHERE audience=? ORDER BY section", (audience,)
            ).fetchall()
        return [dict(r) for r in rows]

    def put_private_context(self, section: str, content: str, source: str, audience: str) -> None:
        if not section or not audience:
            raise ValueError("section and audience are required")
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO private_context(section,content,source,updated_at,audience)"
                " VALUES (?,?,?,?,?)",
                (section, str(content or ""), str(source or ""), self._now(), audience),
            )

    # ------------------------------------------------------------------
    # observed posts
    # ------------------------------------------------------------------

    def add_observed_post(self, **fields: Any) -> bool:
        """保存观察到的频道/群消息；(chat_id,msg_id) 重复返回 False"""
        unknown = set(fields) - set(_POST_FIELDS)
        if unknown:
            raise ValueError(f"add_observed_post unknown fields: {sorted(unknown)}")
        for k in ("chat_id", "msg_id", "posted_at", "author_kind", "via"):
            if fields.get(k) in (None, ""):
                raise ValueError(f"add_observed_post missing {k}")
        _require(fields["author_kind"], m.AUTHOR_KINDS, "author_kind")
        with self._tx() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO observed_posts(chat_id,msg_id,posted_at,sender_id,author_kind,"
                "is_forward,fwd_from,text,via,ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    int(fields["chat_id"]),
                    int(fields["msg_id"]),
                    float(fields["posted_at"]),
                    str(fields.get("sender_id") or ""),
                    fields["author_kind"],
                    _flag(fields.get("is_forward")),
                    str(fields.get("fwd_from") or ""),
                    str(fields.get("text") or ""),
                    str(fields["via"]),
                    self._now(),
                ),
            )
            return cur.rowcount == 1

    def posts_since(
        self, chat_ids: Sequence[int], since_ts: float, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """posted_at > since_ts 的帖子（旧→新）"""
        ids = _int_list(chat_ids)
        if not ids:
            return []
        with self._lock:
            rows = self._conn().execute(
                f"SELECT * FROM observed_posts WHERE chat_id IN ({','.join('?' * len(ids))})"
                " AND posted_at>? ORDER BY posted_at, chat_id, msg_id LIMIT ?",
                ids + [float(since_ts), max(0, int(limit))],
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # cursors / counters
    # ------------------------------------------------------------------

    def get_cursor(self, name: str) -> Optional[str]:
        with self._lock:
            row = self._conn().execute("SELECT value FROM cursors WHERE name=?", (name,)).fetchone()
        return None if row is None else str(row[0])

    def set_cursor(self, name: str, value: Any) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO cursors(name,value,updated_at) VALUES (?,?,?)",
                (name, str(value), self._now()),
            )

    def bump_counter(self, name: str, n: int = 1, day: Optional[str] = None) -> int:
        """计数器 +n；day 缺省为 Asia/Shanghai 当天；返回新值"""
        day = day or m.local_day(self._now())
        with self._tx() as db:
            db.execute(
                "INSERT INTO counters(day,name,value) VALUES (?,?,?)"
                " ON CONFLICT(day,name) DO UPDATE SET value=value+excluded.value",
                (day, name, int(n)),
            )
            return int(
                db.execute("SELECT value FROM counters WHERE day=? AND name=?", (day, name)).fetchone()[0]
            )

    def get_counters(self, day: Optional[str] = None) -> Dict[str, int]:
        day = day or m.local_day(self._now())
        with self._lock:
            rows = self._conn().execute(
                "SELECT name, value FROM counters WHERE day=? ORDER BY name", (day,)
            ).fetchall()
        return {r["name"]: int(r["value"]) for r in rows}
