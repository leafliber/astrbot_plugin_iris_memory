"""
Iris Chat Memory - 主人限定的表情包库（stickers_library.db）

- 主人在私聊里回复一张 Telegram 贴纸并发 ``/sticker add <标签> [说明]``
  把它登记进库；bot 只能从库里挑表情包发送，不能自己找图。
- 库存的是 Telegram 文档引用（doc_id / access_hash / file_reference）与
  来源消息坐标；file_reference 过期时回源消息刷新后重试一次。
- 生成侧：``prompt_block`` 只给标签与文字说明（DESIGN §12：不出现任何
  数字/计数），模型可在回复最末尾附 ``[表情:标签]``；``parse_marker``
  剥离全部标记并返回第一个允许的标签。
- 限流：同一会话上一张 bot 表情包距今不足 COOLDOWN_SECONDS 则不发；
  除此之外不设任何频率规则（人格不应有固定的表情频率）。
- 隐私：scope='private' 只在主人私聊可用；scope='all' 在开启群聊表情时
  也可用于群；频道与静默会话永远不可用。

同步 sqlite3（check_same_thread=False + RLock），WAL、busy_timeout、
PRAGMA user_version 迁移。async ``send`` 内部的数据库写入走 to_thread。
日志只记录标签 / id，不记录消息正文。
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # 运行环境（AstrBot 容器）
    from astrbot.api import logger
except Exception:  # pragma: no cover - 独立导入时退化
    import logging

    logger = logging.getLogger("iris.stickers")

SCHEMA_VERSION = 1
DB_FILENAME = "stickers_library.db"

LABEL_MAX_LEN = 12
NOTE_MAX_LEN = 40
COOLDOWN_SECONDS = 120.0

SCOPE_PRIVATE = "private"
SCOPE_ALL = "all"
SCOPES = (SCOPE_PRIVATE, SCOPE_ALL)

SEND_SENT = "sent"
SEND_FAILED = "failed"

# 频道永不作为发送目标（DESIGN §2）；调用方可再追加 silent_sessions。
DEFAULT_BLOCKED_SCOPES: frozenset = frozenset()  # 部署方按需填写

_LABEL_RE = re.compile(r"^[0-9a-z㐀-䶿一-鿿]+$")

# 标记：[表情:x] / 【表情：x】 / ［表情：x］ / [表情 : x ] / [表情]（无标签）
_OPEN = r"[\[【［〔]"
_CLOSE = r"[\]】］〕]"
_MARKER_RE = re.compile(
    _OPEN + r"\s*表情(?:包)?\s*(?:[:：]\s*([^\]】］〕\[【［〔\n]{0,40}?))?\s*" + _CLOSE
)
# 回复末尾未闭合的残缺标记：…[表情:x  /  …【表情：x
_TRAILING_RE = re.compile(_OPEN + r"\s*表情(?:包)?\s*(?:[:：][^\]】］〕\n]{0,40})?\s*$")

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS library(
  label TEXT PRIMARY KEY,
  note TEXT NOT NULL DEFAULT '',
  doc_id INTEGER NOT NULL,
  access_hash INTEGER NOT NULL,
  file_reference BLOB NOT NULL DEFAULT x'',
  source_chat_id INTEGER,
  source_msg_id INTEGER,
  emoji TEXT NOT NULL DEFAULT '',
  scope TEXT NOT NULL DEFAULT 'private' CHECK(scope IN ('private','all')),
  enabled INTEGER NOT NULL DEFAULT 1,
  added_by TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  use_count INTEGER NOT NULL DEFAULT 0,
  last_used_at REAL
);
CREATE TABLE IF NOT EXISTS sends(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope_key TEXT NOT NULL,
  label TEXT NOT NULL,
  tg_msg_id INTEGER,
  status TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sends_scope_time ON sends(scope_key, status, created_at);
"""

# user_version=N 的库执行 _MIGRATIONS[N] 后升级到 N+1。只增不删。
_MIGRATIONS: Dict[int, str] = {0: _SCHEMA_V1}


class StickerError(Exception):
    """表情包库的可预期错误（标签非法、不是贴纸、未找到等），消息可直接回给主人。"""


class StickerSendError(Exception):
    """发送失败（含 file_reference 刷新后仍失败）。"""


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def normalize_label(label: Any) -> str:
    """NFKC + 去空白 + ASCII 小写；不校验。"""
    s = unicodedata.normalize("NFKC", str(label or ""))
    s = re.sub(r"\s+", "", s).lower()
    return s


def validate_label(label: Any) -> str:
    """返回规范化标签；非法则抛 StickerError。"""
    s = normalize_label(label)
    if not s:
        raise StickerError("标签不能为空")
    if len(s) > LABEL_MAX_LEN:
        raise StickerError(f"标签太长了，最多 {LABEL_MAX_LEN} 个字")
    if not _LABEL_RE.match(s):
        raise StickerError("标签只能用中文、字母或数字")
    return s


def clean_note(note: Any) -> str:
    s = unicodedata.normalize("NFKC", str(note or ""))
    s = re.sub(r"[\[\]【】［］〔〕]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > NOTE_MAX_LEN:
        raise StickerError(f"说明太长了，最多 {NOTE_MAX_LEN} 个字")
    return s


def _sticker_doc(raw_message: Any) -> Any:
    """Telethon Message.sticker（Document 且带 DocumentAttributeSticker）或 None。"""
    if raw_message is None:
        return None
    try:
        doc = getattr(raw_message, "sticker", None)
    except Exception:
        doc = None
    if doc is None:
        return None
    if getattr(doc, "id", None) is None or getattr(doc, "access_hash", None) is None:
        return None
    return doc


def _sticker_emoji(doc: Any) -> str:
    for attr in getattr(doc, "attributes", None) or []:
        alt = getattr(attr, "alt", None)
        if type(attr).__name__ == "DocumentAttributeSticker" and alt:
            return str(alt)[:16]
    return ""


def _message_ids(result: Any) -> List[int]:
    items = result if isinstance(result, (list, tuple)) else [result]
    ids: List[int] = []
    for it in items:
        mid = getattr(it, "id", None)
        if isinstance(mid, int) and mid > 0:
            ids.append(mid)
    return ids


def _is_file_reference_error(exc: BaseException) -> bool:
    try:
        from telethon import errors as tg_errors

        ref_types = tuple(
            t
            for t in (
                getattr(tg_errors, "FileReferenceExpiredError", None),
                getattr(tg_errors, "FileReferenceInvalidError", None),
                getattr(tg_errors, "FileReferenceEmptyError", None),
            )
            if t is not None
        )
        if ref_types and isinstance(exc, ref_types):
            return True
    except Exception:  # pragma: no cover
        pass
    return "FILE_REFERENCE" in str(exc).upper()


def _base_scope(scope_key: str) -> str:
    """去掉 forum 话题后缀 '#thread'。"""
    return str(scope_key or "").split("#", 1)[0]


def prompt_block(entries: Sequence[Dict[str, Any]]) -> str:
    """给生成 prompt 的中文说明。只含标签与文字说明，不含任何数字/计数。"""
    lines: List[str] = []
    seen = set()
    for e in entries or []:
        label = str(e.get("label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        note = str(e.get("note") or "").strip()
        lines.append(f"- {label}：{note}" if note else f"- {label}")
    if not lines:
        return ""
    return "\n".join(
        [
            "【可用表情包】",
            "主人给你准备了下面这些表情包（标签：什么时候适合用）：",
            *lines,
            "用法：如果某一句回复里正好自然地适合配一个表情包，可以在整条回复的最末尾"
            "附上一个标记 [表情:标签]，系统会把它换成对应的表情包发出去，标记本身不会被看到。",
            "注意：只能用上面列出的标签，不要自己编新标签；不必每次都用，大多数时候不用也很好；"
            "一条回复里只放一个，并且放在最后；表情包不能代替回答，该说的话照常说完。",
        ]
    )


_HWS = " \t\u3000"


def _marker_spans(text: str) -> List[Tuple[int, int, Optional[str]]]:
    """所有标记的 (start, end, raw_label)；末尾残缺标记只在最后一个完整标记之后找。"""
    spans: List[Tuple[int, int, Optional[str]]] = [
        (mt.start(), mt.end(), mt.group(1)) for mt in _MARKER_RE.finditer(text)
    ]
    offset = spans[-1][1] if spans else 0
    tail = _TRAILING_RE.search(text, offset)
    if tail:
        mt2 = re.search(r"[:：](.*)$", tail.group(0))
        spans.append((tail.start(), tail.end(), mt2.group(1) if mt2 else None))
    return spans


def has_marker(text: str) -> bool:
    """文本里是否真有 [表情:…] 标记（含残缺变体）；只出现“表情”二字不算。"""
    return bool(text) and "表情" in text and bool(_marker_spans(text))


def parse_marker(text: str, allowed_labels: Iterable[str]) -> Tuple[str, Optional[str]]:
    """剥离所有 [表情:…] 标记（含全角/残缺变体），返回 (clean_text, 第一个允许的标签|None)。

    没有任何标记时原样返回（逐字节相同）；有标记时只收拾被删片段紧邻的空白，
    其他位置的缩进、连续空格、空行一律不动（代码块等不受影响）。
    """
    if not text:
        return text or "", None
    spans = _marker_spans(text)
    if not spans:
        return text, None
    allowed = {normalize_label(x) for x in (allowed_labels or []) if x}
    found: Optional[str] = None
    for _, _, raw_label in spans:
        if found is None and raw_label:
            cand = normalize_label(raw_label)
            if cand and cand in allowed:
                found = cand

    # 相邻标记之间只有空白时合并为一个被删片段
    merged: List[List[int]] = []
    for s0, e0, _ in spans:
        if merged and not text[merged[-1][1]:s0].strip():
            merged[-1][1] = e0
        else:
            merged.append([s0, e0])

    chunks: List[str] = []
    prev = 0
    for s0, e0 in merged:
        chunks.append(text[prev:s0])
        prev = e0
    chunks.append(text[prev:])

    # 结尾的标记：去掉它前面紧邻的空白
    if not chunks[-1].strip():
        chunks[-1] = ""
        chunks[-2] = chunks[-2].rstrip()
    # 开头的标记：去掉它后面紧邻的空白（只到第一行的缩进为止，保留缩进）
    if not chunks[0].strip():
        chunks = chunks[1:]
        chunks[0] = re.sub(r"^[ \t\u3000]*\n*", "", chunks[0]) if chunks[0].strip() else ""

    out = chunks[0]
    for right in chunks[1:]:
        left = out.rstrip(_HWS)
        r = right.lstrip(_HWS)
        had_space = len(left) != len(out) or len(r) != len(right)
        if not left or not r:
            out = left + r
        elif left.endswith("\n") and r.startswith("\n"):
            out = left + r[1:]  # 标记独占一行：去掉多出的那一行
        elif left.endswith("\n") or r.startswith("\n"):
            out = left + r
        else:
            out = left + (" " if had_space else "") + r
    return out, found


# ---------------------------------------------------------------------------
# 库
# ---------------------------------------------------------------------------


class StickerLibrary:
    def __init__(
        self,
        db_path: "str | Path",
        clock: Optional[Callable[[], float]] = None,
        *,
        busy_timeout_ms: int = 5000,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self._clock = clock or time.time
        self._busy_timeout_ms = int(busy_timeout_ms)
        self.cooldown_seconds = float(cooldown_seconds)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None
        self._migrate()

    # 纯函数也挂在实例上，方便调用方只持有 library 对象
    prompt_block = staticmethod(prompt_block)
    parse_marker = staticmethod(parse_marker)

    # -- 连接 / 迁移 ---------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        with self._lock:
            if self._db is None:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                db = sqlite3.connect(
                    str(self.db_path), check_same_thread=False, isolation_level=None
                )
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
                self._db = db
            return self._db

    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn().execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        with self._lock:
            db = self._conn()
            version = self.schema_version()
            while version < SCHEMA_VERSION:
                script = _MIGRATIONS[version]
                db.execute("BEGIN IMMEDIATE")
                try:
                    for stmt in [s.strip() for s in script.split(";") if s.strip()]:
                        db.execute(stmt)
                    db.execute(f"PRAGMA user_version={version + 1}")
                    db.execute("COMMIT")
                except Exception:
                    db.execute("ROLLBACK")
                    raise
                version += 1

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                finally:
                    self._db = None

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        fr = d.get("file_reference")
        d["file_reference"] = bytes(fr) if fr is not None else b""
        d["enabled"] = bool(d.get("enabled"))
        return d

    # -- 增删改查 ------------------------------------------------------------

    def get(self, label: str) -> Optional[Dict[str, Any]]:
        key = normalize_label(label)
        with self._lock:
            row = self._conn().execute("SELECT * FROM library WHERE label=?", (key,)).fetchone()
        return self._row(row)

    def add_from_message(
        self, label: str, note: str, raw_message: Any, added_by: Any
    ) -> Dict[str, Any]:
        """登记 raw_message（Telethon Message）里的贴纸；同标签覆盖（保留 scope/计数）。"""
        key = validate_label(label)
        note_s = clean_note(note)
        doc = _sticker_doc(raw_message)
        if doc is None:
            raise StickerError("这条消息不是贴纸，请回复一张贴纸再发 /sticker add")
        fr = getattr(doc, "file_reference", None) or b""
        chat_id = getattr(raw_message, "chat_id", None)
        msg_id = getattr(raw_message, "id", None)
        now = float(self._clock())
        with self._lock:
            db = self._conn()
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    """
                    INSERT INTO library(label, note, doc_id, access_hash, file_reference,
                        source_chat_id, source_msg_id, emoji, scope, enabled, added_by,
                        created_at, updated_at, use_count, last_used_at)
                    VALUES(?,?,?,?,?,?,?,?,'private',1,?,?,?,0,NULL)
                    ON CONFLICT(label) DO UPDATE SET
                        note=excluded.note, doc_id=excluded.doc_id,
                        access_hash=excluded.access_hash,
                        file_reference=excluded.file_reference,
                        source_chat_id=excluded.source_chat_id,
                        source_msg_id=excluded.source_msg_id,
                        emoji=excluded.emoji, enabled=1,
                        added_by=excluded.added_by, updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        note_s,
                        int(doc.id),
                        int(doc.access_hash),
                        sqlite3.Binary(bytes(fr)),
                        int(chat_id) if chat_id is not None else None,
                        int(msg_id) if msg_id is not None else None,
                        _sticker_emoji(doc),
                        str(added_by or ""),
                        now,
                        now,
                    ),
                )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        logger.info(f"[stickers] 登记表情包 label={key} doc={int(doc.id)}")
        entry = self.get(key)
        assert entry is not None
        return entry

    def remove(self, label: str) -> bool:
        key = normalize_label(label)
        with self._lock:
            cur = self._conn().execute("DELETE FROM library WHERE label=?", (key,))
        return cur.rowcount > 0

    def list(self, scope_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """全部（含停用）条目；给了 scope_key 时只返回该作用域可见 scope 的启用条目
        （私聊 'private:*' → private+all；其他 → all）。权限判定请用 allowed_for。"""
        sql = "SELECT * FROM library"
        params: Tuple[Any, ...] = ()
        if scope_key is not None:
            if str(scope_key).startswith("private:"):
                sql += " WHERE enabled=1"
            else:
                sql += " WHERE enabled=1 AND scope='all'"
        sql += " ORDER BY created_at, label"
        with self._lock:
            rows = self._conn().execute(sql, params).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def set_scope(self, label: str, scope: str) -> Dict[str, Any]:
        key = normalize_label(label)
        sc = str(scope or "").strip().lower()
        if sc not in SCOPES:
            raise StickerError("范围只能是 private 或 all")
        with self._lock:
            cur = self._conn().execute(
                "UPDATE library SET scope=?, updated_at=? WHERE label=?",
                (sc, float(self._clock()), key),
            )
        if cur.rowcount == 0:
            raise StickerError(f"没有叫「{key}」的表情包")
        entry = self.get(key)
        assert entry is not None
        return entry

    def set_enabled(self, label: str, enabled: bool) -> bool:
        key = normalize_label(label)
        with self._lock:
            cur = self._conn().execute(
                "UPDATE library SET enabled=?, updated_at=? WHERE label=?",
                (1 if enabled else 0, float(self._clock()), key),
            )
        return cur.rowcount > 0

    def allowed_for(
        self,
        scope_key: str,
        owner_ids: Iterable[Any],
        groups_enabled: bool,
        *,
        blocked_scopes: Iterable[Any] = (),
    ) -> List[Dict[str, Any]]:
        """该会话可用的表情包。

        - 主人私聊 ``private:<owner>`` → scope private + all；
        - 其他私聊与群 → 仅 scope='all'，且仅当 groups_enabled；
        - 频道（DEFAULT_BLOCKED_SCOPES）与 blocked_scopes（silent_sessions 等）→ 永远为空。
        """
        sk = str(scope_key or "").strip()
        if not sk:
            return []
        blocked = {str(x) for x in DEFAULT_BLOCKED_SCOPES} | {str(x) for x in blocked_scopes or ()}
        if sk in blocked or _base_scope(sk) in blocked:
            return []
        owners = {str(x) for x in owner_ids or ()}
        if sk.startswith("private:"):
            uid = sk.split(":", 1)[1]
            if uid in owners:
                return self.list(scope_key=sk)
        if not groups_enabled:
            return []
        with self._lock:
            rows = self._conn().execute(
                "SELECT * FROM library WHERE enabled=1 AND scope='all' ORDER BY created_at, label"
            ).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    # -- 发送记账 / 限流 ------------------------------------------------------

    def should_send(self, scope_key: str, now: Optional[float] = None) -> bool:
        """上一张 bot 表情包（status=sent）在此会话距今 < cooldown 则 False。无其他频率规则。"""
        t = float(self._clock() if now is None else now)
        with self._lock:
            row = self._conn().execute(
                "SELECT MAX(created_at) FROM sends WHERE scope_key=? AND status=?",
                (str(scope_key), SEND_SENT),
            ).fetchone()
        last = row[0] if row else None
        return last is None or (t - float(last)) >= self.cooldown_seconds

    def record_send(
        self,
        scope_key: str,
        label: str,
        tg_msg_ids: Sequence[int],
        status: str = SEND_SENT,
        now: Optional[float] = None,
    ) -> None:
        t = float(self._clock() if now is None else now)
        key = normalize_label(label)
        ids: List[Optional[int]] = [int(x) for x in tg_msg_ids] or [None]
        with self._lock:
            db = self._conn()
            db.execute("BEGIN IMMEDIATE")
            try:
                for mid in ids:
                    db.execute(
                        "INSERT INTO sends(scope_key, label, tg_msg_id, status, created_at)"
                        " VALUES(?,?,?,?,?)",
                        (str(scope_key), key, mid, status, t),
                    )
                if status == SEND_SENT:
                    db.execute(
                        "UPDATE library SET use_count=use_count+1, last_used_at=? WHERE label=?",
                        (t, key),
                    )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise

    def refresh_document(self, label: str, doc: Any) -> Dict[str, Any]:
        key = normalize_label(label)
        fr = getattr(doc, "file_reference", None) or b""
        with self._lock:
            self._conn().execute(
                "UPDATE library SET doc_id=?, access_hash=?, file_reference=?, updated_at=?"
                " WHERE label=?",
                (
                    int(doc.id),
                    int(doc.access_hash),
                    sqlite3.Binary(bytes(fr)),
                    float(self._clock()),
                    key,
                ),
            )
        entry = self.get(key)
        if entry is None:
            raise StickerError(f"没有叫「{key}」的表情包")
        return entry

    # -- 发送 ----------------------------------------------------------------

    async def send(
        self,
        client: Any,
        peer: Any,
        entry: Dict[str, Any],
        *,
        scope_key: Optional[str] = None,
        reply_to: Optional[int] = None,
    ) -> List[int]:
        """经 Telethon client 发送 entry 对应的贴纸，返回 Telegram message id 列表（发送证据）。

        file_reference 过期/无效 → 回源消息刷新库内引用后重试一次。
        失败抛 StickerSendError。给了 scope_key 时记账（sent / failed）。
        """
        from telethon.tl.types import InputDocument

        label = str(entry.get("label") or "")
        if client is None:
            raise StickerSendError("telethon client 不可用")

        def _input(e: Dict[str, Any]) -> Any:
            return InputDocument(
                id=int(e["doc_id"]),
                access_hash=int(e["access_hash"]),
                file_reference=bytes(e.get("file_reference") or b""),
            )

        async def _once(e: Dict[str, Any]) -> List[int]:
            kwargs: Dict[str, Any] = {"file": _input(e)}
            if reply_to:
                kwargs["reply_to"] = reply_to
            result = await client.send_file(peer, **kwargs)
            ids = _message_ids(result)
            if not ids:
                raise StickerSendError("Telegram 没有返回消息，无法确认已发送")
            return ids

        try:
            try:
                ids = await _once(entry)
            except Exception as exc:
                if not _is_file_reference_error(exc):
                    raise
                logger.info(f"[stickers] file_reference 失效，回源刷新 label={label}")
                fresh = await self._refetch(client, entry)
                ids = await _once(fresh)
        except Exception as exc:
            if scope_key is not None:
                try:
                    await asyncio.to_thread(self.record_send, scope_key, label, [], SEND_FAILED)
                except Exception:  # pragma: no cover
                    pass
            if isinstance(exc, StickerSendError):
                raise
            raise StickerSendError(f"表情包发送失败: {type(exc).__name__}") from exc

        if scope_key is not None:
            try:
                await asyncio.to_thread(self.record_send, scope_key, label, ids, SEND_SENT)
            except Exception as exc:  # 记账失败不影响已发送事实
                logger.warning(f"[stickers] 记账失败 label={label}: {type(exc).__name__}")
        return ids

    async def _refetch(self, client: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
        chat_id = entry.get("source_chat_id")
        msg_id = entry.get("source_msg_id")
        if chat_id is None or msg_id is None:
            raise StickerSendError("没有来源消息，无法刷新 file_reference")
        msg = await client.get_messages(int(chat_id), ids=int(msg_id))
        if isinstance(msg, (list, tuple)):
            msg = msg[0] if msg else None
        doc = _sticker_doc(msg)
        if doc is None:
            raise StickerSendError("来源消息已不存在或不再是贴纸，请重新登记")
        return await asyncio.to_thread(self.refresh_document, str(entry["label"]), doc)

    # -- 主人命令 ------------------------------------------------------------

    def handle_command(
        self,
        args: Sequence[str],
        replied_raw_message: Any,
        sender_id: Any,
        owner_ids: Iterable[Any],
        *,
        is_group: bool = False,
    ) -> str:
        """处理 ``/sticker <子命令>``（args 不含 'sticker' 本身），返回回复文本。"""
        owners = {str(x) for x in owner_ids or ()}
        if is_group:
            return "表情包只能在和主人的私聊里管理。"
        if str(sender_id) not in owners:
            return "只有主人可以管理表情包。"
        argv = [str(a) for a in (args or []) if str(a).strip()]
        sub = argv[0].lower() if argv else "help"
        try:
            if sub in ("add", "添加"):
                if len(argv) < 2:
                    return "用法：回复一张贴纸，发 /sticker add <标签> [说明]"
                if replied_raw_message is None:
                    return "请先回复一张贴纸，再发 /sticker add <标签> [说明]"
                label = argv[1]
                note = " ".join(argv[2:])
                existed = self.get(normalize_label(label)) is not None
                e = self.add_from_message(label, note, replied_raw_message, sender_id)
                verb = "已更新" if existed else "已添加"
                scope_txt = "仅私聊" if e["scope"] == SCOPE_PRIVATE else "私聊和群"
                return f"{verb}表情包「{e['label']}」（{scope_txt}）" + (
                    f"：{e['note']}" if e["note"] else ""
                )
            if sub in ("list", "ls", "列表"):
                rows = self.list()
                if not rows:
                    return "表情包库还是空的。回复一张贴纸发 /sticker add <标签> [说明] 来添加。"
                lines = ["表情包库："]
                for e in rows:
                    flag = "仅私聊" if e["scope"] == SCOPE_PRIVATE else "私聊和群"
                    if not e["enabled"]:
                        flag += "，已停用"
                    note = f" — {e['note']}" if e["note"] else ""
                    lines.append(f"· {e['label']}{e['emoji'] and ' ' + e['emoji']}（{flag}）{note}")
                return "\n".join(lines)
            if sub in ("del", "rm", "remove", "删除"):
                if len(argv) < 2:
                    return "用法：/sticker del <标签>"
                key = normalize_label(argv[1])
                return f"已删除「{key}」" if self.remove(key) else f"没有叫「{key}」的表情包"
            if sub in ("scope", "范围"):
                if len(argv) < 3:
                    return "用法：/sticker scope <标签> private|all"
                e = self.set_scope(argv[1], argv[2])
                scope_txt = "仅私聊" if e["scope"] == SCOPE_PRIVATE else "私聊和群"
                return f"「{e['label']}」现在可用于：{scope_txt}"
            return _HELP
        except StickerError as exc:
            return str(exc)


_HELP = (
    "表情包管理（仅主人私聊）：\n"
    "/sticker add <标签> [说明] — 回复一张贴纸来添加（同名覆盖）\n"
    "/sticker list — 查看全部\n"
    "/sticker del <标签> — 删除\n"
    "/sticker scope <标签> private|all — private 仅私聊，all 也可用于群\n"
    "标签：中文、字母或数字，最多 12 个字；说明最多 40 个字。"
)
