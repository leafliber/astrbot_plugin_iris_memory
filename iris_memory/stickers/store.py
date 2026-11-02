"""
Iris Chat Memory - 贴纸理解缓存（stickers_desc.db）

收到的 Telegram 贴纸按 ``doc_id`` 缓存一句文字描述，命中后不再调用视觉模型。
与表情包库（``library.py`` / stickers_library.db）分文件存放，互不迁移。

- 表 ``sticker_desc``：doc_id → 描述；``hits`` 仅用于程序内部统计，
  不进入任何 prompt（DESIGN §12）。
- 表 ``describe_quota``：按自然日（UTC+8）记录未命中缓存而实际调用模型的次数，
  用于每日上限。

同步 sqlite3（check_same_thread=False + RLock），WAL、busy_timeout、
PRAGMA user_version 迁移。调用方在事件循环中应通过 ``asyncio.to_thread`` 调用。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

SCHEMA_VERSION = 1
DB_DIRNAME = "stickers"
DB_FILENAME = "stickers_desc.db"

KINDS = ("static", "tgs", "video")

# 中国无夏令时，固定 UTC+8 即可，不依赖 tzdata。
_DAY_TZ = timezone(timedelta(hours=8))

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS sticker_desc(
  doc_id INTEGER PRIMARY KEY,
  set_id INTEGER,
  set_short_name TEXT,
  emoji TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'static' CHECK(kind IN ('static','tgs','video')),
  description TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sticker_desc_set ON sticker_desc(set_id);
CREATE TABLE IF NOT EXISTS describe_quota(
  day TEXT PRIMARY KEY,
  misses INTEGER NOT NULL DEFAULT 0
)
"""

# user_version=N 的库执行 _MIGRATIONS[N] 后升级到 N+1。只增不删。
_MIGRATIONS: Dict[int, str] = {0: _SCHEMA_V1}


def day_key(ts: float) -> str:
    """把时间戳换算成 UTC+8 的自然日 ``YYYY-MM-DD``。"""
    return datetime.fromtimestamp(float(ts), _DAY_TZ).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class StickerDescription:
    doc_id: int
    set_id: Optional[int]
    set_short_name: Optional[str]
    emoji: str
    kind: str
    description: str
    model: str
    created_at: float
    hits: int


class StickerStore:
    """贴纸描述缓存 + 每日未命中计数。"""

    def __init__(
        self,
        db_path: "str | Path",
        clock: Optional[Callable[[], float]] = None,
        *,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self._clock = clock or time.time
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None
        self._migrate()

    @classmethod
    def for_data_dir(
        cls, data_dir: "str | Path", clock: Optional[Callable[[], float]] = None
    ) -> "StickerStore":
        """``<data_dir>/stickers/stickers_desc.db``。"""
        return cls(Path(data_dir) / DB_DIRNAME / DB_FILENAME, clock)

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

    # -- 描述缓存 -------------------------------------------------------------

    def get(self, doc_id: int, *, count_hit: bool = True) -> Optional[StickerDescription]:
        with self._lock:
            db = self._conn()
            row = db.execute(
                "SELECT * FROM sticker_desc WHERE doc_id=?", (int(doc_id),)
            ).fetchone()
            if row is None:
                return None
            hits = int(row["hits"])
            if count_hit:
                db.execute(
                    "UPDATE sticker_desc SET hits=hits+1 WHERE doc_id=?", (int(doc_id),)
                )
                hits += 1
            return StickerDescription(
                doc_id=int(row["doc_id"]),
                set_id=None if row["set_id"] is None else int(row["set_id"]),
                set_short_name=row["set_short_name"],
                emoji=row["emoji"] or "",
                kind=row["kind"],
                description=row["description"],
                model=row["model"] or "",
                created_at=float(row["created_at"]),
                hits=hits,
            )

    def put(
        self,
        doc_id: int,
        description: str,
        *,
        set_id: Optional[int] = None,
        set_short_name: Optional[str] = None,
        emoji: str = "",
        kind: str = "static",
        model: str = "",
        created_at: Optional[float] = None,
    ) -> None:
        if kind not in KINDS:
            kind = "static"
        ts = float(self._clock() if created_at is None else created_at)
        with self._lock:
            self._conn().execute(
                """
                INSERT INTO sticker_desc(doc_id, set_id, set_short_name, emoji, kind,
                                         description, model, created_at, hits)
                VALUES(?,?,?,?,?,?,?,?,0)
                ON CONFLICT(doc_id) DO UPDATE SET
                  set_id=excluded.set_id,
                  set_short_name=excluded.set_short_name,
                  emoji=excluded.emoji,
                  kind=excluded.kind,
                  description=excluded.description,
                  model=excluded.model,
                  created_at=excluded.created_at
                """,
                (
                    int(doc_id),
                    None if set_id is None else int(set_id),
                    set_short_name or None,
                    emoji or "",
                    kind,
                    description,
                    model or "",
                    ts,
                ),
            )

    def delete(self, doc_id: int) -> bool:
        with self._lock:
            cur = self._conn().execute(
                "DELETE FROM sticker_desc WHERE doc_id=?", (int(doc_id),)
            )
            return cur.rowcount > 0

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn().execute("SELECT COUNT(*) FROM sticker_desc").fetchone()[0]
            )

    # -- 每日未命中额度 -------------------------------------------------------

    def misses_on(self, day: str) -> int:
        with self._lock:
            row = self._conn().execute(
                "SELECT misses FROM describe_quota WHERE day=?", (day,)
            ).fetchone()
            return int(row["misses"]) if row else 0

    def try_consume_miss(self, cap: int, *, now: Optional[float] = None) -> bool:
        """当日计数 < cap 时 +1 并返回 True；否则返回 False。cap<=0 视为关闭。"""
        cap = int(cap)
        if cap <= 0:
            return False
        day = day_key(self._clock() if now is None else now)
        with self._lock:
            db = self._conn()
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT misses FROM describe_quota WHERE day=?", (day,)
                ).fetchone()
                used = int(row["misses"]) if row else 0
                if used >= cap:
                    db.execute("COMMIT")
                    return False
                db.execute(
                    "INSERT INTO describe_quota(day, misses) VALUES(?,1) "
                    "ON CONFLICT(day) DO UPDATE SET misses=misses+1",
                    (day,),
                )
                # 只保留最近一段时间的计数行
                db.execute(
                    "DELETE FROM describe_quota WHERE day < ?",
                    (day_key((self._clock() if now is None else now) - 30 * 86400),),
                )
                db.execute("COMMIT")
                return True
            except Exception:
                db.execute("ROLLBACK")
                raise
