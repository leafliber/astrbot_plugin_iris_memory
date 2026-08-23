"""Dream 跨轮持久化游标。

状态以 cycle/persona/stage 三元组记录。一个 cycle 内已经完成的阶段不会因
预算耗尽或进程重启而重跑；全部 persona 和全局阶段完成后才推进到下一 cycle。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


class DreamCursorStore:
    """小型 SQLite 游标仓库；每个操作独立事务，可安全跨任务恢复。"""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self._db_path), timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def _connection(self):
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _init_schema(self) -> None:
        with self._lock, self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS dream_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dream_stage_cursor (
                    cycle INTEGER NOT NULL,
                    persona_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(cycle, persona_id, stage_name)
                );
                CREATE INDEX IF NOT EXISTS idx_dream_stage_persona
                    ON dream_stage_cursor(cycle, persona_id, completed);
                INSERT OR IGNORE INTO dream_meta(key,value) VALUES('cycle','1');
                INSERT OR IGNORE INTO dream_meta(key,value) VALUES('next_persona','');
                """
            )

    def get_cycle(self) -> int:
        with self._lock, self._connection() as db:
            row = db.execute(
                "SELECT value FROM dream_meta WHERE key='cycle'"
            ).fetchone()
            return max(1, int(row["value"] if row else 1))

    def get_next_persona(self) -> str:
        with self._lock, self._connection() as db:
            row = db.execute(
                "SELECT value FROM dream_meta WHERE key='next_persona'"
            ).fetchone()
            return str(row["value"] if row else "")

    def set_next_persona(self, persona_id: str) -> None:
        with self._lock, self._connection() as db:
            db.execute(
                "INSERT INTO dream_meta(key,value) VALUES('next_persona',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (persona_id,),
            )

    def completed_stages(self, cycle: int, persona_id: str) -> set[str]:
        with self._lock, self._connection() as db:
            rows = db.execute(
                "SELECT stage_name FROM dream_stage_cursor "
                "WHERE cycle=? AND persona_id=? AND completed=1",
                (cycle, persona_id),
            ).fetchall()
            return {str(row["stage_name"]) for row in rows}

    def mark_stage(
        self, cycle: int, persona_id: str, stage_name: str, *, completed: bool
    ) -> None:
        with self._lock, self._connection() as db:
            db.execute(
                "INSERT INTO dream_stage_cursor"
                "(cycle,persona_id,stage_name,completed,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(cycle,persona_id,stage_name) DO UPDATE SET "
                "completed=excluded.completed,updated_at=excluded.updated_at",
                (cycle, persona_id, stage_name, int(completed), time.time()),
            )

    def persona_complete(
        self, cycle: int, persona_id: str, stage_names: Iterable[str]
    ) -> bool:
        required = set(stage_names)
        return required.issubset(self.completed_stages(cycle, persona_id))

    def advance_cycle(self, cycle: int, next_persona: str = "") -> int:
        """CAS 推进 cycle；旧两轮以前的游标一并清理。"""
        with self._lock, self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM dream_meta WHERE key='cycle'"
            ).fetchone()
            current = max(1, int(row["value"] if row else 1))
            if current != cycle:
                db.rollback()
                return current
            next_cycle = current + 1
            db.execute(
                "UPDATE dream_meta SET value=? WHERE key='cycle'",
                (str(next_cycle),),
            )
            db.execute(
                "UPDATE dream_meta SET value=? WHERE key='next_persona'",
                (next_persona,),
            )
            db.execute(
                "DELETE FROM dream_stage_cursor WHERE cycle<?",
                (max(1, next_cycle - 1),),
            )
            db.commit()
            return next_cycle
