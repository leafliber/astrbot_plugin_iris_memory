"""L1 总结结果的持久化 Outbox。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .models import ContextMessage


@dataclass
class SummaryOutboxJob:
    job_id: str
    queue_key: str
    group_id: str
    summary: str
    messages: list[ContextMessage]
    attempt_count: int
    next_attempt_at: float
    last_error: str
    l2_done: bool
    profile_done: bool


class SummaryOutbox:
    """SQLite Outbox；总结结果落盘后 L1 才允许 rotate。"""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_outbox (
                    job_id TEXT PRIMARY KEY,
                    queue_key TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                    ,l2_done INTEGER NOT NULL DEFAULT 0
                    ,profile_done INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_summary_outbox_due "
                "ON summary_outbox(status, next_attempt_at, created_at)"
            )
            columns = {
                row[1]
                for row in self._db.execute(
                    "PRAGMA table_info(summary_outbox)"
                ).fetchall()
            }
            if "l2_done" not in columns:
                self._db.execute(
                    "ALTER TABLE summary_outbox ADD COLUMN l2_done INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            if "profile_done" not in columns:
                self._db.execute(
                    "ALTER TABLE summary_outbox ADD COLUMN profile_done INTEGER "
                    "NOT NULL DEFAULT 0"
                )
            self._db.commit()

    def enqueue(
        self,
        *,
        queue_key: str,
        group_id: str,
        summary: str,
        messages: list[ContextMessage],
    ) -> str:
        job_id = f"summary_{uuid.uuid4().hex}"
        now = time.time()
        payload = json.dumps(
            [message.to_dict() for message in messages],
            ensure_ascii=False,
            default=str,
        )
        with self._lock:
            self._db.execute(
                "INSERT INTO summary_outbox "
                "(job_id, queue_key, group_id, summary, messages_json, "
                "status, attempt_count, next_attempt_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)",
                (job_id, queue_key, group_id, summary, payload, now, now),
            )
            self._db.commit()
        return job_id

    @staticmethod
    def _from_row(row: sqlite3.Row) -> SummaryOutboxJob:
        return SummaryOutboxJob(
            job_id=row["job_id"],
            queue_key=row["queue_key"],
            group_id=row["group_id"],
            summary=row["summary"],
            messages=[
                ContextMessage.from_dict(item)
                for item in json.loads(row["messages_json"])
            ],
            attempt_count=int(row["attempt_count"]),
            next_attempt_at=float(row["next_attempt_at"]),
            last_error=row["last_error"],
            l2_done=bool(row["l2_done"]),
            profile_done=bool(row["profile_done"]),
        )

    def get(self, job_id: str) -> SummaryOutboxJob | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM summary_outbox WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def list_due(self, *, limit: int = 100) -> list[SummaryOutboxJob]:
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM summary_outbox "
                "WHERE status IN ('pending', 'retry_wait') AND next_attempt_at <= ? "
                "ORDER BY created_at LIMIT ?",
                (now, max(1, limit)),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def complete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM summary_outbox WHERE job_id = ?", (job_id,)
            )
            self._db.commit()

    def mark_stage_done(self, job_id: str, stage: str) -> None:
        if stage not in {"l2", "profile"}:
            raise ValueError(f"未知 Outbox 阶段：{stage}")
        column = "l2_done" if stage == "l2" else "profile_done"
        with self._lock:
            self._db.execute(
                f"UPDATE summary_outbox SET {column}=1, updated_at=? WHERE job_id=?",
                (time.time(), job_id),
            )
            self._db.commit()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            row = self._db.execute(
                "SELECT attempt_count FROM summary_outbox WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return
            attempt = int(row[0]) + 1
            intervals = (300, 1800, 7200, 43200)
            delay = intervals[min(attempt - 1, len(intervals) - 1)]
            self._db.execute(
                "UPDATE summary_outbox SET status='retry_wait', attempt_count=?, "
                "next_attempt_at=?, last_error=?, updated_at=? WHERE job_id=?",
                (attempt, time.time() + delay, error[:1000], time.time(), job_id),
            )
            self._db.commit()

    def count(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM summary_outbox").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._db.close()
