"""
反馈数据存储（SQLite）

管理场景权重、回复记录、反馈记录、每日统计和 LLM 限流数据。
所有表结构按照 PROACTIVE_REDESIGN.md 第五节定义。
"""

from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

from iris_memory.proactive.core.models import (
    ReplyFeedback,
    ReplyRecord,
    SceneWeight,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.feedback_store")

# topic_vector 序列化 / 反序列化（BLOB）
_FLOAT_FMT = "f"  # 4 bytes per float32


def _encode_vector(vec: Optional[List[float]]) -> Optional[bytes]:
    if vec is None:
        return None
    return struct.pack(f"<{len(vec)}{_FLOAT_FMT}", *vec)


def _decode_vector(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    n = len(blob) // struct.calcsize(_FLOAT_FMT)
    return list(struct.unpack(f"<{n}{_FLOAT_FMT}", blob))


class FeedbackStore:
    """SQLite 反馈数据存储

    管理以下表：
    - scene_weights: 场景权重
    - reply_records: 回复记录
    - feedback_records: 反馈记录
    - scene_weight_history: 权重变更历史
    - daily_stats: 每日统计
    - llm_quota: LLM 限流
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[Any] = None
        self._lock = asyncio.Lock()

    # ========== 生命周期 ==========

    async def initialize(self) -> None:
        """初始化数据库连接和表结构"""
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._create_tables()
        logger.debug(f"FeedbackStore initialized at {self._db_path}")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _create_tables(self) -> None:
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS scene_weights (
                scene_id TEXT PRIMARY KEY,
                success_rate REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reply_records (
                record_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                session_type TEXT,
                scene_ids TEXT,
                decision_type TEXT,
                urgency TEXT,
                reply_type TEXT,
                confidence REAL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_summary TEXT,
                topic_vector BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_reply_records_session_time
            ON reply_records(session_key, sent_at DESC);

            CREATE TABLE IF NOT EXISTS feedback_records (
                feedback_id TEXT PRIMARY KEY,
                record_id TEXT REFERENCES reply_records(record_id),
                user_replied BOOLEAN DEFAULT FALSE,
                reply_within_window BOOLEAN DEFAULT FALSE,
                engagement_score REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS scene_weight_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT,
                old_success_rate REAL,
                new_success_rate REAL,
                update_reason TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_scene_weight_history_scene
            ON scene_weight_history(scene_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_detections INTEGER DEFAULT 0,
                rule_hits INTEGER DEFAULT 0,
                vector_hits INTEGER DEFAULT 0,
                llm_hits INTEGER DEFAULT 0,
                replies_sent INTEGER DEFAULT 0,
                avg_engagement REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS llm_quota (
                session_key TEXT PRIMARY KEY,
                hour INTEGER,
                count INTEGER DEFAULT 0,
                updated_at TIMESTAMP
            );
        """)
        await self._conn.commit()

    # ========== scene_weights ==========

    async def get_scene_weight(self, scene_id: str) -> Optional[SceneWeight]:
        assert self._conn
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT * FROM scene_weights WHERE scene_id = ?", (scene_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return SceneWeight(
                scene_id=row["scene_id"],
                success_rate=row["success_rate"],
                usage_count=row["usage_count"],
                last_used=_parse_ts(row["last_used"]),
                updated_at=_parse_ts(row["updated_at"]),
            )

    async def get_scene_weights_batch(self, scene_ids: List[str]) -> Dict[str, SceneWeight]:
        """批量获取场景权重"""
        if not scene_ids:
            return {}
        assert self._conn
        placeholders = ",".join("?" for _ in scene_ids)
        async with self._lock:
            cursor = await self._conn.execute(
                f"SELECT * FROM scene_weights WHERE scene_id IN ({placeholders})",
                scene_ids,
            )
            rows = await cursor.fetchall()
        return {
            row["scene_id"]: SceneWeight(
                scene_id=row["scene_id"],
                success_rate=row["success_rate"],
                usage_count=row["usage_count"],
                last_used=_parse_ts(row["last_used"]),
                updated_at=_parse_ts(row["updated_at"]),
            )
            for row in rows
        }

    async def upsert_scene_weight(
        self,
        scene_id: str,
        success_rate: float,
        usage_count: int,
        last_used: Optional[datetime] = None,
    ) -> None:
        assert self._conn
        now = datetime.now().isoformat()
        lu = last_used.isoformat() if last_used else now
        async with self._lock:
            await self._conn.execute(
                """INSERT INTO scene_weights (scene_id, success_rate, usage_count, last_used, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(scene_id) DO UPDATE SET
                     success_rate=excluded.success_rate,
                     usage_count=excluded.usage_count,
                     last_used=excluded.last_used,
                     updated_at=excluded.updated_at""",
                (scene_id, success_rate, usage_count, lu, now),
            )
            await self._conn.commit()

    async def batch_update_usage_counts(self, updates: List[Tuple[str, int]]) -> None:
        """批量更新 usage_count"""
        if not updates:
            return
        assert self._conn
        now = datetime.now().isoformat()
        async with self._lock:
            await self._conn.executemany(
                """INSERT INTO scene_weights (scene_id, usage_count, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(scene_id) DO UPDATE SET
                     usage_count=excluded.usage_count,
                     updated_at=excluded.updated_at""",
                [(sid, cnt, now) for sid, cnt in updates],
            )
            await self._conn.commit()

    async def record_weight_change(
        self,
        scene_id: str,
        old_rate: float,
        new_rate: float,
        reason: str,
    ) -> None:
        assert self._conn
        async with self._lock:
            await self._conn.execute(
                """INSERT INTO scene_weight_history
                   (scene_id, old_success_rate, new_success_rate, update_reason)
                   VALUES (?, ?, ?, ?)""",
                (scene_id, old_rate, new_rate, reason),
            )
            await self._conn.commit()

    # ========== reply_records ==========

    async def record_reply(
        self,
        record: ReplyRecord,
    ) -> None:
        assert self._conn
        async with self._lock:
            await self._conn.execute(
                """INSERT OR REPLACE INTO reply_records
                   (record_id, session_key, session_type, scene_ids, decision_type,
                    urgency, reply_type, confidence, sent_at, content_summary, topic_vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.session_key,
                    record.session_type,
                    json.dumps(record.scene_ids),
                    record.decision_type,
                    record.urgency,
                    record.reply_type,
                    record.confidence,
                    record.sent_at.isoformat(),
                    record.content_summary,
                    _encode_vector(record.topic_vector),
                ),
            )
            await self._conn.commit()

    async def get_last_reply(self, session_key: str) -> Optional[ReplyRecord]:
        assert self._conn
        async with self._lock:
            cursor = await self._conn.execute(
                """SELECT * FROM reply_records
                   WHERE session_key = ?
                   ORDER BY sent_at DESC LIMIT 1""",
                (session_key,),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_reply_record(row)

    async def get_recent_replies(
        self, session_key: str, limit: int = 10
    ) -> List[ReplyRecord]:
        assert self._conn
        async with self._lock:
            cursor = await self._conn.execute(
                """SELECT * FROM reply_records
                   WHERE session_key = ?
                   ORDER BY sent_at DESC LIMIT ?""",
                (session_key, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_reply_record(r) for r in rows]

    async def get_recent_followup_counts(
        self, window_seconds: int = 120
    ) -> List[Dict[str, Any]]:
        """获取最近 window_seconds 秒内每个 session 的 followup 回复数"""
        assert self._conn
        cutoff = (datetime.now() - timedelta(seconds=window_seconds)).isoformat()
        async with self._lock:
            cursor = await self._conn.execute(
                """SELECT session_key, COUNT(*) as followup_count,
                          MAX(sent_at) as last_updated
                   FROM reply_records
                   WHERE reply_type = 'followup' AND sent_at >= ?
                   GROUP BY session_key""",
                (cutoff,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "session_key": r["session_key"],
                "followup_count": r["followup_count"],
                "last_updated": _parse_ts(r["last_updated"]),
            }
            for r in rows
        ]

    # ========== feedback_records ==========

    async def record_feedback(self, feedback: ReplyFeedback) -> None:
        assert self._conn
        async with self._lock:
            await self._conn.execute(
                """INSERT OR REPLACE INTO feedback_records
                   (feedback_id, record_id, user_replied, reply_within_window,
                    engagement_score, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    feedback.feedback_id,
                    feedback.record_id,
                    feedback.user_replied,
                    feedback.reply_within_window,
                    feedback.engagement_score,
                    feedback.recorded_at.isoformat(),
                ),
            )
            await self._conn.commit()

    # ========== daily_stats ==========

    async def increment_daily_stats(
        self,
        detection_type: str = "",
        reply_sent: bool = False,
    ) -> None:
        assert self._conn
        today = datetime.now().strftime("%Y-%m-%d")
        col_map = {
            "rule": "rule_hits",
            "vector": "vector_hits",
            "llm": "llm_hits",
        }
        async with self._lock:
            await self._conn.execute(
                """INSERT INTO daily_stats (date, total_detections)
                   VALUES (?, 1)
                   ON CONFLICT(date) DO UPDATE SET
                     total_detections = total_detections + 1""",
                (today,),
            )
            if detection_type in col_map:
                col = col_map[detection_type]
                await self._conn.execute(
                    f"""UPDATE daily_stats SET {col} = {col} + 1 WHERE date = ?""",
                    (today,),
                )
            if reply_sent:
                await self._conn.execute(
                    """UPDATE daily_stats SET replies_sent = replies_sent + 1
                       WHERE date = ?""",
                    (today,),
                )
            await self._conn.commit()

    async def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        assert self._conn
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT * FROM daily_stats WHERE date >= ? ORDER BY date DESC",
                (cutoff,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ========== llm_quota ==========

    async def get_llm_quotas_for_hour(self, hour: int) -> List[Dict[str, Any]]:
        assert self._conn
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT session_key, count FROM llm_quota WHERE hour = ?",
                (hour,),
            )
            rows = await cursor.fetchall()
        return [
            {"session_key": r["session_key"], "count": r["count"]}
            for r in rows
        ]

    async def update_llm_quota(
        self, session_key: str, hour: int, count: int
    ) -> None:
        assert self._conn
        now = datetime.now().isoformat()
        async with self._lock:
            await self._conn.execute(
                """INSERT INTO llm_quota (session_key, hour, count, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_key) DO UPDATE SET
                     hour=excluded.hour,
                     count=excluded.count,
                     updated_at=excluded.updated_at""",
                (session_key, hour, count, now),
            )
            await self._conn.commit()

    # ========== 数据清理 ==========

    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """清理过期数据"""
        assert self._conn
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        total_deleted = 0
        async with self._lock:
            for table, col in [
                ("reply_records", "sent_at"),
                ("feedback_records", "recorded_at"),
                ("scene_weight_history", "updated_at"),
            ]:
                cursor = await self._conn.execute(
                    f"DELETE FROM {table} WHERE {col} < ?", (cutoff,)
                )
                total_deleted += cursor.rowcount
            await self._conn.commit()
        logger.debug(f"Cleaned {total_deleted} old records (retention={retention_days}d)")
        return total_deleted


# ========== 辅助函数 ==========


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _row_to_reply_record(row: Any) -> ReplyRecord:
    scene_ids_raw = row["scene_ids"]
    try:
        scene_ids = json.loads(scene_ids_raw) if scene_ids_raw else []
    except (json.JSONDecodeError, TypeError):
        scene_ids = []
    return ReplyRecord(
        record_id=row["record_id"],
        session_key=row["session_key"],
        session_type=row["session_type"] or "",
        scene_ids=scene_ids,
        decision_type=row["decision_type"] or "",
        urgency=row["urgency"] or "",
        reply_type=row["reply_type"] or "",
        confidence=row["confidence"] or 0.0,
        sent_at=_parse_ts(row["sent_at"]) or datetime.now(),
        content_summary=row["content_summary"] or "",
        topic_vector=_decode_vector(row["topic_vector"]),
    )
