"""
Iris Chat Memory - 学习模块存储层

使用独立 SQLite 库（learning.db）持久化三类学习产物：
- expression_pattern：从对话对规则提取的"场景→表达"模式
- few_shot：user→bot 对话对样例（经 LLM 审查后注入）
- jargon_candidate / jargon_candidate_daily：自动暗语候选及滚动证据
- jargon：经过严格自动鉴别或手工录入的正式暗语词典

所有方法为同步方法，sqlite 操作很快，调用方在 async 侧直接调用即可；
写操作由组件级 asyncio.Lock 保证事件循环内串行，内部再以
threading.Lock 兜底（check_same_thread=False 连接）。
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger

logger = get_logger("learning.storage")

_SCHEMA_VERSION = 2

# JSON 备份格式版本。导出使用原始数据库字段，便于完整保留自动暗语漏斗数据。
LEARNING_EXPORT_VERSION = "1.0"

_EXPORT_COLUMNS = {
    "few_shot": (
        "id", "group_id", "user_id", "user_text", "bot_text", "message_id",
        "status", "created_at",
    ),
    "expression_pattern": (
        "id", "group_id", "scene", "expression", "source_pair_id", "hit_count",
        "status", "created_at", "last_hit_at",
    ),
    "jargon": (
        "id", "group_id", "term", "aliases_json", "meaning", "confidence",
        "status", "category", "evidence_count", "approved_at", "last_seen_at",
        "dormant_at", "created_at", "updated_at",
    ),
    "jargon_candidate": (
        "id", "group_id", "term", "state", "first_seen_at", "last_seen_at",
        "local_score", "evidence_count_at_review", "llm_attempts", "last_llm_at",
        "next_review_at", "review_token", "category", "canonical_term", "meaning",
        "confidence", "verdict_reason", "created_at", "updated_at",
    ),
    "jargon_candidate_daily": (
        "candidate_id", "day", "message_count", "user_stats_json",
        "left_neighbors_json", "right_neighbors_json", "support_hashes_json",
        "contexts_json",
    ),
    "jargon_group_daily": ("group_id", "day", "users_json"),
    "jargon_llm_usage": ("day", "call_count", "candidate_count", "last_call_at"),
    "jargon_reactivation": ("jargon_id", "user_id", "contribution_count", "last_at"),
}

# 表达模式 / few_shot 的统一状态
STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_DISABLED = "disabled"

# jargon 额外的正常状态（词条未达审查语义，用 active 表示生效中）
STATUS_ACTIVE = "active"
STATUS_DORMANT = "dormant"

# 允许 update_status 操作的表白名单，防止 SQL 拼接注入
_TABLES = ("expression_pattern", "few_shot", "jargon")

# 各表允许通过 update_row 修改的字段白名单（Web 管理用）
_UPDATABLE_FIELDS = {
    "expression_pattern": ("scene", "expression", "status"),
    "few_shot": ("user_text", "bot_text", "status"),
    "jargon": ("term", "aliases_json", "meaning", "confidence", "status", "category"),
}

# 各表合法的 status 取值（暗语候选状态独立存放，正式词典支持休眠）
_VALID_STATUSES = {
    "expression_pattern": (STATUS_PENDING, STATUS_APPROVED, STATUS_DISABLED),
    "few_shot": (STATUS_PENDING, STATUS_APPROVED, STATUS_DISABLED),
    "jargon": (STATUS_ACTIVE, STATUS_DORMANT, STATUS_DISABLED),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS expression_pattern (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL DEFAULT '',
    scene TEXT NOT NULL DEFAULT '',
    expression TEXT NOT NULL,
    source_pair_id INTEGER,
    hit_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending_review',
    created_at REAL,
    last_hit_at REAL
);

CREATE TABLE IF NOT EXISTS few_shot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    user_text TEXT NOT NULL,
    bot_text TEXT NOT NULL,
    message_id TEXT,
    status TEXT DEFAULT 'pending_review',
    created_at REAL
);

CREATE TABLE IF NOT EXISTS jargon (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL DEFAULT '',
    term TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    meaning TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT DEFAULT 'active',
    category TEXT NOT NULL DEFAULT 'manual',
    evidence_count INTEGER NOT NULL DEFAULT 0,
    approved_at REAL,
    last_seen_at REAL,
    dormant_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(group_id, term)
);

CREATE TABLE IF NOT EXISTS jargon_candidate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    term TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'collecting',
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    local_score REAL NOT NULL DEFAULT 0,
    evidence_count_at_review INTEGER NOT NULL DEFAULT 0,
    llm_attempts INTEGER NOT NULL DEFAULT 0,
    last_llm_at REAL,
    next_review_at REAL,
    review_token TEXT,
    category TEXT,
    canonical_term TEXT,
    meaning TEXT,
    confidence REAL,
    verdict_reason TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(group_id, term)
);

CREATE TABLE IF NOT EXISTS jargon_candidate_daily (
    candidate_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    user_stats_json TEXT NOT NULL DEFAULT '{}',
    left_neighbors_json TEXT NOT NULL DEFAULT '{}',
    right_neighbors_json TEXT NOT NULL DEFAULT '{}',
    support_hashes_json TEXT NOT NULL DEFAULT '[]',
    contexts_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(candidate_id, day),
    FOREIGN KEY(candidate_id) REFERENCES jargon_candidate(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jargon_group_daily (
    group_id TEXT NOT NULL,
    day TEXT NOT NULL,
    users_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(group_id, day)
);

CREATE TABLE IF NOT EXISTS jargon_llm_usage (
    day TEXT PRIMARY KEY,
    call_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    last_call_at REAL
);

CREATE TABLE IF NOT EXISTS jargon_reactivation (
    jargon_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    contribution_count INTEGER NOT NULL DEFAULT 0,
    last_at REAL NOT NULL,
    PRIMARY KEY(jargon_id, user_id),
    FOREIGN KEY(jargon_id) REFERENCES jargon(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pattern_group_status ON expression_pattern(group_id, status);
CREATE INDEX IF NOT EXISTS idx_fewshot_group_status ON few_shot(group_id, status);
CREATE INDEX IF NOT EXISTS idx_fewshot_user ON few_shot(user_id);
CREATE INDEX IF NOT EXISTS idx_jargon_group ON jargon(group_id);
CREATE INDEX IF NOT EXISTS idx_jargon_candidate_state ON jargon_candidate(state, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_jargon_candidate_group ON jargon_candidate(group_id);
CREATE INDEX IF NOT EXISTS idx_jargon_daily_day ON jargon_candidate_daily(day);
"""


class LearningStorage:
    """学习模块 SQLite 存储

    负责 learning.db 的建表与全部读写操作。
    非线程安全场景下由内部 threading.Lock 保护，
    async 侧的并发串行化由组件级 asyncio.Lock 负责。
    """

    def __init__(self, db_path: Path):
        """初始化存储

        Args:
            db_path: 数据库文件路径（父目录自动创建）
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        """创建 V2 表结构；旧暗语 schema 明确报错，不做隐式迁移。"""
        with self._lock:
            existing = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jargon'"
            ).fetchone()
            if existing:
                columns = {
                    row["name"] for row in self._db.execute("PRAGMA table_info(jargon)").fetchall()
                }
                required = {"aliases_json", "evidence_count", "approved_at", "updated_at"}
                if not required.issubset(columns):
                    raise RuntimeError(
                        "learning.db 暗语表仍是旧结构；本版本不提供兼容迁移，"
                        "请先自行处理并重建 learning.db"
                    )
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            self._db.commit()

    def close(self) -> None:
        """关闭数据库连接"""
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # few_shot 对话对
    # ------------------------------------------------------------------

    def insert_pair(
        self,
        group_id: str,
        user_id: str,
        user_text: str,
        bot_text: str,
        message_id: Optional[str] = None,
    ) -> int:
        """插入一条 user→bot 对话对（status=pending_review）

        Returns:
            新行 id
        """
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO few_shot (group_id, user_id, user_text, bot_text,"
                " message_id, status, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    group_id,
                    user_id,
                    user_text,
                    bot_text,
                    message_id,
                    STATUS_PENDING,
                    time.time(),
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def get_pending_pairs(self, limit: int) -> List[Dict[str, Any]]:
        """取待审查对话对（按创建时间升序）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM few_shot WHERE status=? ORDER BY created_at LIMIT ?",
                (STATUS_PENDING, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_approved_few_shots(
        self, group_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """取本群已通过的对话样例（最新优先）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM few_shot WHERE group_id=? AND status=?"
                " ORDER BY created_at DESC LIMIT ?",
                (group_id, STATUS_APPROVED, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # expression_pattern 表达模式
    # ------------------------------------------------------------------

    def insert_pattern(
        self,
        group_id: str,
        scene: str,
        expression: str,
        source_pair_id: Optional[int] = None,
    ) -> int:
        """插入一条表达模式候选（status=pending_review）

        Returns:
            新行 id
        """
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO expression_pattern"
                " (group_id, scene, expression, source_pair_id, hit_count,"
                " status, created_at, last_hit_at) VALUES (?,?,?,?,0,?,?,NULL)",
                (group_id, scene, expression, source_pair_id, STATUS_PENDING, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def get_pending_patterns(self, limit: int) -> List[Dict[str, Any]]:
        """取待审查表达模式（按创建时间升序）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM expression_pattern WHERE status=?"
                " ORDER BY created_at LIMIT ?",
                (STATUS_PENDING, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_approved_patterns(
        self, group_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """取本群已通过的表达模式（按命中数降序）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM expression_pattern WHERE group_id=? AND status=?"
                " ORDER BY hit_count DESC, created_at DESC LIMIT ?",
                (group_id, STATUS_APPROVED, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_pattern_hit(self, ids: List[int]) -> None:
        """记录表达模式命中（hit_count+1，刷新 last_hit_at）"""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._db.execute(
                f"UPDATE expression_pattern SET hit_count=hit_count+1,"
                f" last_hit_at=? WHERE id IN ({placeholders})",
                (time.time(), *ids),
            )
            self._db.commit()

    def decay_patterns(self, decay_days: int, max_count: int) -> int:
        """衰减淘汰表达模式

        规则：
        1. hit_count=0 且创建时间超过 decay_days → 删除；
        2. last_hit_at 距今超过 decay_days 且命中数 ≤1 → 删除；
        3. 总量（含 approved）超过 max_count 时，按命中率（hit_count 升序）
           淘汰超出的部分。

        Returns:
            删除的总条数
        """
        now = time.time()
        cutoff = now - decay_days * 86400
        removed = 0
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM expression_pattern WHERE status=? AND"
                " ((hit_count=0 AND created_at<?) OR"
                "  (hit_count<=1 AND last_hit_at IS NOT NULL AND last_hit_at<?))",
                (STATUS_APPROVED, cutoff, cutoff),
            )
            removed += cur.rowcount

            # 总量超限，按命中率淘汰 approved 中最不常用的
            row = self._db.execute(
                "SELECT COUNT(*) AS c FROM expression_pattern WHERE status=?",
                (STATUS_APPROVED,),
            ).fetchone()
            overflow = int(row["c"]) - max_count
            if overflow > 0:
                cur = self._db.execute(
                    "DELETE FROM expression_pattern WHERE id IN ("
                    " SELECT id FROM expression_pattern WHERE status=?"
                    " ORDER BY hit_count ASC, created_at ASC LIMIT ?)",
                    (STATUS_APPROVED, overflow),
                )
                removed += cur.rowcount
            self._db.commit()
        return removed

    # ------------------------------------------------------------------
    # jargon 暗语
    # ------------------------------------------------------------------

    def get_active_jargon(self, group_id: str) -> List[Dict[str, Any]]:
        """取本群正式生效的暗语词典。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jargon WHERE group_id=? AND status=?"
                " AND meaning IS NOT NULL AND meaning != ''",
                (group_id, STATUS_ACTIVE),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item["aliases"] = json.loads(item.get("aliases_json") or "[]")
                except (TypeError, json.JSONDecodeError):
                    item["aliases"] = []
                result.append(item)
            return result

    @staticmethod
    def _loads(raw: Any, default: Any) -> Any:
        try:
            return json.loads(raw) if raw else default
        except (TypeError, json.JSONDecodeError):
            return default

    def record_jargon_group_activity(
        self, group_id: str, user_id: str, now: float
    ) -> None:
        """记录滚动窗口内的群活跃发送者，用于自适应多用户门槛。"""
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        with self._lock:
            row = self._db.execute(
                "SELECT users_json FROM jargon_group_daily WHERE group_id=? AND day=?",
                (group_id, day),
            ).fetchone()
            users = set(self._loads(row["users_json"], []) if row else [])
            users.add(user_id)
            self._db.execute(
                "INSERT INTO jargon_group_daily(group_id,day,users_json) VALUES(?,?,?)"
                " ON CONFLICT(group_id,day) DO UPDATE SET users_json=excluded.users_json",
                (group_id, day, json.dumps(sorted(users), ensure_ascii=False)),
            )
            self._db.commit()

    def record_jargon_observations(
        self,
        group_id: str,
        user_id: str,
        message_hash: str,
        context: str,
        observations: List[Dict[str, Any]],
        now: float,
        cooldown_seconds: float,
    ) -> int:
        """按消息频次记录候选证据；同用户同词在冷却期内不重复贡献。"""
        if not observations:
            return 0
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        accepted = 0
        with self._lock:
            for obs in observations:
                term = str(obs.get("term") or "").strip()
                if not term:
                    continue
                self._db.execute(
                    "INSERT INTO jargon_candidate"
                    " (group_id,term,state,first_seen_at,last_seen_at,created_at,updated_at)"
                    " VALUES(?,?, 'collecting',?,?,?,?)"
                    " ON CONFLICT(group_id,term) DO UPDATE SET"
                    " last_seen_at=excluded.last_seen_at, updated_at=excluded.updated_at"
                    " WHERE jargon_candidate.state IN ('collecting','deferred')",
                    (group_id, term, now, now, now, now),
                )
                candidate = self._db.execute(
                    "SELECT id,state FROM jargon_candidate WHERE group_id=? AND term=?",
                    (group_id, term),
                ).fetchone()
                if not candidate or candidate["state"] in ("rejected", "expired", "promoted"):
                    continue
                cid = int(candidate["id"])
                daily = self._db.execute(
                    "SELECT * FROM jargon_candidate_daily WHERE candidate_id=? AND day=?",
                    (cid, day),
                ).fetchone()
                user_stats = self._loads(daily["user_stats_json"], {}) if daily else {}
                prior = user_stats.get(user_id) or {"count": 0, "last_at": 0}
                if now - float(prior.get("last_at", 0)) < cooldown_seconds:
                    continue
                prior["count"] = int(prior.get("count", 0)) + 1
                prior["last_at"] = now
                user_stats[user_id] = prior

                left = self._loads(daily["left_neighbors_json"], {}) if daily else {}
                right = self._loads(daily["right_neighbors_json"], {}) if daily else {}
                for char in obs.get("left", []) or ["<B>"]:
                    left[str(char)] = int(left.get(str(char), 0)) + 1
                for char in obs.get("right", []) or ["<E>"]:
                    right[str(char)] = int(right.get(str(char), 0)) + 1

                hashes = self._loads(daily["support_hashes_json"], []) if daily else []
                if message_hash not in hashes:
                    hashes.append(message_hash)
                    hashes = hashes[-64:]
                contexts = self._loads(daily["contexts_json"], []) if daily else []
                if not any(c.get("user_id") == user_id for c in contexts):
                    contexts.append({"user_id": user_id, "text": context[:240]})
                    contexts = contexts[-6:]

                count = int(daily["message_count"]) + 1 if daily else 1
                self._db.execute(
                    "INSERT INTO jargon_candidate_daily"
                    " (candidate_id,day,message_count,user_stats_json,left_neighbors_json,"
                    " right_neighbors_json,support_hashes_json,contexts_json)"
                    " VALUES(?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(candidate_id,day) DO UPDATE SET"
                    " message_count=excluded.message_count,"
                    " user_stats_json=excluded.user_stats_json,"
                    " left_neighbors_json=excluded.left_neighbors_json,"
                    " right_neighbors_json=excluded.right_neighbors_json,"
                    " support_hashes_json=excluded.support_hashes_json,"
                    " contexts_json=excluded.contexts_json",
                    (
                        cid, day, count,
                        json.dumps(user_stats, ensure_ascii=False),
                        json.dumps(left, ensure_ascii=False),
                        json.dumps(right, ensure_ascii=False),
                        json.dumps(hashes, ensure_ascii=False),
                        json.dumps(contexts, ensure_ascii=False),
                    ),
                )
                accepted += 1
            self._db.commit()
        return accepted

    def get_jargon_candidate_snapshots(
        self, window_days: int, now: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """聚合滚动窗口内 collecting/deferred 候选及其有限证据。"""
        now = now or time.time()
        cutoff_day = time.strftime(
            "%Y-%m-%d", time.localtime(now - max(1, window_days) * 86400)
        )
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*,d.day,d.message_count,d.user_stats_json,"
                " d.left_neighbors_json,d.right_neighbors_json,"
                " d.support_hashes_json,d.contexts_json"
                " FROM jargon_candidate c JOIN jargon_candidate_daily d ON d.candidate_id=c.id"
                " WHERE c.state IN ('collecting','deferred') AND d.day>=?"
                " ORDER BY c.group_id,c.id,d.day",
                (cutoff_day,),
            ).fetchall()
            group_rows = self._db.execute(
                "SELECT group_id,users_json FROM jargon_group_daily WHERE day>=?",
                (cutoff_day,),
            ).fetchall()

        active_users: Dict[str, set] = {}
        for row in group_rows:
            active_users.setdefault(row["group_id"], set()).update(
                self._loads(row["users_json"], [])
            )
        merged: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            cid = int(row["id"])
            item = merged.setdefault(cid, {
                **{k: row[k] for k in row.keys() if k not in {
                    "day", "message_count", "user_stats_json", "left_neighbors_json",
                    "right_neighbors_json", "support_hashes_json", "contexts_json"
                }},
                "message_count": 0, "user_counts": {}, "left_neighbors": {},
                "right_neighbors": {}, "support_hashes": set(), "contexts": [],
                "active_group_users": len(active_users.get(row["group_id"], set())),
            })
            item["message_count"] += int(row["message_count"])
            for uid, stats in self._loads(row["user_stats_json"], {}).items():
                item["user_counts"][uid] = item["user_counts"].get(uid, 0) + int(stats.get("count", 0))
            for field in ("left_neighbors", "right_neighbors"):
                raw_field = f"{field}_json"
                for key, value in self._loads(row[raw_field], {}).items():
                    item[field][key] = item[field].get(key, 0) + int(value)
            item["support_hashes"].update(self._loads(row["support_hashes_json"], []))
            for ctx in self._loads(row["contexts_json"], []):
                if not any(x.get("user_id") == ctx.get("user_id") for x in item["contexts"]):
                    item["contexts"].append(ctx)
                    item["contexts"] = item["contexts"][-6:]
        for item in merged.values():
            item["support_hashes"] = sorted(item["support_hashes"])
        return list(merged.values())

    def set_candidate_scores(self, scores: Dict[int, float]) -> None:
        if not scores:
            return
        now = time.time()
        with self._lock:
            self._db.executemany(
                "UPDATE jargon_candidate SET local_score=?,updated_at=? WHERE id=?",
                [(score, now, cid) for cid, score in scores.items()],
            )
            self._db.commit()

    def claim_jargon_candidates(
        self, ids: List[int], token: str, now: float
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._db.execute(
                f"UPDATE jargon_candidate SET state='queued',review_token=?,updated_at=?"
                f" WHERE id IN ({placeholders}) AND state IN ('collecting','deferred')",
                (token, now, *ids),
            )
            rows = self._db.execute(
                f"SELECT * FROM jargon_candidate WHERE id IN ({placeholders})"
                " AND state='queued' AND review_token=?",
                (*ids, token),
            ).fetchall()
            self._db.commit()
            return [dict(r) for r in rows]

    def apply_jargon_verdict(
        self,
        candidate_ids: List[int],
        token: str,
        decision: str,
        category: str,
        canonical_term: str,
        meaning: str,
        confidence: float,
        reason: str,
        evidence_count: int,
        next_review_at: Optional[float] = None,
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """按 review_token 比较回写；批准时原子晋升正式词典。"""
        if not candidate_ids:
            return False
        now = time.time()
        placeholders = ",".join("?" for _ in candidate_ids)
        state = {"approve": "promoted", "reject": "rejected"}.get(decision, "deferred")
        with self._lock:
            rows = self._db.execute(
                f"SELECT id,group_id,term,llm_attempts FROM jargon_candidate"
                f" WHERE id IN ({placeholders}) AND state='queued' AND review_token=?",
                (*candidate_ids, token),
            ).fetchall()
            if len(rows) != len(candidate_ids):
                return False
            group_id = rows[0]["group_id"]
            self._db.execute(
                f"UPDATE jargon_candidate SET state=?,category=?,canonical_term=?,"
                f" meaning=?,confidence=?,verdict_reason=?,llm_attempts=llm_attempts+1,"
                f" evidence_count_at_review=?,last_llm_at=?,next_review_at=?,"
                f" review_token=NULL,updated_at=? WHERE id IN ({placeholders})"
                f" AND state='queued' AND review_token=?",
                (
                    state, category, canonical_term or None, meaning or None,
                    confidence, reason[:500], evidence_count, now, next_review_at,
                    now, *candidate_ids, token,
                ),
            )
            if decision == "approve":
                clean_aliases = sorted({a for a in (aliases or []) if a and a != canonical_term})
                self._db.execute(
                    "INSERT INTO jargon"
                    " (group_id,term,aliases_json,meaning,confidence,status,category,"
                    " evidence_count,approved_at,last_seen_at,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,'active',?,?,?,?,?,?)"
                    " ON CONFLICT(group_id,term) DO UPDATE SET"
                    " aliases_json=excluded.aliases_json,meaning=excluded.meaning,"
                    " confidence=excluded.confidence,status='active',category=excluded.category,"
                    " evidence_count=excluded.evidence_count,updated_at=excluded.updated_at",
                    (
                        group_id, canonical_term,
                        json.dumps(clean_aliases, ensure_ascii=False), meaning,
                        confidence, category, evidence_count, now, now, now, now,
                    ),
                )
            self._db.commit()
            return True

    def release_jargon_claim(self, token: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jargon_candidate SET state='deferred',review_token=NULL,updated_at=?"
                " WHERE state='queued' AND review_token=?",
                (time.time(), token),
            )
            self._db.commit()

    def reserve_jargon_llm_call(
        self, day: str, candidate_count: int, daily_limit: int, min_interval: float, now: float
    ) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT call_count,last_call_at FROM jargon_llm_usage WHERE day=?", (day,)
            ).fetchone()
            if row and int(row["call_count"]) >= daily_limit:
                return False
            if row and row["last_call_at"] and now - float(row["last_call_at"]) < min_interval:
                return False
            self._db.execute(
                "INSERT INTO jargon_llm_usage(day,call_count,candidate_count,last_call_at)"
                " VALUES(?,1,?,?) ON CONFLICT(day) DO UPDATE SET"
                " call_count=call_count+1,candidate_count=candidate_count+excluded.candidate_count,"
                " last_call_at=excluded.last_call_at",
                (day, candidate_count, now),
            )
            self._db.commit()
            return True

    def get_jargon_usage(self, day: str) -> Dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jargon_llm_usage WHERE day=?", (day,)
            ).fetchone()
            return dict(row) if row else {"day": day, "call_count": 0, "candidate_count": 0}

    def list_jargon_candidates(
        self, group_id: Optional[str] = None, state: Optional[str] = None,
        limit: int = 50, offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """分页列出候选及证据摘要，供管理页只读观察。"""
        clauses, params = [], []
        if group_id:
            clauses.append("c.group_id=?")
            params.append(group_id)
        if state:
            clauses.append("c.state=?")
            params.append(state)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*,COALESCE(SUM(d.message_count),0) AS message_count"
                " FROM jargon_candidate c LEFT JOIN jargon_candidate_daily d"
                f" ON d.candidate_id=c.id{where} GROUP BY c.id"
                " ORDER BY c.local_score DESC,c.last_seen_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                daily = self._db.execute(
                    "SELECT user_stats_json FROM jargon_candidate_daily WHERE candidate_id=?",
                    (item["id"],),
                ).fetchall()
                users = set()
                for entry in daily:
                    users.update(self._loads(entry["user_stats_json"], {}).keys())
                item["user_count"] = len(users)
                result.append(item)
            return result

    def count_jargon_candidates(
        self, group_id: Optional[str] = None, state: Optional[str] = None
    ) -> int:
        clauses, params = [], []
        if group_id:
            clauses.append("group_id=?")
            params.append(group_id)
        if state:
            clauses.append("state=?")
            params.append(state)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            row = self._db.execute(
                f"SELECT COUNT(*) AS c FROM jargon_candidate{where}", params
            ).fetchone()
            return int(row["c"])

    def maintain_jargon(self, now: float, window_days: int, expire_days: int,
                        rejected_days: int, dormant_days: int, max_per_group: int) -> Dict[str, int]:
        """清理滚动证据、候选并让长期未使用的正式暗语休眠。"""
        cutoff_day = time.strftime("%Y-%m-%d", time.localtime(now - max(window_days, 1) * 86400))
        expired_cutoff = now - max(expire_days, 1) * 86400
        rejected_cutoff = now - max(rejected_days, 1) * 86400
        dormant_cutoff = now - max(dormant_days, 1) * 86400
        result = {"daily": 0, "expired": 0, "rejected": 0, "dormant": 0, "overflow": 0}
        with self._lock:
            result["daily"] += self._db.execute(
                "DELETE FROM jargon_candidate_daily WHERE day<?", (cutoff_day,)
            ).rowcount
            result["daily"] += self._db.execute(
                "DELETE FROM jargon_group_daily WHERE day<?", (cutoff_day,)
            ).rowcount
            result["expired"] = self._db.execute(
                "UPDATE jargon_candidate SET state='expired',updated_at=?"
                " WHERE state IN ('collecting','deferred') AND last_seen_at<?",
                (now, expired_cutoff),
            ).rowcount
            result["rejected"] = self._db.execute(
                "DELETE FROM jargon_candidate WHERE state IN ('rejected','expired')"
                " AND updated_at<?", (rejected_cutoff,)
            ).rowcount
            result["dormant"] = self._db.execute(
                "UPDATE jargon SET status='dormant',dormant_at=?,updated_at=?"
                " WHERE status='active' AND last_seen_at<?",
                (now, now, dormant_cutoff),
            ).rowcount
            groups = self._db.execute(
                "SELECT group_id,COUNT(*) AS c FROM jargon_candidate"
                " WHERE state IN ('collecting','deferred') GROUP BY group_id HAVING c>?",
                (max_per_group,),
            ).fetchall()
            for group in groups:
                overflow = int(group["c"]) - max_per_group
                result["overflow"] += self._db.execute(
                    "DELETE FROM jargon_candidate WHERE id IN (SELECT id FROM jargon_candidate"
                    " WHERE group_id=? AND state IN ('collecting','deferred')"
                    " ORDER BY local_score ASC,last_seen_at ASC LIMIT ?)",
                    (group["group_id"], overflow),
                ).rowcount
            self._db.commit()
        return result

    def record_jargon_hits(self, ids: List[int], now: Optional[float] = None) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        ts = now or time.time()
        with self._lock:
            self._db.execute(
                f"UPDATE jargon SET last_seen_at=?,updated_at=? WHERE id IN ({placeholders})",
                (ts, ts, *ids),
            )
            self._db.commit()

    def observe_formal_jargon(
        self, group_id: str, text: str, user_id: str, now: float,
        cooldown_seconds: float, window_seconds: float,
    ) -> List[str]:
        """刷新正式暗语使用时间，并让休眠词在多用户再次使用后自动恢复。"""
        if not text:
            return []
        matched_terms: List[str] = []
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jargon WHERE group_id=? AND status IN ('active','dormant')",
                (group_id,),
            ).fetchall()
            for row in rows:
                aliases = self._loads(row["aliases_json"], [])
                if not any(term and term in text for term in [row["term"], *aliases]):
                    continue
                matched_terms.extend([term for term in [row["term"], *aliases] if term and term in text])
                jid = int(row["id"])
                if row["status"] == STATUS_ACTIVE:
                    self._db.execute(
                        "UPDATE jargon SET last_seen_at=?,updated_at=? WHERE id=?",
                        (now, now, jid),
                    )
                    continue
                # 休眠恢复只统计最近窗口，避免跨很久的零散命中累积。
                self._db.execute(
                    "DELETE FROM jargon_reactivation WHERE jargon_id=? AND last_at<?",
                    (jid, now - window_seconds),
                )
                prior = self._db.execute(
                    "SELECT contribution_count,last_at FROM jargon_reactivation"
                    " WHERE jargon_id=? AND user_id=?", (jid, user_id),
                ).fetchone()
                if prior and now - float(prior["last_at"]) < cooldown_seconds:
                    continue
                count = int(prior["contribution_count"]) + 1 if prior else 1
                self._db.execute(
                    "INSERT INTO jargon_reactivation(jargon_id,user_id,contribution_count,last_at)"
                    " VALUES(?,?,?,?) ON CONFLICT(jargon_id,user_id) DO UPDATE SET"
                    " contribution_count=excluded.contribution_count,last_at=excluded.last_at",
                    (jid, user_id, count, now),
                )
                evidence = self._db.execute(
                    "SELECT COUNT(*) AS users,SUM(contribution_count) AS messages"
                    " FROM jargon_reactivation WHERE jargon_id=?", (jid,),
                ).fetchone()
                if int(evidence["users"] or 0) >= 2 and int(evidence["messages"] or 0) >= 3:
                    self._db.execute(
                        "UPDATE jargon SET status='active',dormant_at=NULL,last_seen_at=?,updated_at=?"
                        " WHERE id=? AND status='dormant'", (now, now, jid),
                    )
                    self._db.execute("DELETE FROM jargon_reactivation WHERE jargon_id=?", (jid,))
            self._db.commit()
        return matched_terms

    def set_jargon_status(self, group_id: str, term: str, status: str) -> None:
        """手动设置正式词条状态。"""
        if status not in _VALID_STATUSES["jargon"]:
            raise ValueError(f"非法暗语状态：{status}")
        with self._lock:
            now = time.time()
            if status == STATUS_ACTIVE:
                self._db.execute(
                    "UPDATE jargon SET status=?,last_seen_at=?,dormant_at=NULL,updated_at=?"
                    " WHERE group_id=? AND term=?",
                    (status, now, now, group_id, term),
                )
            else:
                self._db.execute(
                    "UPDATE jargon SET status=?,updated_at=? WHERE group_id=? AND term=?",
                    (status, now, group_id, term),
                )
            self._db.commit()

    # ------------------------------------------------------------------
    # 通用
    # ------------------------------------------------------------------

    def update_status(
        self,
        table: str,
        ids: List[int],
        status: str,
        expected_status: Optional[str] = None,
    ) -> int:
        """批量更新行状态

        Args:
            table: 表名（expression_pattern/few_shot/jargon 白名单内）
            ids: 行 id 列表
            status: 目标状态（须为该表合法取值）
            expected_status: 比较更新条件——仅当行当前状态等于该值
                才更新（后台审查回写用，防止覆盖期间的管理员修改）

        Returns:
            实际更新的行数
        """
        if table not in _TABLES:
            raise ValueError(f"不允许更新的表：{table}")
        if status not in _VALID_STATUSES[table]:
            raise ValueError(
                f"表 {table} 非法的状态：{status}"
                f"（允许：{', '.join(_VALID_STATUSES[table])}）"
            )
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        conds, params = "", []
        if expected_status is not None:
            conds = " AND status=?"
            params = [expected_status]
        with self._lock:
            if table == "jargon":
                now = time.time()
                if status == STATUS_ACTIVE:
                    cur = self._db.execute(
                        f"UPDATE jargon SET status=?,last_seen_at=?,dormant_at=NULL,updated_at=?"
                        f" WHERE id IN ({placeholders}){conds}",
                        (status, now, now, *ids, *params),
                    )
                else:
                    cur = self._db.execute(
                        f"UPDATE jargon SET status=?,updated_at=?"
                        f" WHERE id IN ({placeholders}){conds}",
                        (status, now, *ids, *params),
                    )
            else:
                cur = self._db.execute(
                    f"UPDATE {table} SET status=? WHERE id IN ({placeholders}){conds}",
                    (status, *ids, *params),
                )
            self._db.commit()
            return cur.rowcount

    def clear_all(self) -> None:
        """清空全部学习数据（保留兼容命令层的无返回值接口）。"""
        self.delete_all()

    def delete_all(self) -> Dict[str, int]:
        """清空学习模块的全部主数据、候选证据和用量记录。"""
        delete_order = (
            "jargon_candidate_daily",
            "jargon_reactivation",
            "jargon_group_daily",
            "jargon_llm_usage",
            "expression_pattern",
            "few_shot",
            "jargon_candidate",
            "jargon",
        )
        deleted: Dict[str, int] = {}
        with self._lock:
            for table in delete_order:
                cur = self._db.execute(f"DELETE FROM {table}")
                deleted[table] = max(0, int(cur.rowcount))
            self._db.commit()
        deleted["total"] = sum(deleted.values())
        return deleted

    def export_all(self) -> Dict[str, Any]:
        """导出学习模块完整 JSON 快照。"""
        from datetime import datetime

        tables: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            for table, columns in _EXPORT_COLUMNS.items():
                order = "id" if "id" in columns else ", ".join(columns[:2])
                rows = self._db.execute(
                    f"SELECT {', '.join(columns)} FROM {table} ORDER BY {order}"
                ).fetchall()
                tables[table] = [dict(row) for row in rows]
        return {
            "version": LEARNING_EXPORT_VERSION,
            "export_time": datetime.now().isoformat(),
            "tables": tables,
            "stats": {table: len(rows) for table, rows in tables.items()},
        }

    def import_from_data(
        self, data: Dict[str, Any], skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """导入 :meth:`export_all` 快照，并重映射所有行 ID 与引用。"""
        if not isinstance(data, dict):
            raise ValueError("学习模块导入数据必须是字典")
        tables = data.get("tables")
        if not isinstance(tables, dict):
            raise ValueError("学习模块导入数据缺少 tables 字段")
        for table, rows in tables.items():
            if table not in _EXPORT_COLUMNS:
                raise ValueError(f"学习模块导入数据包含未知表：{table}")
            if not isinstance(rows, list):
                raise ValueError(f"表 {table} 的数据必须是列表")

        per_table = {
            table: {"imported": 0, "skipped": 0, "errors": 0}
            for table in _EXPORT_COLUMNS
        }
        pair_ids: Dict[int, int] = {}
        jargon_ids: Dict[int, int] = {}
        candidate_ids: Dict[int, int] = {}

        def rows_for(table: str) -> List[Dict[str, Any]]:
            rows = tables.get(table) or []
            return rows if isinstance(rows, list) else []

        def insert_raw(
            table: str,
            row: Dict[str, Any],
            overrides: Optional[Dict[str, Any]] = None,
        ) -> int:
            if not isinstance(row, dict):
                raise ValueError("导入记录必须是字典")
            values = dict(row)
            values.pop("id", None)
            if overrides:
                values.update(overrides)
            allowed = [c for c in _EXPORT_COLUMNS[table] if c != "id"]
            columns = [c for c in allowed if c in values]
            if not columns:
                raise ValueError(f"表 {table} 的记录为空")
            placeholders = ",".join("?" for _ in columns)
            cur = self._db.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                tuple(values[c] for c in columns),
            )
            return int(cur.lastrowid)

        def mark(table: str, result: str) -> None:
            per_table[table][result] += 1

        with self._lock:
            # 父表先导入，随后才能安全重映射 source/candidate/jargon 引用。
            for row in rows_for("few_shot"):
                try:
                    old_id = int(row.get("id") or 0)
                    existing = None
                    if skip_duplicates:
                        existing = self._db.execute(
                            "SELECT id FROM few_shot WHERE group_id=? AND user_id=?"
                            " AND user_text=? AND bot_text=? AND message_id IS ? LIMIT 1",
                            (
                                row.get("group_id", ""), row.get("user_id", ""),
                                row.get("user_text", ""), row.get("bot_text", ""),
                                row.get("message_id"),
                            ),
                        ).fetchone()
                    if existing:
                        new_id = int(existing["id"])
                        mark("few_shot", "skipped")
                    else:
                        new_id = insert_raw("few_shot", row)
                        mark("few_shot", "imported")
                    if old_id:
                        pair_ids[old_id] = new_id
                except Exception as e:
                    logger.warning(f"导入学习表 few_shot 失败：{e}")
                    mark("few_shot", "errors")

            for row in rows_for("expression_pattern"):
                try:
                    old_source = row.get("source_pair_id")
                    source_id = pair_ids.get(int(old_source)) if old_source else None
                    existing = None
                    if skip_duplicates:
                        existing = self._db.execute(
                            "SELECT id FROM expression_pattern WHERE group_id=? AND scene=?"
                            " AND expression=? AND source_pair_id IS ? LIMIT 1",
                            (
                                row.get("group_id", ""), row.get("scene", ""),
                                row.get("expression", ""), source_id,
                            ),
                        ).fetchone()
                    if existing:
                        mark("expression_pattern", "skipped")
                    else:
                        insert_raw(
                            "expression_pattern", row, {"source_pair_id": source_id}
                        )
                        mark("expression_pattern", "imported")
                except Exception as e:
                    logger.warning(f"导入学习表 expression_pattern 失败：{e}")
                    mark("expression_pattern", "errors")

            for table, id_map in (
                ("jargon", jargon_ids),
                ("jargon_candidate", candidate_ids),
            ):
                for row in rows_for(table):
                    try:
                        old_id = int(row.get("id") or 0)
                        existing = self._db.execute(
                            f"SELECT id FROM {table} WHERE group_id=? AND term=?",
                            (row.get("group_id", ""), row.get("term", "")),
                        ).fetchone()
                        if existing:
                            new_id = int(existing["id"])
                            if skip_duplicates:
                                mark(table, "skipped")
                            else:
                                mark(table, "errors")
                        else:
                            new_id = insert_raw(table, row)
                            mark(table, "imported")
                        if old_id:
                            id_map[old_id] = new_id
                    except Exception as e:
                        logger.warning(f"导入学习表 {table} 失败：{e}")
                        mark(table, "errors")

            child_specs = (
                ("jargon_candidate_daily", "candidate_id", candidate_ids),
                ("jargon_group_daily", None, None),
                ("jargon_llm_usage", None, None),
                ("jargon_reactivation", "jargon_id", jargon_ids),
            )
            for table, fk_column, id_map in child_specs:
                for row in rows_for(table):
                    try:
                        overrides = None
                        if fk_column and id_map is not None:
                            old_fk = int(row.get(fk_column) or 0)
                            new_fk = id_map.get(old_fk)
                            if not new_fk:
                                raise ValueError(f"无法映射 {fk_column}={old_fk}")
                            overrides = {fk_column: new_fk}
                        try:
                            insert_raw(table, row, overrides)
                            mark(table, "imported")
                        except sqlite3.IntegrityError:
                            mark(table, "skipped" if skip_duplicates else "errors")
                    except Exception as e:
                        logger.warning(f"导入学习表 {table} 失败：{e}")
                        mark(table, "errors")
            self._db.commit()

        return {
            "tables": per_table,
            "imported_count": sum(v["imported"] for v in per_table.values()),
            "skipped_count": sum(v["skipped"] for v in per_table.values()),
            "error_count": sum(v["errors"] for v in per_table.values()),
        }

    def list_by_group(
        self, table: str, group_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """按群列出条目（供指令层 show 使用，不限状态）

        Args:
            table: 表名（expression_pattern/few_shot/jargon 白名单内）
            group_id: 群 ID
            limit: 最多返回条数

        Returns:
            条目字典列表（jargon 按 evidence_count 降序，其余按创建时间降序）
        """
        if table not in _TABLES:
            raise ValueError(f"不允许查询的表：{table}")
        order = "evidence_count DESC" if table == "jargon" else "created_at DESC"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM {table} WHERE group_id=? ORDER BY {order} LIMIT ?",
                (group_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_by_group(self, group_id: str) -> None:
        """清空指定群的全部学习数据"""
        with self._lock:
            self._db.execute("DELETE FROM jargon_candidate WHERE group_id=?", (group_id,))
            self._db.execute("DELETE FROM jargon_group_daily WHERE group_id=?", (group_id,))
            for table in _TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE group_id=?", (group_id,))
            self._db.commit()

    def clear_by_user(self, user_id: str, group_id: Optional[str] = None) -> None:
        """清空指定用户的学习数据

        仅 few_shot 表有用户维度，expression_pattern/jargon 无 user 字段，
        不随用户清理。
        """
        with self._lock:
            if group_id:
                self._db.execute(
                    "DELETE FROM few_shot WHERE user_id=? AND group_id=?",
                    (user_id, group_id),
                )
            else:
                self._db.execute("DELETE FROM few_shot WHERE user_id=?", (user_id,))
            self._db.commit()

    def get_stats(self) -> Dict[str, Any]:
        """统计三表计数与 pending/approved 分布"""
        stats: Dict[str, Any] = {}
        with self._lock:
            for table in _TABLES:
                rows = self._db.execute(
                    f"SELECT status, COUNT(*) AS c FROM {table} GROUP BY status"
                ).fetchall()
                dist = {r["status"]: int(r["c"]) for r in rows}
                stats[table] = {
                    "total": sum(dist.values()),
                    "by_status": dist,
                }
            rows = self._db.execute(
                "SELECT state,COUNT(*) AS c FROM jargon_candidate GROUP BY state"
            ).fetchall()
            candidate_dist = {r["state"]: int(r["c"]) for r in rows}
            stats["jargon_candidate"] = {
                "total": sum(candidate_dist.values()), "by_status": candidate_dist
            }
        return stats

    # ------------------------------------------------------------------
    # Web 管理 CRUD
    # ------------------------------------------------------------------

    def list_rows(
        self,
        table: str,
        group_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """分页查询条目（可选群/状态筛选，供 Web 管理列表）

        jargon 按 evidence_count 降序，其余按创建时间降序。
        """
        if table not in _TABLES:
            raise ValueError(f"不允许查询的表：{table}")
        clauses, params = [], []
        if group_id:
            clauses.append("group_id=?")
            params.append(group_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "evidence_count DESC" if table == "jargon" else "created_at DESC"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM {table}{where} ORDER BY {order} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_rows(
        self,
        table: str,
        group_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """统计条目数（筛选条件同 list_rows，配合分页返回 total）"""
        if table not in _TABLES:
            raise ValueError(f"不允许查询的表：{table}")
        clauses, params = [], []
        if group_id:
            clauses.append("group_id=?")
            params.append(group_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            row = self._db.execute(
                f"SELECT COUNT(*) AS c FROM {table}{where}", params
            ).fetchone()
            return int(row["c"])

    def delete_rows(self, table: str, ids: List[int]) -> int:
        """按 id 批量删除条目

        Returns:
            实际删除的行数
        """
        if table not in _TABLES:
            raise ValueError(f"不允许删除的表：{table}")
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            cur = self._db.execute(
                f"DELETE FROM {table} WHERE id IN ({placeholders})", ids
            )
            self._db.commit()
            return cur.rowcount

    def update_row(self, table: str, row_id: int, fields: Dict[str, Any]) -> bool:
        """按 id 更新条目字段（字段白名单校验，供 Web 管理编辑）

        Args:
            table: 表名（白名单内）
            row_id: 行 id
            fields: 待更新字段，仅允许 _UPDATABLE_FIELDS 中列出的键

        Returns:
            是否有行被更新
        """
        if table not in _TABLES:
            raise ValueError(f"不允许更新的表：{table}")
        bad = set(fields) - set(_UPDATABLE_FIELDS[table])
        if bad:
            raise ValueError(f"表 {table} 不允许更新的字段：{sorted(bad)}")
        if not fields:
            return False
        if "status" in fields and fields["status"] not in _VALID_STATUSES[table]:
            raise ValueError(
                f"表 {table} 非法的状态：{fields['status']}"
                f"（允许：{', '.join(_VALID_STATUSES[table])}）"
            )
        if table == "jargon":
            fields = dict(fields)
            now = time.time()
            fields["updated_at"] = now
            if fields.get("status") == STATUS_ACTIVE:
                fields["last_seen_at"] = now
                fields["dormant_at"] = None
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE {table} SET {sets} WHERE id=?",
                (*fields.values(), row_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def insert_jargon(
        self,
        group_id: str,
        term: str,
        meaning: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """手动新增或更新正式暗语词条。

        Returns:
            词条行 id
        """
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jargon"
                " (group_id,term,meaning,confidence,status,category,evidence_count,"
                " approved_at,last_seen_at,created_at,updated_at)"
                " VALUES (?,?,?,?,?,'manual',0,?,?,?,?)"
                " ON CONFLICT(group_id, term) DO UPDATE SET"
                " meaning=COALESCE(NULLIF(excluded.meaning,''),jargon.meaning),"
                " confidence=CASE WHEN ? THEN jargon.confidence ELSE excluded.confidence END,"
                " status=?,category='manual',updated_at=excluded.updated_at",
                (
                    group_id, term.strip(), meaning or "", confidence or 0.0,
                    STATUS_ACTIVE, now, now, now, now, confidence is None, STATUS_ACTIVE,
                ),
            )
            row = self._db.execute(
                "SELECT id FROM jargon WHERE group_id=? AND term=?",
                (group_id, term),
            ).fetchone()
            self._db.commit()
            return int(row["id"])

    def list_groups(self) -> List[str]:
        """列出三表中出现过的全部群 ID（去空串、排序，供筛选下拉）"""
        groups = set()
        with self._lock:
            for table in _TABLES:
                rows = self._db.execute(
                    f"SELECT DISTINCT group_id FROM {table}"
                ).fetchall()
                groups.update(r["group_id"] for r in rows)
            rows = self._db.execute(
                "SELECT DISTINCT group_id FROM jargon_candidate"
            ).fetchall()
            groups.update(r["group_id"] for r in rows)
        groups.discard("")
        return sorted(groups)
