"""
Iris Chat Memory - 学习模块存储层

使用独立 SQLite 库（learning.db）持久化三类学习产物：
- expression_pattern：从对话对规则提取的"场景→表达"模式
- few_shot：user→bot 对话对样例（经 LLM 审查后注入）
- jargon：圈内暗语词频统计与含义推断结果

所有方法为同步方法，sqlite 操作很快，调用方在 async 侧直接调用即可；
写操作由组件级 asyncio.Lock 保证事件循环内串行，内部再以
threading.Lock 兜底（check_same_thread=False 连接）。
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger

logger = get_logger("learning.storage")

# 表达模式 / few_shot 的统一状态
STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_DISABLED = "disabled"

# jargon 额外的正常状态（词条未达审查语义，用 active 表示生效中）
STATUS_ACTIVE = "active"

# 允许 update_status 操作的表白名单，防止 SQL 拼接注入
_TABLES = ("expression_pattern", "few_shot", "jargon")

# 各表允许通过 update_row 修改的字段白名单（Web 管理用）
_UPDATABLE_FIELDS = {
    "expression_pattern": ("scene", "expression", "status"),
    "few_shot": ("user_text", "bot_text", "status"),
    "jargon": ("term", "meaning", "confidence", "status"),
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
    count INTEGER DEFAULT 0,
    meaning TEXT,
    confidence REAL,
    status TEXT DEFAULT 'active',
    last_inferred_at REAL,
    created_at REAL,
    UNIQUE(group_id, term)
);

CREATE INDEX IF NOT EXISTS idx_pattern_group_status ON expression_pattern(group_id, status);
CREATE INDEX IF NOT EXISTS idx_fewshot_group_status ON few_shot(group_id, status);
CREATE INDEX IF NOT EXISTS idx_fewshot_user ON few_shot(user_id);
CREATE INDEX IF NOT EXISTS idx_jargon_group ON jargon(group_id);
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
            self._db.execute("PRAGMA journal_mode=WAL")

    def init_schema(self) -> None:
        """创建表结构（幂等）"""
        with self._lock:
            self._db.executescript(_SCHEMA)
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

    def upsert_jargon_count(self, group_id: str, term: str, delta: int) -> int:
        """累加词条计数（不存在则插入）

        Returns:
            累加后的新计数
        """
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jargon (group_id, term, count, status, created_at)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(group_id, term) DO UPDATE SET count=count+?",
                (group_id, term, delta, STATUS_ACTIVE, now, delta),
            )
            row = self._db.execute(
                "SELECT count FROM jargon WHERE group_id=? AND term=?",
                (group_id, term),
            ).fetchone()
            self._db.commit()
            return int(row["count"]) if row else delta

    def load_all_jargon_counts(self) -> Dict[tuple, int]:
        """加载全部词条计数到内存（组件启动时调用）

        Returns:
            {(group_id, term): count}
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT group_id, term, count FROM jargon"
            ).fetchall()
            return {(r["group_id"], r["term"]): int(r["count"]) for r in rows}

    def get_active_jargon(self, group_id: str) -> List[Dict[str, Any]]:
        """取本群已推断且生效的暗语（有含义且未禁用）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jargon WHERE group_id=? AND status=?"
                " AND meaning IS NOT NULL AND meaning != ''",
                (group_id, STATUS_ACTIVE),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_jargon_terms_for_inference(
        self, thresholds: List[int]
    ) -> List[Dict[str, Any]]:
        """取计数达到最低阈值档位、可参与含义推断的词条

        只按 count >= min(thresholds) 且状态生效粗筛；
        是否跨档需要重新推断由 JargonLearner 的内存档位状态
        （每词上次推断档位）进一步过滤。

        Returns:
            词条字典列表（含 id/group_id/term/count/meaning/last_inferred_at）
        """
        if not thresholds:
            return []
        min_t = min(thresholds)
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jargon WHERE status IN (?, ?) AND count>=?",
                (STATUS_ACTIVE, STATUS_PENDING, min_t),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_jargon_inferred(
        self, jargon_id: int, meaning: str, confidence: float
    ) -> None:
        """记录词条含义推断结果"""
        with self._lock:
            self._db.execute(
                "UPDATE jargon SET meaning=?, confidence=?, last_inferred_at=?"
                " WHERE id=?",
                (meaning, confidence, time.time(), jargon_id),
            )
            self._db.commit()

    def set_jargon_status(self, group_id: str, term: str, status: str) -> None:
        """手动设置词条状态（active/disabled）"""
        with self._lock:
            self._db.execute(
                "UPDATE jargon SET status=? WHERE group_id=? AND term=?",
                (status, group_id, term),
            )
            self._db.commit()

    # ------------------------------------------------------------------
    # 通用
    # ------------------------------------------------------------------

    def update_status(self, table: str, ids: List[int], status: str) -> None:
        """批量更新行状态

        Args:
            table: 表名（expression_pattern/few_shot/jargon 白名单内）
            ids: 行 id 列表
            status: 目标状态
        """
        if table not in _TABLES:
            raise ValueError(f"不允许更新的表：{table}")
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self._db.execute(
                f"UPDATE {table} SET status=? WHERE id IN ({placeholders})",
                (status, *ids),
            )
            self._db.commit()

    def clear_all(self) -> None:
        """清空全部三张表"""
        with self._lock:
            for table in _TABLES:
                self._db.execute(f"DELETE FROM {table}")
            self._db.commit()

    def list_by_group(
        self, table: str, group_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """按群列出条目（供指令层 show 使用，不限状态）

        Args:
            table: 表名（expression_pattern/few_shot/jargon 白名单内）
            group_id: 群 ID
            limit: 最多返回条数

        Returns:
            条目字典列表（jargon 按 count 降序，其余按创建时间降序）
        """
        if table not in _TABLES:
            raise ValueError(f"不允许查询的表：{table}")
        order = "count DESC" if table == "jargon" else "created_at DESC"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM {table} WHERE group_id=? ORDER BY {order} LIMIT ?",
                (group_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_by_group(self, group_id: str) -> None:
        """清空指定群的全部学习数据"""
        with self._lock:
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

        jargon 按 count 降序，其余按创建时间降序。
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
        order = "count DESC" if table == "jargon" else "created_at DESC"
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
        """手动新增暗语词条（status=active，count 从 0 起）

        同群同词已存在时更新其含义与置信度。

        Returns:
            词条行 id
        """
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jargon"
                " (group_id, term, count, meaning, confidence, status, created_at)"
                " VALUES (?,?,0,?,?,?,?)"
                " ON CONFLICT(group_id, term) DO UPDATE SET"
                " meaning=COALESCE(excluded.meaning, jargon.meaning),"
                " confidence=COALESCE(excluded.confidence, jargon.confidence)",
                (group_id, term, meaning, confidence, STATUS_ACTIVE, now),
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
        groups.discard("")
        return sorted(groups)
