"""
Iris Chat Memory - 人格自迭代存储层

使用独立 SQLite 库（persona_evolution.db）持久化：
- evolution_jobs：迭代任务（persona_id 唯一）
- style_samples：脱敏风格语料（dedupe_hash 唯一）
- evolution_runs：迭代执行审计
- persona_revisions：人格版本快照（(job_id, version) 唯一）
- revision_samples：版本-语料引用（sample_id 可置空保留 hash）

与仓库其他库的差异（本组件首次引入）：
- PRAGMA busy_timeout：WAL 下读写竞争时等待而非立即 SQLITE_BUSY；
- PRAGMA user_version 迁移框架：版本化 schema，迁移在同一事务内执行，
  迁移前备份数据库文件，失败回滚并抛出，由组件降级处理。

所有方法为同步方法，sqlite 操作很快，调用方在 async 侧直接调用即可；
写操作由组件级 asyncio.Lock 保证事件循环内串行，内部再以
threading.Lock 兜底（check_same_thread=False 连接）。
"""

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.core import get_logger
from .models import (
    ApprovalMode,
    EditMode,
    EvolutionJob,
    EvolutionRun,
    JobStatus,
    PersonaRevision,
    RevisionStatus,
)

logger = get_logger("persona_evolution.storage")

# 当前 schema 版本（迁移按版本号顺序执行）
SCHEMA_VERSION = 2

# 组件级导出格式版本（文档 §19，与全量备份 1.1 对齐）
PE_EXPORT_VERSION = "1.1"

# update_job 允许修改的字段白名单，防止 SQL 拼接注入
_JOB_UPDATABLE_FIELDS = (
    "name",
    "goal_preset_id",
    "custom_goal",
    "source_group_ids_json",
    "source_user_ids_json",
    "edit_mode",
    "approval_mode",
    "status",
    "trigger_sample_count",
    "min_interval_hours",
    "provider_id",
    "reviewer_provider_id",
    "protected_fragments_json",
    "last_success_at",
    "last_sample_cursor",
    "last_applied_revision_id",
    "consecutive_failures",
)

# update_run 允许修改的字段白名单
_RUN_UPDATABLE_FIELDS = (
    "status",
    "sample_cursor_to",
    "eligible_count",
    "selected_count",
    "finished_at",
    "error_code",
    "error_message",
    "analysis_tokens",
    "generation_tokens",
    "review_tokens",
)

# update_revision 允许修改的字段白名单（JSON 字段传 Python 对象自动序列化）
_REVISION_UPDATABLE_FIELDS = (
    "status",
    "parent_revision_id",
    "base_prompt",
    "result_prompt",
    "base_hash",
    "result_hash",
    "goal_snapshot_json",
    "style_profile_json",
    "change_summary_json",
    "rationale",
    "decision_reason",
    "confidence",
    "validation_json",
    "review_json",
    "provider_snapshot_json",
    "applied_at",
)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS evolution_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    goal_preset_id TEXT NOT NULL DEFAULT 'natural',
    custom_goal TEXT NOT NULL DEFAULT '',
    source_group_ids_json TEXT NOT NULL DEFAULT '[]',
    source_user_ids_json TEXT NOT NULL DEFAULT '[]',
    edit_mode TEXT NOT NULL DEFAULT 'managed_block',
    approval_mode TEXT NOT NULL DEFAULT 'auto',
    status TEXT NOT NULL DEFAULT 'active',
    trigger_sample_count INTEGER NOT NULL DEFAULT 100,
    min_interval_hours INTEGER NOT NULL DEFAULT 24,
    provider_id TEXT NOT NULL DEFAULT '',
    reviewer_provider_id TEXT NOT NULL DEFAULT '',
    protected_fragments_json TEXT NOT NULL DEFAULT '[]',
    last_success_at REAL,
    last_sample_cursor INTEGER NOT NULL DEFAULT 0,
    last_applied_revision_id INTEGER,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS style_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL DEFAULT '',
    group_id TEXT NOT NULL DEFAULT '',
    group_name TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    user_name TEXT NOT NULL DEFAULT '',
    normalized_text TEXT NOT NULL,
    message_id TEXT,
    dedupe_hash TEXT NOT NULL UNIQUE,
    char_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_pe_samples_created ON style_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_pe_samples_group_created ON style_samples(group_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pe_samples_user_created ON style_samples(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pe_samples_group_user_created ON style_samples(group_id, user_id, created_at);

CREATE TABLE IF NOT EXISTS evolution_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    trigger_type TEXT NOT NULL DEFAULT 'auto',
    status TEXT NOT NULL DEFAULT 'running',
    sample_cursor_from INTEGER NOT NULL DEFAULT 0,
    sample_cursor_to INTEGER NOT NULL DEFAULT 0,
    eligible_count INTEGER NOT NULL DEFAULT 0,
    selected_count INTEGER NOT NULL DEFAULT 0,
    started_at REAL,
    finished_at REAL,
    error_code TEXT,
    error_message TEXT,
    analysis_tokens INTEGER NOT NULL DEFAULT 0,
    generation_tokens INTEGER NOT NULL DEFAULT 0,
    review_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pe_runs_job ON evolution_runs(job_id, started_at);

CREATE TABLE IF NOT EXISTS persona_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    parent_revision_id INTEGER,
    status TEXT NOT NULL DEFAULT 'candidate',
    trigger_type TEXT NOT NULL DEFAULT 'auto',
    edit_mode TEXT NOT NULL DEFAULT 'managed_block',
    approval_mode TEXT NOT NULL DEFAULT 'auto',
    base_prompt TEXT,
    result_prompt TEXT,
    base_hash TEXT,
    result_hash TEXT,
    goal_snapshot_json TEXT,
    style_profile_json TEXT,
    change_summary_json TEXT,
    rationale TEXT,
    confidence REAL,
    validation_json TEXT,
    review_json TEXT,
    provider_snapshot_json TEXT,
    created_at REAL,
    applied_at REAL,
    UNIQUE(job_id, version)
);

CREATE INDEX IF NOT EXISTS idx_pe_revisions_job ON persona_revisions(job_id, created_at);

CREATE TABLE IF NOT EXISTS revision_samples (
    revision_id INTEGER NOT NULL,
    sample_id INTEGER,
    sample_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pe_revision_samples_revision ON revision_samples(revision_id);
"""

# 版本号 -> 迁移 SQL（在 user_version=N 的库上执行后升级到 N+1）
_MIGRATIONS: Dict[int, str] = {
    0: _SCHEMA_V1,
    # v2：persona_revisions 增加管理决策原因列（拒绝/回滚/采纳基线，文档 §13.1）
    1: "ALTER TABLE persona_revisions ADD COLUMN decision_reason TEXT NOT NULL DEFAULT ''",
}


class PersonaEvolutionStorage:
    """人格自迭代 SQLite 存储

    负责 persona_evolution.db 的建库迁移与全部读写操作。
    非线程安全场景下由内部 threading.Lock 保护，
    async 侧的并发串行化由组件级 asyncio.Lock 负责。
    """

    def __init__(self, db_path: Path, busy_timeout_ms: int = 5000):
        """初始化存储

        Args:
            db_path: 数据库文件路径（父目录自动创建）
            busy_timeout_ms: 写竞争时的等待毫秒数（busy_timeout）
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

    # ------------------------------------------------------------------
    # schema 迁移
    # ------------------------------------------------------------------

    def get_schema_version(self) -> int:
        """读取当前 schema 版本（PRAGMA user_version）"""
        with self._lock:
            row = self._db.execute("PRAGMA user_version").fetchone()
            return int(row[0])

    def init_schema(self) -> None:
        """执行 schema 迁移（幂等）

        从当前 user_version 起按版本顺序执行迁移：
        - 迁移前备份数据库文件（persona_evolution.backup_v{N}.db）；
        - 每个迁移在同一事务内执行并更新 user_version；
        - 失败回滚并抛出，由组件捕获后置 _init_error 降级，
          不影响 Iris Memory 其他功能。
        """
        current = self.get_schema_version()
        if current >= SCHEMA_VERSION:
            return

        self._backup_before_migration(current)

        for version in range(current, SCHEMA_VERSION):
            migration = _MIGRATIONS.get(version)
            if migration is None:
                raise RuntimeError(f"缺少 schema 迁移脚本：v{version} -> v{version + 1}")
            with self._lock:
                # 逐句执行以保证迁移在同一事务内（executescript 会隐式 COMMIT）
                try:
                    self._db.execute("BEGIN")
                    for statement in migration.split(";"):
                        statement = statement.strip()
                        if statement:
                            self._db.execute(statement)
                    self._db.execute(f"PRAGMA user_version={version + 1}")
                    self._db.commit()
                except Exception:
                    self._db.rollback()
                    raise
            logger.info(f"persona_evolution.db schema 已迁移到 v{version + 1}")

    def _backup_before_migration(self, current_version: int) -> None:
        """迁移前备份数据库文件（仅对已有数据的库）"""
        if not self._db_path.exists() or self._db_path.stat().st_size == 0:
            return
        backup = self._db_path.with_name(
            f"{self._db_path.stem}.backup_v{current_version}.db"
        )
        try:
            with self._lock:
                # WAL 模式下先 checkpoint，保证备份包含全部已提交数据
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            shutil.copy2(self._db_path, backup)
            logger.info(f"迁移前备份完成：{backup}")
        except Exception as e:
            logger.warning(f"迁移前备份失败（继续迁移）：{e}")

    def close(self) -> None:
        """关闭数据库连接"""
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # style_samples 语料
    # ------------------------------------------------------------------

    def insert_sample(
        self,
        *,
        platform: str,
        group_id: str,
        group_name: str,
        user_id: str,
        user_name: str,
        normalized_text: str,
        dedupe_hash: str,
        message_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> Optional[int]:
        """插入一条语料样本（dedupe_hash 冲突时忽略）

        Returns:
            新行 id；重复样本返回 None
        """
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO style_samples"
                " (platform, group_id, group_name, user_id, user_name,"
                " normalized_text, message_id, dedupe_hash, char_count, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    platform,
                    group_id,
                    group_name,
                    user_id,
                    user_name,
                    normalized_text,
                    message_id,
                    dedupe_hash,
                    len(normalized_text),
                    created_at if created_at is not None else time.time(),
                ),
            )
            self._db.commit()
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)

    def count_samples(
        self,
        group_ids: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
        since_id: int = 0,
    ) -> int:
        """范围计数：空数组（或 None）表示该维度不限

        四种组合（文档 §7.1）：
        - 群空 + 用户空：全部语料
        - 群指定 + 用户空：指定群的全部真人发言
        - 群空 + 用户指定：这些用户在全部群的发言
        - 群指定 + 用户指定：取交集

        Args:
            group_ids: 群 ID 过滤列表
            user_ids: 用户 ID 过滤列表
            since_id: 只统计 id 大于该值的样本（自动触发增量计数）
        """
        where, params = self._build_scope_where(group_ids, user_ids, since_id)
        with self._lock:
            row = self._db.execute(
                f"SELECT COUNT(*) AS c FROM style_samples{where}", params
            ).fetchone()
            return int(row["c"])

    def fetch_samples(
        self,
        group_ids: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
        since_id: int = 0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按范围取语料（按 id 升序，供均衡抽样）"""
        where, params = self._build_scope_where(group_ids, user_ids, since_id)
        sql = f"SELECT * FROM style_samples{where} ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def _build_scope_where(
        group_ids: Optional[List[str]],
        user_ids: Optional[List[str]],
        since_id: int,
    ) -> tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if group_ids:
            placeholders = ",".join("?" for _ in group_ids)
            clauses.append(f"group_id IN ({placeholders})")
            params.extend(group_ids)
        if user_ids:
            placeholders = ",".join("?" for _ in user_ids)
            clauses.append(f"user_id IN ({placeholders})")
            params.extend(user_ids)
        if since_id > 0:
            clauses.append("id > ?")
            params.append(since_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def get_latest_sample_id(self) -> int:
        """取当前最大 Sample ID（Job 创建基线用），空表返回 0"""
        with self._lock:
            row = self._db.execute("SELECT MAX(id) AS m FROM style_samples").fetchone()
            return int(row["m"]) if row["m"] is not None else 0

    def prune_samples(self, retention_days: int, max_count: int) -> int:
        """语料保留清理：超期删除 + 超限最旧先删

        删除后把 revision_samples 中引用已删样本的 sample_id 置空
        （保留 sample_hash），避免悬空引用，Revision 不受影响。

        Returns:
            删除的总条数
        """
        now = time.time()
        cutoff = now - retention_days * 86400
        removed = 0
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM style_samples WHERE created_at < ?", (cutoff,)
            )
            removed += cur.rowcount

            row = self._db.execute("SELECT COUNT(*) AS c FROM style_samples").fetchone()
            overflow = int(row["c"]) - max_count
            if overflow > 0:
                cur = self._db.execute(
                    "DELETE FROM style_samples WHERE id IN ("
                    " SELECT id FROM style_samples ORDER BY id ASC LIMIT ?)",
                    (overflow,),
                )
                removed += cur.rowcount

            if removed:
                self._db.execute(
                    "UPDATE revision_samples SET sample_id=NULL"
                    " WHERE sample_id IS NOT NULL AND sample_id NOT IN"
                    " (SELECT id FROM style_samples)"
                )
            self._db.commit()
        if removed:
            logger.info(f"语料保留清理完成，删除 {removed} 条")
        return removed

    def clear_samples(
        self,
        group_ids: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
    ) -> int:
        """清除语料（管理操作）：都不传则清空全部

        Returns:
            删除的条数
        """
        where, params = self._build_scope_where(group_ids, user_ids, 0)
        with self._lock:
            cur = self._db.execute(f"DELETE FROM style_samples{where}", params)
            self._db.execute(
                "UPDATE revision_samples SET sample_id=NULL"
                " WHERE sample_id IS NOT NULL AND sample_id NOT IN"
                " (SELECT id FROM style_samples)"
            )
            self._db.commit()
            return cur.rowcount

    def delete_all(self) -> Dict[str, int]:
        """删除自迭代模块的全部任务、历史版本、审计记录和脱敏语料。"""
        delete_order = (
            "revision_samples",
            "persona_revisions",
            "evolution_runs",
            "evolution_jobs",
            "style_samples",
        )
        deleted: Dict[str, int] = {}
        with self._lock:
            for table in delete_order:
                cur = self._db.execute(f"DELETE FROM {table}")
                deleted[table] = max(0, int(cur.rowcount))
            self._db.commit()
        deleted["total"] = sum(deleted.values())
        return deleted

    # ------------------------------------------------------------------
    # evolution_jobs CRUD
    # ------------------------------------------------------------------

    def create_job(self, job: EvolutionJob) -> int:
        """创建迭代任务

        Args:
            job: Job 对象（persona_id 必须唯一）

        Returns:
            新行 id

        Raises:
            ValueError: persona_id 已存在 Job 或枚举值非法
        """
        now = time.time()
        try:
            with self._lock:
                cur = self._db.execute(
                    "INSERT INTO evolution_jobs"
                    " (persona_id, name, goal_preset_id, custom_goal,"
                    " source_group_ids_json, source_user_ids_json,"
                    " edit_mode, approval_mode, status,"
                    " trigger_sample_count, min_interval_hours,"
                    " provider_id, reviewer_provider_id, protected_fragments_json,"
                    " last_success_at, last_sample_cursor, last_applied_revision_id,"
                    " consecutive_failures, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job.persona_id,
                        job.name,
                        job.goal_preset_id,
                        job.custom_goal,
                        json.dumps(job.source_group_ids, ensure_ascii=False),
                        json.dumps(job.source_user_ids, ensure_ascii=False),
                        job.edit_mode,
                        job.approval_mode,
                        job.status,
                        job.trigger_sample_count,
                        job.min_interval_hours,
                        job.provider_id,
                        job.reviewer_provider_id,
                        json.dumps(job.protected_fragments, ensure_ascii=False),
                        job.last_success_at,
                        job.last_sample_cursor,
                        job.last_applied_revision_id,
                        job.consecutive_failures,
                        now,
                        now,
                    ),
                )
                self._db.commit()
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError(f"persona_id 已存在迭代 Job：{job.persona_id}") from e

    def get_job(self, job_id: int) -> Optional[EvolutionJob]:
        """按 id 取 Job"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM evolution_jobs WHERE id=?", (job_id,)
            ).fetchone()
            return self._row_to_job(row) if row else None

    def get_job_by_persona(self, persona_id: str) -> Optional[EvolutionJob]:
        """按 persona_id 取 Job"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM evolution_jobs WHERE persona_id=?", (persona_id,)
            ).fetchone()
            return self._row_to_job(row) if row else None

    def list_jobs(self) -> List[EvolutionJob]:
        """列出全部 Job（按创建时间升序）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM evolution_jobs ORDER BY created_at, id"
            ).fetchall()
            return [self._row_to_job(r) for r in rows]

    def update_job(self, job_id: int, fields: Dict[str, Any]) -> bool:
        """按 id 更新 Job 字段（字段白名单校验）

        列表类型字段（source_group_ids/source_user_ids/protected_fragments）
        传 Job 字段名即可，内部序列化为 *_json 列。

        Returns:
            是否有行被更新
        """
        list_fields = {
            "source_group_ids": "source_group_ids_json",
            "source_user_ids": "source_user_ids_json",
            "protected_fragments": "protected_fragments_json",
        }
        columns: Dict[str, Any] = {}
        for key, value in fields.items():
            column = list_fields.get(key, key)
            if column not in _JOB_UPDATABLE_FIELDS:
                raise ValueError(f"evolution_jobs 不允许更新的字段：{key}")
            if column in list_fields.values() and isinstance(value, (list, tuple)):
                value = json.dumps(list(value), ensure_ascii=False)
            columns[column] = value
        if not columns:
            return False
        columns["updated_at"] = time.time()
        sets = ", ".join(f"{k}=?" for k in columns)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE evolution_jobs SET {sets} WHERE id=?",
                (*columns.values(), job_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> EvolutionJob:
        def _loads(raw: Optional[str]) -> List[str]:
            if not raw:
                return []
            try:
                value = json.loads(raw)
                return list(value) if isinstance(value, list) else []
            except Exception:
                return []

        return EvolutionJob(
            id=int(row["id"]),
            persona_id=row["persona_id"],
            name=row["name"],
            goal_preset_id=row["goal_preset_id"],
            custom_goal=row["custom_goal"],
            source_group_ids=_loads(row["source_group_ids_json"]),
            source_user_ids=_loads(row["source_user_ids_json"]),
            edit_mode=row["edit_mode"] or EditMode.MANAGED_BLOCK.value,
            approval_mode=row["approval_mode"] or ApprovalMode.AUTO.value,
            status=row["status"] or JobStatus.ACTIVE.value,
            trigger_sample_count=int(row["trigger_sample_count"]),
            min_interval_hours=int(row["min_interval_hours"]),
            provider_id=row["provider_id"],
            reviewer_provider_id=row["reviewer_provider_id"],
            protected_fragments=_loads(row["protected_fragments_json"]),
            last_success_at=row["last_success_at"],
            last_sample_cursor=int(row["last_sample_cursor"]),
            last_applied_revision_id=row["last_applied_revision_id"],
            consecutive_failures=int(row["consecutive_failures"]),
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
        )

    # ------------------------------------------------------------------
    # evolution_runs 执行审计
    # ------------------------------------------------------------------

    def create_run(self, run: EvolutionRun) -> int:
        """创建一条执行记录（初始 status=running）

        Returns:
            新行 id
        """
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO evolution_runs"
                " (job_id, trigger_type, status, sample_cursor_from,"
                " sample_cursor_to, eligible_count, selected_count,"
                " started_at, finished_at, error_code, error_message,"
                " analysis_tokens, generation_tokens, review_tokens)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.job_id,
                    run.trigger_type,
                    run.status,
                    run.sample_cursor_from,
                    run.sample_cursor_to,
                    run.eligible_count,
                    run.selected_count,
                    run.started_at or now,
                    run.finished_at,
                    run.error_code,
                    run.error_message,
                    run.analysis_tokens,
                    run.generation_tokens,
                    run.review_tokens,
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def update_run(self, run_id: int, fields: Dict[str, Any]) -> bool:
        """更新 Run 字段（白名单校验）"""
        columns = {
            k: v for k, v in fields.items() if k in _RUN_UPDATABLE_FIELDS
        }
        unknown = set(fields) - set(columns)
        if unknown:
            raise ValueError(f"evolution_runs 不允许更新的字段：{sorted(unknown)}")
        if not columns:
            return False
        sets = ", ".join(f"{k}=?" for k in columns)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE evolution_runs SET {sets} WHERE id=?",
                (*columns.values(), run_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def get_run(self, run_id: int) -> Optional[EvolutionRun]:
        """按 id 取 Run"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM evolution_runs WHERE id=?", (run_id,)
            ).fetchone()
            return self._row_to_run(row) if row else None

    def list_runs(self, job_id: int, limit: int = 20) -> List[EvolutionRun]:
        """取 Job 最近的执行记录（按开始时间倒序）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM evolution_runs WHERE job_id=?"
                " ORDER BY started_at DESC, id DESC LIMIT ?",
                (job_id, int(limit)),
            ).fetchall()
            return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> EvolutionRun:
        return EvolutionRun(
            id=int(row["id"]),
            job_id=int(row["job_id"]),
            trigger_type=row["trigger_type"],
            status=row["status"],
            sample_cursor_from=int(row["sample_cursor_from"]),
            sample_cursor_to=int(row["sample_cursor_to"]),
            eligible_count=int(row["eligible_count"]),
            selected_count=int(row["selected_count"]),
            started_at=row["started_at"] or 0.0,
            finished_at=row["finished_at"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            analysis_tokens=int(row["analysis_tokens"]),
            generation_tokens=int(row["generation_tokens"]),
            review_tokens=int(row["review_tokens"]),
        )

    # ------------------------------------------------------------------
    # persona_revisions 版本快照
    # ------------------------------------------------------------------

    def create_revision(self, revision: PersonaRevision) -> int:
        """创建人格版本（version 自动取 Job 内最大值 +1）

        Returns:
            新行 id
        """
        now = time.time()
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(version) AS v FROM persona_revisions WHERE job_id=?",
                (revision.job_id,),
            ).fetchone()
            version = int(row["v"]) + 1 if row["v"] is not None else 1
            cur = self._db.execute(
                "INSERT INTO persona_revisions"
                " (job_id, version, parent_revision_id, status, trigger_type,"
                " edit_mode, approval_mode, base_prompt, result_prompt,"
                " base_hash, result_hash, goal_snapshot_json, style_profile_json,"
                " change_summary_json, rationale, confidence, validation_json,"
                " review_json, provider_snapshot_json, created_at, applied_at,"
                " decision_reason)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    revision.job_id,
                    version,
                    revision.parent_revision_id,
                    revision.status,
                    revision.trigger_type,
                    revision.edit_mode,
                    revision.approval_mode,
                    revision.base_prompt,
                    revision.result_prompt,
                    revision.base_hash,
                    revision.result_hash,
                    json.dumps(revision.goal_snapshot, ensure_ascii=False),
                    json.dumps(revision.style_profile, ensure_ascii=False),
                    json.dumps(revision.change_summary, ensure_ascii=False),
                    revision.rationale,
                    revision.confidence,
                    json.dumps(revision.validation, ensure_ascii=False),
                    json.dumps(revision.review, ensure_ascii=False),
                    json.dumps(revision.provider_snapshot, ensure_ascii=False),
                    now,
                    revision.applied_at,
                    revision.decision_reason,
                ),
            )
            self._db.commit()
            revision_id = int(cur.lastrowid)
            revision.id = revision_id
            revision.version = version
            revision.created_at = now
            return revision_id

    def update_revision(self, revision_id: int, fields: Dict[str, Any]) -> bool:
        """更新 Revision 字段（白名单校验）

        JSON 快照字段可传 PersonaRevision 字段名（goal_snapshot /
        style_profile / change_summary / validation / review /
        provider_snapshot），值为 Python 对象，内部序列化为 *_json 列。
        """
        json_fields = {
            "goal_snapshot": "goal_snapshot_json",
            "style_profile": "style_profile_json",
            "change_summary": "change_summary_json",
            "validation": "validation_json",
            "review": "review_json",
            "provider_snapshot": "provider_snapshot_json",
        }
        columns: Dict[str, Any] = {}
        for key, value in fields.items():
            column = json_fields.get(key, key)
            if column not in _REVISION_UPDATABLE_FIELDS:
                raise ValueError(f"persona_revisions 不允许更新的字段：{key}")
            if column in json_fields.values() and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            columns[column] = value
        if not columns:
            return False
        sets = ", ".join(f"{k}=?" for k in columns)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE persona_revisions SET {sets} WHERE id=?",
                (*columns.values(), revision_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def get_revision(self, revision_id: int) -> Optional[PersonaRevision]:
        """按 id 取 Revision"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM persona_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            return self._row_to_revision(row) if row else None

    def get_revision_by_version(
        self, job_id: int, version: int
    ) -> Optional[PersonaRevision]:
        """按 (job_id, version) 取 Revision"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM persona_revisions WHERE job_id=? AND version=?",
                (job_id, version),
            ).fetchone()
            return self._row_to_revision(row) if row else None

    def list_revisions(
        self,
        job_id: int,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[PersonaRevision]:
        """取 Job 的版本历史（按版本号倒序），可按状态过滤"""
        sql = "SELECT * FROM persona_revisions WHERE job_id=?"
        params: List[Any] = [job_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY version DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
            return [self._row_to_revision(r) for r in rows]

    def list_revisions_by_status(
        self, status: str, limit: int = 100
    ) -> List[PersonaRevision]:
        """按状态取全部 Job 的 Revision（启动恢复对账用）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM persona_revisions WHERE status=?"
                " ORDER BY created_at, id LIMIT ?",
                (status, int(limit)),
            ).fetchall()
            return [self._row_to_revision(r) for r in rows]

    def get_latest_applied_revision(
        self, job_id: int
    ) -> Optional[PersonaRevision]:
        """取 Job 最近一次已发布的 Revision"""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM persona_revisions WHERE job_id=? AND status=?"
                " ORDER BY applied_at DESC, id DESC LIMIT 1",
                (job_id, RevisionStatus.APPLIED.value),
            ).fetchone()
            return self._row_to_revision(row) if row else None

    def mark_revision_applied(
        self,
        revision_id: int,
        job_id: int,
        applied_at: float,
        status: str = RevisionStatus.APPLIED.value,
    ) -> None:
        """原子置已发布：Revision 状态与 Job 发布基线在同一事务更新

        Args:
            status: 发布后的 Revision 状态；常规发布为 applied，
                回滚发布为 rollback（保留时间线上的回滚标识，§13.3）
        """
        with self._lock:
            try:
                self._db.execute("BEGIN")
                self._db.execute(
                    "UPDATE persona_revisions SET status=?, applied_at=?"
                    " WHERE id=?",
                    (status, applied_at, revision_id),
                )
                self._db.execute(
                    "UPDATE evolution_jobs SET last_applied_revision_id=?,"
                    " last_success_at=?, updated_at=? WHERE id=?",
                    (revision_id, applied_at, applied_at, job_id),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def insert_revision_samples(
        self, revision_id: int, samples: List[Dict[str, Any]]
    ) -> int:
        """记录 Revision 引用的语料（只存 id 与 hash，不复制原文）

        Args:
            samples: 语料字典列表（含 id / dedupe_hash）

        Returns:
            写入条数
        """
        with self._lock:
            self._db.executemany(
                "INSERT INTO revision_samples (revision_id, sample_id, sample_hash)"
                " VALUES (?,?,?)",
                [
                    (
                        revision_id,
                        s.get("id"),
                        s.get("dedupe_hash") or "",
                    )
                    for s in samples
                ],
            )
            self._db.commit()
            return len(samples)

    @staticmethod
    def _row_to_revision(row: sqlite3.Row) -> PersonaRevision:
        def _loads(raw: Optional[str], default: Any) -> Any:
            if not raw:
                return default
            try:
                value = json.loads(raw)
                return value if isinstance(value, type(default)) else default
            except Exception:
                return default

        return PersonaRevision(
            id=int(row["id"]),
            job_id=int(row["job_id"]),
            version=int(row["version"]),
            parent_revision_id=row["parent_revision_id"],
            status=row["status"] or RevisionStatus.CANDIDATE.value,
            trigger_type=row["trigger_type"],
            edit_mode=row["edit_mode"] or EditMode.MANAGED_BLOCK.value,
            approval_mode=row["approval_mode"] or ApprovalMode.AUTO.value,
            base_prompt=row["base_prompt"],
            result_prompt=row["result_prompt"],
            base_hash=row["base_hash"],
            result_hash=row["result_hash"],
            goal_snapshot=_loads(row["goal_snapshot_json"], {}),
            style_profile=_loads(row["style_profile_json"], {}),
            change_summary=_loads(row["change_summary_json"], []),
            rationale=row["rationale"] or "",
            decision_reason=row["decision_reason"] or "",
            confidence=row["confidence"],
            validation=_loads(row["validation_json"], {}),
            review=_loads(row["review_json"], {}),
            provider_snapshot=_loads(row["provider_snapshot_json"], {}),
            created_at=row["created_at"] or 0.0,
            applied_at=row["applied_at"],
        )

    # ------------------------------------------------------------------
    # Revision 语料引用查询（审批复核用）
    # ------------------------------------------------------------------

    def list_revision_sample_ids(self, revision_id: int) -> List[int]:
        """取 Revision 引用且仍存在语料表中的 Sample ID（已删语料自动排除）"""
        with self._lock:
            rows = self._db.execute(
                "SELECT rs.sample_id AS sid FROM revision_samples rs"
                " JOIN style_samples s ON s.id = rs.sample_id"
                " WHERE rs.revision_id=? AND rs.sample_id IS NOT NULL",
                (revision_id,),
            ).fetchall()
            return [int(r["sid"]) for r in rows]

    def fetch_samples_by_ids(self, sample_ids: List[int]) -> List[Dict[str, Any]]:
        """按 ID 列表取语料（批准候选时重建复核语料集）"""
        if not sample_ids:
            return []
        placeholders = ",".join("?" for _ in sample_ids)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM style_samples WHERE id IN ({placeholders})"
                " ORDER BY id",
                list(sample_ids),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 语料统计（Web samples/stats，不返回原文）
    # ------------------------------------------------------------------

    def get_sample_stats(self, top_n: int = 50, days: int = 30) -> Dict[str, Any]:
        """语料分布统计：群/用户/按天计数与总量，不含任何原文

        Args:
            top_n: 群/用户分布各返回的最大条目数（按计数降序）
            days: 时间分布覆盖的最近天数
        """
        now = time.time()
        cutoff = now - days * 86400
        with self._lock:
            total = int(
                self._db.execute("SELECT COUNT(*) AS c FROM style_samples")
                .fetchone()["c"]
            )
            by_group = [
                dict(r)
                for r in self._db.execute(
                    "SELECT group_id, group_name, COUNT(*) AS count"
                    " FROM style_samples GROUP BY group_id"
                    " ORDER BY count DESC, group_id LIMIT ?",
                    (int(top_n),),
                ).fetchall()
            ]
            by_user = [
                dict(r)
                for r in self._db.execute(
                    "SELECT user_id, user_name, COUNT(*) AS count"
                    " FROM style_samples GROUP BY user_id"
                    " ORDER BY count DESC, user_id LIMIT ?",
                    (int(top_n),),
                ).fetchall()
            ]
            day_rows = self._db.execute(
                "SELECT strftime('%Y-%m-%d', created_at, 'unixepoch', 'localtime')"
                " AS day, COUNT(*) AS count FROM style_samples"
                " WHERE created_at >= ? GROUP BY day ORDER BY day",
                (cutoff,),
            ).fetchall()
            latest = self._db.execute(
                "SELECT MAX(id) AS m FROM style_samples"
            ).fetchone()
        return {
            "total": total,
            "by_group": by_group,
            "by_user": by_user,
            "by_day": [{"day": r["day"], "count": int(r["count"])} for r in day_rows],
            "latest_sample_id": int(latest["m"]) if latest["m"] is not None else 0,
            "days": int(days),
        }

    # ------------------------------------------------------------------
    # 组件级导出导入（文档 §19 / 全量备份 1.1）
    # ------------------------------------------------------------------

    def export_all(self, include_samples: bool = False) -> Dict[str, Any]:
        """导出组件完整快照（文档 §19）

        默认包含 schema 版本、Job、Run、Revision 完整快照（含风格画像/
        校验/审查结果）与 Revision-语料引用（仅 id 与 hash），
        不包含语料原文；include_samples=True 时附带已脱敏语料。

        Args:
            include_samples: 是否附带脱敏语料原文（UI 需显示隐私警告）
        """
        from dataclasses import asdict
        from datetime import datetime

        with self._lock:
            jobs = [
                asdict(self._row_to_job(r))
                for r in self._db.execute(
                    "SELECT * FROM evolution_jobs ORDER BY id"
                ).fetchall()
            ]
            runs = [
                asdict(self._row_to_run(r))
                for r in self._db.execute(
                    "SELECT * FROM evolution_runs ORDER BY id"
                ).fetchall()
            ]
            revisions = [
                asdict(self._row_to_revision(r))
                for r in self._db.execute(
                    "SELECT * FROM persona_revisions ORDER BY job_id, version"
                ).fetchall()
            ]
            refs = [
                dict(r)
                for r in self._db.execute(
                    "SELECT revision_id, sample_id, sample_hash"
                    " FROM revision_samples ORDER BY revision_id"
                ).fetchall()
            ]
            samples: List[Dict[str, Any]] = []
            if include_samples:
                samples = [
                    dict(r)
                    for r in self._db.execute(
                        "SELECT * FROM style_samples ORDER BY id"
                    ).fetchall()
                ]
        return {
            "version": PE_EXPORT_VERSION,
            "export_time": datetime.now().isoformat(),
            "include_samples": bool(include_samples),
            "jobs": jobs,
            "runs": runs,
            "revisions": revisions,
            "revision_refs": refs,
            "samples": samples,
            "stats": {
                "job_count": len(jobs),
                "run_count": len(runs),
                "revision_count": len(revisions),
                "sample_count": len(samples),
            },
        }

    def import_from_data(
        self, data: Dict[str, Any], skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """从导出快照导入（绝不修改 AstrBot Persona，文档 §19）

        ID 全部重映射：Job 按 persona_id 去重（已存在时其 Run/Revision
        一并跳过）；Revision 版本号按导出值保留，(job_id, version)
        冲突跳过；语料按 dedupe_hash 天然去重。导入的 Revision 历史
        不触发任何发布，恢复哪个版本需管理员显式操作。

        Args:
            data: export_all 产出的数据字典
            skip_duplicates: True 时重复 Job 跳过，False 时计为错误

        Returns:
            导入统计字典

        Raises:
            ValueError: 数据结构非法
        """
        if not isinstance(data, dict):
            raise ValueError("导入数据必须是字典")
        stats = {
            "imported_jobs": 0,
            "skipped_jobs": 0,
            "imported_runs": 0,
            "imported_revisions": 0,
            "skipped_revisions": 0,
            "imported_samples": 0,
            "imported_refs": 0,
            "error_count": 0,
        }

        job_id_map: Dict[int, Optional[int]] = {}
        revision_id_map: Dict[int, int] = {}
        sample_id_map: Dict[int, int] = {}
        pending_baseline: List[tuple[int, Any]] = []

        # ---- 语料（dedupe_hash 天然去重，先于引用映射）----
        for sd in data.get("samples") or []:
            try:
                old_id = int(sd.get("id") or 0)
                new_id = self.insert_sample(
                    platform=str(sd.get("platform") or ""),
                    group_id=str(sd.get("group_id") or ""),
                    group_name=str(sd.get("group_name") or ""),
                    user_id=str(sd.get("user_id") or ""),
                    user_name=str(sd.get("user_name") or ""),
                    normalized_text=str(sd.get("normalized_text") or ""),
                    dedupe_hash=str(sd.get("dedupe_hash") or ""),
                    message_id=sd.get("message_id"),
                    created_at=sd.get("created_at"),
                )
                if new_id is None:
                    with self._lock:
                        row = self._db.execute(
                            "SELECT id FROM style_samples WHERE dedupe_hash=?",
                            (str(sd.get("dedupe_hash") or ""),),
                        ).fetchone()
                    new_id = int(row["id"]) if row else 0
                else:
                    stats["imported_samples"] += 1
                if old_id and new_id:
                    sample_id_map[old_id] = new_id
            except Exception as e:
                logger.warning(f"导入语料失败：{e}")
                stats["error_count"] += 1

        # ---- Job（persona_id 唯一，重复跳过）----
        for jd in data.get("jobs") or []:
            try:
                old_id = int(jd.get("id") or 0)
                persona_id = str(jd.get("persona_id") or "")
                if not persona_id:
                    raise ValueError("缺少 persona_id")
                if self.get_job_by_persona(persona_id) is not None:
                    job_id_map[old_id] = None
                    if skip_duplicates:
                        stats["skipped_jobs"] += 1
                    else:
                        stats["error_count"] += 1
                    continue
                new_id = self.create_job(
                    EvolutionJob(
                        persona_id=persona_id,
                        name=str(jd.get("name") or ""),
                        goal_preset_id=str(jd.get("goal_preset_id") or "natural"),
                        custom_goal=str(jd.get("custom_goal") or ""),
                        source_group_ids=list(jd.get("source_group_ids") or []),
                        source_user_ids=list(jd.get("source_user_ids") or []),
                        edit_mode=str(jd.get("edit_mode") or EditMode.MANAGED_BLOCK.value),
                        approval_mode=str(
                            jd.get("approval_mode") or ApprovalMode.AUTO.value
                        ),
                        status=str(jd.get("status") or JobStatus.ACTIVE.value),
                        trigger_sample_count=int(jd.get("trigger_sample_count") or 100),
                        min_interval_hours=int(jd.get("min_interval_hours") or 24),
                        provider_id=str(jd.get("provider_id") or ""),
                        reviewer_provider_id=str(jd.get("reviewer_provider_id") or ""),
                        protected_fragments=list(jd.get("protected_fragments") or []),
                        last_success_at=jd.get("last_success_at"),
                        last_sample_cursor=int(jd.get("last_sample_cursor") or 0),
                        consecutive_failures=int(jd.get("consecutive_failures") or 0),
                    )
                )
                job_id_map[old_id] = new_id
                pending_baseline.append((new_id, jd.get("last_applied_revision_id")))
                stats["imported_jobs"] += 1
            except Exception as e:
                logger.warning(f"导入 Job 失败：{e}")
                stats["error_count"] += 1

        # ---- Run（job_id 重映射，新 id）----
        for rd in data.get("runs") or []:
            try:
                new_job_id = job_id_map.get(int(rd.get("job_id") or 0))
                if not new_job_id:
                    continue
                self.create_run(
                    EvolutionRun(
                        job_id=new_job_id,
                        trigger_type=str(rd.get("trigger_type") or "auto"),
                        status=str(rd.get("status") or "success"),
                        sample_cursor_from=int(rd.get("sample_cursor_from") or 0),
                        sample_cursor_to=int(rd.get("sample_cursor_to") or 0),
                        eligible_count=int(rd.get("eligible_count") or 0),
                        selected_count=int(rd.get("selected_count") or 0),
                        started_at=float(rd.get("started_at") or 0.0),
                        finished_at=rd.get("finished_at"),
                        error_code=rd.get("error_code"),
                        error_message=rd.get("error_message"),
                        analysis_tokens=int(rd.get("analysis_tokens") or 0),
                        generation_tokens=int(rd.get("generation_tokens") or 0),
                        review_tokens=int(rd.get("review_tokens") or 0),
                    )
                )
                stats["imported_runs"] += 1
            except Exception as e:
                logger.warning(f"导入 Run 失败：{e}")
                stats["error_count"] += 1

        # ---- Revision（版本号保留，父版本与 Job 重映射）----
        exported_revisions = sorted(
            data.get("revisions") or [],
            key=lambda r: (int(r.get("job_id") or 0), int(r.get("version") or 0)),
        )
        for rd in exported_revisions:
            try:
                new_job_id = job_id_map.get(int(rd.get("job_id") or 0))
                if not new_job_id:
                    stats["skipped_revisions"] += 1
                    continue
                old_parent = rd.get("parent_revision_id")
                revision = PersonaRevision(
                    job_id=new_job_id,
                    version=int(rd.get("version") or 0),
                    parent_revision_id=(
                        revision_id_map.get(int(old_parent)) if old_parent else None
                    ),
                    status=str(rd.get("status") or RevisionStatus.CANDIDATE.value),
                    trigger_type=str(rd.get("trigger_type") or "auto"),
                    edit_mode=str(rd.get("edit_mode") or EditMode.MANAGED_BLOCK.value),
                    approval_mode=str(
                        rd.get("approval_mode") or ApprovalMode.AUTO.value
                    ),
                    base_prompt=rd.get("base_prompt"),
                    result_prompt=rd.get("result_prompt"),
                    base_hash=rd.get("base_hash"),
                    result_hash=rd.get("result_hash"),
                    goal_snapshot=dict(rd.get("goal_snapshot") or {}),
                    style_profile=dict(rd.get("style_profile") or {}),
                    change_summary=list(rd.get("change_summary") or []),
                    rationale=str(rd.get("rationale") or ""),
                    decision_reason=str(rd.get("decision_reason") or ""),
                    confidence=rd.get("confidence"),
                    validation=dict(rd.get("validation") or {}),
                    review=dict(rd.get("review") or {}),
                    provider_snapshot=dict(rd.get("provider_snapshot") or {}),
                    applied_at=rd.get("applied_at"),
                )
                new_rev_id = self._insert_revision_imported(
                    revision, float(rd.get("created_at") or 0.0) or None
                )
                if new_rev_id is None:
                    stats["skipped_revisions"] += 1
                    continue
                revision_id_map[int(rd.get("id") or 0)] = new_rev_id
                stats["imported_revisions"] += 1
            except Exception as e:
                logger.warning(f"导入 Revision 失败：{e}")
                stats["error_count"] += 1

        # ---- Revision-语料引用（revision/sample 重映射，hash 兜底）----
        for rr in data.get("revision_refs") or []:
            try:
                new_rev_id = revision_id_map.get(int(rr.get("revision_id") or 0))
                if not new_rev_id:
                    continue
                old_sample = rr.get("sample_id")
                new_sample = (
                    sample_id_map.get(int(old_sample)) if old_sample else None
                )
                self.insert_revision_samples(
                    new_rev_id,
                    [{"id": new_sample, "dedupe_hash": rr.get("sample_hash") or ""}],
                )
                stats["imported_refs"] += 1
            except Exception as e:
                logger.warning(f"导入 Revision 语料引用失败：{e}")
                stats["error_count"] += 1

        # ---- Job 发布基线重映射 ----
        for new_job_id, old_baseline in pending_baseline:
            if not old_baseline:
                continue
            mapped = revision_id_map.get(int(old_baseline))
            if mapped:
                self.update_job(new_job_id, {"last_applied_revision_id": mapped})

        logger.info(f"人格自迭代导入完成：{stats}")
        return stats

    def _insert_revision_imported(
        self, revision: PersonaRevision, created_at: Optional[float] = None
    ) -> Optional[int]:
        """按导出版本号直接插入 Revision（导入专用）

        Returns:
            新行 id；(job_id, version) 冲突返回 None
        """
        now = time.time()
        with self._lock:
            try:
                cur = self._db.execute(
                    "INSERT INTO persona_revisions"
                    " (job_id, version, parent_revision_id, status, trigger_type,"
                    " edit_mode, approval_mode, base_prompt, result_prompt,"
                    " base_hash, result_hash, goal_snapshot_json, style_profile_json,"
                    " change_summary_json, rationale, confidence, validation_json,"
                    " review_json, provider_snapshot_json, created_at, applied_at,"
                    " decision_reason)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        revision.job_id,
                        revision.version,
                        revision.parent_revision_id,
                        revision.status,
                        revision.trigger_type,
                        revision.edit_mode,
                        revision.approval_mode,
                        revision.base_prompt,
                        revision.result_prompt,
                        revision.base_hash,
                        revision.result_hash,
                        json.dumps(revision.goal_snapshot, ensure_ascii=False),
                        json.dumps(revision.style_profile, ensure_ascii=False),
                        json.dumps(revision.change_summary, ensure_ascii=False),
                        revision.rationale,
                        revision.confidence,
                        json.dumps(revision.validation, ensure_ascii=False),
                        json.dumps(revision.review, ensure_ascii=False),
                        json.dumps(revision.provider_snapshot, ensure_ascii=False),
                        created_at or now,
                        revision.applied_at,
                        revision.decision_reason,
                    ),
                )
                self._db.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return None
