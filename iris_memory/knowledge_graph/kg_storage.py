"""
知识图谱 SQLite + FTS5 存储层

提供：
- 节点/边的 CRUD 操作
- FTS5 全文检索（中英文实体名、关系标签）
- 基于 scope 的读写隔离（私聊/群聊/全局）
- 实体去重与合并（基于 normalized name + user_id/group_id）
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
)
from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import CacheDefaults

logger = get_logger("kg_storage")

# ── 常量 ──
_SCHEMA_VERSION = 2
_DEFAULT_DB_NAME = "knowledge_graph.db"
_NODE_CACHE_TTL = 300  # 5 分钟缓存 TTL
_NODE_CACHE_MAX_SIZE = CacheDefaults.KG_NODE_CACHE_MAX_SIZE

# 中文字符范围正则
_CN_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


class KGStorage:
    """SQLite + FTS5 知识图谱存储"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        # ── 节点缓存层（减少重复查询）──
        self._node_cache: Dict[str, Tuple[KGNode, float]] = {}  # cache_key -> (node, timestamp)
        self._cache_ttl = _NODE_CACHE_TTL
        self._cache_max_size = _NODE_CACHE_MAX_SIZE

    # ================================================================
    # 生命周期
    # ================================================================

    async def initialize(self, db_path: Path) -> None:
        """初始化数据库（建表 + FTS5 + 自动迁移）"""
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._auto_migrate()
        logger.debug(f"KGStorage initialized: {db_path}")

    def _create_tables(self) -> None:
        """建表（幂等）"""
        with self._tx() as cur:
            # ── 节点表 ──
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    id            TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    display_name  TEXT NOT NULL DEFAULT '',
                    node_type     TEXT NOT NULL DEFAULT 'unknown',
                    user_id       TEXT NOT NULL DEFAULT '',
                    group_id      TEXT,
                    persona_id    TEXT DEFAULT 'default',
                    aliases       TEXT NOT NULL DEFAULT '[]',
                    properties    TEXT NOT NULL DEFAULT '{}',
                    mention_count INTEGER NOT NULL DEFAULT 1,
                    confidence    REAL NOT NULL DEFAULT 0.5,
                    created_time  TEXT NOT NULL,
                    updated_time  TEXT NOT NULL
                )
            """)

            # ── 边表 ──
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id             TEXT PRIMARY KEY,
                    source_id      TEXT NOT NULL,
                    target_id      TEXT NOT NULL,
                    relation_type  TEXT NOT NULL DEFAULT 'related_to',
                    relation_label TEXT NOT NULL DEFAULT '',
                    memory_id      TEXT,
                    user_id        TEXT NOT NULL DEFAULT '',
                    group_id       TEXT,
                    persona_id     TEXT DEFAULT 'default',
                    confidence     REAL NOT NULL DEFAULT 0.5,
                    weight         REAL NOT NULL DEFAULT 1.0,
                    properties     TEXT NOT NULL DEFAULT '{}',
                    created_time   TEXT NOT NULL,
                    updated_time   TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES kg_nodes(id) ON DELETE CASCADE
                )
            """)

            # ── 索引 ──
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON kg_nodes(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_user ON kg_nodes(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_group ON kg_nodes(group_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON kg_nodes(node_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON kg_edges(source_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON kg_edges(target_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_rel ON kg_edges(relation_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_memory ON kg_edges(memory_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_user ON kg_edges(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_group ON kg_edges(group_id)")

            # ── FTS5 虚拟表 ──
            # tokenize='unicode61' 支持中英文
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kg_nodes_fts
                USING fts5(
                    name,
                    display_name,
                    aliases,
                    content='kg_nodes',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kg_edges_fts
                USING fts5(
                    relation_label,
                    content='kg_edges',
                    content_rowid='rowid',
                    tokenize='unicode61'
                )
            """)

            # ── FTS 触发器（自动同步） ──
            cur.executescript("""
                CREATE TRIGGER IF NOT EXISTS kg_nodes_ai AFTER INSERT ON kg_nodes BEGIN
                    INSERT INTO kg_nodes_fts(rowid, name, display_name, aliases)
                    VALUES (new.rowid, new.name, new.display_name, new.aliases);
                END;
                CREATE TRIGGER IF NOT EXISTS kg_nodes_ad AFTER DELETE ON kg_nodes BEGIN
                    INSERT INTO kg_nodes_fts(kg_nodes_fts, rowid, name, display_name, aliases)
                    VALUES ('delete', old.rowid, old.name, old.display_name, old.aliases);
                END;
                CREATE TRIGGER IF NOT EXISTS kg_nodes_au AFTER UPDATE ON kg_nodes BEGIN
                    INSERT INTO kg_nodes_fts(kg_nodes_fts, rowid, name, display_name, aliases)
                    VALUES ('delete', old.rowid, old.name, old.display_name, old.aliases);
                    INSERT INTO kg_nodes_fts(rowid, name, display_name, aliases)
                    VALUES (new.rowid, new.name, new.display_name, new.aliases);
                END;

                CREATE TRIGGER IF NOT EXISTS kg_edges_ai AFTER INSERT ON kg_edges BEGIN
                    INSERT INTO kg_edges_fts(rowid, relation_label)
                    VALUES (new.rowid, new.relation_label);
                END;
                CREATE TRIGGER IF NOT EXISTS kg_edges_ad AFTER DELETE ON kg_edges BEGIN
                    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, relation_label)
                    VALUES ('delete', old.rowid, old.relation_label);
                END;
                CREATE TRIGGER IF NOT EXISTS kg_edges_au AFTER UPDATE ON kg_edges BEGIN
                    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, relation_label)
                    VALUES ('delete', old.rowid, old.relation_label);
                    INSERT INTO kg_edges_fts(rowid, relation_label)
                    VALUES (new.rowid, new.relation_label);
                END;
            """)

            # ── 版本标记 ──
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cur.execute(
                "INSERT OR REPLACE INTO kg_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _auto_migrate(self) -> None:
        """自动迁移：检查 schema_version 并执行必要的 DDL 升级"""
        assert self._conn
        try:
            row = self._conn.execute(
                "SELECT value FROM kg_meta WHERE key = 'schema_version'"
            ).fetchone()
            current = int(row["value"]) if row else 1

            if current < _SCHEMA_VERSION:
                logger.info(f"KG schema migration: v{current} → v{_SCHEMA_VERSION}")

                if current < 2:
                    self._migrate_v1_to_v2()

                self._conn.execute(
                    "INSERT OR REPLACE INTO kg_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(_SCHEMA_VERSION)),
                )
                self._conn.commit()
                logger.info(f"KG schema migration completed: v{current} → v{_SCHEMA_VERSION}")

            self._ensure_persona_indexes()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"KG schema migration failed: {e}")
            raise RuntimeError(f"Database migration failed: {e}") from e

    def _ensure_persona_indexes(self) -> None:
        """确保 persona_id 相关索引存在（新旧数据库都需要）"""
        assert self._conn
        try:
            cols = {info[1] for info in self._conn.execute("PRAGMA table_info(kg_nodes)").fetchall()}
            if "persona_id" in cols:
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_persona ON kg_nodes(persona_id)")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_user_persona ON kg_nodes(user_id, persona_id)")
            cols_e = {info[1] for info in self._conn.execute("PRAGMA table_info(kg_edges)").fetchall()}
            if "persona_id" in cols_e:
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_persona ON kg_edges(persona_id)")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_user_persona ON kg_edges(user_id, persona_id)")
        except sqlite3.Error as e:
            logger.warning(f"Failed to create persona indexes: {e}")

    def _migrate_v1_to_v2(self) -> None:
        """v1 → v2：添加 persona_id 列（DEFAULT 'default' 自动回填旧数据）"""
        assert self._conn
        with self._tx() as cur:
            cols = {info[1] for info in cur.execute("PRAGMA table_info(kg_nodes)").fetchall()}
            if "persona_id" not in cols:
                cur.execute("ALTER TABLE kg_nodes ADD COLUMN persona_id TEXT DEFAULT 'default'")
                logger.info("Migration v1→v2: added persona_id to kg_nodes")
            cols_e = {info[1] for info in cur.execute("PRAGMA table_info(kg_edges)").fetchall()}
            if "persona_id" not in cols_e:
                cur.execute("ALTER TABLE kg_edges ADD COLUMN persona_id TEXT DEFAULT 'default'")
                logger.info("Migration v1→v2: added persona_id to kg_edges")

    @contextmanager
    def _tx(self):
        """简化事务上下文"""
        assert self._conn is not None, "KGStorage not initialized"
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ================================================================
    # 节点 CRUD
    # ================================================================

    async def upsert_node(self, node: KGNode) -> KGNode:
        """插入或更新节点（按 normalized name + user_id + group_id 去重）

        如果同名节点已存在则更新提及次数和置信度，返回现有节点。
        使用内存缓存减少数据库查询。
        """
        async with self._lock:
            cache_key = self._node_cache_key(node.name, node.user_id, node.group_id)

            # 先查缓存
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                existing = cached
            else:
                existing = self._find_node_by_name(node.name, node.user_id, node.group_id)

            if existing:
                # 合并：增加提及次数，提升置信度
                existing.mention_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.updated_time = datetime.now()
                # 合并别名
                alias_set = set(existing.aliases)
                if node.display_name and node.display_name not in alias_set and node.display_name != existing.name:
                    existing.aliases.append(node.display_name)
                for a in node.aliases:
                    if a not in alias_set:
                        existing.aliases.append(a)
                # 合并属性（新覆盖旧）
                existing.properties.update(node.properties)
                self._update_node(existing)
                self._put_to_cache(cache_key, existing)
                return existing
            else:
                self._insert_node(node)
                self._put_to_cache(cache_key, node)
                return node

    def _find_node_by_name(
        self, name: str, user_id: str, group_id: Optional[str]
    ) -> Optional[KGNode]:
        """按 normalized name 查找节点"""
        assert self._conn
        normalized = self._normalize_name(name)
        if group_id:
            row = self._conn.execute(
                "SELECT * FROM kg_nodes WHERE name = ? AND (user_id = ? OR group_id = ?)",
                (normalized, user_id, group_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM kg_nodes WHERE name = ? AND user_id = ? AND group_id IS NULL",
                (normalized, user_id),
            ).fetchone()
        if row:
            return KGNode.from_row(dict(row))
        return None

    def _insert_node(self, node: KGNode) -> None:
        """插入节点"""
        node.name = self._normalize_name(node.name)
        d = node.to_dict()
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO kg_nodes
                   (id, name, display_name, node_type, user_id, group_id, persona_id,
                    aliases, properties, mention_count, confidence,
                    created_time, updated_time)
                   VALUES (:id, :name, :display_name, :node_type, :user_id, :group_id, :persona_id,
                           :aliases, :properties, :mention_count, :confidence,
                           :created_time, :updated_time)""",
                d,
            )

    def _update_node(self, node: KGNode) -> None:
        """更新节点"""
        d = node.to_dict()
        with self._tx() as cur:
            cur.execute(
                """UPDATE kg_nodes SET
                    display_name=:display_name, node_type=:node_type,
                    aliases=:aliases, properties=:properties,
                    mention_count=:mention_count, confidence=:confidence,
                    updated_time=:updated_time
                   WHERE id=:id""",
                d,
            )

    async def get_node(self, node_id: str) -> Optional[KGNode]:
        """按 ID 获取节点"""
        async with self._lock:
            assert self._conn
            row = self._conn.execute(
                "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            return KGNode.from_row(dict(row)) if row else None

    async def search_nodes(
        self,
        query: str,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[KGNodeType] = None,
        limit: int = 20,
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """FTS5 全文搜索节点

        Args:
            query: 搜索文本
            user_id: 限定用户
            group_id: 限定群组
            node_type: 限定节点类型
            limit: 最大返回数
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            匹配的节点列表（按 rank 排序）
        """
        async with self._lock:
            return self._search_nodes_sync(query, user_id, group_id, node_type, limit, persona_id)

    def _search_nodes_sync(
        self,
        query: str,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[KGNodeType] = None,
        limit: int = 20,
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """同步版全文搜索

        对中文查询优先使用精确 LIKE 匹配（FTS5 unicode61 对中文按字符切分，
        搜索"张三"会匹配"张三丰"、"李张三"等，精度不足）；
        对英文查询使用 FTS5 全文检索。
        """
        assert self._conn

        # 中文查询优先使用 LIKE 精确匹配
        if self._is_chinese(query):
            results = self._search_nodes_like_exact(query, user_id, group_id, node_type, limit, persona_id)
            if results:
                return results
            # 精确匹配无结果时退回模糊 LIKE
            return self._search_nodes_like(query, user_id, group_id, node_type, limit, persona_id)

        # 英文使用 FTS5
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return self._search_nodes_like(query, user_id, group_id, node_type, limit, persona_id)

        try:
            rows = self._conn.execute(
                """SELECT n.* FROM kg_nodes n
                   JOIN kg_nodes_fts f ON n.rowid = f.rowid
                   WHERE kg_nodes_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, limit * 3),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS 查询语法错误时 fallback
            return self._search_nodes_like(query, user_id, group_id, node_type, limit, persona_id)

        nodes = [KGNode.from_row(dict(r)) for r in rows]
        return self._filter_nodes_by_scope(nodes, user_id, group_id, node_type, limit, persona_id)

    def _search_nodes_like_exact(
        self,
        query: str,
        user_id: Optional[str],
        group_id: Optional[str],
        node_type: Optional[KGNodeType],
        limit: int,
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """精确名称匹配搜索（用于中文实体）

        优先匹配名称完全相等的节点，再匹配别名包含查询词的节点。
        避免 FTS5 unicode61 对中文按字分词导致的误匹配。
        """
        assert self._conn
        normalized = self._normalize_name(query)
        # 精确匹配 name
        rows = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE name = ? LIMIT ?",
            (normalized, limit * 3),
        ).fetchall()
        # 补充 display_name 精确匹配
        if len(rows) < limit:
            extra = self._conn.execute(
                "SELECT * FROM kg_nodes WHERE display_name = ? AND name != ? LIMIT ?",
                (query, normalized, limit * 3 - len(rows)),
            ).fetchall()
            rows = list(rows) + list(extra)
        nodes = [KGNode.from_row(dict(r)) for r in rows]
        return self._filter_nodes_by_scope(nodes, user_id, group_id, node_type, limit, persona_id)

    def _search_nodes_like(
        self,
        query: str,
        user_id: Optional[str],
        group_id: Optional[str],
        node_type: Optional[KGNodeType],
        limit: int,
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """LIKE 回退搜索"""
        assert self._conn
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE name LIKE ? OR display_name LIKE ? LIMIT ?",
            (pattern, pattern, limit * 3),
        ).fetchall()
        nodes = [KGNode.from_row(dict(r)) for r in rows]
        return self._filter_nodes_by_scope(nodes, user_id, group_id, node_type, limit, persona_id)

    def _filter_nodes_by_scope(
        self,
        nodes: List[KGNode],
        user_id: Optional[str],
        group_id: Optional[str],
        node_type: Optional[KGNodeType],
        limit: int,
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """根据 scope 和 persona 过滤节点

        安全约束：当 user_id 为空时返回空列表，防止数据泄露。
        persona_id 非 None 时启用 persona 过滤（只返回匹配的 persona）。
        """
        if not user_id:
            logger.warning("Scope filtering skipped: no user_id provided")
            return []

        result = []
        for n in nodes:
            if node_type and n.node_type != node_type:
                continue
            # persona 过滤
            if persona_id is not None and n.persona_id != persona_id:
                continue
            if group_id:
                # 群聊场景：自己的 + 群共享的
                if n.user_id == user_id or n.group_id == group_id:
                    result.append(n)
            else:
                # 私聊场景
                if n.user_id == user_id and n.group_id is None:
                    result.append(n)
            if len(result) >= limit:
                break
        return result

    # ================================================================
    # 边 CRUD
    # ================================================================

    async def upsert_edge(self, edge: KGEdge) -> KGEdge:
        """插入或更新边（按 source_id + target_id + relation_type 去重）"""
        async with self._lock:
            existing = self._find_edge(edge.source_id, edge.target_id, edge.relation_type)
            if existing:
                existing.weight += 0.5
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.updated_time = datetime.now()
                if edge.relation_label and edge.relation_label != existing.relation_label:
                    existing.relation_label = edge.relation_label
                existing.properties.update(edge.properties)
                self._update_edge(existing)
                return existing
            else:
                self._insert_edge(edge)
                return edge

    def _find_edge(
        self, source_id: str, target_id: str, relation_type: KGRelationType
    ) -> Optional[KGEdge]:
        """查找现有边"""
        assert self._conn
        row = self._conn.execute(
            """SELECT * FROM kg_edges
               WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
            (source_id, target_id, relation_type.value),
        ).fetchone()
        return KGEdge.from_row(dict(row)) if row else None

    def _insert_edge(self, edge: KGEdge) -> None:
        d = edge.to_dict()
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO kg_edges
                   (id, source_id, target_id, relation_type, relation_label,
                    memory_id, user_id, group_id, persona_id, confidence, weight,
                    properties, created_time, updated_time)
                   VALUES (:id, :source_id, :target_id, :relation_type, :relation_label,
                           :memory_id, :user_id, :group_id, :persona_id, :confidence, :weight,
                           :properties, :created_time, :updated_time)""",
                d,
            )

    def _update_edge(self, edge: KGEdge) -> None:
        d = edge.to_dict()
        with self._tx() as cur:
            cur.execute(
                """UPDATE kg_edges SET
                    relation_label=:relation_label, confidence=:confidence,
                    weight=:weight, properties=:properties, updated_time=:updated_time
                   WHERE id=:id""",
                d,
            )

    async def get_edges_from(
        self,
        node_id: str,
        relation_type: Optional[KGRelationType] = None,
        limit: int = 50,
    ) -> List[KGEdge]:
        """获取从 node_id 出发的所有边"""
        async with self._lock:
            return self._get_edges_from_sync(node_id, relation_type, limit)

    def _get_edges_from_sync(
        self,
        node_id: str,
        relation_type: Optional[KGRelationType] = None,
        limit: int = 50,
    ) -> List[KGEdge]:
        assert self._conn
        if relation_type:
            rows = self._conn.execute(
                "SELECT * FROM kg_edges WHERE source_id = ? AND relation_type = ? LIMIT ?",
                (node_id, relation_type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM kg_edges WHERE source_id = ? LIMIT ?",
                (node_id, limit),
            ).fetchall()
        return [KGEdge.from_row(dict(r)) for r in rows]

    async def get_edges_to(
        self,
        node_id: str,
        relation_type: Optional[KGRelationType] = None,
        limit: int = 50,
    ) -> List[KGEdge]:
        """获取指向 node_id 的所有边"""
        async with self._lock:
            return self._get_edges_to_sync(node_id, relation_type, limit)

    def _get_edges_to_sync(
        self,
        node_id: str,
        relation_type: Optional[KGRelationType] = None,
        limit: int = 50,
    ) -> List[KGEdge]:
        assert self._conn
        if relation_type:
            rows = self._conn.execute(
                "SELECT * FROM kg_edges WHERE target_id = ? AND relation_type = ? LIMIT ?",
                (node_id, relation_type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM kg_edges WHERE target_id = ? LIMIT ?",
                (node_id, limit),
            ).fetchall()
        return [KGEdge.from_row(dict(r)) for r in rows]

    async def get_neighbors(
        self,
        node_id: str,
        limit: int = 50,
    ) -> List[Tuple[KGEdge, KGNode]]:
        """获取邻居节点（双向）"""
        async with self._lock:
            return self._get_neighbors_sync(node_id, limit)

    def _get_neighbors_sync(
        self,
        node_id: str,
        limit: int = 50,
    ) -> List[Tuple[KGEdge, KGNode]]:
        """同步获取邻居"""
        assert self._conn
        results: List[Tuple[KGEdge, KGNode]] = []

        # 出边
        rows = self._conn.execute(
            """SELECT e.*, n.id as n_id, n.name as n_name, n.display_name as n_display,
                      n.node_type as n_type, n.user_id as n_user, n.group_id as n_group,
                      n.persona_id as n_persona,
                      n.aliases as n_aliases, n.properties as n_props,
                      n.mention_count as n_mention, n.confidence as n_conf,
                      n.created_time as n_created, n.updated_time as n_updated
               FROM kg_edges e JOIN kg_nodes n ON e.target_id = n.id
               WHERE e.source_id = ?
               LIMIT ?""",
            (node_id, limit),
        ).fetchall()
        for r in rows:
            rd = dict(r)
            edge = KGEdge.from_row(rd)
            node = KGNode.from_row({
                "id": rd["n_id"], "name": rd["n_name"],
                "display_name": rd["n_display"], "node_type": rd["n_type"],
                "user_id": rd["n_user"], "group_id": rd["n_group"],
                "persona_id": rd.get("n_persona", "default"),
                "aliases": rd["n_aliases"], "properties": rd["n_props"],
                "mention_count": rd["n_mention"], "confidence": rd["n_conf"],
                "created_time": rd["n_created"], "updated_time": rd["n_updated"],
            })
            results.append((edge, node))

        # 入边
        rows = self._conn.execute(
            """SELECT e.*, n.id as n_id, n.name as n_name, n.display_name as n_display,
                      n.node_type as n_type, n.user_id as n_user, n.group_id as n_group,
                      n.persona_id as n_persona,
                      n.aliases as n_aliases, n.properties as n_props,
                      n.mention_count as n_mention, n.confidence as n_conf,
                      n.created_time as n_created, n.updated_time as n_updated
               FROM kg_edges e JOIN kg_nodes n ON e.source_id = n.id
               WHERE e.target_id = ?
               LIMIT ?""",
            (node_id, limit),
        ).fetchall()
        for r in rows:
            rd = dict(r)
            edge = KGEdge.from_row(rd)
            node = KGNode.from_row({
                "id": rd["n_id"], "name": rd["n_name"],
                "display_name": rd["n_display"], "node_type": rd["n_type"],
                "user_id": rd["n_user"], "group_id": rd["n_group"],
                "persona_id": rd.get("n_persona", "default"),
                "aliases": rd["n_aliases"], "properties": rd["n_props"],
                "mention_count": rd["n_mention"], "confidence": rd["n_conf"],
                "created_time": rd["n_created"], "updated_time": rd["n_updated"],
            })
            results.append((edge, node))

        return results[:limit]

    # ================================================================
    # 批量操作
    # ================================================================

    async def delete_by_memory_id(self, memory_id: str) -> int:
        """删除与指定记忆关联的所有边"""
        async with self._lock:
            with self._tx() as cur:
                cur.execute("DELETE FROM kg_edges WHERE memory_id = ?", (memory_id,))
                return cur.rowcount

    async def delete_user_data_by_group(self, group_id: str) -> int:
        """删除指定群组的所有知识图谱数据（不限定用户）"""
        async with self._lock:
            count = 0
            with self._tx() as cur:
                cur.execute("DELETE FROM kg_edges WHERE group_id = ?", (group_id,))
                count += cur.rowcount
                cur.execute("DELETE FROM kg_nodes WHERE group_id = ?", (group_id,))
                count += cur.rowcount
            self._invalidate_cache()
            return count

    async def delete_user_data(self, user_id: str, group_id: Optional[str] = None) -> int:
        """删除用户的所有知识图谱数据"""
        async with self._lock:
            count = 0
            with self._tx() as cur:
                if group_id:
                    cur.execute("DELETE FROM kg_edges WHERE user_id = ? AND group_id = ?", (user_id, group_id))
                    count += cur.rowcount
                    cur.execute("DELETE FROM kg_nodes WHERE user_id = ? AND group_id = ?", (user_id, group_id))
                    count += cur.rowcount
                else:
                    cur.execute("DELETE FROM kg_edges WHERE user_id = ?", (user_id,))
                    count += cur.rowcount
                    cur.execute("DELETE FROM kg_nodes WHERE user_id = ?", (user_id,))
                    count += cur.rowcount
            # 失效缓存
            self._invalidate_cache(user_id=user_id)
            return count

    async def delete_all(self) -> int:
        """删除全部"""
        async with self._lock:
            count = 0
            with self._tx() as cur:
                cur.execute("SELECT COUNT(*) FROM kg_edges")
                count += cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM kg_nodes")
                count += cur.fetchone()[0]
                cur.execute("DELETE FROM kg_edges")
                cur.execute("DELETE FROM kg_nodes")
            # 失效全部缓存
            self._invalidate_cache()
            return count

    async def get_stats(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """获取统计信息"""
        async with self._lock:
            assert self._conn
            params: List[Any] = []
            node_where = ""
            edge_where = ""
            if user_id:
                node_where = " WHERE user_id = ?"
                edge_where = " WHERE user_id = ?"
                params = [user_id]
                if group_id:
                    node_where += " AND group_id = ?"
                    edge_where += " AND group_id = ?"
                    params.append(group_id)

            node_count = self._conn.execute(
                f"SELECT COUNT(*) FROM kg_nodes{node_where}", params
            ).fetchone()[0]
            edge_count = self._conn.execute(
                f"SELECT COUNT(*) FROM kg_edges{edge_where}", params
            ).fetchone()[0]

            return {"nodes": node_count, "edges": edge_count}

    # ================================================================
    # 缓存方法
    # ================================================================

    def _node_cache_key(self, name: str, user_id: str, group_id: Optional[str]) -> str:
        """生成节点缓存 key"""
        normalized = self._normalize_name(name)
        return f"{normalized}:{user_id}:{group_id or ''}"

    def _get_from_cache(self, key: str) -> Optional[KGNode]:
        """从缓存获取节点，返回 None 表示未命中或已过期"""
        entry = self._node_cache.get(key)
        if entry is None:
            return None
        node, ts = entry
        if time.monotonic() - ts > self._cache_ttl:
            del self._node_cache[key]
            return None
        return node

    def _put_to_cache(self, key: str, node: KGNode) -> None:
        """写入缓存，超过容量时淘汰最旧条目"""
        if len(self._node_cache) >= self._cache_max_size:
            # 淘汰最旧的10%条目
            items = sorted(self._node_cache.items(), key=lambda x: x[1][1])
            for k, _ in items[:max(1, self._cache_max_size // 10)]:
                del self._node_cache[k]
        self._node_cache[key] = (node, time.monotonic())

    def _invalidate_cache(self, name: Optional[str] = None, user_id: Optional[str] = None,
                          group_id: Optional[str] = None) -> None:
        """失效缓存条目"""
        if name and user_id:
            key = self._node_cache_key(name, user_id, group_id)
            self._node_cache.pop(key, None)
        elif user_id:
            keys_to_remove = [k for k in self._node_cache if f":{user_id}:" in k]
            for k in keys_to_remove:
                del self._node_cache[k]
        else:
            self._node_cache.clear()

    # ================================================================
    # 图遍历基础（供 reasoning 模块调用）
    # ================================================================

    async def get_node_by_id(self, node_id: str) -> Optional[KGNode]:
        """按 ID 获取节点（无锁版，供内部使用）"""
        return await self.get_node(node_id)

    def get_node_by_id_sync(self, node_id: str) -> Optional[KGNode]:
        """同步按 ID 获取节点"""
        assert self._conn
        row = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return KGNode.from_row(dict(row)) if row else None

    def get_neighbors_sync(self, node_id: str, limit: int = 50) -> List[Tuple[KGEdge, KGNode]]:
        """同步获取邻居（供 BFS 使用）"""
        return self._get_neighbors_sync(node_id, limit)

    # ================================================================
    # 维护支持方法（供 kg_maintenance / kg_consistency / kg_quality 使用）
    # ================================================================

    async def get_all_nodes(self, limit: int = 10000) -> List[KGNode]:
        """获取所有节点（供维护/质量检测使用）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT * FROM kg_nodes LIMIT ?", (limit,)
            ).fetchall()
            return [KGNode.from_row(dict(r)) for r in rows]

    async def get_all_edges(self, limit: int = 50000) -> List[KGEdge]:
        """获取所有边（供维护/质量检测使用）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT * FROM kg_edges LIMIT ?", (limit,)
            ).fetchall()
            return [KGEdge.from_row(dict(r)) for r in rows]

    async def get_orphan_node_ids(self) -> List[str]:
        """获取孤立节点 ID（无任何边连接的节点）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                """SELECT n.id FROM kg_nodes n
                   WHERE NOT EXISTS (
                       SELECT 1 FROM kg_edges e WHERE e.source_id = n.id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM kg_edges e WHERE e.target_id = n.id
                   )"""
            ).fetchall()
            return [r[0] for r in rows]

    async def get_dangling_edges(self) -> List[KGEdge]:
        """获取悬空边（指向不存在节点的边）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                """SELECT e.* FROM kg_edges e
                   WHERE NOT EXISTS (
                       SELECT 1 FROM kg_nodes n WHERE n.id = e.source_id
                   )
                   OR NOT EXISTS (
                       SELECT 1 FROM kg_nodes n WHERE n.id = e.target_id
                   )"""
            ).fetchall()
            return [KGEdge.from_row(dict(r)) for r in rows]

    async def get_self_referencing_edges(self) -> List[KGEdge]:
        """获取自引用边（source_id == target_id）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT * FROM kg_edges WHERE source_id = target_id"
            ).fetchall()
            return [KGEdge.from_row(dict(r)) for r in rows]

    async def get_edges_between(self, source_id: str, target_id: str) -> List[KGEdge]:
        """获取两个节点之间的所有边（双向）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                """SELECT * FROM kg_edges
                   WHERE (source_id = ? AND target_id = ?)
                      OR (source_id = ? AND target_id = ?)""",
                (source_id, target_id, target_id, source_id),
            ).fetchall()
            return [KGEdge.from_row(dict(r)) for r in rows]

    async def delete_nodes_by_ids(self, node_ids: List[str]) -> int:
        """批量删除节点"""
        if not node_ids:
            return 0
        async with self._lock:
            with self._tx() as cur:
                placeholders = ",".join("?" * len(node_ids))
                cur.execute(
                    f"DELETE FROM kg_nodes WHERE id IN ({placeholders})",
                    node_ids,
                )
                count = cur.rowcount
            self._invalidate_cache()
            return count

    async def delete_edges_by_ids(self, edge_ids: List[str]) -> int:
        """批量删除边"""
        if not edge_ids:
            return 0
        async with self._lock:
            with self._tx() as cur:
                placeholders = ",".join("?" * len(edge_ids))
                cur.execute(
                    f"DELETE FROM kg_edges WHERE id IN ({placeholders})",
                    edge_ids,
                )
                return cur.rowcount

    async def get_node_ids_set(self) -> set:
        """获取所有节点 ID 集合（高效用于存在性检查）"""
        async with self._lock:
            assert self._conn
            rows = self._conn.execute("SELECT id FROM kg_nodes").fetchall()
            return {r[0] for r in rows}

    # ================================================================
    # SQL 聚合方法（避免全量加载到 Python 内存）
    # ================================================================

    async def get_node_count(self) -> int:
        """获取节点总数"""
        async with self._lock:
            assert self._conn
            row = self._conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()
            return row[0] if row else 0

    async def get_edge_count(self) -> int:
        """获取边总数"""
        async with self._lock:
            assert self._conn
            row = self._conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()
            return row[0] if row else 0

    async def get_avg_confidence(self) -> Dict[str, float]:
        """使用 SQL 聚合计算平均置信度

        Returns:
            {"nodes": 平均节点置信度, "edges": 平均边置信度}
        """
        async with self._lock:
            assert self._conn
            nr = self._conn.execute(
                "SELECT AVG(confidence) FROM kg_nodes"
            ).fetchone()
            er = self._conn.execute(
                "SELECT AVG(confidence) FROM kg_edges"
            ).fetchone()
            return {
                "nodes": nr[0] if nr and nr[0] is not None else 0.0,
                "edges": er[0] if er and er[0] is not None else 0.0,
            }

    async def get_low_confidence_counts(self, threshold: float = 0.3) -> Dict[str, int]:
        """使用 SQL 计算低置信度节点和边数量

        Args:
            threshold: 低置信度阈值

        Returns:
            {"nodes": 低置信度节点数, "edges": 低置信度边数}
        """
        async with self._lock:
            assert self._conn
            nr = self._conn.execute(
                "SELECT COUNT(*) FROM kg_nodes WHERE confidence < ?",
                (threshold,),
            ).fetchone()
            er = self._conn.execute(
                "SELECT COUNT(*) FROM kg_edges WHERE confidence < ?",
                (threshold,),
            ).fetchone()
            return {
                "nodes": nr[0] if nr else 0,
                "edges": er[0] if er else 0,
            }

    async def get_node_type_distribution(self) -> Dict[str, int]:
        """使用 SQL GROUP BY 统计节点类型分布

        Returns:
            {node_type_value: count}
        """
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT node_type, COUNT(*) as cnt FROM kg_nodes GROUP BY node_type"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    async def get_relation_type_distribution(self) -> Dict[str, int]:
        """使用 SQL GROUP BY 统计关系类型分布

        Returns:
            {relation_type_value: count}
        """
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT relation_type, COUNT(*) as cnt FROM kg_edges GROUP BY relation_type"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    async def get_low_confidence_stale_edge_ids(
        self,
        confidence_threshold: float = 0.2,
        cutoff_iso: str = "",
    ) -> List[str]:
        """使用 SQL 获取低置信度且过期的边 ID 列表

        Args:
            confidence_threshold: 置信度阈值
            cutoff_iso: ISO 格式的截止时间字符串

        Returns:
            待清理的边 ID 列表
        """
        async with self._lock:
            assert self._conn
            rows = self._conn.execute(
                "SELECT id FROM kg_edges WHERE confidence < ? AND updated_time < ?",
                (confidence_threshold, cutoff_iso),
            ).fetchall()
            return [r[0] for r in rows]

    async def detect_contradictions_sql(
        self,
        relation_pairs: List[tuple],
    ) -> List[Dict[str, Any]]:
        """使用 SQL JOIN 检测矛盾关系对

        Args:
            relation_pairs: [(rel_a, rel_b), ...] 互斥的关系类型值对

        Returns:
            匹配的矛盾边对信息列表
        """
        if not relation_pairs:
            return []

        async with self._lock:
            assert self._conn
            results: List[Dict[str, Any]] = []
            for rel_a, rel_b in relation_pairs:
                rows = self._conn.execute(
                    """SELECT e1.id as e1_id, e2.id as e2_id,
                              e1.source_id, e1.target_id,
                              e1.relation_type as rel_a, e2.relation_type as rel_b
                       FROM kg_edges e1
                       JOIN kg_edges e2
                         ON e1.source_id = e2.source_id
                        AND e1.target_id = e2.target_id
                       WHERE e1.relation_type = ? AND e2.relation_type = ?
                         AND e1.id < e2.id""",
                    (rel_a, rel_b),
                ).fetchall()
                for r in rows:
                    results.append(dict(r))
            return results

    # ================================================================
    # 分批迭代方法（大图谱内存优化）
    # ================================================================

    async def iter_nodes_batch(self, batch_size: int = 500) -> List[List[KGNode]]:
        """分批获取所有节点

        对于需要遍历所有节点的场景，分批加载减少内存峰值。

        Args:
            batch_size: 每批数量

        Returns:
            节点批次列表
        """
        batches: List[List[KGNode]] = []
        async with self._lock:
            assert self._conn
            offset = 0
            while True:
                rows = self._conn.execute(
                    "SELECT * FROM kg_nodes LIMIT ? OFFSET ?",
                    (batch_size, offset),
                ).fetchall()
                if not rows:
                    break
                batches.append([KGNode.from_row(dict(r)) for r in rows])
                offset += batch_size
        return batches

    async def iter_edges_batch(self, batch_size: int = 500) -> List[List[KGEdge]]:
        """分批获取所有边

        Args:
            batch_size: 每批数量

        Returns:
            边批次列表
        """
        batches: List[List[KGEdge]] = []
        async with self._lock:
            assert self._conn
            offset = 0
            while True:
                rows = self._conn.execute(
                    "SELECT * FROM kg_edges LIMIT ? OFFSET ?",
                    (batch_size, offset),
                ).fetchall()
                if not rows:
                    break
                batches.append([KGEdge.from_row(dict(r)) for r in rows])
                offset += batch_size
        return batches

    # ================================================================
    # 工具方法
    # ================================================================

    @staticmethod
    def _normalize_name(name: str) -> str:
        """规范化实体名称：去首尾空格、统一全角半角、小写"""
        name = name.strip()
        # 全角转半角（常见标点）
        name = name.replace("，", ",").replace("。", ".").replace("！", "!")
        return name.lower()

    @staticmethod
    def _is_chinese(text: str) -> bool:
        """判断文本是否主要为中文"""
        if not text:
            return False
        cn_count = len(_CN_CHAR_RE.findall(text))
        return cn_count > 0 and cn_count / len(text) > 0.3

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """构建 FTS5 查询表达式

        对于英文，按空格切词并用 OR 连接以提高召回。
        """
        tokens = query.strip().split()
        if not tokens:
            return ""
        # 转义双引号
        safe_tokens = [t.replace('"', '""') for t in tokens if t]
        if len(safe_tokens) == 1:
            return f'"{safe_tokens[0]}"'
        return " OR ".join(f'"{t}"' for t in safe_tokens)
