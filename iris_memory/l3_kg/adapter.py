"""SQLite 图谱适配器"""

import json
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

from iris_memory.config import get_config
from iris_memory.core import Component, InitMode, get_logger
from .models import GraphEdge, GraphNode

logger = get_logger("l3_kg")


async def build_profile_alias_map(profile_storage, persona_id: str = "default") -> Dict[str, List[str]]:
    """从用户画像构建 {user_id: [昵称/曾用名...]} 别名映射

    供 merge_person_nodes_by_user_id 做存量 Person 节点归一化修复：
    昵称命名且无 user_id 标记的节点按映射吸收/打标。数据来源为
    list_all_users（含当前昵称）与各用户画像的 historical_names。
    画像不可用或无数据时返回空映射（修复退化为仅按节点自带别名）。
    """
    alias_map: Dict[str, List[str]] = {}
    try:
        users = await profile_storage.list_all_users(persona_id)
        for entry in users:
            user_id = entry.get("user_id")
            if not user_id:
                continue
            names = alias_map.setdefault(user_id, [])
            nickname = entry.get("nickname")
            if nickname and nickname != user_id and nickname not in names:
                names.append(nickname)
            try:
                profile = await profile_storage.get_user_profile(
                    user_id, entry.get("group_id") or "default", persona_id
                )
            except Exception as e:
                logger.debug(f"读取用户画像失败：{user_id}, {e}")
                profile = None
            if profile:
                for hist in getattr(profile, "historical_names", None) or []:
                    if hist and hist != user_id and hist not in names:
                        names.append(hist)
        return alias_map
    except Exception as e:
        logger.warning(f"从画像构建别名映射失败：{e}")
        return {}


class L3KGAdapter(Component):
    """SQLite 图谱适配器

    使用 SQLite 存储实体关系图谱。
    支持：
    - 动态节点类型（通过 label 字段）
    - 动态关系类型（通过 relation_type 字段）
    - JSON 存储扩展属性
    - BFS 路径扩展检索
    """

    def __init__(self):
        super().__init__()
        self._init_mode = InitMode.EAGER
        self._db_lock = threading.RLock()

    @property
    def name(self) -> str:
        return "l3_kg"

    async def initialize(self) -> None:
        """初始化 SQLite 数据库"""
        config = get_config()

        if not config.get("l3_kg.enable"):
            logger.info("L3 知识图谱未启用")
            self._is_available = False
            return

        self._persist_dir = config.data_dir / "l3_graph"
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            db_path = self._persist_dir / "graph.db"
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            with self._db_lock:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA foreign_keys=ON")
                self._create_schema_unlocked()

            self._is_available = True
            logger.info(f"SQLite 图谱初始化成功：{self._persist_dir}")
        except Exception as e:
            logger.error(f"SQLite 图谱初始化失败：{e}")
            self._is_available = False

    def _create_schema_unlocked(self):
        """创建数据库表结构（调用方需持有 _db_lock）"""
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_access_time TEXT,
                created_time TEXT,
                source_memory_id TEXT,
                group_id TEXT,
                persona_id TEXT NOT NULL DEFAULT 'default',
                properties TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_access_time TEXT,
                created_time TEXT,
                source_memory_id TEXT,
                persona_id TEXT NOT NULL DEFAULT 'default',
                properties TEXT DEFAULT '{}',
                PRIMARY KEY (source_id, target_id, relation_type),
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_label_name ON nodes(label, name);
            CREATE INDEX IF NOT EXISTS idx_nodes_group ON nodes(group_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_source_mem ON nodes(source_memory_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        """)
        # 增量迁移：旧图谱无 persona_id 时归入 default 命名空间。
        node_cols = {
            row[1] for row in self._db.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "persona_id" not in node_cols:
            self._db.execute(
                "ALTER TABLE nodes ADD COLUMN persona_id TEXT NOT NULL DEFAULT 'default'"
            )
        edge_cols = {
            row[1] for row in self._db.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "persona_id" not in edge_cols:
            self._db.execute(
                "ALTER TABLE edges ADD COLUMN persona_id TEXT NOT NULL DEFAULT 'default'"
            )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_persona ON nodes(persona_id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_persona ON edges(persona_id)"
        )
        self._db.commit()
        logger.debug("SQLite schema 创建完成")

    async def add_node(self, node: GraphNode) -> bool:
        """添加节点，如果节点已存在则合并信息

        合并策略：
        - content: 追加新描述（去重）
        - confidence: 取最大值
        - access_count: 保留原值
        - source_memory_id: 合并到 properties.source_memory_ids
        - group_id: 合并到 properties.group_ids
        - properties: 合并

        Args:
            node: 图谱节点对象

        Returns:
            成功返回 True，失败返回 False
        """
        if not self._is_available:
            return False

        try:
            existing = self._get_node_by_id(node.id)

            if existing:
                merged_content = self._merge_node_content(
                    existing["content"], node.content
                )
                merged_confidence = max(existing["confidence"], node.confidence)
                existing_access_count = existing["access_count"]

                merged_properties = dict(existing["properties"])
                merged_properties.update(node.properties)

                existing_ids = merged_properties.get("source_memory_ids", "")
                existing_list = [
                    x.strip() for x in existing_ids.split(",") if x.strip()
                ]
                if node.source_memory_id and node.source_memory_id not in existing_list:
                    existing_list.append(node.source_memory_id)
                merged_properties["source_memory_ids"] = ",".join(existing_list)

                existing_groups = merged_properties.get("group_ids", "")
                group_list = [
                    x.strip() for x in existing_groups.split(",") if x.strip()
                ]
                existing_group_id = existing.get("group_id", "")
                if existing_group_id and existing_group_id not in group_list:
                    group_list.insert(0, existing_group_id)
                if node.group_id and node.group_id not in group_list:
                    group_list.append(node.group_id)
                merged_properties["group_ids"] = ",".join(group_list)

                # aliases 并集合并：update 会用本批次的单一昵称直接覆盖历史累积
                # 的别名列表，导致旧昵称丢失、别名检索失效。此处从原始 existing
                # 属性读取历史别名并去重追加本批次别名。
                merged_alias_list = [
                    a.strip()
                    for a in existing["properties"].get("aliases", "").split(",")
                    if a.strip()
                ]
                for alias in [
                    a.strip()
                    for a in node.properties.get("aliases", "").split(",")
                    if a.strip()
                ]:
                    if alias not in merged_alias_list:
                        merged_alias_list.append(alias)
                if merged_alias_list:
                    merged_properties["aliases"] = ",".join(merged_alias_list)

                self._db_write(
                    """UPDATE nodes SET
                        content = ?, confidence = ?, access_count = ?,
                        last_access_time = ?, group_id = ?, persona_id = ?, properties = ?
                    WHERE id = ?""",
                    (
                        merged_content,
                        merged_confidence,
                        existing_access_count,
                        datetime.now().isoformat(),
                        node.group_id or existing.get("group_id", ""),
                        node.persona_id,
                        json.dumps(merged_properties, ensure_ascii=False),
                        node.id,
                    ),
                )
                logger.debug(f"节点合并成功：{node.id}")
            else:
                properties = dict(node.properties)
                if node.source_memory_id:
                    properties["source_memory_ids"] = node.source_memory_id
                if node.group_id:
                    properties["group_ids"] = node.group_id

                self._db_write(
                    """INSERT INTO nodes
                        (id, label, name, content, confidence, access_count,
                         last_access_time, created_time, source_memory_id,
                         group_id, persona_id, properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        node.id,
                        node.label,
                        node.name,
                        node.content,
                        node.confidence,
                        node.access_count,
                        (node.last_access_time or datetime.now()).isoformat(),
                        node.created_time.isoformat()
                        if isinstance(node.created_time, datetime)
                        else node.created_time,
                        node.source_memory_id,
                        node.group_id,
                        node.persona_id,
                        json.dumps(properties, ensure_ascii=False),
                    ),
                )
                logger.debug(f"节点添加成功：{node.id}")

            return True
        except Exception as e:
            logger.error(f"添加节点失败：{e}")
            return False

    def _merge_node_content(self, existing_content: str, new_content: str) -> str:
        """合并节点内容描述

        将新描述追加到已有描述后，对按分号拆分的子句进行去重。
        """
        if not existing_content:
            return new_content
        if not new_content:
            return existing_content

        if new_content in existing_content:
            return existing_content
        if existing_content in new_content:
            return new_content

        existing_parts = [p.strip() for p in existing_content.split("；") if p.strip()]
        new_parts = [p.strip() for p in new_content.split("；") if p.strip()]

        deduped_parts = list(existing_parts)
        for new_part in new_parts:
            is_dup = False
            for existing_part in existing_parts:
                if new_part in existing_part or existing_part in new_part:
                    if len(new_part) <= len(existing_part):
                        is_dup = True
                        break
            if not is_dup:
                deduped_parts.append(new_part)

        return "；".join(deduped_parts)

    def _get_node_by_id(self, node_id: str) -> Optional[dict]:
        """根据 ID 获取节点"""
        try:
            row = self._db_fetchone(
                """SELECT content, confidence, access_count, group_id, properties
                FROM nodes WHERE id = ?""",
                (node_id,),
            )

            if row:
                props = row["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                return {
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "access_count": row["access_count"],
                    "group_id": row["group_id"],
                    "properties": props if isinstance(props, dict) else {},
                }
            return None
        except Exception as e:
            logger.debug(f"查询节点失败：{e}")
            return None

    async def add_edge(self, edge: GraphEdge) -> bool:
        """添加关系边

        如果边已存在，会合并信息：
        - source_memory_ids: 追加新的来源记忆ID
        - weight: 累加权重（最大1.0）
        - confidence: 取最大置信度
        """
        if not self._is_available:
            return False

        try:
            existing_edge = self._get_edge(
                edge.source_id, edge.target_id, edge.relation_type
            )

            if existing_edge:
                existing_ids = existing_edge.get("source_memory_ids", "")
                existing_list = [
                    x.strip() for x in existing_ids.split(",") if x.strip()
                ]

                if edge.source_memory_id and edge.source_memory_id not in existing_list:
                    existing_list.append(edge.source_memory_id)

                new_weight = min(
                    1.0, existing_edge.get("weight", 0.5) + edge.weight * 0.1
                )
                new_confidence = max(
                    existing_edge.get("confidence", 0.5), edge.confidence
                )

                merged_properties = dict(existing_edge.get("properties", {}))
                merged_properties.update(edge.properties)
                merged_properties["source_memory_ids"] = ",".join(existing_list)

                self._db_write(
                    """UPDATE edges SET
                        weight = ?, confidence = ?,
                        access_count = access_count + 1,
                        last_access_time = ?, properties = ?
                    WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
                    (
                        new_weight,
                        new_confidence,
                        datetime.now().isoformat(),
                        json.dumps(merged_properties, ensure_ascii=False),
                        edge.source_id,
                        edge.target_id,
                        edge.relation_type,
                    ),
                )
                logger.debug(f"边合并成功：{edge.generate_id()}")
            else:
                properties = dict(edge.properties)
                if edge.source_memory_id:
                    properties["source_memory_ids"] = edge.source_memory_id

                self._db_write(
                    """INSERT OR IGNORE INTO edges
                        (source_id, target_id, relation_type, weight, confidence,
                         access_count, last_access_time, created_time,
                         source_memory_id, persona_id, properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        edge.source_id,
                        edge.target_id,
                        edge.relation_type,
                        edge.weight,
                        edge.confidence,
                        edge.access_count,
                        (edge.last_access_time or datetime.now()).isoformat(),
                        edge.created_time.isoformat()
                        if isinstance(edge.created_time, datetime)
                        else edge.created_time,
                        edge.source_memory_id,
                        edge.persona_id,
                        json.dumps(properties, ensure_ascii=False),
                    ),
                )
                logger.debug(f"边添加成功：{edge.generate_id()}")

            return True
        except Exception as e:
            logger.error(f"添加边失败：{e}")
            return False

    def _get_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> Optional[dict]:
        """获取已存在的边"""
        try:
            row = self._db_fetchone(
                """SELECT weight, confidence, properties
                FROM edges
                WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
                (source_id, target_id, relation_type),
            )

            if row:
                props = row["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                props = props if isinstance(props, dict) else {}
                return {
                    "weight": row["weight"],
                    "confidence": row["confidence"],
                    "properties": props,
                    "source_memory_ids": props.get("source_memory_ids", ""),
                }
            return None
        except Exception as e:
            logger.debug(f"查询边失败：{e}")
            return None

    async def expand_from_nodes(
        self,
        node_ids: list[str],
        max_depth: int = 2,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        max_nodes: int = 100,
        max_edges: int = 200,
    ) -> tuple[list[dict], list[dict]]:
        """BFS 路径扩展检索"""
        if not self._is_available:
            return [], []

        try:
            visited = set(node_ids)
            frontier = list(node_ids)
            nodes_map: dict[str, dict] = {}
            edges_list: list[dict] = []

            # 种子节点也必须按 group_id 过滤，否则跨群节点作为种子泄漏
            seed_conditions = [f"id IN ({','.join('?' * len(node_ids))})"]
            seed_params: list = [*node_ids]
            if group_id:
                seed_conditions.append("group_id = ?")
                seed_params.append(group_id)
            if persona_id is not None:
                seed_conditions.append("persona_id = ?")
                seed_params.append(persona_id)
            seed_rows = self._db_fetchall(
                f"SELECT * FROM nodes WHERE {' AND '.join(seed_conditions)}",
                seed_params,
            )
            for row in seed_rows:
                nodes_map[row["id"]] = dict(row)

            for _ in range(max_depth):
                if (
                    not frontier
                    or len(nodes_map) >= max_nodes
                    or len(edges_list) >= max_edges
                ):
                    break

                placeholders = ",".join("?" * len(frontier))

                if group_id:
                    query = f"""
                        SELECT e.source_id, e.target_id, e.relation_type,
                               e.weight, e.confidence, e.access_count,
                               e.last_access_time, e.created_time,
                               e.source_memory_id, e.persona_id, e.properties
                        FROM edges e
                        WHERE (e.source_id IN ({placeholders})
                               OR e.target_id IN ({placeholders}))
                        AND (
                            e.source_id IN (SELECT id FROM nodes WHERE group_id = ?)
                            AND e.target_id IN (SELECT id FROM nodes WHERE group_id = ?)
                        )
                    """
                    query += " AND e.persona_id = ?" if persona_id is not None else ""
                    params = [*frontier, *frontier, group_id, group_id]
                    if persona_id is not None:
                        params.append(persona_id)
                    rows = self._db_fetchall(query, params)
                else:
                    query = f"""
                        SELECT source_id, target_id, relation_type,
                               weight, confidence, access_count,
                               last_access_time, created_time,
                               source_memory_id, persona_id, properties
                        FROM edges
                        WHERE (source_id IN ({placeholders})
                           OR target_id IN ({placeholders}))
                    """
                    if persona_id is not None:
                        query += " AND persona_id = ?"
                    params = [*frontier, *frontier]
                    if persona_id is not None:
                        params.append(persona_id)
                    rows = self._db_fetchall(query, params)

                next_frontier = []
                frontier_set = set(frontier)
                seen_edge_keys: set[tuple[str, str, str]] = {
                    (e["source"], e["target"], e["relation_type"])
                    for e in edges_list
                }

                for row in rows:
                    if len(edges_list) >= max_edges:
                        break

                    source_id = row["source_id"]
                    target_id = row["target_id"]
                    relation_type = row["relation_type"]

                    # 跨层去重：同一条边可能在多个 depth 的 frontier 中被重复查出
                    edge_key = (source_id, target_id, relation_type)
                    if edge_key in seen_edge_keys:
                        continue
                    seen_edge_keys.add(edge_key)

                    edge_props = row["properties"]
                    if isinstance(edge_props, str):
                        edge_props = json.loads(edge_props)

                    edges_list.append(
                        {
                            "source": source_id,
                            "target": target_id,
                            "_src": source_id,
                            "_dst": target_id,
                            "relation_type": row["relation_type"],
                            "weight": row["weight"],
                            "confidence": row["confidence"],
                            "access_count": row["access_count"],
                            "last_access_time": row["last_access_time"],
                            "created_time": row["created_time"],
                            "source_memory_id": row["source_memory_id"],
                            "persona_id": row["persona_id"],
                            "properties": edge_props
                            if isinstance(edge_props, dict)
                            else {},
                        }
                    )

                    neighbor_id = target_id if source_id in frontier_set else source_id
                    if neighbor_id not in visited and len(nodes_map) < max_nodes:
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)

                if next_frontier:
                    ph = ",".join("?" * len(next_frontier))
                    neighbor_query = f"SELECT * FROM nodes WHERE id IN ({ph})"
                    neighbor_params: list = [*next_frontier]
                    if persona_id is not None:
                        neighbor_query += " AND persona_id = ?"
                        neighbor_params.append(persona_id)
                    neighbor_rows = self._db_fetchall(
                        neighbor_query, neighbor_params
                    )
                    for node_row in neighbor_rows:
                        nodes_map[node_row["id"]] = dict(node_row)

                frontier = next_frontier

            nodes = list(nodes_map.values())

            seen = set()
            unique_edges = []
            for e in edges_list:
                key = f"{e['source']}-{e['relation_type']}-{e['target']}"
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)
            edges = unique_edges

            logger.info(f"路径扩展检索完成：{len(nodes)} 个节点，{len(edges)} 条边")
            return nodes, edges
        except Exception as e:
            logger.error(f"路径扩展检索失败：{e}")
            return [], []

    async def update_node_access(self, node_ids: list[str]) -> None:
        """更新节点访问计数和最后访问时间"""
        if not self._is_available:
            return

        try:
            now = datetime.now().isoformat()
            with self._db_lock:
                for node_id in node_ids:
                    self._db.execute(
                        """UPDATE nodes SET access_count = access_count + 1,
                            last_access_time = ? WHERE id = ?""",
                        (now, node_id),
                    )
                self._db.commit()
            logger.debug(f"更新了 {len(node_ids)} 个节点的访问计数")
        except Exception as e:
            logger.error(f"更新节点访问计数失败：{e}")

    async def get_stats(self) -> dict:
        """获取图谱统计信息"""
        if not self._is_available:
            return {
                "available": False,
                "node_count": 0,
                "edge_count": 0,
                "node_types": {},
                "relation_types": {},
            }

        try:
            node_count = self._db_fetchone("SELECT COUNT(*) FROM nodes")[0]
            edge_count = self._db_fetchone("SELECT COUNT(*) FROM edges")[0]

            node_types = {}
            for row in self._db_fetchall(
                "SELECT label, COUNT(*) as cnt FROM nodes GROUP BY label"
            ):
                if row["label"]:
                    node_types[row["label"]] = row["cnt"]

            relation_types = {}
            for row in self._db_fetchall(
                "SELECT relation_type, COUNT(*) as cnt FROM edges GROUP BY relation_type"
            ):
                if row["relation_type"]:
                    relation_types[row["relation_type"]] = row["cnt"]

            return {
                "available": True,
                "node_count": node_count,
                "edge_count": edge_count,
                "node_types": node_types,
                "relation_types": relation_types,
                "persist_dir": str(self._persist_dir),
            }
        except Exception as e:
            logger.error(f"获取图谱统计失败：{e}")
            return {
                "available": False,
                "node_count": 0,
                "edge_count": 0,
                "node_types": {},
                "relation_types": {},
            }

    async def get_all_nodes(
        self,
        limit: int = 100,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> list[dict]:
        """获取节点（用于前端展示）

        Args:
            limit: 最大返回数量
            group_id: 群聊ID，传入时仅返回该群节点，None 则返回所有群节点
        """
        if not self._is_available:
            return []

        try:
            conditions: list[str] = []
            params: list = []
            if group_id:
                conditions.append("group_id = ?")
                params.append(group_id)
            if persona_id is not None:
                conditions.append("persona_id = ?")
                params.append(persona_id)
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)
            rows = self._db_fetchall(
                f"""SELECT id, label, name, content, confidence,
                           access_count, last_access_time, created_time,
                           source_memory_id, group_id, persona_id, properties
                    FROM nodes{where} LIMIT ?""",
                params,
            )

            nodes = []
            for row in rows:
                props = row["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                nodes.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "name": row["name"],
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "access_count": row["access_count"],
                        "last_access_time": row["last_access_time"],
                        "created_time": row["created_time"],
                        "source_memory_id": row["source_memory_id"],
                        "group_id": row["group_id"],
                        "persona_id": row["persona_id"],
                        "properties": props,
                    }
                )

            logger.debug(f"获取到 {len(nodes)} 个节点")
            return nodes
        except Exception as e:
            logger.error(f"获取所有节点失败：{e}")
            return []

    async def search_nodes(
        self,
        keyword: str,
        limit: int = 20,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> list[dict]:
        """搜索节点（匹配 name 或 content）

        Args:
            keyword: 搜索关键词
            limit: 最大返回数量
            group_id: 群聊ID，传入时仅返回该群的节点，None 则返回所有群的节点
        """
        if not self._is_available:
            return []

        try:
            pattern = f"%{keyword}%"
            conditions = ["(name LIKE ? OR content LIKE ? OR properties LIKE ?)"]
            params: list = [pattern, pattern, pattern]
            if group_id:
                conditions.append("group_id = ?")
                params.append(group_id)
            if persona_id is not None:
                conditions.append("persona_id = ?")
                params.append(persona_id)
            params.append(limit)
            rows = self._db_fetchall(
                f"""SELECT id, label, name, content, confidence, persona_id
                    FROM nodes WHERE {' AND '.join(conditions)} LIMIT ?""",
                params,
            )

            nodes = [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "name": row["name"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "persona_id": row["persona_id"],
                }
                for row in rows
            ]

            logger.debug(f"搜索节点 '{keyword}' 找到 {len(nodes)} 个结果")
            return nodes
        except Exception as e:
            logger.error(f"搜索节点失败：{e}")
            return []

    async def search_nodes_detailed(
        self,
        query: str,
        label: Optional[str] = None,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """搜索节点（完整信息，支持 label 和 group_id 过滤）

        用于 LLM Tool 搜索知识图谱。
        """
        if not self._is_available:
            return []

        try:
            pattern = f"%{query}%"
            conditions = ["(name LIKE ? OR content LIKE ? OR properties LIKE ?)"]
            params: list = [pattern, pattern, pattern]

            if label:
                conditions.append("label = ?")
                params.append(label)

            if group_id:
                conditions.append("group_id = ?")
                params.append(group_id)

            if persona_id is not None:
                conditions.append("persona_id = ?")
                params.append(persona_id)

            where = " AND ".join(conditions)
            params.append(limit)

            rows = self._db_fetchall(
                f"""SELECT id, label, name, content, confidence, group_id, persona_id
                    FROM nodes WHERE {where} LIMIT ?""",
                params,
            )

            return [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "name": row["name"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "group_id": row["group_id"],
                    "persona_id": row["persona_id"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"搜索节点失败：{e}")
            return []

    async def update_node_content_by_source_memory(
        self, memory_id: str, new_content: str, new_source_memory_id: str = None
    ) -> Optional[str]:
        """根据 source_memory_id 更新节点内容

        用于记忆修正 Tool，返回更新的节点 ID 或 None。
        可选传入 new_source_memory_id 将节点的 source_memory_id 重指向新记忆，
        避免 L2↔L3 引用断裂（旧记忆已删除，source_memory_id 指向已删 ID）。
        """
        if not self._is_available:
            return None

        try:
            # 先查 source_memory_id 列（单来源节点）
            row = self._db_fetchone(
                "SELECT id FROM nodes WHERE source_memory_id = ?",
                (memory_id,),
            )

            # 列未命中时，回退查 properties.source_memory_ids（批量提取的
            # 多来源节点，source_memory_id 列仅存逗号连接的完整列表或首条 ID）
            if not row:
                rows = self._db_fetchall(
                    "SELECT id, properties FROM nodes WHERE properties LIKE ?",
                    (f'%"{memory_id}"%',),
                )
                for r in rows:
                    try:
                        props = json.loads(r["properties"]) if isinstance(
                            r["properties"], str
                        ) else r["properties"]
                        if not isinstance(props, dict):
                            continue
                        ids_str = props.get("source_memory_ids", "")
                        if ids_str:
                            ids_list = [
                                s.strip() for s in ids_str.split(",") if s.strip()
                            ]
                            if memory_id in ids_list:
                                row = r
                                break
                    except (json.JSONDecodeError, TypeError):
                        continue

            if not row:
                return None

            node_id = row["id"]
            props_str = self._db_fetchone(
                "SELECT properties FROM nodes WHERE id = ?", (node_id,)
            )["properties"]
            props = json.loads(props_str) if isinstance(props_str, str) else props_str
            if not isinstance(props, dict):
                props = {}
            props["corrected"] = "true"
            props["correction_time"] = datetime.now().isoformat()

            if new_source_memory_id:
                self._db_write(
                    """UPDATE nodes SET content = ?, confidence = 1.0,
                       source_memory_id = ?, properties = ?
                    WHERE id = ?""",
                    (
                        new_content,
                        new_source_memory_id,
                        json.dumps(props, ensure_ascii=False),
                        node_id,
                    ),
                )
            else:
                self._db_write(
                    """UPDATE nodes SET content = ?, confidence = 1.0, properties = ?
                    WHERE id = ?""",
                    (new_content, json.dumps(props, ensure_ascii=False), node_id),
                )
            logger.info(f"已更新图谱节点: node_id={node_id}")
            return node_id
        except Exception as e:
            logger.warning(f"根据来源记忆更新节点失败：{e}")
            return None

    async def get_node_ids_by_source_memory_ids(
        self, memory_ids: list[str], persona_id: Optional[str] = None
    ) -> list[str]:
        """根据来源记忆 ID 反向查找图谱节点 ID"""
        if not self._is_available or not memory_ids:
            return []

        try:
            placeholders = ",".join("?" * len(memory_ids))
            persona_clause = " AND persona_id = ?" if persona_id is not None else ""
            params = [*memory_ids]
            if persona_id is not None:
                params.append(persona_id)
            rows = self._db_fetchall(
                f"SELECT id FROM nodes WHERE source_memory_id IN ({placeholders})"
                f"{persona_clause}",
                params,
            )

            node_ids: set[str] = {row["id"] for row in rows}

            for mid in memory_ids:
                pattern = f"%{mid}%"
                extra_params: list = [pattern]
                if persona_id is not None:
                    extra_params.append(persona_id)
                extra_rows = self._db_fetchall(
                    "SELECT id, properties FROM nodes WHERE properties LIKE ?"
                    + (" AND persona_id = ?" if persona_id is not None else ""),
                    extra_params,
                )
                for row in extra_rows:
                    try:
                        props = json.loads(row["properties"])
                        ids_list = [
                            x.strip()
                            for x in props.get("source_memory_ids", "").split(",")
                            if x.strip()
                        ]
                        if mid in ids_list:
                            node_ids.add(row["id"])
                    except Exception:
                        pass

            if node_ids:
                logger.debug(f"根据来源记忆 ID 反向查找找到 {len(node_ids)} 个图谱节点")
            return list(node_ids)
        except Exception as e:
            logger.error(f"根据来源记忆 ID 查找节点失败: {e}")
            return []

    async def search_edges(self, keyword: str, limit: int = 20) -> list[dict]:
        """搜索边（匹配 relation_type）"""
        if not self._is_available:
            return []

        try:
            pattern = f"%{keyword}%"
            rows = self._db_fetchall(
                """SELECT e.source_id, e.target_id, e.relation_type, e.confidence,
                          src.name as src_name, src.label as src_label,
                          tgt.name as tgt_name, tgt.label as tgt_label
                   FROM edges e
                   JOIN nodes src ON e.source_id = src.id
                   JOIN nodes tgt ON e.target_id = tgt.id
                   WHERE e.relation_type LIKE ?
                   LIMIT ?""",
                (pattern, limit),
            )

            return [
                {
                    "source": {
                        "id": row["source_id"],
                        "label": row["src_label"],
                        "name": row["src_name"],
                    },
                    "target": {
                        "id": row["target_id"],
                        "label": row["tgt_label"],
                        "name": row["tgt_name"],
                    },
                    "relation": row["relation_type"],
                    "confidence": row["confidence"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"搜索边失败：{e}")
            return []

    async def get_all_edges(
        self, limit: int = 100, group_id: Optional[str] = None
    ) -> list[dict]:
        """获取所有边（用于前端展示）

        Args:
            limit: 最大返回数量
            group_id: 群聊ID，传入时仅返回该群内的边（两端节点都属于该群）
        """
        if not self._is_available:
            return []

        try:
            if group_id:
                rows = self._db_fetchall(
                    """SELECT e.source_id, e.relation_type, e.confidence,
                              e.weight, e.access_count, e.created_time,
                              src.label as src_label, src.name as src_name,
                              tgt.id as tgt_id, tgt.label as tgt_label, tgt.name as tgt_name
                       FROM edges e
                       JOIN nodes src ON e.source_id = src.id
                       JOIN nodes tgt ON e.target_id = tgt.id
                       WHERE src.group_id = ? AND tgt.group_id = ?
                       LIMIT ?""",
                    (group_id, group_id, limit),
                )
            else:
                rows = self._db_fetchall(
                    """SELECT e.source_id, e.relation_type, e.confidence,
                              e.weight, e.access_count, e.created_time,
                              src.label as src_label, src.name as src_name,
                              tgt.id as tgt_id, tgt.label as tgt_label, tgt.name as tgt_name
                       FROM edges e
                       JOIN nodes src ON e.source_id = src.id
                       JOIN nodes tgt ON e.target_id = tgt.id
                       LIMIT ?""",
                    (limit,),
                )

            return [
                {
                    "source": {
                        "id": row["source_id"],
                        "label": row["src_label"],
                        "name": row["src_name"],
                    },
                    "target": {
                        "id": row["tgt_id"],
                        "label": row["tgt_label"],
                        "name": row["tgt_name"],
                    },
                    "relation": row["relation_type"],
                    "confidence": row["confidence"],
                    "weight": row["weight"],
                    "access_count": row["access_count"],
                    "created_time": row["created_time"],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"获取所有边失败：{e}")
            return []

    async def delete_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> bool:
        """删除指定的边"""
        if not self._is_available:
            return False

        try:
            self._db_write(
                """DELETE FROM edges
                WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
                (source_id, target_id, relation_type),
            )
            logger.info(f"已删除边：{source_id} -[{relation_type}]-> {target_id}")
            return True
        except Exception as e:
            logger.error(f"删除边失败：{e}")
            return False

    async def get_random_person_node(self) -> Optional[dict]:
        if not self._is_available:
            return None

        try:
            import random

            rows = self._db_fetchall(
                """SELECT id, label, name, content, confidence
                FROM nodes WHERE label = 'Person'"""
            )

            nodes = [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "name": row["name"],
                    "content": row["content"],
                    "confidence": row["confidence"],
                }
                for row in rows
            ]

            return random.choice(nodes) if nodes else None
        except Exception as e:
            logger.error(f"获取随机Person节点失败：{e}")
            return None

    async def expand_from_node(
        self,
        node_id: str,
        depth: int = 2,
        max_nodes: int = 50,
        max_edges: int = 100,
        group_id: Optional[str] = None,
    ) -> tuple[list[dict], list[dict]]:
        """从单个节点出发 BFS 扩展

        Args:
            node_id: 起始节点 ID
            depth: BFS 扩展深度（1-3）
            max_nodes: 最大返回节点数
            max_edges: 最大返回边数
            group_id: 群聊ID，传入时仅扩展该群内的节点，None 则跨群扩展
        """
        if not self._is_available:
            return [], []

        depth = min(max(1, depth), 3)

        try:
            nodes_map: dict[str, dict] = {}
            node_ids_to_query = [node_id]
            all_visited = {node_id}

            if group_id:
                seed_row = self._db_fetchone(
                    "SELECT * FROM nodes WHERE id = ? AND group_id = ?",
                    (node_id, group_id),
                )
            else:
                seed_row = self._db_fetchone("SELECT * FROM nodes WHERE id = ?", (node_id,))
            if seed_row:
                nodes_map[node_id] = dict(seed_row)

            for _ in range(depth):
                if len(nodes_map) >= max_nodes or not node_ids_to_query:
                    break

                remaining = max_nodes - len(nodes_map)
                placeholders = ",".join("?" * len(node_ids_to_query))

                if group_id:
                    rows = self._db_fetchall(
                        f"""SELECT DISTINCT n.id, n.label, n.name, n.content, n.confidence
                        FROM edges e
                        JOIN nodes n ON (
                            (e.source_id IN ({placeholders}) AND n.id = e.target_id)
                            OR (e.target_id IN ({placeholders}) AND n.id = e.source_id)
                        )
                        WHERE n.id NOT IN ({",".join("?" * len(all_visited))})
                        AND n.group_id = ?
                        LIMIT ?""",
                        (*node_ids_to_query, *node_ids_to_query, *all_visited, group_id, remaining),
                    )
                else:
                    rows = self._db_fetchall(
                        f"""SELECT DISTINCT n.id, n.label, n.name, n.content, n.confidence
                        FROM edges e
                        JOIN nodes n ON (
                            (e.source_id IN ({placeholders}) AND n.id = e.target_id)
                            OR (e.target_id IN ({placeholders}) AND n.id = e.source_id)
                        )
                        WHERE n.id NOT IN ({",".join("?" * len(all_visited))})
                        LIMIT ?""",
                        (*node_ids_to_query, *node_ids_to_query, *all_visited, remaining),
                    )

                node_ids_to_query = []
                for row in rows:
                    nid = row["id"]
                    if nid not in all_visited:
                        all_visited.add(nid)
                        node_ids_to_query.append(nid)
                        nodes_map[nid] = {
                            "id": nid,
                            "label": row["label"],
                            "name": row["name"],
                            "content": row["content"],
                            "confidence": row["confidence"],
                        }

            edges_list = []
            seen_edges: set[str] = set()

            if nodes_map:
                node_ids = list(nodes_map.keys())
                placeholders = ",".join("?" * len(node_ids))

                edge_rows = self._db_fetchall(
                    f"""SELECT source_id, target_id, relation_type, confidence
                    FROM edges
                    WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
                    LIMIT ?""",
                    (*node_ids, *node_ids, max_edges),
                )

                for row in edge_rows:
                    edge_key = f"{row['source_id']}->{row['target_id']}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges_list.append(
                            {
                                "source": row["source_id"],
                                "target": row["target_id"],
                                "relation": row["relation_type"],
                                "confidence": row["confidence"],
                            }
                        )

            logger.debug(
                f"从节点 {node_id} 拓展深度 {depth}，"
                f"获取 {len(nodes_map)} 节点，{len(edges_list)} 边"
            )
            return list(nodes_map.values()), edges_list
        except Exception as e:
            logger.error(f"从节点拓展失败：{e}")
            return [], []

    async def get_node_connection_counts(self) -> dict[str, int]:
        """获取所有节点的连接数"""
        if not self._is_available:
            return {}

        try:
            rows = self._db_fetchall("""
                SELECT n.id, (
                    SELECT COUNT(*) FROM edges e
                    WHERE e.source_id = n.id OR e.target_id = n.id
                ) as conn_count
                FROM nodes n
            """)

            return {row["id"]: row["conn_count"] for row in rows}
        except Exception as e:
            logger.error(f"获取节点连接数失败：{e}")
            return {}

    async def find_orphaned_subject_nodes(
        self,
        subject_labels: set[str] | None = None,
        max_confidence: float = 0.5,
    ) -> list[dict]:
        """查找无 Person 关联且置信度低于阈值的主体绑定类型节点

        这些节点（如 Preference/Trait/Belief/Goal/Skill）描述的是"某人的属性"，
        但没有任何边连接到 Person 节点，因此无法确定主体是谁。
        仅返回置信度低于 max_confidence 的节点，避免误删高置信度的有用信息。

        Args:
            subject_labels: 需要绑定 Person 的节点类型集合。
                            默认为 Preference/Trait/Belief/Goal/Skill。
            max_confidence: 置信度上限，仅清理低于此值的节点。默认 0.5。

        Returns:
            无主节点列表，每项包含 id/label/name/content/confidence
        """
        if not self._is_available:
            return []

        if subject_labels is None:
            subject_labels = {
                "Preference", "Trait", "Belief", "Goal", "Skill"
            }

        try:
            label_placeholders = ",".join("?" * len(subject_labels))
            labels_list = list(subject_labels)

            rows = self._db_fetchall(f"""
                SELECT n.id, n.label, n.name, n.content, n.confidence
                FROM nodes n
                WHERE n.label IN ({label_placeholders})
                  AND n.confidence < ?
                  AND n.id NOT IN (
                    SELECT CASE
                        WHEN e.source_id IN (
                            SELECT id FROM nodes WHERE label = 'Person'
                        ) THEN e.target_id
                        ELSE e.source_id
                    END
                    FROM edges e
                    WHERE e.source_id IN (SELECT id FROM nodes WHERE label = 'Person')
                       OR e.target_id IN (SELECT id FROM nodes WHERE label = 'Person')
                  )
            """, labels_list + [max_confidence])

            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"查找无主节点失败：{e}")
            return []

    async def update_node_properties(self, node_id: str, properties: dict) -> bool:
        """更新节点扩展属性"""
        if not self._is_available:
            return False

        try:
            self._db_write(
                "UPDATE nodes SET properties = ? WHERE id = ?",
                (json.dumps(properties, ensure_ascii=False), node_id),
            )
            return True
        except Exception as e:
            logger.error(f"更新节点属性失败：{e}")
            return False

    async def evict_nodes(self, node_ids: list[str]) -> int:
        """淘汰节点及关联边（CASCADE 自动删边）"""
        if not self._is_available or not node_ids:
            return 0

        try:
            placeholders = ",".join("?" * len(node_ids))
            self._db_write(f"DELETE FROM nodes WHERE id IN ({placeholders})", node_ids)

            logger.info(f"已淘汰 {len(node_ids)} 个节点及其关联边")
            return len(node_ids)
        except Exception as e:
            logger.error(f"淘汰节点失败：{e}")
            return 0

    async def delete_by_group(self, group_id: str, persona_id: Optional[str] = None) -> int:
        """删除指定群聊的所有节点和边

        跨群共享节点（节点 ID 全局确定，合并写入后 group_id 列可能被
        其他群覆盖）通过 properties.group_ids CSV 匹配：列命中或 CSV
        含该群即删（彻底删除语义）。properties 可能存在损坏 JSON，
        json_valid 守卫避免 json_extract 抛错导致整次删除失败。
        """
        if not self._is_available:
            return 0

        try:
            where = (
                "(group_id = ? OR (json_valid(properties) "
                "AND (',' || json_extract(properties, '$.group_ids') || ',') "
                "LIKE ?))"
            )
            params: list = [group_id, f"%,{group_id},%"]
            if persona_id is not None:
                where += " AND persona_id = ?"
                params.append(persona_id)

            count_row = self._db_fetchone(
                f"SELECT COUNT(*) FROM nodes WHERE {where}", params
            )
            node_count = count_row[0]

            if node_count == 0:
                logger.debug(f"群聊 {group_id} 没有知识图谱节点")
                return 0

            self._db_write(
                f"DELETE FROM nodes WHERE {where}", params
            )

            logger.info(f"已删除群聊 {group_id} 的 {node_count} 个节点及其关联边")
            return node_count
        except Exception as e:
            logger.error(f"删除群聊知识图谱失败: {e}", exc_info=True)
            return 0

    async def delete_all(self) -> int:
        """删除所有节点和边"""
        if not self._is_available:
            return 0

        try:
            count = self._db_fetchone("SELECT COUNT(*) FROM nodes")[0]

            if count == 0:
                return 0

            self._db_writes(
                [
                    ("DELETE FROM edges", ()),
                    ("DELETE FROM nodes", ()),
                ]
            )

            logger.info(f"已删除所有知识图谱节点，共 {count} 个")
            return count
        except Exception as e:
            logger.error(f"删除所有知识图谱失败: {e}", exc_info=True)
            return 0

    async def delete_by_user(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> int:
        """删除与指定用户相关的节点

        命中条件：name == user_id（归一化形态），或
        properties.user_id == user_id（历史标记）。存量节点 properties
        可能损坏，json_valid 守卫避免 json_extract 抛错。群过滤同样
        覆盖跨群共享节点（group_id 列或 properties.group_ids CSV 命中
        即删，彻底删除语义）。count 与 DELETE 使用同一 WHERE。
        """
        if not self._is_available:
            return 0

        try:
            group_match = (
                "(group_id = ? OR (json_valid(properties) "
                "AND (',' || json_extract(properties, '$.group_ids') || ',') "
                "LIKE ?))"
            )
            user_match = (
                "(name = ? OR (json_valid(properties) "
                "AND json_extract(properties, '$.user_id') = ?))"
            )

            conditions = [user_match]
            params: list = [user_id, user_id]
            if group_id:
                conditions.insert(0, group_match)
                params = [group_id, f"%,{group_id},%", user_id, user_id]

            if persona_id is not None:
                conditions.append("persona_id = ?")
                params.append(persona_id)

            where = " AND ".join(conditions)

            count_row = self._db_fetchone(
                f"SELECT COUNT(*) FROM nodes WHERE {where}", params
            )
            node_count = count_row[0]
            if node_count == 0:
                logger.debug(f"用户 {user_id} 没有知识图谱节点")
                return 0

            self._db_write(
                f"DELETE FROM nodes WHERE {where}", params
            )

            logger.info(f"已删除用户 {user_id} 的 {node_count} 个知识图谱节点")
            return node_count
        except Exception as e:
            logger.error(f"删除用户知识图谱失败: {e}", exc_info=True)
            return 0

    async def merge_duplicate_nodes(self) -> tuple[int, int]:
        """合并同名同 label 的重复节点

        Returns:
            (合并的节点组数, 删除的重复节点数)
        """
        if not self._is_available:
            return 0, 0

        with self._db_lock:
            try:
                rows = self._db.execute(
                    """SELECT id, label, name, content, confidence,
                              access_count, created_time, group_id, persona_id, properties
                       FROM nodes"""
                ).fetchall()

                name_groups: dict[str, list[dict]] = {}
                for row in rows:
                    props = row["properties"]
                    if isinstance(props, str):
                        try:
                            props = json.loads(props)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"节点 {row['id']} properties JSON 损坏，已回退为空字典"
                            )
                            props = {}
                    node_data = {
                        "id": row["id"],
                        "label": row["label"],
                        "name": row["name"],
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "access_count": row["access_count"],
                        "created_time": row["created_time"],
                        "group_id": row["group_id"],
                        "persona_id": row["persona_id"],
                        "properties": props if isinstance(props, dict) else {},
                    }
                    key = f"{row['persona_id']}:{row['label']}:{row['name']}"
                    name_groups.setdefault(key, []).append(node_data)

                merged_count = 0
                deleted_count = 0

                for _key, nodes in name_groups.items():
                    if len(nodes) <= 1:
                        continue

                    nodes.sort(key=lambda n: n.get("created_time", "") or "")
                    keep_node = nodes[0]
                    duplicate_nodes = nodes[1:]

                    merged_content = keep_node["content"] or ""
                    merged_confidence = keep_node["confidence"] or 0.5
                    merged_access_count = keep_node["access_count"] or 0
                    merged_properties = dict(keep_node.get("properties", {}))

                    source_ids = [
                        x.strip()
                        for x in merged_properties.get("source_memory_ids", "").split(
                            ","
                        )
                        if x.strip()
                    ]
                    group_ids = [
                        x.strip()
                        for x in merged_properties.get("group_ids", "").split(",")
                        if x.strip()
                    ]
                    if (
                        keep_node.get("group_id")
                        and keep_node["group_id"] not in group_ids
                    ):
                        group_ids.insert(0, keep_node["group_id"])

                    merged_aliases = [
                        a.strip()
                        for a in merged_properties.get("aliases", "").split(",")
                        if a.strip()
                    ]

                    dup_ids = []
                    for dup in duplicate_nodes:
                        merged_content = self._merge_node_content(
                            merged_content, dup["content"]
                        )

                        merged_confidence = max(
                            merged_confidence, dup.get("confidence", 0.5)
                        )
                        merged_access_count += dup.get("access_count", 0)

                        dup_props = dup.get("properties", {})
                        if isinstance(dup_props, dict):
                            for k, v in dup_props.items():
                                if k not in (
                                    "source_memory_ids",
                                    "group_ids",
                                    "aliases",
                                ):
                                    merged_properties[k] = v
                            for alias in [
                                a.strip()
                                for a in dup_props.get("aliases", "").split(",")
                                if a.strip()
                            ]:
                                if alias not in merged_aliases:
                                    merged_aliases.append(alias)
                            dup_source_ids = [
                                x.strip()
                                for x in dup_props.get("source_memory_ids", "").split(
                                    ","
                                )
                                if x.strip()
                            ]
                            source_ids.extend(
                                sid for sid in dup_source_ids if sid not in source_ids
                            )
                            dup_group_ids = [
                                x.strip()
                                for x in dup_props.get("group_ids", "").split(",")
                                if x.strip()
                            ]
                            group_ids.extend(
                                gid for gid in dup_group_ids if gid not in group_ids
                            )

                        if dup.get("group_id") and dup["group_id"] not in group_ids:
                            group_ids.append(dup["group_id"])

                        dup_ids.append(dup["id"])

                    keep_id = keep_node["id"]

                    if dup_ids:
                        self._redirect_edges_and_delete_nodes(dup_ids, keep_id)
                        deleted_count += len(dup_ids)

                    if source_ids:
                        merged_properties["source_memory_ids"] = ",".join(source_ids)
                    if group_ids:
                        merged_properties["group_ids"] = ",".join(group_ids)
                    if merged_aliases:
                        merged_properties["aliases"] = ",".join(merged_aliases)

                    merged_group_id = keep_node.get("group_id", "")
                    if not merged_group_id and group_ids:
                        merged_group_id = group_ids[0]

                    self._db.execute(
                        """UPDATE nodes SET content = ?, confidence = ?,
                           access_count = ?, group_id = ?, properties = ?
                        WHERE id = ?""",
                        (
                            merged_content,
                            merged_confidence,
                            merged_access_count,
                            merged_group_id,
                            json.dumps(merged_properties, ensure_ascii=False),
                            keep_id,
                        ),
                    )

                    merged_count += 1

                self._db.commit()

                if merged_count > 0:
                    logger.info(
                        f"节点合并完成：合并了 {merged_count} 组重复节点，"
                        f"删除了 {deleted_count} 个重复节点"
                    )

                return merged_count, deleted_count
            except Exception as e:
                # 循环内任一异常（如损坏 JSON、约束冲突）必须回滚未提交的
                # 半合并 DELETE/INSERT/UPDATE，否则这些脏数据会被下一个无关
                # _db_write 的 commit 一并刷盘，造成节点/边不一致的半提交损坏。
                self._db.rollback()
                logger.error(f"合并重复节点失败：{e}", exc_info=True)
                return 0, 0

    def _redirect_edges_and_delete_nodes(
        self, dup_ids: list[str], keep_id: str
    ) -> None:
        """将重复节点的边重定向到保留节点，去自环，并删除重复节点

        调用方需持有 _db_lock 且处于事务中；本方法不提交，
        由调用方在合并循环结束后统一 commit，异常时 rollback。
        """
        if not dup_ids:
            return

        dup_ids_set = set(dup_ids)
        dup_placeholders = ",".join("?" * len(dup_ids))

        dup_edges = self._db.execute(
            f"""SELECT source_id, target_id, relation_type, weight,
                       confidence, access_count, last_access_time,
                       created_time, source_memory_id, persona_id, properties
                FROM edges
                WHERE source_id IN ({dup_placeholders})
                   OR target_id IN ({dup_placeholders})""",
            (*dup_ids, *dup_ids),
        ).fetchall()

        self._db.execute(
            f"""DELETE FROM edges
                WHERE source_id IN ({dup_placeholders})
                   OR target_id IN ({dup_placeholders})""",
            (*dup_ids, *dup_ids),
        )

        for edge_row in dup_edges:
            new_source = (
                keep_id
                if edge_row["source_id"] in dup_ids_set
                else edge_row["source_id"]
            )
            new_target = (
                keep_id
                if edge_row["target_id"] in dup_ids_set
                else edge_row["target_id"]
            )

            if new_source == new_target:
                continue

            edge_props = edge_row["properties"]
            if isinstance(edge_props, str):
                try:
                    edge_props = json.loads(edge_props)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        f"边 {edge_row['source_id']}-"
                        f"{edge_row['relation_type']}-"
                        f"{edge_row['target_id']} properties JSON 损坏，"
                        "已回退为空字典"
                    )
                    edge_props = {}

            existing = self._db.execute(
                """SELECT weight, confidence, access_count, properties
                   FROM edges
                   WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
                (new_source, new_target, edge_row["relation_type"]),
            ).fetchone()

            if existing:
                existing_props = existing["properties"]
                if isinstance(existing_props, str):
                    try:
                        existing_props = json.loads(existing_props)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("已有边 properties JSON 损坏，已回退为空字典")
                        existing_props = {}
                existing_props.update(edge_props)
                self._db.execute(
                    """UPDATE edges SET
                        weight = MAX(weight, ?),
                        confidence = MAX(confidence, ?),
                        access_count = access_count + ?,
                        properties = ?
                    WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
                    (
                        edge_row["weight"],
                        edge_row["confidence"],
                        edge_row["access_count"],
                        json.dumps(existing_props, ensure_ascii=False),
                        new_source,
                        new_target,
                        edge_row["relation_type"],
                    ),
                )
            else:
                self._db.execute(
                    """INSERT INTO edges
                        (source_id, target_id, relation_type, weight,
                         confidence, access_count, last_access_time,
                         created_time, source_memory_id, persona_id, properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_source,
                        new_target,
                        edge_row["relation_type"],
                        edge_row["weight"],
                        edge_row["confidence"],
                        edge_row["access_count"],
                        edge_row["last_access_time"],
                        edge_row["created_time"],
                        edge_row["source_memory_id"],
                        edge_row["persona_id"],
                        json.dumps(edge_props, ensure_ascii=False),
                    ),
                )

        self._db.execute(
            f"DELETE FROM nodes WHERE id IN ({dup_placeholders})",
            dup_ids,
        )

    async def merge_person_nodes_by_user_id(
        self, alias_map: Optional[Dict[str, List[str]]] = None
    ) -> tuple[int, int]:
        """按 properties.user_id 合并分裂的 Person 节点（存量修复）

        同一用户的 Person 节点可能因昵称/QQ号命名差异而分裂。本方法按
        (group_id, properties.user_id) 分组（尊重群隔离，不跨群合并），
        优先保留 name == user_id 的节点，合并 content/aliases/来源与群信息，
        重定向边并去自环，删除重复节点。

        分组以带 properties.user_id 标记的节点（提取阶段归一化后写入）为锚点；
        此外会按群隔离吸收 name 恰等于某已知 user_id/别名 的无标记历史遗留
        节点进同组合并，修复归一化重命名（生成新 ID）后残留的持久分裂。

        Args:
            alias_map: 外部别名映射 {user_id: [昵称/曾用名...]}，通常来自
                用户画像（build_profile_alias_map）。无标记昵称节点命中映射
                时归组合并；同组无已标记锚点时仅为该节点补打 user_id 标记
                （改名会产生新节点 ID 导致边重定向，这里不做改名）。

        Returns:
            (合并的节点组数, 删除的重复节点数)
        """
        if not self._is_available:
            return 0, 0

        with self._db_lock:
            try:
                rows = self._db.execute(
                    """SELECT id, name, content, confidence, access_count,
                              created_time, group_id, persona_id, properties
                       FROM nodes WHERE label = 'Person'"""
                ).fetchall()

                parsed: list[tuple[dict, str]] = []
                user_groups: dict[tuple[str, str, str], list[dict]] = {}
                for row in rows:
                    props = row["properties"]
                    if isinstance(props, str):
                        try:
                            props = json.loads(props)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"节点 {row['id']} properties JSON 损坏，已回退为空字典"
                            )
                            props = {}
                    props = props if isinstance(props, dict) else {}
                    node_data = {
                        "id": row["id"],
                        "name": row["name"],
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "access_count": row["access_count"],
                        "created_time": row["created_time"],
                        "group_id": row["group_id"] or "",
                        "persona_id": row["persona_id"],
                        "properties": props,
                    }
                    user_id = str(props["user_id"]) if props.get("user_id") else ""
                    parsed.append((node_data, user_id))
                    if user_id:
                        key = (
                            node_data["persona_id"],
                            node_data["group_id"],
                            user_id,
                        )
                        user_groups.setdefault(key, []).append(node_data)

                # 由已标记节点构建别名索引：(group_id, 昵称或user_id) -> user_id。
                # 归一化重命名会生成新 ID 的 user_id 节点，历史遗留的无标记昵称
                # 节点（name 恰为某已知别名）若不吸收进同组合并，将形成自动合并
                # 永远无法修复的持久分裂。此处按群隔离将其归并到对应 user_id 组。
                alias_index: dict[tuple[str, str, str], str] = {}
                for (persona_id, group_id, user_id), group_nodes in user_groups.items():
                    for node in group_nodes:
                        match_names = {user_id}
                        match_names.update(
                            a.strip()
                            for a in node["properties"].get("aliases", "").split(",")
                            if a.strip()
                        )
                        for match_name in match_names:
                            alias_index.setdefault(
                                (persona_id, group_id, match_name), user_id
                            )

                # 外部别名映射（画像驱动）并入吸收索引：昵称命中映射的无标记
                # 节点同样归组到对应 user_id，合并时统一打标。
                external_lookup: dict[str, str] = {}
                if alias_map:
                    for ext_uid, ext_aliases in alias_map.items():
                        if not ext_uid:
                            continue
                        ext_names = {ext_uid}
                        for a in ext_aliases or []:
                            if a:
                                ext_names.add(a)
                        for n in ext_names:
                            external_lookup.setdefault(n, ext_uid)

                # 无锚点可归组、但命中外部映射的单节点：合并后统一补打
                # user_id 标记（见循环后的 singleton 处理）
                externally_tagged_singletons: list[tuple[str, dict]] = []

                for node_data, user_id in parsed:
                    if user_id:
                        continue
                    group_key = (
                        node_data["persona_id"],
                        node_data["group_id"],
                        node_data["name"],
                    )
                    target_uid = alias_index.get(group_key)
                    if target_uid is None and external_lookup:
                        target_uid = external_lookup.get(node_data["name"])
                    if not target_uid:
                        continue

                    target_key = (
                        node_data["persona_id"],
                        node_data["group_id"],
                        target_uid,
                    )
                    group_nodes = user_groups.get(target_key)
                    if group_nodes is None:
                        # 同组无已标记锚点：不产生合并，仅登记待补打标记
                        externally_tagged_singletons.append((target_uid, node_data))
                        continue
                    group_nodes.append(node_data)

                merged_count = 0
                deleted_count = 0

                for (_persona_id, _group_id, user_id), nodes in user_groups.items():
                    if len(nodes) <= 1:
                        continue

                    # 优先保留 name == user_id 的节点，其次按创建时间升序
                    nodes.sort(
                        key=lambda n: (
                            n["name"] != user_id,
                            n.get("created_time", "") or "",
                        )
                    )
                    keep_node = nodes[0]
                    duplicate_nodes = nodes[1:]

                    merged_content = keep_node["content"] or ""
                    merged_confidence = keep_node["confidence"] or 0.5
                    merged_access_count = keep_node["access_count"] or 0
                    merged_properties = dict(keep_node.get("properties", {}))

                    merged_aliases = [
                        a.strip()
                        for a in merged_properties.get("aliases", "").split(",")
                        if a.strip()
                    ]
                    if (
                        keep_node["name"]
                        and keep_node["name"] != user_id
                        and keep_node["name"] not in merged_aliases
                    ):
                        merged_aliases.append(keep_node["name"])

                    source_ids = [
                        x.strip()
                        for x in merged_properties.get("source_memory_ids", "").split(
                            ","
                        )
                        if x.strip()
                    ]
                    group_ids = [
                        x.strip()
                        for x in merged_properties.get("group_ids", "").split(",")
                        if x.strip()
                    ]
                    if (
                        keep_node.get("group_id")
                        and keep_node["group_id"] not in group_ids
                    ):
                        group_ids.insert(0, keep_node["group_id"])

                    dup_ids = []
                    for dup in duplicate_nodes:
                        merged_content = self._merge_node_content(
                            merged_content, dup["content"]
                        )
                        merged_confidence = max(
                            merged_confidence, dup.get("confidence", 0.5)
                        )
                        merged_access_count += dup.get("access_count", 0)

                        dup_props = dup.get("properties", {})
                        if isinstance(dup_props, dict):
                            for a in [
                                x.strip()
                                for x in dup_props.get("aliases", "").split(",")
                                if x.strip()
                            ]:
                                if a not in merged_aliases:
                                    merged_aliases.append(a)
                            for k, v in dup_props.items():
                                if k not in (
                                    "source_memory_ids",
                                    "group_ids",
                                    "aliases",
                                ):
                                    merged_properties[k] = v
                            dup_source_ids = [
                                x.strip()
                                for x in dup_props.get("source_memory_ids", "").split(
                                    ","
                                )
                                if x.strip()
                            ]
                            source_ids.extend(
                                sid for sid in dup_source_ids if sid not in source_ids
                            )
                            dup_group_ids = [
                                x.strip()
                                for x in dup_props.get("group_ids", "").split(",")
                                if x.strip()
                            ]
                            group_ids.extend(
                                gid for gid in dup_group_ids if gid not in group_ids
                            )

                        if (
                            dup["name"]
                            and dup["name"] != user_id
                            and dup["name"] not in merged_aliases
                        ):
                            merged_aliases.append(dup["name"])

                        if dup.get("group_id") and dup["group_id"] not in group_ids:
                            group_ids.append(dup["group_id"])

                        dup_ids.append(dup["id"])

                    if merged_aliases:
                        merged_properties["aliases"] = ",".join(merged_aliases)
                    merged_properties["user_id"] = user_id
                    if source_ids:
                        merged_properties["source_memory_ids"] = ",".join(source_ids)
                    if group_ids:
                        merged_properties["group_ids"] = ",".join(group_ids)

                    keep_id = keep_node["id"]

                    if dup_ids:
                        self._redirect_edges_and_delete_nodes(dup_ids, keep_id)
                        deleted_count += len(dup_ids)

                    merged_group_id = keep_node.get("group_id", "")
                    if not merged_group_id and group_ids:
                        merged_group_id = group_ids[0]

                    self._db.execute(
                        """UPDATE nodes SET content = ?, confidence = ?,
                           access_count = ?, group_id = ?, properties = ?
                        WHERE id = ?""",
                        (
                            merged_content,
                            merged_confidence,
                            merged_access_count,
                            merged_group_id,
                            json.dumps(merged_properties, ensure_ascii=False),
                            keep_id,
                        ),
                    )

                    merged_count += 1

                # 命中外部别名映射但无同组锚点的单节点：就地补打 user_id
                # 标记。不改名（改名会更换节点 ID 并需边重定向），标记后
                # 按用户搜索/删除即可命中。
                tagged_singletons = 0
                for uid, node_data in externally_tagged_singletons:
                    props = dict(node_data.get("properties") or {})
                    if props.get("user_id"):
                        continue
                    props["user_id"] = uid
                    self._db.execute(
                        "UPDATE nodes SET properties = ? WHERE id = ?",
                        (
                            json.dumps(props, ensure_ascii=False),
                            node_data["id"],
                        ),
                    )
                    tagged_singletons += 1

                self._db.commit()

                if merged_count > 0:
                    logger.info(
                        f"Person 节点按 user_id 合并完成：合并了 {merged_count} 组，"
                        f"删除了 {deleted_count} 个重复节点"
                    )
                if tagged_singletons > 0:
                    logger.info(
                        f"画像别名归一化：为 {tagged_singletons} 个存量 "
                        f"Person 节点补打 user_id 标记"
                    )

                return merged_count, deleted_count
            except Exception as e:
                self._db.rollback()
                logger.error(f"按 user_id 合并 Person 节点失败：{e}", exc_info=True)
                return 0, 0

    async def export_all(self) -> dict:
        """导出所有知识图谱数据"""
        if not self._is_available:
            return {
                "version": "1.0",
                "nodes": [],
                "edges": [],
                "export_time": "",
                "stats": {"node_count": 0, "edge_count": 0},
            }

        try:
            node_rows = self._db_fetchall(
                """SELECT id, label, name, content, confidence,
                          access_count, last_access_time, created_time,
                          source_memory_id, group_id, persona_id, properties
                   FROM nodes"""
            )

            nodes = []
            for row in node_rows:
                props = row["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                nodes.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "name": row["name"],
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "access_count": row["access_count"],
                        "last_access_time": row["last_access_time"],
                        "created_time": row["created_time"],
                        "source_memory_id": row["source_memory_id"],
                        "group_id": row["group_id"],
                        "persona_id": row["persona_id"],
                        "properties": props if isinstance(props, dict) else {},
                    }
                )

            edge_rows = self._db_fetchall(
                """SELECT source_id, target_id, relation_type, weight, confidence,
                          access_count, last_access_time, created_time,
                          source_memory_id, persona_id, properties
                   FROM edges"""
            )

            edges = []
            for row in edge_rows:
                props = row["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                edges.append(
                    {
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "relation_type": row["relation_type"],
                        "weight": row["weight"],
                        "confidence": row["confidence"],
                        "access_count": row["access_count"],
                        "last_access_time": row["last_access_time"],
                        "created_time": row["created_time"],
                        "source_memory_id": row["source_memory_id"],
                        "persona_id": row["persona_id"],
                        "properties": props if isinstance(props, dict) else {},
                    }
                )

            logger.info(f"知识图谱导出完成：{len(nodes)} 个节点，{len(edges)} 条边")

            return {
                "version": "1.0",
                "export_time": datetime.now().isoformat(),
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                },
            }
        except Exception as e:
            logger.error(f"导出知识图谱失败：{e}", exc_info=True)
            return {
                "version": "1.0",
                "nodes": [],
                "edges": [],
                "export_time": "",
                "stats": {"node_count": 0, "edge_count": 0},
            }

    async def import_from_data(self, data: dict, skip_duplicates: bool = True) -> dict:
        """从数据字典导入知识图谱"""
        if not self._is_available:
            return {
                "imported_nodes": 0,
                "imported_edges": 0,
                "skipped_nodes": 0,
                "error_count": 0,
            }

        nodes_data = data.get("nodes", [])
        edges_data = data.get("edges", [])

        imported_nodes = 0
        imported_edges = 0
        skipped_nodes = 0
        skipped_edges = 0
        error_count = 0

        for node_data in nodes_data:
            try:
                node = GraphNode(
                    id=node_data.get("id", ""),
                    label=node_data.get("label", "Entity"),
                    name=node_data.get("name", ""),
                    content=node_data.get("content", ""),
                    confidence=node_data.get("confidence", 0.5),
                    access_count=node_data.get("access_count", 0),
                    group_id=node_data.get("group_id"),
                    source_memory_id=node_data.get("source_memory_id"),
                    properties=node_data.get("properties", {}),
                    persona_id=node_data.get("persona_id", "default"),
                )

                if not node.id:
                    node.id = node.generate_id()

                success = await self.add_node(node)
                if success:
                    imported_nodes += 1
                else:
                    skipped_nodes += 1

            except Exception as e:
                logger.error(f"导入节点失败：{e}")
                error_count += 1

        for edge_data in edges_data:
            try:
                edge = GraphEdge(
                    source_id=edge_data.get("source_id", ""),
                    target_id=edge_data.get("target_id", ""),
                    relation_type=edge_data.get("relation_type", "RELATED_TO"),
                    weight=edge_data.get("weight", 1.0),
                    confidence=edge_data.get("confidence", 0.5),
                    access_count=edge_data.get("access_count", 0),
                    source_memory_id=edge_data.get("source_memory_id"),
                    properties=edge_data.get("properties", {}),
                    persona_id=edge_data.get("persona_id", "default"),
                )

                if not edge.source_id or not edge.target_id:
                    # 边跳过应计入 skipped_edges，此前误计入 skipped_nodes
                    skipped_edges += 1
                    continue

                success = await self.add_edge(edge)
                if success:
                    imported_edges += 1
                else:
                    skipped_edges += 1

            except Exception as e:
                logger.error(f"导入边失败：{e}")
                error_count += 1

        logger.info(
            f"知识图谱导入完成：节点 {imported_nodes}/{len(nodes_data)}，"
            f"边 {imported_edges}/{len(edges_data)}，"
            f"跳过节点 {skipped_nodes}，跳过边 {skipped_edges}，错误 {error_count}"
        )

        return {
            "imported_nodes": imported_nodes,
            "imported_edges": imported_edges,
            "skipped_nodes": skipped_nodes,
            "skipped_edges": skipped_edges,
            "error_count": error_count,
        }

    # ========================================================================
    # 线程安全的 DB 辅助方法
    # ========================================================================

    def _db_execute(self, sql: str, params=()):
        """线程安全的 DB 读操作"""
        with self._db_lock:
            return self._db.execute(sql, params)

    def _db_write(self, sql: str, params=()):
        """线程安全的 DB 写操作（INSERT/UPDATE/DELETE + COMMIT）"""
        with self._db_lock:
            self._db.execute(sql, params)
            self._db.commit()

    def _db_writes(self, statements: list[tuple[str, tuple]]):
        """线程安全的批量写操作"""
        with self._db_lock:
            for sql, params in statements:
                self._db.execute(sql, params)
            self._db.commit()

    def _db_fetchone(self, sql: str, params=()):
        """线程安全的 DB 单行查询"""
        with self._db_lock:
            return self._db.execute(sql, params).fetchone()

    def _db_fetchall(self, sql: str, params=()):
        """线程安全的 DB 多行查询"""
        with self._db_lock:
            return self._db.execute(sql, params).fetchall()

    async def shutdown(self) -> None:
        """关闭数据库连接"""
        if hasattr(self, "_db") and self._db:
            self._db.close()
        self._reset_state()
        logger.info("SQLite 图谱已关闭")
