"""legacy_migration 测试共享配置与测试替身"""

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# 测试替身
# ============================================================================


class FakeStar:
    """模拟 AstrBot Star 的 KV 存储接口（dict 后端）"""

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}

    async def get_kv_data(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    async def put_kv_data(self, key: str, value: Any) -> None:
        self.kv[key] = value

    async def delete_kv_data(self, key: str) -> None:
        self.kv.pop(key, None)


class FakeL2Adapter:
    """模拟 L2MemoryAdapter（记录 add_memory 调用）"""

    def __init__(self, available: bool = True) -> None:
        self.is_available = available
        self.added: List[Dict[str, Any]] = []

    async def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        skip_dedup: bool = False,
        persona_id: str = "default",
    ) -> Optional[str]:
        if not self.is_available:
            return None
        memory_id = f"mem_{len(self.added):06d}"
        self.added.append(
            {
                "id": memory_id,
                "content": content,
                "metadata": metadata or {},
                "persona_id": persona_id,
                "skip_dedup": skip_dedup,
            }
        )
        return memory_id


class FakeL3Adapter:
    """模拟 L3KGAdapter（记录 add_node/add_edge 调用）"""

    def __init__(self, available: bool = True) -> None:
        self.is_available = available
        self.nodes: Dict[str, Any] = {}
        self.edges: List[Any] = []
        self.fail_on_add = False

    async def add_node(self, node: Any) -> bool:
        if self.fail_on_add:
            raise RuntimeError("模拟写入失败")
        self.nodes[node.id] = node
        return True

    async def add_edge(self, edge: Any) -> bool:
        if self.fail_on_add:
            raise RuntimeError("模拟写入失败")
        self.edges.append(edge)
        return True


class FakeProfileStorage:
    """模拟 ProfileStorage（dict 后端）"""

    def __init__(self, available: bool = True) -> None:
        self.is_available = available
        self.profiles: Dict[Any, Any] = {}

    async def get_user_profile(
        self, user_id: str, group_id: str = "default", persona_id: str = "default"
    ) -> Any:
        return self.profiles.get((user_id, group_id, persona_id))

    async def save_user_profile(
        self,
        profile: Any,
        group_id: str = "default",
        increment_version: bool = True,
        persona_id: str = "default",
    ) -> None:
        self.profiles[(profile.user_id, group_id, persona_id)] = profile


class FakeComponentManager:
    """模拟 ComponentManager（按名字返回组件）"""

    def __init__(self, **components: Any) -> None:
        self._components = dict(components)
        self.wait_calls: List[Any] = []

    def get_component(self, name: str, expected_type: Any = None) -> Any:
        return self._components.get(name)

    async def wait_for_background_init(self, timeout: Any = None) -> None:
        self.wait_calls.append(timeout)


class FakeUserConfig(dict):
    """模拟 AstrBotConfig：dict 子类 + save_config()"""

    def __init__(self, *args: Any, save_raises: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_calls = 0
        self._save_raises = save_raises

    def save_config(self, replace_config: Any = None) -> None:
        self.save_calls += 1
        if self._save_raises:
            raise OSError("模拟只读文件系统")


# ============================================================================
# 旧版 knowledge_graph.db 构造工具
# ============================================================================

LEGACY_KG_SCHEMA = """
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
);
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
    updated_time   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kg_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

DEFAULT_NODE = {
    "id": "n1",
    "name": "小明",
    "display_name": "小明",
    "node_type": "person",
    "user_id": "u1",
    "group_id": "g1",
    "persona_id": "default",
    "aliases": "[]",
    "properties": "{}",
    "mention_count": 1,
    "confidence": 0.5,
    "created_time": "2025-01-01T00:00:00",
    "updated_time": "2025-01-02T00:00:00",
}

DEFAULT_EDGE = {
    "id": "e1",
    "source_id": "n1",
    "target_id": "n2",
    "relation_type": "lives_in",
    "relation_label": "居住在",
    "memory_id": "old_mem_1",
    "user_id": "u1",
    "group_id": "g1",
    "persona_id": "default",
    "confidence": 0.7,
    "weight": 0.9,
    "properties": "{}",
    "created_time": "2025-01-01T00:00:00",
    "updated_time": "2025-01-03T00:00:00",
}


def make_legacy_kg_db(
    path: Path,
    nodes: Optional[List[Dict[str, Any]]] = None,
    edges: Optional[List[Dict[str, Any]]] = None,
    schema_sql: Optional[str] = LEGACY_KG_SCHEMA,
) -> Path:
    """现场构造旧 schema 的 knowledge_graph.db

    Args:
        path: 数据库文件路径
        nodes: kg_nodes 行（缺省列由 DEFAULT_NODE 补全）
        edges: kg_edges 行（缺省列由 DEFAULT_EDGE 补全）
        schema_sql: 建表 SQL；传 None 表示不建表（用于无表分支测试）
    """
    conn = sqlite3.connect(str(path))
    try:
        if schema_sql is not None:
            conn.executescript(schema_sql)
            conn.execute(
                "INSERT OR REPLACE INTO kg_meta (key, value) VALUES ('schema_version', '2')"
            )
        for node in nodes or []:
            row = {**DEFAULT_NODE, **node}
            conn.execute(
                """INSERT OR REPLACE INTO kg_nodes
                   (id, name, display_name, node_type, user_id, group_id, persona_id,
                    aliases, properties, mention_count, confidence, created_time, updated_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row[k] for k in DEFAULT_NODE),
            )
        for edge in edges or []:
            row = {**DEFAULT_EDGE, **edge}
            conn.execute(
                """INSERT OR REPLACE INTO kg_edges
                   (id, source_id, target_id, relation_type, relation_label, memory_id,
                    user_id, group_id, persona_id, confidence, weight, properties,
                    created_time, updated_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row[k] for k in DEFAULT_EDGE),
            )
        conn.commit()
    finally:
        conn.close()
    return path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def star() -> FakeStar:
    return FakeStar()


@pytest.fixture
def l2_adapter() -> FakeL2Adapter:
    return FakeL2Adapter()


@pytest.fixture
def l3_adapter() -> FakeL3Adapter:
    return FakeL3Adapter()


@pytest.fixture
def profile_storage() -> FakeProfileStorage:
    return FakeProfileStorage()


@pytest.fixture
def component_manager(
    l2_adapter: FakeL2Adapter,
    l3_adapter: FakeL3Adapter,
    profile_storage: FakeProfileStorage,
) -> FakeComponentManager:
    return FakeComponentManager(
        l2_memory=l2_adapter, l3_kg=l3_adapter, profile=profile_storage
    )


@pytest.fixture
def iris_config(tmp_path: Path):
    """初始化 iris 配置系统（测试后重置全局单例）"""
    from iris_memory.config import init_config
    from iris_memory.config.config import reset_config

    config = init_config(FakeUserConfig(), tmp_path)
    yield config
    reset_config()
