"""l3_migrator 测试（真实 sqlite 构造旧 schema 库）"""

import json

import pytest

from iris_memory.legacy_migration.detector import LegacyDetection
from iris_memory.legacy_migration.l3_migrator import (
    map_legacy_edge,
    map_legacy_node,
    migrate_l3,
)
from iris_memory.l3_kg.models import GraphNode

from .conftest import make_legacy_kg_db


def _kg_detection(path):
    return LegacyDetection(kg_db_path=path)


class TestMapLegacyNode:
    """节点映射正确性"""

    def test_full_mapping(self):
        node = map_legacy_node(
            {
                "id": "n1",
                "name": "xiaoming",
                "display_name": "小明",
                "node_type": "person",
                "user_id": "u1",
                "group_id": "g1",
                "persona_id": "yuki",
                "aliases": '["明哥"]',
                "properties": json.dumps({"description": "后端工程师"}),
                "mention_count": 7,
                "confidence": 0.8,
                "created_time": "2025-01-01T08:00:00",
                "updated_time": "2025-02-01T08:00:00",
            }
        )
        assert node is not None
        assert node.label == "Person"
        assert node.name == "小明"
        assert node.content == "后端工程师"
        assert node.confidence == 0.8
        assert node.access_count == 7
        assert node.group_id == "g1"
        assert node.created_time.isoformat() == "2025-01-01T08:00:00"
        assert node.last_access_time.isoformat() == "2025-02-01T08:00:00"
        # 新 id 按 label+name 哈希生成（与新版 GraphNode 去重规则一致）
        expected = GraphNode(
            id="", label="Person", name="小明", content="x"
        ).generate_id()
        assert node.id == expected
        # legacy 信息保留在 properties
        assert node.properties["legacy_node_id"] == "n1"
        assert node.properties["legacy_node_type"] == "person"
        assert node.properties["legacy_user_id"] == "u1"
        assert node.properties["legacy_persona_id"] == "yuki"
        assert node.properties["aliases"] == '["明哥"]'
        assert node.properties["migrated_from"] == "iris_memory_v2"

    def test_type_mapping(self):
        cases = {
            "person": "Person",
            "location": "Location",
            "organization": "Group",
            "object": "Item",
            "event": "Event",
            "concept": "Concept",
            "time": "Concept",
            "unknown": "Concept",
            "robot": "Robot",  # 未收录类型 → 首字母大写动态 label
        }
        for legacy_type, expected_label in cases.items():
            node = map_legacy_node({"id": "x", "name": "n", "node_type": legacy_type})
            assert node is not None
            assert node.label == expected_label, legacy_type

    def test_empty_name_returns_none(self):
        assert map_legacy_node({"id": "x", "name": "", "display_name": ""}) is None

    def test_content_fallback_to_name(self):
        node = map_legacy_node({"id": "x", "name": "苹果", "properties": "{}"})
        assert node is not None
        assert node.content == "苹果"

    def test_default_persona_not_recorded(self):
        node = map_legacy_node({"id": "x", "name": "n", "persona_id": "default"})
        assert node is not None
        assert "legacy_persona_id" not in node.properties


class TestMapLegacyEdge:
    """边映射正确性"""

    def _id_map(self):
        return {"n1": "person_aaa", "n2": "location_bbb"}

    def test_full_mapping(self):
        edge, dangling = map_legacy_edge(
            {
                "source_id": "n1",
                "target_id": "n2",
                "relation_type": "lives_in",
                "relation_label": "居住在",
                "memory_id": "old_mem_1",
                "user_id": "u1",
                "confidence": 0.7,
                "weight": 0.9,
                "created_time": "2025-01-01T08:00:00",
                "updated_time": "2025-01-03T08:00:00",
            },
            self._id_map(),
        )
        assert not dangling
        assert edge is not None
        assert edge.source_id == "person_aaa"
        assert edge.target_id == "location_bbb"
        assert edge.relation_type == "LOCATED_AT"
        assert edge.weight == 0.9
        assert edge.confidence == 0.7
        # 旧 memory_id 不写入 source_memory_id（新 L2 id 已重生成，直接引用必然悬空）
        assert edge.source_memory_id is None
        assert edge.properties["legacy_memory_id"] == "old_mem_1"
        assert edge.properties["legacy_relation_type"] == "lives_in"
        assert edge.properties["relation_label"] == "居住在"
        assert edge.created_time.isoformat() == "2025-01-01T08:00:00"
        assert edge.last_access_time.isoformat() == "2025-01-03T08:00:00"

    def test_relation_mapping(self):
        cases = {
            "friend_of": "KNOWS",
            "likes": "HAS_PREFERENCE",
            "dislikes": "HAS_PREFERENCE",
            "is": "HAS_TRAIT",
            "wants": "HAS_GOAL",
            "works_at": "PART_OF",
            "participated_in": "PARTICIPATED_IN",
            "happened_at": "HAPPENED_AT",
            "caused_by": "LEADS_TO",
            "something_custom": "RELATED_TO",
        }
        for legacy_rel, expected in cases.items():
            edge, _ = map_legacy_edge(
                {"source_id": "n1", "target_id": "n2", "relation_type": legacy_rel},
                self._id_map(),
            )
            assert edge is not None
            assert edge.relation_type == expected, legacy_rel

    def test_dislikes_polarity(self):
        edge, _ = map_legacy_edge(
            {"source_id": "n1", "target_id": "n2", "relation_type": "dislikes"},
            self._id_map(),
        )
        assert edge.properties["polarity"] == "dislike"

    def test_dangling_edge(self):
        edge, dangling = map_legacy_edge(
            {"source_id": "n1", "target_id": "n999"}, self._id_map()
        )
        assert edge is None
        assert dangling


class TestMigrateL3:
    """迁移流程（真实 sqlite 旧库）"""

    @pytest.mark.asyncio
    async def test_skipped_no_data(self, component_manager):
        stats = await migrate_l3(LegacyDetection(), component_manager)
        assert stats["status"] == "skipped_no_data"

    @pytest.mark.asyncio
    async def test_skipped_adapter_unavailable(
        self, tmp_path, component_manager, l3_adapter
    ):
        l3_adapter.is_available = False
        db = make_legacy_kg_db(tmp_path / "knowledge_graph.db", nodes=[{"id": "n1"}])
        stats = await migrate_l3(_kg_detection(db), component_manager)
        assert stats["status"] == "skipped_adapter_unavailable"

    @pytest.mark.asyncio
    async def test_skipped_no_tables(self, tmp_path, component_manager):
        db = make_legacy_kg_db(tmp_path / "knowledge_graph.db", schema_sql=None)
        stats = await migrate_l3(_kg_detection(db), component_manager)
        assert stats["status"] == "skipped_no_tables"

    @pytest.mark.asyncio
    async def test_migrate_nodes_and_edges(self, tmp_path, component_manager):
        db = make_legacy_kg_db(
            tmp_path / "knowledge_graph.db",
            nodes=[
                {
                    "id": "n1",
                    "name": "小明",
                    "display_name": "小明",
                    "node_type": "person",
                    "mention_count": 5,
                    "confidence": 0.9,
                    "aliases": '["明哥"]',
                    "properties": json.dumps({"description": "后端工程师"}),
                },
                {
                    "id": "n2",
                    "name": "北京",
                    "display_name": "北京",
                    "node_type": "location",
                    "group_id": "g1",
                },
                {
                    "id": "n3",
                    "name": "",
                    "display_name": "",
                    "node_type": "concept",
                },  # 无名节点 → 错误计数
            ],
            edges=[
                {"id": "e1", "source_id": "n1", "target_id": "n2", "relation_type": "lives_in"},
                {"id": "e2", "source_id": "n1", "target_id": "n9", "relation_type": "knows"},  # 悬空
                {"id": "e3", "source_id": "n2", "target_id": "n1", "relation_type": "related_to"},
            ],
        )

        stats = await migrate_l3(_kg_detection(db), component_manager)

        assert stats["status"] == "ok"
        assert stats["nodes_total"] == 3
        assert stats["nodes_imported"] == 2
        assert stats["errors"] == 1  # 无名节点
        assert stats["edges_total"] == 3
        assert stats["edges_imported"] == 2
        assert stats["edges_dangling"] == 1

        l3 = component_manager.get_component("l3_kg")
        assert len(l3.nodes) == 2
        labels = {n.label for n in l3.nodes.values()}
        assert labels == {"Person", "Location"}

        person = next(n for n in l3.nodes.values() if n.label == "Person")
        assert person.access_count == 5
        assert person.properties["legacy_node_id"] == "n1"
        assert person.properties["aliases"] == '["明哥"]'

        assert len(l3.edges) == 2
        rel_types = {e.relation_type for e in l3.edges}
        assert rel_types == {"LOCATED_AT", "RELATED_TO"}
        # 边端点已重指向新 id
        new_ids = set(l3.nodes.keys())
        for edge in l3.edges:
            assert edge.source_id in new_ids
            assert edge.target_id in new_ids

    @pytest.mark.asyncio
    async def test_same_name_nodes_merge_to_same_id(self, tmp_path, component_manager):
        """不同旧 id 但同名同类型的节点应坍缩到同一新 id（由 add_node 合并）"""
        db = make_legacy_kg_db(
            tmp_path / "knowledge_graph.db",
            nodes=[
                {"id": "n1", "name": "小明", "display_name": "小明", "node_type": "person"},
                {"id": "n2", "name": "小明", "display_name": "小明", "node_type": "person", "user_id": "u2"},
            ],
            edges=[
                {"id": "e1", "source_id": "n1", "target_id": "n2", "relation_type": "knows"},
            ],
        )

        stats = await migrate_l3(_kg_detection(db), component_manager)
        assert stats["nodes_imported"] == 2  # 两次 add_node 都成功（合并语义）
        l3 = component_manager.get_component("l3_kg")
        assert len(l3.nodes) == 1  # 但只有一个新 id
        # 自环边：两端坍缩到同一节点
        assert len(l3.edges) == 1
        edge = l3.edges[0]
        assert edge.source_id == edge.target_id

    @pytest.mark.asyncio
    async def test_add_failure_isolated(self, tmp_path, component_manager, l3_adapter):
        db = make_legacy_kg_db(
            tmp_path / "knowledge_graph.db",
            nodes=[{"id": "n1"}, {"id": "n2", "name": "北京"}],
        )
        l3_adapter.fail_on_add = True

        stats = await migrate_l3(_kg_detection(db), component_manager)
        # 每个节点失败都被隔离计数，迁移器整体不抛异常
        assert stats["status"] == "ok"
        assert stats["errors"] == 2
        assert stats["nodes_imported"] == 0
