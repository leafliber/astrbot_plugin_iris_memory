"""L3 知识图谱数据模型测试"""

from datetime import datetime
from iris_memory.l3_kg import (
    GraphNode,
    GraphEdge,
    ExtractionResult,
    NODE_TYPE_WHITELIST,
    RELATION_TYPE_WHITELIST,
)


class TestGraphNode:
    """GraphNode 测试"""

    def test_create_node(self):
        """测试创建节点"""
        node = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )

        assert node.label == "Person"
        assert node.name == "Alice"
        assert node.content == "Alice is a software engineer"
        assert node.confidence == 1.0
        assert node.access_count == 0
        assert isinstance(node.created_time, datetime)

    def test_generate_id(self):
        """测试 ID 生成"""
        node = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )

        node_id = node.generate_id()

        # ID 格式：{label_lower}_{hash_prefix}
        assert node_id.startswith("person_")
        assert len(node_id) == len("person_") + 12  # 12 位 hash

    def test_id_consistency(self):
        """测试相同内容生成相同 ID"""
        node1 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node2 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )

        assert node1.generate_id() == node2.generate_id()

    def test_id_different_for_different_content(self):
        """测试不同实体生成不同 ID"""
        node1 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node2 = GraphNode(
            id="", label="Person", name="Bob", content="Bob is a data scientist"
        )

        assert node1.generate_id() != node2.generate_id()

    def test_id_same_for_same_name_different_content(self):
        """测试同 label+name 不同 content 生成相同 ID（去重合并关键）"""
        node1 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node2 = GraphNode(
            id="", label="Person", name="Alice", content="Alice likes coding"
        )

        assert node1.generate_id() == node2.generate_id()

    def test_id_different_for_different_label_same_name(self):
        """测试同 name 不同 label 生成不同 ID"""
        node1 = GraphNode(
            id="", label="Person", name="Python", content="Python the person"
        )
        node2 = GraphNode(
            id="", label="Concept", name="Python", content="Python the language"
        )

        assert node1.generate_id() != node2.generate_id()

    def test_to_dict(self):
        """测试转换为字典"""
        node = GraphNode(
            id="person_abc123",
            label="Person",
            name="Alice",
            content="Alice is a software engineer",
            confidence=0.9,
            group_id="group_123",
        )

        node_dict = node.to_dict()

        assert node_dict["id"] == "person_abc123"
        assert node_dict["label"] == "Person"
        assert node_dict["name"] == "Alice"
        assert node_dict["confidence"] == 0.9
        assert node_dict["group_id"] == "group_123"
        assert isinstance(node_dict["properties"], dict)


class TestGraphEdge:
    """GraphEdge 测试"""

    def test_create_edge(self):
        """测试创建边"""
        edge = GraphEdge(
            source_id="person_abc123",
            target_id="event_def456",
            relation_type="ATTENDED",
        )

        assert edge.source_id == "person_abc123"
        assert edge.target_id == "event_def456"
        assert edge.relation_type == "ATTENDED"
        assert edge.weight == 1.0
        assert edge.confidence == 1.0

    def test_generate_id(self):
        """测试边 ID 生成"""
        edge = GraphEdge(
            source_id="person_abc123",
            target_id="event_def456",
            relation_type="ATTENDED",
        )

        edge_id = edge.generate_id()

        # ID 格式：{source_id}_{relation_type}_{target_id}
        assert edge_id == "person_abc123_ATTENDED_event_def456"

    def test_to_dict(self):
        """测试转换为字典"""
        edge = GraphEdge(
            source_id="person_abc123",
            target_id="event_def456",
            relation_type="ATTENDED",
            confidence=0.8,
        )

        edge_dict = edge.to_dict()

        assert edge_dict["source_id"] == "person_abc123"
        assert edge_dict["target_id"] == "event_def456"
        assert edge_dict["relation_type"] == "ATTENDED"
        assert edge_dict["confidence"] == 0.8


class TestExtractionResult:
    """ExtractionResult 测试"""

    def test_empty_result(self):
        """测试空结果"""
        result = ExtractionResult()

        assert result.is_empty()
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    def test_non_empty_result(self):
        """测试非空结果"""
        node = GraphNode(
            id="person_abc123",
            label="Person",
            name="Alice",
            content="Alice is a software engineer",
        )

        result = ExtractionResult(nodes=[node])

        assert not result.is_empty()
        assert len(result.nodes) == 1

    def test_to_dict(self):
        """测试转换为字典"""
        node = GraphNode(
            id="person_abc123",
            label="Person",
            name="Alice",
            content="Alice is a software engineer",
        )
        edge = GraphEdge(
            source_id="person_abc123",
            target_id="event_def456",
            relation_type="ATTENDED",
        )

        result = ExtractionResult(
            nodes=[node], edges=[edge], extraction_confidence=0.85
        )

        result_dict = result.to_dict()

        assert len(result_dict["nodes"]) == 1
        assert len(result_dict["edges"]) == 1
        assert result_dict["extraction_confidence"] == 0.85


class TestWhitelists:
    """白名单常量测试"""

    def test_node_type_whitelist(self):
        """测试节点类型白名单"""
        assert "Person" in NODE_TYPE_WHITELIST
        assert "Event" in NODE_TYPE_WHITELIST
        assert "Concept" in NODE_TYPE_WHITELIST
        assert "Location" in NODE_TYPE_WHITELIST
        assert "Item" in NODE_TYPE_WHITELIST
        assert "Topic" in NODE_TYPE_WHITELIST

    def test_relation_type_whitelist(self):
        """测试关系类型白名单"""
        assert "KNOWS" in RELATION_TYPE_WHITELIST
        assert "HAS_PREFERENCE" in RELATION_TYPE_WHITELIST
        assert "HAS_SKILL" in RELATION_TYPE_WHITELIST
        assert "HAS_TRAIT" in RELATION_TYPE_WHITELIST
        assert "HAS_GOAL" in RELATION_TYPE_WHITELIST
        assert "HAS_BELIEF" in RELATION_TYPE_WHITELIST
        assert "RELATED_TO" in RELATION_TYPE_WHITELIST
        assert "PART_OF" in RELATION_TYPE_WHITELIST
        assert "LEADS_TO" in RELATION_TYPE_WHITELIST
        assert "CONTRADICTS" in RELATION_TYPE_WHITELIST
        assert "SUPPORTS" in RELATION_TYPE_WHITELIST
