"""
知识图谱数据模型单元测试
"""

import pytest

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGPath,
    KGRelationType,
    KGTriple,
)


class TestKGNodeType:
    """节点类型枚举测试"""

    def test_values(self):
        assert KGNodeType.PERSON.value == "person"
        assert KGNodeType.LOCATION.value == "location"
        assert KGNodeType.UNKNOWN.value == "unknown"

    def test_from_string(self):
        assert KGNodeType("person") == KGNodeType.PERSON
        assert KGNodeType("unknown") == KGNodeType.UNKNOWN


class TestKGRelationType:
    """关系类型枚举测试"""

    def test_interpersonal(self):
        assert KGRelationType.FRIEND_OF.value == "friend_of"
        assert KGRelationType.BOSS_OF.value == "boss_of"

    def test_attribute(self):
        assert KGRelationType.LIVES_IN.value == "lives_in"
        assert KGRelationType.WORKS_AT.value == "works_at"

    def test_behavior(self):
        assert KGRelationType.LIKES.value == "likes"
        assert KGRelationType.DISLIKES.value == "dislikes"


class TestKGNode:
    """节点数据模型测试"""

    def test_default_node(self):
        node = KGNode()
        assert node.id  # auto-generated UUID
        assert node.name == ""
        assert node.node_type == KGNodeType.UNKNOWN
        assert node.mention_count == 1
        assert node.confidence == 0.5

    def test_node_with_values(self):
        node = KGNode(
            name="张三",
            display_name="张三",
            node_type=KGNodeType.PERSON,
            user_id="user1",
            group_id="group1",
            aliases=["小张"],
            confidence=0.8,
        )
        assert node.name == "张三"
        assert node.node_type == KGNodeType.PERSON
        assert "小张" in node.aliases
        assert node.user_id == "user1"

    def test_to_dict(self):
        node = KGNode(name="test", display_name="Test", node_type=KGNodeType.CONCEPT)
        d = node.to_dict()
        assert d["name"] == "test"
        assert d["display_name"] == "Test"
        assert d["node_type"] == "concept"
        assert isinstance(d["aliases"], str)  # JSON string
        assert isinstance(d["properties"], str)

    def test_from_row_roundtrip(self):
        original = KGNode(
            name="李四",
            display_name="李四",
            node_type=KGNodeType.PERSON,
            user_id="u1",
            group_id="g1",
            aliases=["老李"],
            properties={"age": 30},
            mention_count=5,
            confidence=0.9,
        )
        d = original.to_dict()
        restored = KGNode.from_row(d)
        assert restored.name == original.name
        assert restored.node_type == original.node_type
        assert restored.aliases == original.aliases
        assert restored.properties == original.properties
        assert restored.mention_count == original.mention_count

    def test_unique_ids(self):
        n1 = KGNode()
        n2 = KGNode()
        assert n1.id != n2.id


class TestKGEdge:
    """边数据模型测试"""

    def test_default_edge(self):
        edge = KGEdge()
        assert edge.id
        assert edge.relation_type == KGRelationType.RELATED_TO
        assert edge.weight == 1.0

    def test_edge_with_values(self):
        edge = KGEdge(
            source_id="s1",
            target_id="t1",
            relation_type=KGRelationType.FRIEND_OF,
            relation_label="好朋友",
            memory_id="m1",
            user_id="u1",
            confidence=0.8,
        )
        assert edge.source_id == "s1"
        assert edge.target_id == "t1"
        assert edge.relation_label == "好朋友"

    def test_to_dict_from_row_roundtrip(self):
        original = KGEdge(
            source_id="s1",
            target_id="t1",
            relation_type=KGRelationType.LIKES,
            relation_label="喜欢",
            memory_id="m1",
            confidence=0.7,
        )
        d = original.to_dict()
        restored = KGEdge.from_row(d)
        assert restored.source_id == original.source_id
        assert restored.relation_type == original.relation_type
        assert restored.relation_label == original.relation_label


class TestKGTriple:
    """三元组数据模型测试"""

    def test_default_triple(self):
        t = KGTriple()
        assert t.subject == ""
        assert t.predicate == ""
        assert t.object == ""
        assert t.relation_type == KGRelationType.RELATED_TO

    def test_triple_repr(self):
        t = KGTriple(subject="张三", predicate="喜欢", object="编程")
        rep = repr(t)
        assert "张三" in rep
        assert "喜欢" in rep
        assert "编程" in rep


class TestKGPath:
    """路径数据模型测试"""

    def test_empty_path(self):
        p = KGPath()
        assert p.to_text() == ""
        assert p.hop_count == 0

    def test_path_to_text(self):
        n1 = KGNode(name="a", display_name="Alice")
        n2 = KGNode(name="b", display_name="Bob")
        e = KGEdge(
            source_id=n1.id,
            target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF,
            relation_label="朋友",
        )
        p = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.8, hop_count=1)
        text = p.to_text()
        assert "Alice" in text
        assert "Bob" in text
        assert "朋友" in text

    def test_multi_hop_path(self):
        n1 = KGNode(name="a", display_name="A")
        n2 = KGNode(name="b", display_name="B")
        n3 = KGNode(name="c", display_name="C")
        e1 = KGEdge(source_id=n1.id, target_id=n2.id, relation_label="认识")
        e2 = KGEdge(source_id=n2.id, target_id=n3.id, relation_label="住在")
        p = KGPath(nodes=[n1, n2, n3], edges=[e1, e2], hop_count=2)
        text = p.to_text()
        assert "A" in text
        assert "B" in text
        assert "C" in text
        assert "认识" in text
        assert "住在" in text
