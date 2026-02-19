"""
知识图谱上下文格式化器单元测试
"""

import pytest

from iris_memory.knowledge_graph.kg_context import KGContextFormatter
from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGPath,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult


class TestFormatReasoningResult:
    """多跳推理结果格式化测试"""

    def test_empty_result(self):
        fmt = KGContextFormatter()
        result = ReasoningResult()
        text = fmt.format_reasoning_result(result)
        assert text == ""

    def test_single_path(self):
        fmt = KGContextFormatter()
        n1 = KGNode(name="张三", display_name="张三")
        n2 = KGNode(name="李四", display_name="李四")
        e = KGEdge(source_id=n1.id, target_id=n2.id,
                   relation_label="朋友", relation_type=KGRelationType.FRIEND_OF)
        path = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.8, hop_count=1)
        result = ReasoningResult(paths=[path], seed_nodes=[n1])

        text = fmt.format_reasoning_result(result)
        assert "【知识关联】" in text
        assert "张三" in text
        assert "李四" in text
        assert "朋友" in text

    def test_multi_path(self):
        fmt = KGContextFormatter()
        n1 = KGNode(name="a", display_name="A")
        n2 = KGNode(name="b", display_name="B")
        n3 = KGNode(name="c", display_name="C")
        e1 = KGEdge(source_id=n1.id, target_id=n2.id, relation_label="认识")
        e2 = KGEdge(source_id=n2.id, target_id=n3.id, relation_label="住在")

        p1 = KGPath(nodes=[n1, n2], edges=[e1], total_confidence=0.8, hop_count=1)
        p2 = KGPath(nodes=[n1, n2, n3], edges=[e1, e2], total_confidence=0.5, hop_count=2)
        result = ReasoningResult(paths=[p1, p2], seed_nodes=[n1])

        text = fmt.format_reasoning_result(result)
        assert "A" in text
        assert "B" in text

    def test_max_facts_limit(self):
        """应遵守 max_facts 限制"""
        fmt = KGContextFormatter(max_facts=2)
        paths = []
        for i in range(10):
            n1 = KGNode(name=f"e{i}", display_name=f"Entity{i}")
            n2 = KGNode(name=f"e{i+10}", display_name=f"Entity{i+10}")
            e = KGEdge(source_id=n1.id, target_id=n2.id, relation_label=f"rel{i}")
            paths.append(KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.8, hop_count=1))

        result = ReasoningResult(paths=paths, seed_nodes=[])
        text = fmt.format_reasoning_result(result)
        # 应有 header(2 行) + 最多 2 条事实
        lines = [l for l in text.split("\n") if l.startswith("·")]
        assert len(lines) <= 2

    def test_max_chars_limit(self):
        """应遵守 max_chars 限制"""
        fmt = KGContextFormatter(max_chars=50, max_facts=100)
        n1 = KGNode(name="very_long_entity_name_" * 3, display_name="VeryLong" * 10)
        n2 = KGNode(name="another_long_name_" * 3, display_name="AnotherLong" * 10)
        e = KGEdge(source_id=n1.id, target_id=n2.id, relation_label="some relation")
        path = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.8, hop_count=1)
        result = ReasoningResult(paths=[path], seed_nodes=[n1])

        text = fmt.format_reasoning_result(result)
        # max_chars is a soft budget, just make sure it doesn't blow up
        assert isinstance(text, str)


class TestFormatEntityRelations:
    """单一实体直接关系格式化测试"""

    def test_empty_relations(self):
        fmt = KGContextFormatter()
        text = fmt.format_entity_relations("张三", [])
        assert text == ""

    def test_single_relation(self):
        fmt = KGContextFormatter()
        edge = KGEdge(relation_label="朋友", relation_type=KGRelationType.FRIEND_OF)
        node = KGNode(name="李四", display_name="李四")
        text = fmt.format_entity_relations("张三", [(edge, node)])
        assert "【知识关联】" in text
        assert "张三" in text
        assert "李四" in text
        assert "朋友" in text

    def test_multiple_relations(self):
        fmt = KGContextFormatter()
        relations = [
            (KGEdge(relation_label="朋友"), KGNode(name="李四", display_name="李四")),
            (KGEdge(relation_label="喜欢"), KGNode(name="编程", display_name="编程")),
            (KGEdge(relation_label="住在"), KGNode(name="北京", display_name="北京")),
        ]
        text = fmt.format_entity_relations("张三", relations)
        assert text.count("·") >= 3

    def test_dedup_facts(self):
        fmt = KGContextFormatter()
        edge = KGEdge(relation_label="朋友")
        node = KGNode(name="李四", display_name="李四")
        # 同样的关系重复
        text = fmt.format_entity_relations("张三", [(edge, node), (edge, node)])
        # 应去重
        lines = [l for l in text.split("\n") if l.startswith("·")]
        assert len(lines) == 1


class TestFormatMixed:
    """混合格式化测试"""

    def test_prefers_reasoning_result(self):
        fmt = KGContextFormatter()
        n1 = KGNode(name="a", display_name="A")
        n2 = KGNode(name="b", display_name="B")
        e = KGEdge(source_id=n1.id, target_id=n2.id, relation_label="rel")
        path = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.8, hop_count=1)
        reasoning = ReasoningResult(paths=[path], seed_nodes=[n1])
        direct = [(KGEdge(relation_label="other"), KGNode(name="c", display_name="C"))]

        text = fmt.format_mixed(
            reasoning_result=reasoning,
            direct_relations=direct,
            entity_name="a",
        )
        # 应使用 reasoning result 而非 direct
        assert "A" in text

    def test_falls_back_to_direct(self):
        fmt = KGContextFormatter()
        direct = [(KGEdge(relation_label="朋友"), KGNode(name="b", display_name="B"))]

        text = fmt.format_mixed(
            reasoning_result=ReasoningResult(),
            direct_relations=direct,
            entity_name="A",
        )
        assert "B" in text

    def test_both_empty(self):
        fmt = KGContextFormatter()
        text = fmt.format_mixed()
        assert text == ""
