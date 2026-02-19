"""
多跳推理引擎单元测试
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_reasoning import (
    KGReasoning,
    ReasoningResult,
    CONFIDENCE_DECAY,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage


def run(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture
def graph_env():
    """创建带预设图数据的测试环境

    图结构:
        张三 --[friend_of]--> 李四
        李四 --[works_at]--> 腾讯
        腾讯 --[related_to]--> 深圳
        张三 --[likes]--> 编程
    """
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg.db"
        run(s.initialize(db_path))

        # 创建节点
        n_zhang = KGNode(id="n1", name="张三", display_name="张三",
                         node_type=KGNodeType.PERSON, user_id="u1", confidence=0.8)
        n_li = KGNode(id="n2", name="李四", display_name="李四",
                      node_type=KGNodeType.PERSON, user_id="u1", confidence=0.8)
        n_tencent = KGNode(id="n3", name="腾讯", display_name="腾讯",
                           node_type=KGNodeType.ORGANIZATION, user_id="u1", confidence=0.7)
        n_shenzhen = KGNode(id="n4", name="深圳", display_name="深圳",
                            node_type=KGNodeType.LOCATION, user_id="u1", confidence=0.7)
        n_programming = KGNode(id="n5", name="编程", display_name="编程",
                               node_type=KGNodeType.CONCEPT, user_id="u1", confidence=0.6)

        for n in [n_zhang, n_li, n_tencent, n_shenzhen, n_programming]:
            run(s.upsert_node(n))

        # 创建边
        edges = [
            KGEdge(source_id="n1", target_id="n2",
                   relation_type=KGRelationType.FRIEND_OF,
                   relation_label="朋友", user_id="u1", confidence=0.8),
            KGEdge(source_id="n2", target_id="n3",
                   relation_type=KGRelationType.WORKS_AT,
                   relation_label="在腾讯工作", user_id="u1", confidence=0.7),
            KGEdge(source_id="n3", target_id="n4",
                   relation_type=KGRelationType.RELATED_TO,
                   relation_label="位于", user_id="u1", confidence=0.6),
            KGEdge(source_id="n1", target_id="n5",
                   relation_type=KGRelationType.LIKES,
                   relation_label="喜欢", user_id="u1", confidence=0.8),
        ]
        for e in edges:
            run(s.upsert_edge(e))

        reasoning = KGReasoning(storage=s, max_depth=3, max_nodes_per_hop=10, min_confidence=0.1)
        yield s, reasoning
        run(s.close())


class TestReasoningResult:
    """ReasoningResult 辅助方法测试"""

    def test_empty_result(self):
        r = ReasoningResult()
        assert not r.has_results
        assert len(r.get_all_nodes()) == 0
        assert len(r.get_all_edges()) == 0
        assert len(r.get_fact_summary()) == 0

    def test_has_results(self):
        from iris_memory.knowledge_graph.kg_models import KGPath
        r = ReasoningResult(paths=[
            KGPath(nodes=[KGNode(name="a"), KGNode(name="b")],
                   edges=[KGEdge()], hop_count=1)
        ])
        assert r.has_results


class TestBFSReasoning:
    """BFS 推理核心测试"""

    def test_1_hop_reasoning(self, graph_env):
        """1 跳推理：张三的朋友"""
        s, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u1", max_depth=1))
        assert result.has_results
        assert len(result.seed_nodes) >= 1
        # 应找到 李四 和 编程
        all_nodes = result.get_all_nodes()
        node_names = [n.name for n in all_nodes]
        assert "李四" in node_names or "编程" in node_names

    def test_2_hop_reasoning(self, graph_env):
        """2 跳推理：张三 -> 李四 -> 腾讯"""
        s, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u1", max_depth=2))
        assert result.has_results
        all_nodes = result.get_all_nodes()
        node_names = [n.name for n in all_nodes]
        assert "腾讯" in node_names

    def test_3_hop_reasoning(self, graph_env):
        """3 跳推理：张三 -> 李四 -> 腾讯 -> 深圳"""
        s, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u1", max_depth=3))
        assert result.has_results
        all_nodes = result.get_all_nodes()
        node_names = [n.name for n in all_nodes]
        assert "深圳" in node_names

    def test_confidence_decay(self, graph_env):
        """置信度应随跳数衰减"""
        _, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u1", max_depth=3))
        if result.paths:
            # 多跳路径的置信度应低于 1 跳路径
            single_hop = [p for p in result.paths if p.hop_count == 1]
            multi_hop = [p for p in result.paths if p.hop_count >= 2]
            if single_hop and multi_hop:
                assert max(p.total_confidence for p in multi_hop) < max(p.total_confidence for p in single_hop)

    def test_max_depth_respected(self, graph_env):
        """max_depth 应被遵守"""
        _, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u1", max_depth=1))
        for path in result.paths:
            assert path.hop_count <= 1

    def test_no_seed_returns_empty(self, graph_env):
        """找不到种子节点应返回空结果"""
        _, reasoning = graph_env
        result = run(reasoning.reason("不存在的实体XYZ", user_id="u1"))
        assert not result.has_results

    def test_scope_isolation(self, graph_env):
        """不同用户看不到其他用户的图"""
        _, reasoning = graph_env
        result = run(reasoning.reason("张三", user_id="u_other"))
        assert not result.has_results


class TestSeedExtraction:
    """种子实体提取测试"""

    def test_extract_chinese_name(self):
        reasoning = KGReasoning(storage=None)  # storage not needed here
        entities = reasoning._extract_query_entities("张三最近怎么样了")
        assert "张三" in entities

    def test_extract_nickname(self):
        reasoning = KGReasoning(storage=None)
        entities = reasoning._extract_query_entities("小王今天来了吗")
        assert "小王" in entities

    def test_extract_english_proper_noun(self):
        reasoning = KGReasoning(storage=None)
        entities = reasoning._extract_query_entities("Alice is my friend")
        assert "Alice" in entities

    def test_extract_quoted_text(self):
        reasoning = KGReasoning(storage=None)
        entities = reasoning._extract_query_entities("你知道「编程」这件事吗")
        assert "编程" in entities

    def test_extract_relation_query(self):
        reasoning = KGReasoning(storage=None)
        entities = reasoning._extract_query_entities("张三是谁")
        assert "张三" in entities

    def test_extract_dedup(self):
        reasoning = KGReasoning(storage=None)
        entities = reasoning._extract_query_entities("张三和张三的朋友")
        assert entities.count("张三") == 1


class TestEntityRelations:
    """直接关系查询测试"""

    def test_query_entity_relations(self, graph_env):
        s, reasoning = graph_env
        relations = run(reasoning.query_entity_relations("张三", user_id="u1"))
        assert len(relations) >= 1
        neighbor_names = [n.name for _, n in relations]
        assert "李四" in neighbor_names or "编程" in neighbor_names

    def test_query_entity_no_match(self, graph_env):
        _, reasoning = graph_env
        relations = run(reasoning.query_entity_relations("不存在", user_id="u1"))
        assert len(relations) == 0


class TestVisibility:
    """scope 可见性测试"""

    def test_same_user_visible(self):
        reasoning = KGReasoning(storage=None)
        node = KGNode(user_id="u1", group_id=None)
        assert reasoning._is_visible(node, "u1", None)

    def test_different_user_invisible(self):
        reasoning = KGReasoning(storage=None)
        node = KGNode(user_id="u1", group_id=None)
        assert not reasoning._is_visible(node, "u2", None)

    def test_same_group_visible(self):
        reasoning = KGReasoning(storage=None)
        node = KGNode(user_id="u2", group_id="g1")
        assert reasoning._is_visible(node, "u1", "g1")

    def test_different_group_invisible(self):
        reasoning = KGReasoning(storage=None)
        node = KGNode(user_id="u2", group_id="g2")
        assert not reasoning._is_visible(node, "u1", "g1")


class TestBFSTimeout:
    """BFS 超时保护测试"""

    def test_bfs_completes_within_timeout(self, graph_env):
        """普通图遍历应在超时时间内完成"""
        _, reasoning = graph_env
        result = run(reasoning.reason(
            query="张三", user_id="u1",
        ))
        assert result is not None

    def test_bfs_timeout_constant_exists(self):
        """BFS_TIMEOUT 常量应存在且为合理值"""
        from iris_memory.knowledge_graph.kg_reasoning import BFS_TIMEOUT
        assert isinstance(BFS_TIMEOUT, (int, float))
        assert 0 < BFS_TIMEOUT <= 10

    def test_bfs_returns_partial_on_timeout(self, graph_env):
        """超时时应返回已收集的部分结果而非报错"""
        import iris_memory.knowledge_graph.kg_reasoning as mod
        _, reasoning = graph_env

        original_timeout = mod.BFS_TIMEOUT
        try:
            # 设置极短超时
            mod.BFS_TIMEOUT = 0.0
            result = run(reasoning.reason(
                query="张三", user_id="u1",
            ))
            # 即使超时也应返回有效结果对象
            assert result is not None
        finally:
            mod.BFS_TIMEOUT = original_timeout


class TestPathDedup:
    """路径去重测试"""

    def test_dedup_keeps_higher_confidence(self):
        from iris_memory.knowledge_graph.kg_models import KGPath
        reasoning = KGReasoning(storage=None)
        n1 = KGNode(id="a", name="a")
        n2 = KGNode(id="b", name="b")
        e = KGEdge(source_id="a", target_id="b")

        p1 = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.5, hop_count=1)
        p2 = KGPath(nodes=[n1, n2], edges=[e], total_confidence=0.9, hop_count=1)

        result = reasoning._deduplicate_paths([p1, p2])
        assert len(result) == 1
        assert result[0].total_confidence == 0.9

    def test_dedup_different_paths(self):
        from iris_memory.knowledge_graph.kg_models import KGPath
        reasoning = KGReasoning(storage=None)
        n1 = KGNode(id="a", name="a")
        n2 = KGNode(id="b", name="b")
        n3 = KGNode(id="c", name="c")
        e1 = KGEdge(source_id="a", target_id="b")
        e2 = KGEdge(source_id="a", target_id="c")

        p1 = KGPath(nodes=[n1, n2], edges=[e1], total_confidence=0.5, hop_count=1)
        p2 = KGPath(nodes=[n1, n3], edges=[e2], total_confidence=0.7, hop_count=1)

        result = reasoning._deduplicate_paths([p1, p2])
        assert len(result) == 2
