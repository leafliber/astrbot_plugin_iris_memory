"""
知识图谱质量报告生成器单元测试
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
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.knowledge_graph.kg_quality import (
    KGQualityReporter,
    QualityReport,
    LowConfidenceStats,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
)


@pytest.fixture
def storage():
    """创建临时数据库的 KGStorage 实例"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg_quality.db"
        asyncio.get_event_loop().run_until_complete(s.initialize(db_path))
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_node(
    name: str,
    user_id: str = "u1",
    node_type: KGNodeType = KGNodeType.PERSON,
    confidence: float = 0.5,
    **kwargs,
) -> KGNode:
    return KGNode(
        name=name, display_name=name,
        user_id=user_id, node_type=node_type,
        confidence=confidence, **kwargs,
    )


def _make_edge(
    source_id: str,
    target_id: str,
    user_id: str = "u1",
    relation_type: KGRelationType = KGRelationType.RELATED_TO,
    confidence: float = 0.5,
    **kwargs,
) -> KGEdge:
    return KGEdge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        user_id=user_id,
        confidence=confidence,
        **kwargs,
    )


class TestOrphanNodeRatio:
    """孤立节点比例测试"""

    def test_no_orphans(self, storage):
        """所有节点连接时比例为 0"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        reporter = KGQualityReporter(storage)
        ratio = run(reporter.get_orphan_node_ratio())
        assert ratio == 0.0

    def test_all_orphans(self, storage):
        """所有节点孤立时比例为 1"""
        run(storage.upsert_node(_make_node("Alice")))
        run(storage.upsert_node(_make_node("Bob")))

        reporter = KGQualityReporter(storage)
        ratio = run(reporter.get_orphan_node_ratio())
        assert ratio == 1.0

    def test_partial_orphans(self, storage):
        """部分节点孤立"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_node(_make_node("Charlie")))  # 孤立
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        reporter = KGQualityReporter(storage)
        ratio = run(reporter.get_orphan_node_ratio())
        assert abs(ratio - 1.0 / 3.0) < 0.01

    def test_empty_graph_zero_ratio(self, storage):
        """空图返回 0"""
        reporter = KGQualityReporter(storage)
        ratio = run(reporter.get_orphan_node_ratio())
        assert ratio == 0.0


class TestLowConfidenceStats:
    """低置信度统计测试"""

    def test_no_low_confidence(self, storage):
        """无低置信度数据"""
        n1 = run(storage.upsert_node(_make_node("Alice", confidence=0.8)))
        n2 = run(storage.upsert_node(_make_node("Bob", confidence=0.9)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, confidence=0.7)))

        reporter = KGQualityReporter(storage)
        stats = run(reporter.get_low_confidence_stats())

        assert stats.low_confidence_node_count == 0
        assert stats.low_confidence_edge_count == 0
        assert stats.low_confidence_node_ratio == 0.0
        assert stats.low_confidence_edge_ratio == 0.0

    def test_all_low_confidence(self, storage):
        """全低置信度"""
        n1 = run(storage.upsert_node(_make_node("Alice", confidence=0.1)))
        n2 = run(storage.upsert_node(_make_node("Bob", confidence=0.2)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, confidence=0.1)))

        reporter = KGQualityReporter(storage)
        stats = run(reporter.get_low_confidence_stats())

        assert stats.low_confidence_node_count == 2
        assert stats.low_confidence_edge_count == 1
        assert stats.low_confidence_node_ratio == 1.0
        assert stats.low_confidence_edge_ratio == 1.0

    def test_custom_threshold(self, storage):
        """自定义阈值"""
        n1 = run(storage.upsert_node(_make_node("Alice", confidence=0.4)))
        n2 = run(storage.upsert_node(_make_node("Bob", confidence=0.6)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, confidence=0.4)))

        reporter = KGQualityReporter(storage)
        stats = run(reporter.get_low_confidence_stats(threshold=0.5))

        assert stats.low_confidence_node_count == 1  # Alice
        assert stats.low_confidence_edge_count == 1
        assert stats.threshold == 0.5

    def test_empty_graph(self, storage):
        """空图统计"""
        reporter = KGQualityReporter(storage)
        stats = run(reporter.get_low_confidence_stats())
        assert stats.low_confidence_node_count == 0
        assert stats.low_confidence_node_ratio == 0.0


class TestRelationTypeDistribution:
    """关系类型分布测试"""

    def test_distribution(self, storage):
        """关系类型分布统计"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        n3 = run(storage.upsert_node(_make_node("Charlie")))

        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n1.id, n3.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(_make_edge(n2.id, n3.id, relation_type=KGRelationType.FRIEND_OF)))

        reporter = KGQualityReporter(storage)
        dist = run(reporter.get_relation_type_distribution())

        assert dist["likes"] == 2
        assert dist["friend_of"] == 1

    def test_empty_graph(self, storage):
        """空图返回空分布"""
        reporter = KGQualityReporter(storage)
        dist = run(reporter.get_relation_type_distribution())
        assert dist == {}


class TestNodeTypeDistribution:
    """节点类型分布测试"""

    def test_distribution(self, storage):
        """节点类型分布统计"""
        run(storage.upsert_node(_make_node("Alice", node_type=KGNodeType.PERSON)))
        run(storage.upsert_node(_make_node("Bob", node_type=KGNodeType.PERSON)))
        run(storage.upsert_node(_make_node("Beijing", node_type=KGNodeType.LOCATION)))

        reporter = KGQualityReporter(storage)
        dist = run(reporter.get_node_type_distribution())

        assert dist["person"] == 2
        assert dist["location"] == 1

    def test_empty_graph(self, storage):
        """空图返回空分布"""
        reporter = KGQualityReporter(storage)
        dist = run(reporter.get_node_type_distribution())
        assert dist == {}


class TestAvgConfidence:
    """平均置信度测试"""

    def test_avg_confidence(self, storage):
        """平均置信度计算"""
        n1 = run(storage.upsert_node(_make_node("Alice", confidence=0.6)))
        n2 = run(storage.upsert_node(_make_node("Bob", confidence=0.8)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, confidence=0.7)))

        reporter = KGQualityReporter(storage)
        avg = run(reporter.get_avg_confidence())

        assert abs(avg["nodes"] - 0.7) < 0.01
        assert abs(avg["edges"] - 0.7) < 0.01

    def test_empty_graph(self, storage):
        """空图置信度为 0"""
        reporter = KGQualityReporter(storage)
        avg = run(reporter.get_avg_confidence())
        assert avg["nodes"] == 0.0
        assert avg["edges"] == 0.0


class TestGenerateReport:
    """完整质量报告测试"""

    def test_report_empty_graph(self, storage):
        """空图报告"""
        reporter = KGQualityReporter(storage)
        report = run(reporter.generate_report())

        assert isinstance(report, QualityReport)
        assert report.total_nodes == 0
        assert report.total_edges == 0
        assert report.orphan_node_count == 0
        assert report.orphan_node_ratio == 0.0
        assert report.avg_node_confidence == 0.0
        assert report.avg_edge_confidence == 0.0
        assert report.avg_edges_per_node == 0.0

    def test_report_with_data(self, storage):
        """有数据的完整报告"""
        n1 = run(storage.upsert_node(_make_node("Alice", node_type=KGNodeType.PERSON, confidence=0.8)))
        n2 = run(storage.upsert_node(_make_node("Bob", node_type=KGNodeType.PERSON, confidence=0.6)))
        n3 = run(storage.upsert_node(_make_node("Beijing", node_type=KGNodeType.LOCATION, confidence=0.1)))

        run(storage.upsert_edge(_make_edge(n1.id, n2.id, relation_type=KGRelationType.FRIEND_OF, confidence=0.9)))
        run(storage.upsert_edge(_make_edge(n1.id, n3.id, relation_type=KGRelationType.LIVES_IN, confidence=0.1)))

        reporter = KGQualityReporter(storage)
        report = run(reporter.generate_report())

        assert report.total_nodes == 3
        assert report.total_edges == 2
        assert report.orphan_node_count == 0  # 所有节点都有边连接

        # 低置信度检查（默认阈值 0.3）
        assert report.low_confidence_stats.low_confidence_node_count == 1  # Beijing
        assert report.low_confidence_stats.low_confidence_edge_count == 1  # lives_in

        # 分布
        assert report.node_type_distribution["person"] == 2
        assert report.node_type_distribution["location"] == 1
        assert report.relation_type_distribution["friend_of"] == 1
        assert report.relation_type_distribution["lives_in"] == 1

        # 平均值
        assert report.avg_edges_per_node > 0

    def test_report_with_orphans(self, storage):
        """带孤立节点的报告"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_node(_make_node("Orphan")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        reporter = KGQualityReporter(storage)
        report = run(reporter.generate_report())

        assert report.orphan_node_count == 1
        assert abs(report.orphan_node_ratio - 1.0 / 3.0) < 0.01

    def test_report_summary_readable(self, storage):
        """报告摘要可读"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        reporter = KGQualityReporter(storage)
        report = run(reporter.generate_report())
        summary = report.summary()

        assert "图谱质量报告" in summary
        assert "节点总数" in summary
        assert "边总数" in summary
        assert "孤立节点" in summary

    def test_report_custom_threshold(self, storage):
        """自定义阈值报告"""
        n1 = run(storage.upsert_node(_make_node("Alice", confidence=0.4)))
        n2 = run(storage.upsert_node(_make_node("Bob", confidence=0.6)))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id, confidence=0.4)))

        reporter = KGQualityReporter(storage)
        report = run(reporter.generate_report(low_confidence_threshold=0.5))

        assert report.low_confidence_stats.threshold == 0.5
        assert report.low_confidence_stats.low_confidence_node_count == 1  # Alice
