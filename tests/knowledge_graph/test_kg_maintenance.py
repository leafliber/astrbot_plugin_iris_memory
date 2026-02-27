"""
知识图谱维护管理器单元测试
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.knowledge_graph.kg_maintenance import (
    KGMaintenanceManager,
    CleanupResult,
    MaintenanceReport,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_STALENESS_DAYS,
)


@pytest.fixture
def storage():
    """创建临时数据库的 KGStorage 实例"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg_maintenance.db"
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


def _make_node(name: str, user_id: str = "u1", **kwargs) -> KGNode:
    return KGNode(name=name, display_name=name, user_id=user_id, **kwargs)


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


class TestFindOrphanNodes:
    """孤立节点检测测试"""

    def test_no_orphans_when_all_connected(self, storage):
        """所有节点都有边连接时，不应有孤立节点"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        orphans = run(mgr.find_orphan_nodes())
        assert len(orphans) == 0

    def test_detect_orphan_node(self, storage):
        """只有节点无边时，应检测为孤立"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        # 只创建节点，不创建边
        run(storage.upsert_node(_make_node("Charlie")))

        # Alice -> Bob 有边
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        orphans = run(mgr.find_orphan_nodes())
        assert len(orphans) == 1  # Charlie 是孤立的

    def test_empty_graph_no_orphans(self, storage):
        """空图不应有孤立节点"""
        mgr = KGMaintenanceManager(storage)
        orphans = run(mgr.find_orphan_nodes())
        assert len(orphans) == 0

    def test_target_node_not_orphan(self, storage):
        """作为边的目标节点不应视为孤立"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        orphans = run(mgr.find_orphan_nodes())
        assert n2.id not in orphans


class TestRemoveOrphanNodes:
    """孤立节点移除测试"""

    def test_remove_orphan_nodes(self, storage):
        """移除孤立节点"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        orphan = run(storage.upsert_node(_make_node("Charlie")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        result = run(mgr.remove_orphan_nodes())

        assert result.removed_count == 1
        assert result.task_name == "孤立节点清理"

        # 验证 Charlie 已被删除
        node = run(storage.get_node(orphan.id))
        assert node is None

    def test_remove_no_orphans(self, storage):
        """无孤立节点时返回 0"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        result = run(mgr.remove_orphan_nodes())

        assert result.removed_count == 0

    def test_remove_multiple_orphans(self, storage):
        """移除多个孤立节点"""
        run(storage.upsert_node(_make_node("Orphan1")))
        run(storage.upsert_node(_make_node("Orphan2")))
        run(storage.upsert_node(_make_node("Orphan3")))

        mgr = KGMaintenanceManager(storage)
        result = run(mgr.remove_orphan_nodes())
        assert result.removed_count == 3


class TestCleanLowConfidenceEdges:
    """低置信度边清理测试"""

    def test_clean_low_confidence_stale_edges(self, storage):
        """清理低置信度且过期的边"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        # 创建低置信度边并手动设置旧的更新时间
        edge = _make_edge(n1.id, n2.id, confidence=0.1)
        edge.updated_time = datetime.now() - timedelta(days=60)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        stale_ids = run(mgr.find_low_confidence_stale_edges())
        assert len(stale_ids) == 1

    def test_skip_recent_low_confidence_edges(self, storage):
        """近期更新的低置信度边不应被清理"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        # 低置信度但刚更新
        edge = _make_edge(n1.id, n2.id, confidence=0.1)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        stale_ids = run(mgr.find_low_confidence_stale_edges())
        assert len(stale_ids) == 0

    def test_skip_high_confidence_stale_edges(self, storage):
        """高置信度但过期的边不应被清理"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        edge = _make_edge(n1.id, n2.id, confidence=0.8)
        edge.updated_time = datetime.now() - timedelta(days=60)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        stale_ids = run(mgr.find_low_confidence_stale_edges())
        assert len(stale_ids) == 0

    def test_clean_edges_with_custom_threshold(self, storage):
        """自定义阈值清理"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        edge = _make_edge(n1.id, n2.id, confidence=0.25)
        edge.updated_time = datetime.now() - timedelta(days=15)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        # 默认阈值 0.2 不会匹配 0.25
        stale_ids = run(mgr.find_low_confidence_stale_edges(
            confidence_threshold=0.3, staleness_days=10
        ))
        assert len(stale_ids) == 1

    def test_clean_low_confidence_edges_method(self, storage):
        """clean_low_confidence_edges 实际删除"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))

        edge = _make_edge(n1.id, n2.id, confidence=0.1)
        edge.updated_time = datetime.now() - timedelta(days=60)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        result = run(mgr.clean_low_confidence_edges())
        assert result.removed_count == 1
        assert result.task_name == "低置信度边清理"


class TestRemoveDanglingEdges:
    """悬空边清理测试"""

    def test_no_dangling_edges(self, storage):
        """无悬空边时返回空"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        mgr = KGMaintenanceManager(storage)
        dangling = run(mgr.find_dangling_edges())
        assert len(dangling) == 0

    def test_detect_dangling_edge_missing_source(self, storage):
        """检测源节点不存在的悬空边"""
        n2 = run(storage.upsert_node(_make_node("Bob")))

        # 临时禁用 FK 以插入悬空边
        edge = _make_edge("nonexistent_source", n2.id)
        assert storage._conn is not None
        storage._conn.execute("PRAGMA foreign_keys=OFF")
        d = edge.to_dict()
        storage._conn.execute(
            """INSERT INTO kg_edges
               (id, source_id, target_id, relation_type, relation_label,
                memory_id, user_id, group_id, persona_id, confidence, weight,
                properties, created_time, updated_time)
               VALUES (:id, :source_id, :target_id, :relation_type, :relation_label,
                       :memory_id, :user_id, :group_id, :persona_id, :confidence, :weight,
                       :properties, :created_time, :updated_time)""",
            d,
        )
        storage._conn.commit()
        storage._conn.execute("PRAGMA foreign_keys=ON")

        mgr = KGMaintenanceManager(storage)
        dangling = run(mgr.find_dangling_edges())
        assert len(dangling) == 1

    def test_remove_dangling_edges(self, storage):
        """移除悬空边"""
        n1 = run(storage.upsert_node(_make_node("Alice")))

        # 临时禁用 FK 以插入悬空边
        edge = _make_edge(n1.id, "nonexistent_target")
        assert storage._conn is not None
        storage._conn.execute("PRAGMA foreign_keys=OFF")
        d = edge.to_dict()
        storage._conn.execute(
            """INSERT INTO kg_edges
               (id, source_id, target_id, relation_type, relation_label,
                memory_id, user_id, group_id, persona_id, confidence, weight,
                properties, created_time, updated_time)
               VALUES (:id, :source_id, :target_id, :relation_type, :relation_label,
                       :memory_id, :user_id, :group_id, :persona_id, :confidence, :weight,
                       :properties, :created_time, :updated_time)""",
            d,
        )
        storage._conn.commit()
        storage._conn.execute("PRAGMA foreign_keys=ON")

        mgr = KGMaintenanceManager(storage)
        result = run(mgr.remove_dangling_edges())
        assert result.removed_count == 1
        assert result.task_name == "悬空边清理"


class TestRunFullCleanup:
    """完整清理流程测试"""

    def test_full_cleanup_empty_graph(self, storage):
        """空图执行完整清理"""
        mgr = KGMaintenanceManager(storage)
        report = run(mgr.run_full_cleanup())

        assert isinstance(report, MaintenanceReport)
        assert report.total_removed == 0
        assert len(report.results) == 3
        assert report.duration_seconds >= 0

    def test_full_cleanup_mixed_issues(self, storage):
        """混合问题的完整清理"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        run(storage.upsert_edge(_make_edge(n1.id, n2.id)))

        # 添加孤立节点
        run(storage.upsert_node(_make_node("Orphan")))

        mgr = KGMaintenanceManager(storage)
        report = run(mgr.run_full_cleanup())

        assert report.total_removed >= 1  # 至少移除了 Orphan

    def test_full_cleanup_order(self, storage):
        """清理顺序正确：悬空边 → 低置信度边 → 孤立节点"""
        mgr = KGMaintenanceManager(storage)
        report = run(mgr.run_full_cleanup())

        assert report.results[0].task_name == "悬空边清理"
        assert report.results[1].task_name == "低置信度边清理"
        assert report.results[2].task_name == "孤立节点清理"

    def test_full_cleanup_summary(self, storage):
        """清理报告摘要可生成"""
        mgr = KGMaintenanceManager(storage)
        report = run(mgr.run_full_cleanup())

        summary = report.summary()
        assert "维护报告" in summary
        assert "共清理" in summary

    def test_cascade_cleanup(self, storage):
        """边删除后产生的孤立节点也应被清理"""
        n1 = run(storage.upsert_node(_make_node("Alice")))
        n2 = run(storage.upsert_node(_make_node("Bob")))
        
        # 创建一条低置信度过期边 —— 这是唯一连接 n1 和 n2 的边
        edge = _make_edge(n1.id, n2.id, confidence=0.05)
        edge.updated_time = datetime.now() - timedelta(days=60)
        run(storage.upsert_edge(edge))

        mgr = KGMaintenanceManager(storage)
        report = run(mgr.run_full_cleanup())

        # 边被清理后，n1 和 n2 变成孤立节点，应被级联清理
        assert report.total_removed >= 3  # 1 edge + 2 nodes
