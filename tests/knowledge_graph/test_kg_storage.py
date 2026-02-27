"""
知识图谱 SQLite + FTS5 存储层单元测试
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


@pytest.fixture
def storage():
    """创建临时数据库的 KGStorage 实例"""
    s = KGStorage()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_kg.db"
        asyncio.get_event_loop().run_until_complete(s.initialize(db_path))
        yield s
        asyncio.get_event_loop().run_until_complete(s.close())


@pytest.fixture
def event_loop():
    """为 async 测试提供事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    """辅助函数：运行协程"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class TestKGStorageInit:
    """初始化测试"""

    def test_initialize_creates_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            s = KGStorage()
            run(s.initialize(db_path))
            assert db_path.exists()
            run(s.close())

    def test_initialize_idempotent(self):
        """重复初始化不应出错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            s = KGStorage()
            run(s.initialize(db_path))
            run(s.close())
            # 再次初始化
            s2 = KGStorage()
            run(s2.initialize(db_path))
            run(s2.close())


class TestKGStorageNodeCRUD:
    """节点 CRUD 测试"""

    def test_upsert_node_insert(self, storage):
        node = KGNode(
            name="张三",
            display_name="张三",
            node_type=KGNodeType.PERSON,
            user_id="u1",
        )
        result = run(storage.upsert_node(node))
        assert result.name == "张三"
        assert result.mention_count == 1

    def test_upsert_node_dedup(self, storage):
        """同名同用户的节点应被去重合并"""
        n1 = KGNode(name="张三", display_name="张三", user_id="u1")
        n2 = KGNode(name="张三", display_name="小张", user_id="u1")
        run(storage.upsert_node(n1))
        result = run(storage.upsert_node(n2))
        # 应该合并，mention_count +1
        assert result.mention_count == 2
        assert "小张" in result.aliases

    def test_upsert_node_different_users(self, storage):
        """不同用户的同名节点应分开存储"""
        n1 = KGNode(name="张三", display_name="张三", user_id="u1")
        n2 = KGNode(name="张三", display_name="张三", user_id="u2")
        r1 = run(storage.upsert_node(n1))
        r2 = run(storage.upsert_node(n2))
        assert r1.id != r2.id

    def test_get_node_by_id(self, storage):
        node = KGNode(name="test", display_name="Test", user_id="u1")
        run(storage.upsert_node(node))
        fetched = run(storage.get_node(node.id))
        assert fetched is not None
        assert fetched.name == "test"

    def test_get_node_not_found(self, storage):
        fetched = run(storage.get_node("nonexistent"))
        assert fetched is None

    def test_upsert_confidence_increments(self, storage):
        """多次 upsert 应增加置信度"""
        node = KGNode(name="test", display_name="test", user_id="u1", confidence=0.5)
        run(storage.upsert_node(node))
        result = run(storage.upsert_node(
            KGNode(name="test", display_name="test", user_id="u1")
        ))
        assert result.confidence > 0.5


class TestKGStorageEdgeCRUD:
    """边 CRUD 测试"""

    def test_upsert_edge(self, storage):
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))

        edge = KGEdge(
            source_id=n1.id,
            target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF,
            relation_label="好朋友",
            user_id="u1",
        )
        result = run(storage.upsert_edge(edge))
        assert result.relation_type == KGRelationType.FRIEND_OF

    def test_upsert_edge_dedup(self, storage):
        """同 source/target/relation_type 边应去重"""
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))

        e1 = KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF, user_id="u1",
        )
        e2 = KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF, user_id="u1",
        )
        run(storage.upsert_edge(e1))
        result = run(storage.upsert_edge(e2))
        assert result.weight > 1.0  # weight should increase

    def test_get_edges_from(self, storage):
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        n3 = KGNode(name="c", display_name="C", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))
        run(storage.upsert_node(n3))

        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF, user_id="u1",
        )))
        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n3.id,
            relation_type=KGRelationType.KNOWS, user_id="u1",
        )))

        edges = run(storage.get_edges_from(n1.id))
        assert len(edges) == 2

    def test_get_edges_to(self, storage):
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))
        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.LIKES, user_id="u1",
        )))

        edges = run(storage.get_edges_to(n2.id))
        assert len(edges) == 1


class TestKGStorageNeighbors:
    """邻居查询测试"""

    def test_get_neighbors_bidirectional(self, storage):
        """双向邻居查询"""
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        n3 = KGNode(name="c", display_name="C", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))
        run(storage.upsert_node(n3))

        # n1 -> n2
        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF, user_id="u1",
        )))
        # n3 -> n2
        run(storage.upsert_edge(KGEdge(
            source_id=n3.id, target_id=n2.id,
            relation_type=KGRelationType.KNOWS, user_id="u1",
        )))

        # n2 的邻居应该有 n1 和 n3
        neighbors = run(storage.get_neighbors(n2.id))
        neighbor_ids = [n.id for _, n in neighbors]
        assert n1.id in neighbor_ids
        assert n3.id in neighbor_ids


class TestKGStorageSearch:
    """FTS5 搜索测试"""

    def test_search_nodes_by_name(self, storage):
        run(storage.upsert_node(KGNode(
            name="张三", display_name="张三",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))
        run(storage.upsert_node(KGNode(
            name="李四", display_name="李四",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))

        results = run(storage.search_nodes("张三", user_id="u1"))
        assert len(results) >= 1
        assert any(n.name == "张三" for n in results)

    def test_search_nodes_scope_filter(self, storage):
        """搜索应遵守 scope 隔离"""
        run(storage.upsert_node(KGNode(
            name="secret", display_name="Secret",
            user_id="u1",
        )))
        run(storage.upsert_node(KGNode(
            name="secret", display_name="Secret",
            user_id="u2",
        )))

        results = run(storage.search_nodes("secret", user_id="u1"))
        assert all(n.user_id == "u1" for n in results)

    def test_search_nodes_empty_query(self, storage):
        results = run(storage.search_nodes("", user_id="u1"))
        assert len(results) == 0

    def test_search_nodes_no_match(self, storage):
        results = run(storage.search_nodes("不存在的实体", user_id="u1"))
        assert len(results) == 0

    def test_search_nodes_by_type(self, storage):
        run(storage.upsert_node(KGNode(
            name="北京", display_name="北京",
            user_id="u1", node_type=KGNodeType.LOCATION,
        )))
        run(storage.upsert_node(KGNode(
            name="北京人", display_name="北京人",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))

        results = run(storage.search_nodes(
            "北京", user_id="u1", node_type=KGNodeType.LOCATION,
        ))
        assert all(n.node_type == KGNodeType.LOCATION for n in results)


class TestKGStorageBulkOps:
    """批量操作测试"""

    def test_delete_by_memory_id(self, storage):
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))
        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.LIKES, memory_id="m1", user_id="u1",
        )))

        count = run(storage.delete_by_memory_id("m1"))
        assert count == 1

    def test_delete_user_data(self, storage):
        run(storage.upsert_node(KGNode(name="a", display_name="A", user_id="u1")))
        run(storage.upsert_node(KGNode(name="b", display_name="B", user_id="u2")))

        count = run(storage.delete_user_data("u1"))
        assert count >= 1

        stats = run(storage.get_stats(user_id="u1"))
        assert stats["nodes"] == 0

    def test_delete_all(self, storage):
        run(storage.upsert_node(KGNode(name="a", display_name="A", user_id="u1")))
        run(storage.upsert_node(KGNode(name="b", display_name="B", user_id="u1")))

        count = run(storage.delete_all())
        assert count >= 2

        stats = run(storage.get_stats())
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_get_stats(self, storage):
        n1 = KGNode(name="a", display_name="A", user_id="u1")
        n2 = KGNode(name="b", display_name="B", user_id="u1")
        run(storage.upsert_node(n1))
        run(storage.upsert_node(n2))
        run(storage.upsert_edge(KGEdge(
            source_id=n1.id, target_id=n2.id,
            relation_type=KGRelationType.FRIEND_OF, user_id="u1",
        )))

        stats = run(storage.get_stats())
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_get_stats_filtered(self, storage):
        run(storage.upsert_node(KGNode(name="a", display_name="A", user_id="u1")))
        run(storage.upsert_node(KGNode(name="b", display_name="B", user_id="u2")))

        stats = run(storage.get_stats(user_id="u1"))
        assert stats["nodes"] == 1


class TestKGStorageNormalization:
    """名称规范化测试"""

    def test_normalize_name(self):
        assert KGStorage._normalize_name("  Hello  ") == "hello"
        assert KGStorage._normalize_name("张三") == "张三"
        assert KGStorage._normalize_name("  Test,Test  ") == "test,test"

    def test_name_normalization_in_upsert(self, storage):
        """名称应在 upsert 时被规范化"""
        n1 = KGNode(name="  Hello  ", display_name="Hello", user_id="u1")
        result = run(storage.upsert_node(n1))
        assert result.name == "hello"


class TestKGStorageChineseSearch:
    """中文搜索精度测试"""

    def test_chinese_exact_match(self, storage):
        """中文搜索应精确匹配，不误匹配子串"""
        run(storage.upsert_node(KGNode(
            name="张三", display_name="张三",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))
        run(storage.upsert_node(KGNode(
            name="张三丰", display_name="张三丰",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))

        results = run(storage.search_nodes("张三", user_id="u1"))
        # 精确匹配应优先返回 "张三"
        assert len(results) >= 1
        assert results[0].name == "张三"

    def test_chinese_fallback_to_like(self, storage):
        """精确匹配无结果时应退回模糊搜索"""
        run(storage.upsert_node(KGNode(
            name="北京大学", display_name="北京大学",
            user_id="u1", node_type=KGNodeType.LOCATION,
        )))

        results = run(storage.search_nodes("北京", user_id="u1"))
        assert len(results) >= 1

    def test_english_uses_fts(self, storage):
        """英文查询应使用 FTS5"""
        run(storage.upsert_node(KGNode(
            name="alice", display_name="Alice",
            user_id="u1", node_type=KGNodeType.PERSON,
        )))

        results = run(storage.search_nodes("alice", user_id="u1"))
        assert len(results) >= 1

    def test_is_chinese(self):
        """中文检测辅助方法"""
        assert KGStorage._is_chinese("张三") is True
        assert KGStorage._is_chinese("Hello") is False
        assert KGStorage._is_chinese("张三abc") is True
        assert KGStorage._is_chinese("") is False


class TestKGStorageScopeFilter:
    """Scope 过滤安全性测试"""

    def test_no_user_id_returns_empty(self, storage):
        """无 user_id 时应返回空列表（安全约束）"""
        run(storage.upsert_node(KGNode(
            name="secret", display_name="Secret", user_id="u1",
        )))

        results = run(storage.search_nodes("secret", user_id=None))
        assert len(results) == 0

    def test_empty_user_id_returns_empty(self, storage):
        """空字符串 user_id 应返回空列表"""
        run(storage.upsert_node(KGNode(
            name="secret", display_name="Secret", user_id="u1",
        )))

        results = run(storage.search_nodes("secret", user_id=""))
        assert len(results) == 0


class TestKGStorageCache:
    """节点缓存测试"""

    def test_cache_hit(self, storage):
        """重复 upsert 应命中缓存"""
        n1 = KGNode(name="张三", display_name="张三", user_id="u1")
        run(storage.upsert_node(n1))

        # 第二次 upsert 应命中缓存
        n2 = KGNode(name="张三", display_name="小张", user_id="u1")
        result = run(storage.upsert_node(n2))
        assert result.mention_count == 2

        # 验证缓存中有数据
        cache_key = storage._node_cache_key("张三", "u1", None)
        cached = storage._get_from_cache(cache_key)
        assert cached is not None
        assert cached.mention_count == 2

    def test_cache_invalidation_on_delete(self, storage):
        """删除操作应失效缓存"""
        run(storage.upsert_node(KGNode(name="a", display_name="A", user_id="u1")))

        # 验证缓存存在
        cache_key = storage._node_cache_key("a", "u1", None)
        assert storage._get_from_cache(cache_key) is not None

        # 删除后缓存应失效
        run(storage.delete_user_data("u1"))
        assert storage._get_from_cache(cache_key) is None

    def test_cache_invalidation_on_delete_all(self, storage):
        """全部删除应清空缓存"""
        run(storage.upsert_node(KGNode(name="a", display_name="A", user_id="u1")))
        run(storage.upsert_node(KGNode(name="b", display_name="B", user_id="u2")))

        assert len(storage._node_cache) >= 2

        run(storage.delete_all())
        assert len(storage._node_cache) == 0


class TestKGStorageDeleteByGroup:
    """按群组删除测试"""

    def test_delete_by_group(self, storage):
        """应能按群组删除图谱数据"""
        run(storage.upsert_node(KGNode(
            name="a", display_name="A", user_id="u1", group_id="g1"
        )))
        run(storage.upsert_node(KGNode(
            name="b", display_name="B", user_id="u2", group_id="g1"
        )))
        run(storage.upsert_node(KGNode(
            name="c", display_name="C", user_id="u1", group_id="g2"
        )))

        count = run(storage.delete_user_data_by_group("g1"))
        assert count >= 2

        stats = run(storage.get_stats())
        assert stats["nodes"] == 1  # 只剩 g2 的节点


class TestKGStorageSQLAggregation:
    """SQL 聚合方法测试"""

    def _make_node(self, name: str, user_id: str = "u1", **kwargs) -> KGNode:
        return KGNode(name=name, display_name=name, user_id=user_id, **kwargs)

    def _make_edge(self, source_id: str, target_id: str, **kwargs) -> KGEdge:
        return KGEdge(source_id=source_id, target_id=target_id, user_id="u1", **kwargs)

    def test_get_node_count(self, storage):
        """节点计数"""
        assert run(storage.get_node_count()) == 0
        run(storage.upsert_node(self._make_node("A")))
        run(storage.upsert_node(self._make_node("B")))
        assert run(storage.get_node_count()) == 2

    def test_get_edge_count(self, storage):
        """边计数"""
        n1 = run(storage.upsert_node(self._make_node("A")))
        n2 = run(storage.upsert_node(self._make_node("B")))
        assert run(storage.get_edge_count()) == 0
        run(storage.upsert_edge(self._make_edge(n1.id, n2.id)))
        assert run(storage.get_edge_count()) == 1

    def test_get_avg_confidence(self, storage):
        """平均置信度"""
        run(storage.upsert_node(self._make_node("A", confidence=0.4)))
        run(storage.upsert_node(self._make_node("B", confidence=0.8)))

        result = run(storage.get_avg_confidence())
        assert abs(result["nodes"] - 0.6) < 0.01

    def test_get_avg_confidence_empty(self, storage):
        """空图的平均置信度"""
        result = run(storage.get_avg_confidence())
        assert result["nodes"] == 0.0
        assert result["edges"] == 0.0

    def test_get_low_confidence_counts(self, storage):
        """低置信度计数"""
        run(storage.upsert_node(self._make_node("A", confidence=0.1)))
        run(storage.upsert_node(self._make_node("B", confidence=0.5)))
        run(storage.upsert_node(self._make_node("C", confidence=0.8)))

        result = run(storage.get_low_confidence_counts(threshold=0.3))
        assert result["nodes"] == 1

    def test_get_node_type_distribution(self, storage):
        """节点类型分布"""
        run(storage.upsert_node(self._make_node("A", node_type=KGNodeType.PERSON)))
        run(storage.upsert_node(self._make_node("B", node_type=KGNodeType.PERSON)))
        run(storage.upsert_node(self._make_node("C", node_type=KGNodeType.LOCATION)))

        dist = run(storage.get_node_type_distribution())
        assert dist["person"] == 2
        assert dist["location"] == 1

    def test_get_relation_type_distribution(self, storage):
        """关系类型分布"""
        n1 = run(storage.upsert_node(self._make_node("A")))
        n2 = run(storage.upsert_node(self._make_node("B")))
        n3 = run(storage.upsert_node(self._make_node("C")))
        run(storage.upsert_edge(self._make_edge(n1.id, n2.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(self._make_edge(n1.id, n3.id, relation_type=KGRelationType.LIKES)))
        run(storage.upsert_edge(self._make_edge(n2.id, n3.id, relation_type=KGRelationType.KNOWS)))

        dist = run(storage.get_relation_type_distribution())
        assert dist["likes"] == 2
        assert dist["knows"] == 1

    def test_iter_nodes_batch(self, storage):
        """分批迭代节点"""
        for i in range(5):
            run(storage.upsert_node(self._make_node(f"N{i}")))

        batches = run(storage.iter_nodes_batch(batch_size=2))
        total = sum(len(b) for b in batches)
        assert total == 5
        assert len(batches) == 3  # 2+2+1

    def test_iter_edges_batch(self, storage):
        """分批迭代边"""
        nodes = []
        for i in range(4):
            nodes.append(run(storage.upsert_node(self._make_node(f"N{i}"))))
        for i in range(3):
            run(storage.upsert_edge(self._make_edge(nodes[i].id, nodes[i+1].id)))

        batches = run(storage.iter_edges_batch(batch_size=2))
        total = sum(len(b) for b in batches)
        assert total == 3

    def test_get_low_confidence_stale_edge_ids(self, storage):
        """低置信度过期边查询"""
        from datetime import datetime, timedelta

        n1 = run(storage.upsert_node(self._make_node("A")))
        n2 = run(storage.upsert_node(self._make_node("B")))

        old_edge = self._make_edge(n1.id, n2.id, confidence=0.1)
        old_edge.updated_time = datetime.now() - timedelta(days=60)
        run(storage.upsert_edge(old_edge))

        cutoff = datetime.now() - timedelta(days=30)
        ids = run(storage.get_low_confidence_stale_edge_ids(0.2, cutoff.isoformat()))
        assert len(ids) == 1
