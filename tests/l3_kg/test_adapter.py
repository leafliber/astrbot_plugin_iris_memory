"""L3 知识图谱适配器测试"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from iris_memory.l3_kg import GraphNode, GraphEdge, L3KGAdapter
from iris_memory.config import init_config


class TestL3KGAdapter:
    """L3KGAdapter 测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest_asyncio.fixture
    async def adapter(self, temp_dir):
        astrbot_config = Mock()
        astrbot_config.__getitem__ = Mock(return_value={"enable": True})
        astrbot_config.__contains__ = Mock(return_value=True)

        init_config(astrbot_config, temp_dir)

        adapter = L3KGAdapter()
        await adapter.initialize()

        yield adapter

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter._is_available
        assert adapter.name == "l3_kg"

    @pytest.mark.asyncio
    async def test_add_node(self, adapter):
        """测试添加节点"""
        node = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node.id = node.generate_id()

        success = await adapter.add_node(node)
        assert success

        # 验证节点已添加
        stats = await adapter.get_stats()
        assert stats["node_count"] == 1

    @pytest.mark.asyncio
    async def test_add_edge(self, adapter):
        """测试添加边"""
        # 创建两个节点
        node1 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node1.id = node1.generate_id()

        node2 = GraphNode(
            id="", label="Event", name="Conference", content="AI Conference 2024"
        )
        node2.id = node2.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)

        # 创建边
        edge = GraphEdge(
            source_id=node1.id, target_id=node2.id, relation_type="ATTENDED"
        )

        success = await adapter.add_edge(edge)
        assert success

        # 验证边已添加
        stats = await adapter.get_stats()
        assert stats["edge_count"] == 1

    @pytest.mark.asyncio
    async def test_expand_from_nodes(self, adapter):
        """测试路径扩展检索"""
        # 创建测试数据
        node1 = GraphNode(
            id="", label="Person", name="Alice", content="Alice is a software engineer"
        )
        node1.id = node1.generate_id()

        node2 = GraphNode(
            id="", label="Person", name="Bob", content="Bob is a data scientist"
        )
        node2.id = node2.generate_id()

        node3 = GraphNode(
            id="", label="Event", name="Conference", content="AI Conference 2024"
        )
        node3.id = node3.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)
        await adapter.add_node(node3)

        # 创建关系链：Alice -> Conference <- Bob
        edge1 = GraphEdge(
            source_id=node1.id, target_id=node3.id, relation_type="ATTENDED"
        )
        edge2 = GraphEdge(
            source_id=node2.id, target_id=node3.id, relation_type="ATTENDED"
        )

        await adapter.add_edge(edge1)
        await adapter.add_edge(edge2)

        # 从 Alice 出发进行路径扩展
        nodes, edges = await adapter.expand_from_nodes(node_ids=[node1.id], max_depth=2)

        assert len(nodes) >= 1
        assert len(edges) >= 1

    @pytest.mark.asyncio
    async def test_expand_from_nodes_filters_seeds_by_group_id(self, adapter):
        """回归：种子节点必须按 group_id 过滤，避免跨群节点泄漏

        此前 expand_from_nodes 未对种子节点按 group_id 过滤，
        传入其他群的节点 ID 作为种子时仍会被返回，导致跨群数据泄漏。
        """
        # 在两个群中各插入一个节点
        node_a = GraphNode(
            id="",
            label="Person",
            name="Alice",
            content="group A member",
            group_id="group_A",
        )
        node_a.id = node_a.generate_id()

        node_b = GraphNode(
            id="",
            label="Person",
            name="Bob",
            content="group B member",
            group_id="group_B",
        )
        node_b.id = node_b.generate_id()

        await adapter.add_node(node_a)
        await adapter.add_node(node_b)

        # 以两个群的节点 ID 作为种子，但只查 group_A
        nodes, edges = await adapter.expand_from_nodes(
            node_ids=[node_a.id, node_b.id], group_id="group_A", max_depth=1
        )

        returned_ids = {n["id"] for n in nodes}
        assert node_a.id in returned_ids
        assert node_b.id not in returned_ids, (
            "group_B 种子节点不应出现在 group_A 检索结果中"
        )

    @pytest.mark.asyncio
    async def test_get_stats(self, adapter):
        """测试获取统计信息"""
        stats = await adapter.get_stats()

        assert stats["available"]
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    # ------------------------------------------------------------------
    # merge_duplicate_nodes 事务原子性 / JSON 容错回归测试
    # ------------------------------------------------------------------

    def _insert_node_row(
        self,
        adapter,
        node_id: str,
        label: str,
        name: str,
        content: str = "",
        confidence: float = 0.5,
        access_count: int = 0,
        created_time: str = "2024-01-01T00:00:00",
        group_id: str = "",
        properties: str = "{}",
    ):
        """直接插入一行节点（绕过 add_node 的同 ID 合并，用于构造重复节点）"""
        adapter._db.execute(
            """INSERT INTO nodes
               (id, label, name, content, confidence, access_count,
                last_access_time, created_time, source_memory_id, group_id, properties)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id,
                label,
                name,
                content,
                confidence,
                access_count,
                None,
                created_time,
                None,
                group_id,
                properties,
            ),
        )
        adapter._db.commit()

    def _insert_edge_row(
        self,
        adapter,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        confidence: float = 0.5,
        access_count: int = 0,
        properties: str = "{}",
    ):
        """直接插入一行边"""
        adapter._db.execute(
            """INSERT INTO edges
               (source_id, target_id, relation_type, weight, confidence,
                access_count, last_access_time, created_time, source_memory_id, properties)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_id,
                target_id,
                relation_type,
                weight,
                confidence,
                access_count,
                None,
                "2024-01-01T00:00:00",
                None,
                properties,
            ),
        )
        adapter._db.commit()

    @pytest.mark.asyncio
    async def test_merge_duplicate_nodes_basic(self, adapter):
        """正常合并：同名同 label 的重复节点合并，边重定向"""
        # 两个同名同 label 但不同 id 的节点（绕过 add_node 合并）
        self._insert_node_row(
            adapter,
            "n_keep",
            "Person",
            "Alice",
            content="desc-1",
            created_time="2024-01-01T00:00:00",
            properties='{"k":"v1"}',
        )
        self._insert_node_row(
            adapter,
            "n_dup",
            "Person",
            "Alice",
            content="desc-2",
            created_time="2024-01-02T00:00:00",
            properties='{"k":"v2"}',
        )
        # 一个无关节点 + 边指向 dup，验证重定向到 keep
        self._insert_node_row(adapter, "n_other", "Event", "Conf", content="conf")
        self._insert_edge_row(
            adapter,
            "n_dup",
            "n_other",
            "ATTENDED",
        )

        merged, deleted = await adapter.merge_duplicate_nodes()

        assert merged == 1
        assert deleted == 1
        # dup 被删除
        remaining = adapter._db_fetchall("SELECT id FROM nodes ORDER BY id")
        ids = [r["id"] for r in remaining]
        assert "n_dup" not in ids
        assert "n_keep" in ids
        # 边重定向到 keep
        edges = adapter._db_fetchall("SELECT source_id, target_id FROM edges")
        assert len(edges) == 1
        assert edges[0]["source_id"] == "n_keep"

    @pytest.mark.asyncio
    async def test_merge_tolerates_corrupt_node_properties_json(self, adapter):
        """损坏的节点 properties JSON 不应中断合并，应回退为空字典继续"""
        self._insert_node_row(
            adapter,
            "n_keep",
            "Person",
            "Bob",
            created_time="2024-01-01T00:00:00",
            properties='{"valid":"yes"}',
        )
        # 损坏 JSON
        self._insert_node_row(
            adapter,
            "n_dup",
            "Person",
            "Bob",
            created_time="2024-01-02T00:00:00",
            properties="{broken json",
        )

        merged, deleted = await adapter.merge_duplicate_nodes()

        # 不抛异常，正常完成合并
        assert merged == 1
        assert deleted == 1
        remaining = adapter._db_fetchall("SELECT id FROM nodes")
        assert len(remaining) == 1
        assert remaining[0]["id"] == "n_keep"

    @pytest.mark.asyncio
    async def test_merge_tolerates_corrupt_edge_properties_json(self, adapter):
        """损坏的边 properties JSON 不应中断合并"""
        self._insert_node_row(
            adapter,
            "n_keep",
            "Person",
            "Carol",
            created_time="2024-01-01T00:00:00",
        )
        self._insert_node_row(
            adapter,
            "n_dup",
            "Person",
            "Carol",
            created_time="2024-01-02T00:00:00",
        )
        self._insert_node_row(adapter, "n_other", "Event", "Meet")
        # 损坏 JSON 的边
        self._insert_edge_row(
            adapter,
            "n_dup",
            "n_other",
            "ATTENDED",
            properties="{bad edge",
        )

        merged, deleted = await adapter.merge_duplicate_nodes()

        assert merged == 1
        assert deleted == 1
        # 边已重定向到 keep
        edges = adapter._db_fetchall("SELECT source_id FROM edges")
        assert len(edges) == 1
        assert edges[0]["source_id"] == "n_keep"

    @pytest.mark.asyncio
    async def test_merge_rolls_back_on_exception(self, adapter):
        """回归：循环内异常时必须 rollback，不得留下半合并的脏数据

        历史 bug：无 rollback，未提交的 DELETE/INSERT 会被下一个无关
        _db_write 的 commit 一并刷盘，造成节点/边不一致。
        """
        self._insert_node_row(
            adapter,
            "n_keep",
            "Person",
            "Dave",
            created_time="2024-01-01T00:00:00",
        )
        self._insert_node_row(
            adapter,
            "n_dup",
            "Person",
            "Dave",
            created_time="2024-01-02T00:00:00",
        )

        # 在合并循环中途注入异常：patch _merge_node_content 抛错
        with patch.object(
            adapter, "_merge_node_content", side_effect=RuntimeError("注入异常")
        ):
            merged, deleted = await adapter.merge_duplicate_nodes()

        assert merged == 0 and deleted == 0

        # 关键断言：异常后图谱应回滚到合并前状态，两个节点都仍在
        remaining = adapter._db_fetchall("SELECT id FROM nodes ORDER BY id")
        ids = [r["id"] for r in remaining]
        assert "n_keep" in ids
        assert "n_dup" in ids
        assert len(ids) == 2, "异常后不应有半合并的脏数据残留"

        # 验证连接仍可用：一次无关写操作不应刷盘脏数据
        await adapter.update_node_access(["n_keep"])
        remaining2 = adapter._db_fetchall("SELECT id FROM nodes ORDER BY id")
        assert len(remaining2) == 2, "后续 commit 不应刷盘已回滚的脏数据"
