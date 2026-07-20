"""L3 知识图谱淘汰策略测试"""

import pytest
import pytest_asyncio
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from iris_memory.l3_kg import (
    GraphNode,
    GraphEdge,
    L3KGAdapter,
)
from iris_memory.config import init_config
from iris_memory.utils.forgetting import should_evict_kg_node


class TestAdapterEviction:
    """L3KGAdapter 淘汰功能测试"""

    @pytest.fixture
    def temp_dir(self):
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest_asyncio.fixture
    async def adapter(self, temp_dir):
        from unittest.mock import Mock

        astrbot_config = Mock()
        astrbot_config.__getitem__ = Mock(return_value={"enable": True})
        astrbot_config.__contains__ = Mock(return_value=True)
        init_config(astrbot_config, temp_dir)

        adapter = L3KGAdapter()
        await adapter.initialize()

        yield adapter

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_evict_nodes(self, adapter):
        node1 = GraphNode(id="", label="Person", name="Alice", content="测试用户")
        node1.id = node1.generate_id()

        node2 = GraphNode(id="", label="Person", name="Bob", content="测试用户")
        node2.id = node2.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)

        deleted = await adapter.evict_nodes([node1.id])

        assert deleted == 1

        stats = await adapter.get_stats()
        assert stats["node_count"] == 1

    @pytest.mark.asyncio
    async def test_evict_nodes_empty_list(self, adapter):
        deleted = await adapter.evict_nodes([])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_evict_nodes_unavailable_adapter(self, adapter):
        adapter._is_available = False

        deleted = await adapter.evict_nodes(["test_id"])
        assert deleted == 0

        adapter._is_available = True

    @pytest.mark.asyncio
    async def test_get_all_nodes(self, adapter):
        node = GraphNode(id="", label="Person", name="TestUser", content="测试用户")
        node.id = node.generate_id()

        await adapter.add_node(node)

        nodes = await adapter.get_all_nodes()

        assert len(nodes) >= 1
        assert any(n["id"] == node.id for n in nodes)

    @pytest.mark.asyncio
    async def test_get_all_nodes_empty(self, adapter):
        nodes = await adapter.get_all_nodes()
        assert nodes == []

    @pytest.mark.asyncio
    async def test_evict_nodes_with_nonexistent_id(self, adapter):
        deleted = await adapter.evict_nodes(["nonexistent_id"])
        assert deleted == len(["nonexistent_id"])

    @pytest.mark.asyncio
    async def test_evict_nodes_with_edges(self, adapter):
        node1 = GraphNode(id="", label="Person", name="Alice", content="用户1")
        node1.id = node1.generate_id()

        node2 = GraphNode(id="", label="Event", name="Conference", content="会议")
        node2.id = node2.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)

        edge = GraphEdge(
            source_id=node1.id, target_id=node2.id, relation_type="ATTENDED"
        )
        await adapter.add_edge(edge)

        deleted = await adapter.evict_nodes([node1.id])
        assert deleted == 1

        stats = await adapter.get_stats()
        assert stats["node_count"] == 1


class TestForgettingUtils:
    """遗忘工具函数测试"""

    @pytest.mark.asyncio
    async def test_should_evict_node_low_score(self):
        result = should_evict_kg_node(
            last_access_time=(datetime.now() - timedelta(days=60)).isoformat(),
            access_count=0,
            confidence=0.1,
            connected_count=0,
            source_memory_count=0,
            threshold=0.3,
            retention_days=30,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_should_evict_node_high_score(self):
        result = should_evict_kg_node(
            last_access_time=datetime.now().isoformat(),
            access_count=50,
            confidence=0.95,
            connected_count=5,
            source_memory_count=3,
            threshold=0.3,
            retention_days=30,
        )
        assert result is False
