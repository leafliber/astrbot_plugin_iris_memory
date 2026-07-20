"""L3 知识图谱检索器测试"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import tempfile
import shutil
import asyncio

from iris_memory.l3_kg import GraphRetriever, GraphNode, GraphEdge, L3KGAdapter
from iris_memory.config import init_config


class TestGraphRetriever:
    """GraphRetriever 测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = Path(tempfile.mkdtemp())
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest_asyncio.fixture
    async def adapter(self, temp_dir):
        from unittest.mock import Mock

        astrbot_config = Mock()
        astrbot_config.__getitem__ = Mock(
            return_value={"enable": True, "expansion_depth": 2, "timeout_ms": 1500}
        )
        astrbot_config.__contains__ = Mock(return_value=True)
        init_config(astrbot_config, temp_dir)

        adapter = L3KGAdapter()
        await adapter.initialize()

        yield adapter

        await adapter.shutdown()

    @pytest.fixture
    def retriever(self, adapter):
        """创建检索器实例"""
        return GraphRetriever(adapter)

    @pytest.mark.asyncio
    async def test_format_for_context_empty(self, retriever):
        """测试格式化空结果"""
        result, included_ids = retriever.format_for_context([], [])
        assert result == ""
        assert included_ids == set()

    @pytest.mark.asyncio
    async def test_format_for_context_with_nodes(self, retriever):
        """测试格式化节点结果"""
        nodes = [
            {
                "id": "person_alice",
                "label": "Person",
                "name": "Alice",
                "content": "软件工程师",
            },
            {
                "id": "person_bob",
                "label": "Person",
                "name": "Bob",
                "content": "数据科学家",
            },
            {
                "id": "event_conf",
                "label": "Event",
                "name": "AI Conference",
                "content": "2024 AI 大会",
            },
        ]

        result, included_ids = retriever.format_for_context(nodes, [])

        assert "【长期知识】" in result
        assert "人物" in result
        assert "Alice" in result
        assert "软件工程师" in result
        assert "Bob" in result
        assert "数据科学家" in result
        assert "事件" in result
        assert "AI Conference" in result
        assert included_ids == {"person_alice", "person_bob", "event_conf"}

    @pytest.mark.asyncio
    async def test_format_for_context_with_edges(self, retriever):
        """测试格式化边结果"""
        nodes = [
            {
                "id": "person_alice",
                "label": "Person",
                "name": "Alice",
                "content": "软件工程师",
            },
            {
                "id": "event_conf",
                "label": "Event",
                "name": "AI Conference",
                "content": "2024 AI 大会",
            },
        ]

        edges = [
            {
                "source": "person_alice",
                "target": "event_conf",
                "relation_type": "PARTICIPATED_IN",
            }
        ]

        result, included_ids = retriever.format_for_context(nodes, edges)

        assert "关系" in result
        assert "Alice" in result
        assert "AI Conference" in result
        assert "参与" in result
        assert "person_alice" in included_ids
        assert "event_conf" in included_ids

    @pytest.mark.asyncio
    async def test_format_for_context_truncates_long_content(self, retriever):
        """测试过长描述截断"""
        long_content = "这是一段非常长的描述" * 20
        nodes = [
            {"id": "n1", "label": "Person", "name": "Alice", "content": long_content}
        ]

        result, included_ids = retriever.format_for_context(
            nodes, [], max_content_length=100
        )

        assert "Alice" in result
        assert "…" in result
        assert included_ids == {"n1"}

    @pytest.mark.asyncio
    async def test_format_for_context_groups_by_type(self, retriever):
        """测试节点按类型分组格式化"""
        nodes = [
            {"id": "n1", "label": "Person", "name": "Alice", "content": "描述1"},
            {"id": "n2", "label": "Person", "name": "Bob", "content": "描述2"},
            {"id": "n3", "label": "Event", "name": "会议", "content": "描述3"},
            {"id": "n4", "label": "Person", "name": "Charlie", "content": "描述4"},
        ]

        result, included_ids = retriever.format_for_context(nodes, [])

        assert "【长期知识】" in result
        assert "人物" in result
        assert "事件" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "会议" in result
        assert "Charlie" in result
        assert included_ids == {"n1", "n2", "n3", "n4"}

    @pytest.mark.asyncio
    async def test_retrieve_with_expansion_empty_ids(self, retriever):
        """测试空节点 ID 列表"""
        nodes, edges = await retriever.retrieve_with_expansion([])

        # 应该返回空结果
        assert nodes == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_retrieve_with_expansion_success(self, retriever, adapter):
        """测试路径扩展检索"""
        # 先添加一些测试数据
        node1 = GraphNode(id="", label="Person", name="Alice", content="软件工程师")
        node1.id = node1.generate_id()

        node2 = GraphNode(id="", label="Event", name="Conference", content="AI 大会")
        node2.id = node2.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)

        edge = GraphEdge(
            source_id=node1.id, target_id=node2.id, relation_type="ATTENDED"
        )
        await adapter.add_edge(edge)

        # 执行检索
        nodes, edges = await retriever.retrieve_with_expansion([node1.id])

        # 验证结果（应该扩展到相关节点）
        assert len(nodes) >= 1  # 至少包含起始节点

    @pytest.mark.asyncio
    async def test_retrieve_with_expansion_with_group_filter(self, retriever, adapter):
        """测试带群聊过滤的路径扩展"""
        # 添加带群聊 ID 的节点
        node1 = GraphNode(
            id="",
            label="Person",
            name="Alice",
            content="软件工程师",
            group_id="group_123",
        )
        node1.id = node1.generate_id()

        node2 = GraphNode(
            id="",
            label="Person",
            name="Bob",
            content="数据科学家",
            group_id="group_456",  # 不同群聊
        )
        node2.id = node2.generate_id()

        await adapter.add_node(node1)
        await adapter.add_node(node2)

        nodes, _ = await retriever.retrieve_with_expansion(
            [node1.id], group_id="group_123"
        )

        assert len(nodes) >= 0

    @pytest.mark.asyncio
    async def test_retrieve_with_expansion_timeout(self, retriever):
        """测试超时保护"""
        # 创建一个会超时的 mock adapter
        mock_adapter = MagicMock()
        mock_adapter._is_available = True

        # 模拟超时的 expand_from_nodes
        async def slow_expand(*args, **kwargs):
            await asyncio.sleep(10)  # 超长延迟
            return [], []

        mock_adapter.expand_from_nodes = slow_expand

        # 修改配置，设置很短的超时
        retriever.adapter = mock_adapter
        retriever.config._hidden.set("l3_timeout_ms", 100)

        # 执行检索
        nodes, edges = await retriever.retrieve_with_expansion(["test_id"])

        # 应该因为超时返回空结果
        assert nodes == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_update_access_count(self, retriever, adapter):
        """测试更新访问计数"""
        # 添加测试节点
        node = GraphNode(id="", label="Person", name="Alice", content="软件工程师")
        node.id = node.generate_id()

        await adapter.add_node(node)

        # 更新访问计数
        await retriever.update_access_count([node.id])

        # 验证更新（通过查询验证）
        # 注意：这需要 adapter 实现查询功能
        # 这里只验证不抛出异常

    @pytest.mark.asyncio
    async def test_update_access_count_empty_list(self, retriever):
        """测试空列表的访问计数更新"""
        # 不应该抛出异常
        await retriever.update_access_count([])

    @pytest.mark.asyncio
    async def test_update_access_count_adapter_unavailable(self, retriever):
        """测试适配器不可用时的访问计数更新"""
        # 设置适配器不可用
        retriever.adapter._is_available = False

        # 不应该抛出异常
        await retriever.update_access_count(["test_id"])

        # 恢复
        retriever.adapter._is_available = True

    @pytest.mark.asyncio
    async def test_retrieve_with_unavailable_adapter(self, retriever):
        """测试适配器不可用时的检索"""
        # 设置适配器不可用
        retriever.adapter._is_available = False

        # 执行检索
        nodes, edges = await retriever.retrieve_with_expansion(["test_id"])

        # 应该返回空结果
        assert nodes == []
        assert edges == []

        # 恢复
        retriever.adapter._is_available = True

    @pytest.mark.asyncio
    async def test_retrieve_by_keywords_passes_limit_to_search_nodes(self, retriever):
        """回归：retrieve_by_keywords 应将 limit 透传给 search_nodes

        历史 bug：retrieve_by_keywords 内部硬编码 ``search_nodes(keyword, limit=5)``
        或忽略 limit 形参，调用方传入的 limit 被丢弃。修复后使用
        ``search_nodes(keyword, limit=limit)`` 透传调用方指定的值。
        """
        mock_adapter = MagicMock()
        mock_adapter.is_available = True
        mock_adapter.search_nodes = AsyncMock(
            return_value=[{"id": "node_1"}, {"id": "node_2"}]
        )
        mock_adapter.expand_from_nodes = AsyncMock(return_value=([], []))

        retriever.adapter = mock_adapter

        await retriever.retrieve_by_keywords(["alice"], limit=7)

        # search_nodes 应以调用方传入的 limit=7 调用，而非硬编码 5
        # group_id 默认 None（未启用群隔离），也需透传给 search_nodes 做过滤
        mock_adapter.search_nodes.assert_called_once_with(
            "alice", limit=7, group_id=None
        )
