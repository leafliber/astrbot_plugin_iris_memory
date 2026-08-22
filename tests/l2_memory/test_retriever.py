"""L2 记忆检索器测试"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.l2_memory.retriever import MemoryRetriever
from iris_memory.l2_memory.models import MemoryEntry, MemorySearchResult
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.core.components import ComponentManager


class TestMemoryRetriever:
    """MemoryRetriever 测试"""

    @pytest.fixture
    def mock_adapter(self):
        """模拟 L2 适配器"""
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = True
        adapter.retrieve = AsyncMock(return_value=[])
        adapter.add_memory = AsyncMock(return_value="mem_001")
        adapter.update_access = AsyncMock(return_value=True)
        return adapter

    @pytest.fixture
    def mock_manager(self, mock_adapter):
        """模拟组件管理器"""
        manager = Mock(spec=ComponentManager)
        manager.get_component = Mock(return_value=mock_adapter)
        return manager

    @pytest.fixture
    def mock_config(self):
        """模拟配置（关闭混合检索，保持纯向量路径语义）"""
        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.top_k": 10,
                "isolation_config.enable_group_memory_isolation": False,
                "l2_memory.relevance_threshold": 0.3,
                "l2_memory.enable_hybrid_retrieval": False,
                "token_budget_max_tokens": 2000,
            }.get(key, default)
        )
        return config

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_manager, mock_config):
        """测试检索成功"""
        # 准备测试数据
        entry = MemoryEntry(
            id="mem_001", content="用户喜欢吃苹果", metadata={"group_id": "group_123"}
        )
        result = MemorySearchResult(entry=entry, score=0.9, distance=0.1)

        mock_adapter = mock_manager.get_component.return_value
        mock_adapter.retrieve = AsyncMock(return_value=[result])

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(mock_manager)
            results = await retriever.retrieve("喜欢吃什么", group_id="group_123")

            assert len(results) == 1
            assert results[0].entry.content == "用户喜欢吃苹果"

    @pytest.mark.asyncio
    async def test_retrieve_unavailable(self, mock_config):
        """测试适配器不可用时检索"""
        manager = Mock(spec=ComponentManager)
        manager.get_component = Mock(return_value=None)

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(manager)
            results = await retriever.retrieve("测试查询")

            assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_group_isolation(self, mock_manager, mock_config):
        """测试群聊隔离检索"""
        mock_config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.top_k": 10,
                "isolation_config.enable_group_memory_isolation": True,
                "l2_memory.relevance_threshold": 0.3,
                "l2_memory.enable_hybrid_retrieval": False,
            }.get(key, default)
        )

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(mock_manager)
            await retriever.retrieve("测试查询", group_id="group_123")

            # 验证传递了 group_id
            mock_adapter = mock_manager.get_component.return_value
            mock_adapter.retrieve.assert_called_once()
            call_args = mock_adapter.retrieve.call_args
            assert call_args[0][1] == "group_123"  # 第二个参数是 group_id

    @pytest.mark.asyncio
    async def test_retrieve_without_group_isolation(self, mock_manager, mock_config):
        """测试关闭群聊隔离检索"""
        mock_config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.top_k": 10,
                "isolation_config.enable_group_memory_isolation": False,
                "l2_memory.relevance_threshold": 0.3,
                "l2_memory.enable_hybrid_retrieval": False,
            }.get(key, default)
        )

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(mock_manager)
            await retriever.retrieve("测试查询", group_id="group_123")

            # 验证未传递 group_id
            mock_adapter = mock_manager.get_component.return_value
            mock_adapter.retrieve.assert_called_once()
            call_args = mock_adapter.retrieve.call_args
            assert call_args[0][1] is None  # 第二个参数应该是 None

    @pytest.mark.asyncio
    async def test_add_from_summary_success(self, mock_manager):
        """测试从总结写入成功"""
        retriever = MemoryRetriever(mock_manager)
        memory_id = await retriever.add_from_summary(
            "用户今天提到了喜欢吃苹果",
            metadata={"group_id": "group_123", "confidence": 0.9},
        )

        assert memory_id == "mem_001"

    @pytest.mark.asyncio
    async def test_add_from_summary_unavailable(self, mock_config):
        """测试适配器不可用时写入"""
        manager = Mock(spec=ComponentManager)
        manager.get_component = Mock(return_value=None)

        retriever = MemoryRetriever(manager)
        memory_id = await retriever.add_from_summary("测试内容")

        assert memory_id is None

    @pytest.mark.asyncio
    async def test_update_access(self, mock_manager):
        """测试更新访问信息"""
        retriever = MemoryRetriever(mock_manager)
        result = await retriever.update_access("mem_001")

        assert result

    @pytest.mark.asyncio
    async def test_retrieve_for_context(self, mock_manager, mock_config):
        """测试格式化为上下文"""
        # 准备测试数据
        entries = [
            MemoryEntry(id=f"mem_{i}", content=f"记忆{i}", metadata={})
            for i in range(3)
        ]
        results = [
            MemorySearchResult(entry=e, score=0.9 - i * 0.1, distance=i * 0.1)
            for i, e in enumerate(entries)
        ]

        mock_adapter = mock_manager.get_component.return_value
        mock_adapter.retrieve = AsyncMock(return_value=results)

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(mock_manager)
            context = await retriever.retrieve_for_context(
                "测试查询", group_id="group_123", max_tokens=1000
            )

            assert "相关记忆" in context
            assert "记忆0" in context
            assert "记忆1" in context
            assert "记忆2" in context

    @pytest.mark.asyncio
    async def test_retrieve_for_context_empty(self, mock_manager, mock_config):
        """测试空结果时格式化上下文"""
        mock_adapter = mock_manager.get_component.return_value
        mock_adapter.retrieve = AsyncMock(return_value=[])

        with patch(
            "iris_memory.l2_memory.retriever.get_config", return_value=mock_config
        ):
            retriever = MemoryRetriever(mock_manager)
            context = await retriever.retrieve_for_context("测试查询")

            assert context == ""


class TestHybridRouting:
    """混合检索路由：启用走 retrieve_hybrid，关闭走向量路 + 阈值过滤"""

    @pytest.fixture
    def adapter_with_mocks(self):
        adapter = Mock(spec=L2MemoryAdapter)
        adapter.is_available = True
        adapter.batch_update_access = AsyncMock(return_value=0)
        manager = Mock(spec=ComponentManager)
        manager.get_component = Mock(return_value=adapter)
        return manager, adapter

    def _config(self, hybrid: bool):
        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.top_k": 10,
                "isolation_config.enable_group_memory_isolation": False,
                "l2_memory.relevance_threshold": 0.3,
                "l2_memory.enable_hybrid_retrieval": hybrid,
            }.get(key, default)
        )
        return config

    @pytest.mark.asyncio
    async def test_hybrid_enabled_routes_to_hybrid(self, adapter_with_mocks):
        manager, adapter = adapter_with_mocks
        entry = MemoryEntry(id="mem_h1", content="混合记忆", metadata={})
        fused = MemorySearchResult(entry=entry, score=0.03, distance=0.97)
        adapter.retrieve_hybrid = AsyncMock(return_value=[fused])
        adapter.retrieve = AsyncMock(return_value=[])

        with patch(
            "iris_memory.l2_memory.retriever.get_config",
            return_value=self._config(True),
        ):
            retriever = MemoryRetriever(manager)
            results = await retriever.retrieve("混合记忆", group_id="g")

        adapter.retrieve_hybrid.assert_called_once()
        # 阈值作为参数传入向量路，融合结果不再被阈值二次过滤
        assert 0.3 in adapter.retrieve_hybrid.call_args[0]
        adapter.retrieve.assert_not_called()
        assert len(results) == 1 and results[0].entry.id == "mem_h1"

    @pytest.mark.asyncio
    async def test_hybrid_disabled_filters_by_threshold(self, adapter_with_mocks):
        manager, adapter = adapter_with_mocks
        low = MemorySearchResult(
            entry=MemoryEntry(id="mem_low", content="低分", metadata={}),
            score=0.1,
            distance=0.9,
        )
        adapter.retrieve = AsyncMock(return_value=[low])
        adapter.retrieve_hybrid = AsyncMock(return_value=[])

        with patch(
            "iris_memory.l2_memory.retriever.get_config",
            return_value=self._config(False),
        ):
            retriever = MemoryRetriever(manager)
            results = await retriever.retrieve("低分", group_id="g")

        adapter.retrieve.assert_called_once()
        adapter.retrieve_hybrid.assert_not_called()
        assert results == []
