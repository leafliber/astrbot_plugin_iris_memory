"""
RelatedMemoryRetriever 相关记忆检索器测试

测试相关记忆检索功能：
- 语义相似检索
- 同群聊检索
- 同用户检索
- 组合检索
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from iris_memory.l3_kg.related_retriever import RelatedMemoryRetriever
from iris_memory.l2_memory.models import MemoryEntry, MemorySearchResult


class TestRelatedMemoryRetriever:
    """RelatedMemoryRetriever 测试类"""

    @pytest.fixture
    def mock_component_manager(self):
        """创建模拟组件管理器"""
        manager = Mock()

        l2_adapter = Mock()
        l2_adapter.is_available = True
        l2_adapter.retrieve = AsyncMock(return_value=[])
        l2_adapter.get_all_entries = AsyncMock(return_value=[])

        def get_component(name, expected_type=None):
            if name == "l2_memory":
                return l2_adapter
            return None

        manager.get_component = get_component

        return manager

    @pytest.fixture
    def retriever(self, mock_component_manager):
        """创建 RelatedMemoryRetriever 实例"""
        return RelatedMemoryRetriever(mock_component_manager)

    @pytest.fixture
    def sample_memory(self):
        """创建示例记忆"""
        return MemoryEntry(
            id="mem_001",
            content="张三喜欢吃苹果",
            metadata={
                "group_id": "group_123",
                "user_id": "user_001",
                "timestamp": datetime.now().isoformat(),
            },
        )

    @pytest.mark.asyncio
    async def test_retrieve_related_l2_unavailable(self, retriever):
        """测试 L2 不可用时返回空列表"""
        l2_adapter = retriever._component_manager.get_component("l2_memory")
        l2_adapter.is_available = False

        memory = MemoryEntry(id="mem_001", content="测试", metadata={})
        result = await retriever.retrieve_related(memory)

        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_related_semantic_only(self, retriever, sample_memory):
        """测试仅语义相似检索"""
        semantic_results = [
            MemorySearchResult(
                entry=MemoryEntry(
                    id="mem_002",
                    content="李四也喜欢吃苹果",
                    metadata={"group_id": "group_456"},
                ),
                score=0.9,
                distance=0.1,
            )
        ]

        l2_adapter = retriever._component_manager.get_component("l2_memory")
        l2_adapter.retrieve = AsyncMock(return_value=semantic_results)
        l2_adapter.get_all_entries = AsyncMock(return_value=[])

        with patch("iris_memory.l3_kg.related_retriever.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key: {
                "kg_extraction_semantic_weight": 1.0,
                "kg_extraction_same_group_weight": 0.0,
                "kg_extraction_same_user_weight": 0.0,
            }.get(key, None)

            result = await retriever.retrieve_related(sample_memory, top_k=5)

            assert len(result) == 1
            assert result[0].id == "mem_002"

    @pytest.mark.asyncio
    async def test_retrieve_related_same_group(self, retriever, sample_memory):
        """测试同群聊检索"""
        group_memories = [
            MemoryEntry(
                id="mem_003",
                content="群聊中的其他记忆",
                metadata={
                    "group_id": "group_123",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]

        l2_adapter = retriever._component_manager.get_component("l2_memory")
        l2_adapter.retrieve = AsyncMock(return_value=[])
        l2_adapter.get_entries_by_group = AsyncMock(return_value=group_memories)

        with patch("iris_memory.l3_kg.related_retriever.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key: {
                "kg_extraction_semantic_weight": 0.0,
                "kg_extraction_same_group_weight": 1.0,
                "kg_extraction_same_user_weight": 0.0,
            }.get(key, None)

            result = await retriever.retrieve_related(sample_memory, top_k=5)

            assert len(result) == 1
            assert result[0].id == "mem_003"

    @pytest.mark.asyncio
    async def test_retrieve_related_same_user(self, retriever, sample_memory):
        """测试同用户检索"""
        user_memories = [
            MemoryEntry(
                id="mem_004",
                content="用户的其他记忆",
                metadata={
                    "group_id": "group_456",
                    "user_id": "user_001",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]

        l2_adapter = retriever._component_manager.get_component("l2_memory")
        l2_adapter.retrieve = AsyncMock(return_value=[])
        l2_adapter.get_entries_by_user = AsyncMock(return_value=user_memories)

        with patch("iris_memory.l3_kg.related_retriever.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key: {
                "kg_extraction_semantic_weight": 0.0,
                "kg_extraction_same_group_weight": 0.0,
                "kg_extraction_same_user_weight": 1.0,
            }.get(key, None)

            result = await retriever.retrieve_related(sample_memory, top_k=5)

            assert len(result) == 1
            assert result[0].id == "mem_004"

    @pytest.mark.asyncio
    async def test_retrieve_related_combined(self, retriever, sample_memory):
        """测试组合检索"""
        semantic_results = [
            MemorySearchResult(
                entry=MemoryEntry(
                    id="mem_002",
                    content="语义相似记忆",
                    metadata={"group_id": "group_456"},
                ),
                score=0.9,
                distance=0.1,
            )
        ]

        group_memories = [
            MemoryEntry(
                id="mem_003",
                content="同群聊记忆",
                metadata={
                    "group_id": "group_123",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]

        user_memories = [
            MemoryEntry(
                id="mem_004",
                content="同用户记忆",
                metadata={
                    "group_id": "group_456",
                    "user_id": "user_001",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]

        all_memories = group_memories + user_memories

        l2_adapter = retriever._component_manager.get_component("l2_memory")

        async def mock_retrieve(query, top_k=10):
            return semantic_results

        async def mock_get_all():
            return all_memories

        l2_adapter.retrieve = mock_retrieve
        l2_adapter.get_all_entries = mock_get_all

        with patch("iris_memory.l3_kg.related_retriever.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key: {
                "kg_extraction_semantic_weight": 0.5,
                "kg_extraction_same_group_weight": 0.3,
                "kg_extraction_same_user_weight": 0.2,
            }.get(key, None)

            result = await retriever.retrieve_related(sample_memory, top_k=5)

            assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_related_excludes_self(self, retriever, sample_memory):
        """测试排除自身记忆"""
        semantic_results = [
            MemorySearchResult(entry=sample_memory, score=1.0, distance=0.0)
        ]

        l2_adapter = retriever._component_manager.get_component("l2_memory")
        l2_adapter.retrieve = AsyncMock(return_value=semantic_results)
        l2_adapter.get_all_entries = AsyncMock(return_value=[])

        with patch("iris_memory.l3_kg.related_retriever.get_config") as mock_config:
            mock_config.return_value.get.side_effect = lambda key: {
                "kg_extraction_semantic_weight": 1.0,
                "kg_extraction_same_group_weight": 0.0,
                "kg_extraction_same_user_weight": 0.0,
            }.get(key, None)

            result = await retriever.retrieve_related(sample_memory, top_k=5)

            assert sample_memory not in result
