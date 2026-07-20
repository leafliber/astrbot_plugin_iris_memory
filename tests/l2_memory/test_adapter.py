"""L2 FAISS + SQLite 适配器测试"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
import pytest

from iris_memory.l2_memory.adapter import L2MemoryAdapter


class TestL2MemoryAdapter:
    """L2MemoryAdapter 测试"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.enable": True,
                "l2_timeout_ms": 2000,
                "l2_max_entries": 10000,
                "l2_similarity_threshold": 0.90,
                "l2_memory.embedding_source": "provider",
                "l2_memory.embedding_provider": "",
                "l2_memory.embedding_model": "BAAI/bge-small-zh-v1.5",
            }.get(key, default)
        )
        config.data_dir = Path(tempfile.mkdtemp())
        return config

    @pytest.fixture
    def mock_faiss_adapter(self, mock_config):
        """创建一个带有 mock FAISS 索引和真实 SQLite 的适配器"""
        adapter = L2MemoryAdapter()
        adapter._is_available = True

        # 创建临时目录
        adapter._persist_dir = Path(tempfile.mkdtemp())
        adapter._embedding_dimensions = 8

        # 创建 mock FAISS 索引
        mock_index = Mock()
        mock_index.ntotal = 0
        mock_index.d = 8

        # 模拟 add_with_ids
        def fake_add_with_ids(vectors, ids):
            mock_index.ntotal += len(ids)

        mock_index.add_with_ids = fake_add_with_ids

        # 模拟 search
        mock_index.search = Mock(
            return_value=(
                np.array([[0.95]]),
                np.array([[0]]),
            )
        )

        # 模拟 remove_ids
        mock_index.remove_ids = Mock()

        adapter._index = mock_index

        # 创建真实 SQLite 数据库
        db_path = adapter._persist_dir / "metadata.db"
        adapter._db = adapter._open_db(db_path)

        adapter._free_list = []
        adapter._dirty = False
        adapter._actual_embedding_model = "test-model"
        adapter._embedding_source = "provider"
        adapter._embedding_provider = None

        return adapter

    @pytest.mark.asyncio
    async def test_adapter_name(self):
        """测试适配器名称"""
        adapter = L2MemoryAdapter()
        assert adapter.name == "l2_memory"

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, mock_config):
        """测试初始化时未启用"""
        mock_config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.enable": False,
            }.get(key, default)
        )

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            adapter = L2MemoryAdapter()
            await adapter.initialize()

            assert not adapter.is_available
            assert "未启用" in adapter.init_error

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """测试关闭适配器"""
        adapter = L2MemoryAdapter()
        adapter._is_available = True
        adapter._index = Mock()
        adapter._db = Mock()

        await adapter.shutdown()

        assert not adapter.is_available
        assert adapter._index is None
        assert adapter._db is None

    @pytest.mark.asyncio
    async def test_add_memory_success(self, mock_faiss_adapter):
        """测试添加记忆成功"""
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value=None)

        # Mock _embed
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])

        memory_id = await adapter.add_memory(
            "测试记忆内容", metadata={"group_id": "group_123"}
        )

        assert memory_id is not None
        assert memory_id.startswith("mem_")

        # 验证 SQLite 中有记录
        count = adapter._count_db()
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_memory_duplicate(self, mock_faiss_adapter):
        """测试添加重复记忆"""
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value="mem_existing")

        # Mock _embed（新逻辑在锁内去重前先计算嵌入）
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])

        memory_id = await adapter.add_memory(
            "测试记忆内容", metadata={"group_id": "group_123"}
        )

        assert memory_id == "mem_existing"

    @pytest.mark.asyncio
    async def test_add_memory_unavailable(self):
        """测试不可用时添加记忆"""
        adapter = L2MemoryAdapter()
        adapter._is_available = False

        memory_id = await adapter.add_memory("测试内容")

        assert memory_id is None

    @pytest.mark.asyncio
    async def test_retrieve_success(self, mock_config):
        """测试检索记忆成功"""
        adapter = L2MemoryAdapter()
        adapter._is_available = True

        # Mock _search_with_vector
        adapter._search_with_vector = Mock(return_value=[])

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.retrieve("测试查询", group_id="group_123", top_k=5)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_unavailable(self):
        """测试不可用时检索"""
        adapter = L2MemoryAdapter()
        adapter._is_available = False

        results = await adapter.retrieve("测试查询")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_timeout(self, mock_config):
        """测试检索超时"""
        mock_config.get = Mock(
            side_effect=lambda key, default=None: {
                "l2_memory.enable": True,
                "l2_timeout_ms": 100,
            }.get(key, default)
        )

        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            adapter = L2MemoryAdapter()
            adapter._is_available = True

            import time

            def slow_search(*args):
                time.sleep(1)
                return []

            adapter._search_with_vector = slow_search

            results = await adapter.retrieve("测试查询")
            assert results == []

    @pytest.mark.asyncio
    async def test_get_entry_count(self, mock_faiss_adapter):
        """测试获取条目数"""
        adapter = mock_faiss_adapter
        count = await adapter.get_entry_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_entry_count_unavailable(self):
        """测试不可用时获取条目数"""
        adapter = L2MemoryAdapter()
        adapter._is_available = False

        count = await adapter.get_entry_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_entries(self, mock_faiss_adapter):
        """测试删除条目"""
        adapter = mock_faiss_adapter

        # 先添加一条记忆
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)
        memory_id = await adapter.add_memory("测试", metadata={"group_id": "g1"})
        assert memory_id is not None

        # 删除
        result = await adapter.delete_entries([memory_id])
        assert result
        assert adapter._count_db() == 0

    @pytest.mark.asyncio
    async def test_delete_entries_empty(self, mock_faiss_adapter):
        """测试删除空列表"""
        adapter = mock_faiss_adapter

        result = await adapter.delete_entries([])
        assert not result

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_faiss_adapter):
        """测试删除 collection"""
        adapter = mock_faiss_adapter
        result = await adapter.delete_collection()
        assert result
        assert adapter._index is None

    @pytest.mark.asyncio
    async def test_delete_collection_no_dir(self):
        """测试无目录时删除 collection"""
        adapter = L2MemoryAdapter()
        adapter._persist_dir = None
        result = await adapter.delete_collection()
        assert not result

    @pytest.mark.asyncio
    async def test_update_access(self, mock_faiss_adapter):
        """测试更新访问信息"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        memory_id = await adapter.add_memory("测试", metadata={"group_id": "g1"})
        assert memory_id is not None

        result = await adapter.update_access(memory_id)
        assert result

    @pytest.mark.asyncio
    async def test_update_access_nonexistent(self, mock_faiss_adapter):
        """测试更新不存在的记忆"""
        adapter = mock_faiss_adapter
        result = await adapter.update_access("mem_nonexistent")
        assert not result

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_faiss_adapter):
        """测试获取统计信息"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory("测试1", metadata={"group_id": "g1"})
        await adapter.add_memory("测试2", metadata={"group_id": "g2"})

        stats = await adapter.get_stats()
        assert stats["total_count"] == 2
        assert stats["group_count"] == 2

    @pytest.mark.asyncio
    async def test_get_entries_by_group(self, mock_faiss_adapter):
        """测试按群聊获取条目"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory("测试1", metadata={"group_id": "g1"})
        await adapter.add_memory("测试2", metadata={"group_id": "g2"})

        entries = await adapter.get_entries_by_group("g1")
        assert len(entries) == 1
        assert entries[0].content == "测试1"

    @pytest.mark.asyncio
    async def test_get_latest_memories(self, mock_faiss_adapter):
        """测试获取最新记忆"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory(
            "旧记忆", metadata={"group_id": "g1", "timestamp": "2024-01-01T00:00:00"}
        )
        await adapter.add_memory(
            "新记忆", metadata={"group_id": "g1", "timestamp": "2024-12-01T00:00:00"}
        )

        results = await adapter.get_latest_memories(limit=1, group_id="g1")
        assert len(results) == 1
        assert results[0].entry.content == "新记忆"

    @pytest.mark.asyncio
    async def test_get_unprocessed_memories(self, mock_faiss_adapter):
        """测试获取未处理记忆"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory("未处理", metadata={"group_id": "g1"})
        await adapter.add_memory(
            "已处理", metadata={"group_id": "g1", "kg_processed": True}
        )

        entries = await adapter.get_unprocessed_memories(limit=10)
        assert len(entries) == 1
        assert entries[0].content == "未处理"

    @pytest.mark.asyncio
    async def test_mark_memories_processed(self, mock_faiss_adapter):
        """测试标记记忆为已处理"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        memory_id = await adapter.add_memory("测试", metadata={"group_id": "g1"})
        result = await adapter.mark_memories_processed([memory_id])
        assert result

        # 验证 kg_processed 标记
        row = adapter._db.execute(
            "SELECT kg_processed FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_delete_by_group(self, mock_faiss_adapter):
        """测试按群聊删除"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory("g1 记忆", metadata={"group_id": "g1"})
        await adapter.add_memory("g2 记忆", metadata={"group_id": "g2"})

        count = await adapter.delete_by_group("g1")
        assert count == 1
        assert adapter._count_db() == 1

    @pytest.mark.asyncio
    async def test_delete_all(self, mock_faiss_adapter):
        """测试删除所有记忆"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        await adapter.add_memory("测试1", metadata={"group_id": "g1"})
        await adapter.add_memory("测试2", metadata={"group_id": "g2"})

        count = await adapter.delete_all()
        assert count == 2
        assert adapter._count_db() == 0
        assert adapter._free_list == []

    @pytest.mark.asyncio
    async def test_update_content(self, mock_faiss_adapter):
        """测试更新记忆内容"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        memory_id = await adapter.add_memory("旧内容", metadata={"group_id": "g1"})
        result = await adapter.update_content(memory_id, "新内容")
        assert result

        entries = await adapter.get_all_entries()
        assert len(entries) == 1
        assert entries[0].content == "新内容"

    @pytest.mark.asyncio
    async def test_update_metadata(self, mock_faiss_adapter):
        """测试更新元数据"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        memory_id = await adapter.add_memory(
            "测试", metadata={"group_id": "g1", "confidence": 0.5}
        )
        result = await adapter.update_metadata(
            memory_id,
            {
                "group_id": "g1",
                "confidence": 0.9,
                "timestamp": "2024-01-01T00:00:00",
            },
        )
        assert result

        entries = await adapter.get_all_entries()
        assert entries[0].metadata["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_free_list_reuse(self, mock_faiss_adapter):
        """测试 free-list 槽位复用"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        mid1 = await adapter.add_memory("记忆1", metadata={"group_id": "g1"})
        await adapter.add_memory("记忆2", metadata={"group_id": "g1"})

        # 删除第一条
        await adapter.delete_entries([mid1])

        # 应该复用 faiss_idx=0 的槽位
        assert 0 in adapter._free_list

        await adapter.add_memory("记忆3", metadata={"group_id": "g1"})
        assert 0 not in adapter._free_list
        assert adapter._count_db() == 2


    @pytest.mark.asyncio
    async def test_batch_retrieve_by_ids(self, mock_faiss_adapter, mock_config):
        """测试按 ID 批量检索：复用已存向量，全程零 embedding 调用"""
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value=None)
        embed_mock = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
        adapter._embed = embed_mock

        mid1 = await adapter.add_memory("记忆一", metadata={})
        mid2 = await adapter.add_memory("记忆二", metadata={})

        # 模拟已存向量重建与检索
        adapter._index.reconstruct = Mock(
            side_effect=lambda i: np.full(8, 0.1, dtype=np.float32)
        )
        adapter._index.search = Mock(
            return_value=(
                np.array([[1.0, 0.9], [1.0, 0.9]]),
                np.array([[0, 1], [0, 1]]),
            )
        )

        embed_mock.reset_mock()
        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.batch_retrieve_by_ids(
                [mid1, mid2, "mem_not_exist"]
            )

        assert len(results) == 3
        # 不存在的 ID 对应空列表
        assert results[2] == []
        # 已存 ID 检索到库中两条记忆
        assert {r.entry.id for r in results[0]} == {mid1, mid2}
        assert {r.entry.id for r in results[1]} == {mid1, mid2}
        # 全程未调用 embedding
        embed_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_retrieve_by_ids_unavailable(self):
        """测试适配器不可用时按 ID 批量检索返回空"""
        adapter = L2MemoryAdapter()
        results = await adapter.batch_retrieve_by_ids(["mem_a", "mem_b"])
        assert results == [[], []]

    @pytest.mark.asyncio
    async def test_batch_retrieve_by_ids_empty_index(
        self, mock_faiss_adapter, mock_config
    ):
        """测试索引为空时按 ID 批量检索返回空"""
        adapter = mock_faiss_adapter
        adapter._index.ntotal = 0
        with patch(
            "iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.batch_retrieve_by_ids(["mem_a"])
        assert results == [[]]
