"""L2 FAISS + SQLite 适配器测试"""

import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import numpy as np
import pytest

from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import L2MemoryAdapter


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
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
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
    async def test_add_memories_bulk_embeds_all_items_once(self, mock_faiss_adapter):
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value=None)
        adapter._embed = AsyncMock(return_value=[[0.1] * 8 for _ in range(10)])
        items = [
            (f"批量记忆 {index}", {"group_id": "g1"}) for index in range(10)
        ]

        memory_ids = await adapter.add_memories_bulk(items)

        assert len([memory_id for memory_id in memory_ids if memory_id]) == 10
        adapter._embed.assert_awaited_once_with(
            [f"批量记忆 {index}" for index in range(10)]
        )
        assert adapter._count_db() == 10

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
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
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
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
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
    async def test_migrate_on_model_change_when_unavailable(self, mock_faiss_adapter):
        """回归测试：初始化阶段（_is_available=False）触发模型迁移应成功。

        复现 issue：initialize() 检测到嵌入模型变更时调用 _migrate_on_model_change，
        此时 _is_available 尚为 False。迁移依赖的导出/导入公共 API（export_all、
        import_from_file、add_memory、get_all_entries）均以 is_available 守卫，
        若不在迁移入口临时置位，导出会因「L2 记忆库不可用」返回 0 条，迁移被
        误判失败，最终导致 L2 记忆库初始化失败。
        """
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value=None)
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])

        # 预置 3 条旧模型记忆，其中 1 条进入归档
        for i in range(3):
            await adapter.add_memory(f"旧记忆 {i}", metadata={"group_id": "g1"})
        archived_id = adapter._db.execute(
            "SELECT memory_id FROM memories ORDER BY faiss_idx LIMIT 1"
        ).fetchone()[0]
        assert await adapter.evict_memories([archived_id]) == 1
        old_count = adapter._count_db()
        assert old_count == 2
        assert await adapter.get_archived_count() == 1

        # 模拟 initialize() 检测到模型变更后的迁移入口状态：
        # _load_existing 尚未执行，_is_available 仍为 False，_db/_index 均未就绪
        adapter._is_available = False
        adapter._db = None
        adapter._index = None
        new_model = "provider:/qwen3-embedding:4b"
        new_dim = 4
        adapter._actual_embedding_model = new_model
        adapter._embedding_dimensions = new_dim

        # 迁移完成须真实落盘，使用真实索引验证归档和正式记忆均能保留。
        # 迁移导入时用新维度重新嵌入
        adapter._embed = AsyncMock(return_value=[[0.2] * new_dim])

        ok = await adapter._migrate_on_model_change(new_model, new_dim)

        assert ok is True
        # 记忆经 导出 -> 重新嵌入 -> 导入 后保留
        assert adapter._count_db() == old_count
        # 软删除归档也随迁移保留，仍可在保留期内恢复
        assert await adapter.get_archived_count() == 1
        assert (await adapter.list_archived_memories())[0]["id"] == archived_id
        # 元数据已更新为新模型/维度
        meta = adapter._load_meta()
        assert meta["embedding_model"] == new_model
        assert meta["embedding_dimensions"] == new_dim
        # 迁移过程中临时置位，最终保持可用
        assert adapter._is_available is True

    @pytest.mark.asyncio
    async def test_update_access(self, mock_faiss_adapter, mock_config):
        """测试更新访问信息"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        adapter._find_similar_unlocked = Mock(return_value=None)

        memory_id = await adapter.add_memory("测试", metadata={"group_id": "g1"})
        assert memory_id is not None

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            result = await adapter.update_access(memory_id)
        assert result

    @pytest.mark.asyncio
    async def test_update_access_nonexistent(self, mock_faiss_adapter, mock_config):
        """测试更新不存在的记忆"""
        adapter = mock_faiss_adapter
        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
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
    async def test_batch_update_contents_uses_one_embedding_and_reopens_kg(
        self, mock_faiss_adapter
    ):
        adapter = mock_faiss_adapter
        adapter._find_similar_unlocked = Mock(return_value=None)
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])
        original_timestamp = "2026-01-02T03:04:05"
        first = await adapter.add_memory(
            "旧一",
            metadata={
                "timestamp": original_timestamp,
                "kg_processed": True,
            },
        )
        second = await adapter.add_memory(
            "旧二", metadata={"timestamp": original_timestamp, "kg_processed": True}
        )

        adapter._embed = AsyncMock(return_value=[[0.2] * 8, [0.3] * 8])
        updated = await adapter.batch_update_contents(
            [(first, "新一"), (second, "新二")]
        )

        assert updated == 2
        adapter._embed.assert_awaited_once()
        assert set(adapter._embed.await_args.args[0]) == {"新一", "新二"}
        entries = await adapter.get_all_entries()
        assert {entry.content for entry in entries} == {"新一", "新二"}
        assert all(entry.metadata["timestamp"] == original_timestamp for entry in entries)
        assert all(entry.metadata["kg_processed"] is False for entry in entries)
        assert all("updated_at" in entry.metadata for entry in entries)

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
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
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
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            results = await adapter.batch_retrieve_by_ids(["mem_a"])
        assert results == [[]]


    @pytest.mark.asyncio
    async def test_initialize_uses_actual_provider_dimension(self, mock_config):
        """Provider 配置维度错误时，以实际向量维度创建索引并可正常写入。"""

        class NemotronProvider:
            model_name = "nvidia/nemotron-3-embed-1b"

            @staticmethod
            def get_dim():
                return 1024

            @staticmethod
            async def get_embeddings(texts):
                return [[0.1] * 2048 for _ in texts]

        context = SimpleNamespace(
            provider_manager=SimpleNamespace(
                embedding_provider_insts=[NemotronProvider()], inst_map={}
            )
        )
        adapter = L2MemoryAdapter(context=context)

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            await adapter.initialize()
            memory_id = await adapter.add_memory(
                "Nemotron 维度回归测试", skip_dedup=True
            )

        assert adapter.is_available
        assert adapter._embedding_dimensions == 2048
        assert adapter._index.d == 2048
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_initialize_migrates_loaded_index_without_dimension_meta(
        self, mock_config
    ):
        """历史元数据缺少维度时，仍应按磁盘索引的真实维度触发迁移。"""

        class NemotronProvider:
            model_name = "nvidia/nemotron-3-embed-1b"

            @staticmethod
            def get_dim():
                return 1024

            @staticmethod
            async def get_embeddings(texts):
                return [[0.1] * 2048 for _ in texts]

        context = SimpleNamespace(
            provider_manager=SimpleNamespace(
                embedding_provider_insts=[NemotronProvider()], inst_map={}
            )
        )
        adapter = L2MemoryAdapter(context=context)
        adapter._load_meta = Mock(return_value={})

        async def load_old_index(_stored_dim):
            adapter._index = SimpleNamespace(d=1024)
            adapter._db = Mock()

        adapter._load_existing = AsyncMock(side_effect=load_old_index)
        adapter._migrate_on_model_change = AsyncMock(return_value=True)
        adapter._count_db = Mock(return_value=1)

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            await adapter.initialize()

        adapter._migrate_on_model_change.assert_awaited_once_with(
            "provider:/nvidia/nemotron-3-embed-1b", 2048
        )
        assert adapter.is_available
        assert adapter._embedding_dimensions == 2048

    @pytest.mark.asyncio
    async def test_initialize_probe_failure_is_recoverable(self, mock_config):
        """维度探测因暂时性网络故障失败时，_try_recover 应能自动重试初始化。"""
        call_count = {"n": 0}

        class FlakyProvider:
            model_name = "nvidia/nemotron-3-embed-1b"

            @staticmethod
            def get_dim():
                return 1024

            @staticmethod
            async def get_embeddings(texts):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise ConnectionError(
                        "Cannot connect to host integrate.api.nvidia.com"
                    )
                return [[0.1] * 2048 for _ in texts]

        context = SimpleNamespace(
            provider_manager=SimpleNamespace(
                embedding_provider_insts=[FlakyProvider()], inst_map={}
            )
        )
        adapter = L2MemoryAdapter(context=context)

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            await adapter.initialize()
            assert not adapter.is_available
            assert "Provider" in adapter._init_error

            assert await adapter._try_recover()
            assert adapter.is_available
            assert adapter._embedding_dimensions == 2048

    @pytest.mark.asyncio
    async def test_add_memory_rejects_runtime_dimension_mismatch(
        self, mock_faiss_adapter
    ):
        """运行期维度漂移应在进入 FAISS 前失败，避免裸 AssertionError。"""
        adapter = mock_faiss_adapter
        adapter._embed = AsyncMock(return_value=[[0.1] * 16])
        adapter._index.add_with_ids = Mock()

        memory_id = await adapter.add_memory("维度漂移", skip_dedup=True)

        assert memory_id is None
        adapter._index.add_with_ids.assert_not_called()

    @pytest.mark.asyncio
    async def test_sparse_index_allocates_after_max_id(self, mock_faiss_adapter):
        """free-list 丢失时不能用 ntotal 覆盖稀疏索引中的现有 ID。"""
        adapter = mock_faiss_adapter
        adapter._upsert_db(0, "mem_0", "记忆0", {}, persona_id="default")
        adapter._upsert_db(2, "mem_2", "记忆2", {}, persona_id="default")
        adapter._index.ntotal = 2
        adapter._free_list = []
        adapter._find_similar_unlocked = Mock(return_value=None)
        adapter._embed = AsyncMock(return_value=[[0.1] * 8])

        memory_id = await adapter.add_memory("记忆3")

        row = adapter._db.execute(
            "SELECT faiss_idx FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        assert row == (3,)
        assert adapter._count_db() == 3

    def test_find_similar_scans_past_other_persona(
        self, mock_faiss_adapter, mock_config
    ):
        """最相似项属于其他人格时，仍应找到当前人格的重复项。"""
        adapter = mock_faiss_adapter
        adapter._upsert_db(0, "mem_other", "相同内容", {}, persona_id="other")
        adapter._upsert_db(1, "mem_target", "相同内容", {}, persona_id="target")
        adapter._index.ntotal = 2
        adapter._index.search = Mock(
            return_value=(
                np.array([[0.99, 0.95]], dtype=np.float32),
                np.array([[0, 1]], dtype=np.int64),
            )
        )

        with patch(
            "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config", return_value=mock_config
        ):
            memory_id = adapter._find_similar_unlocked(
                np.array([[0.1] * 8], dtype=np.float32), persona_id="target"
            )

        assert memory_id == "mem_target"
