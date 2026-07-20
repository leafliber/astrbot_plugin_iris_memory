"""
Token 统计管理器测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from iris_memory.llm.token_stats import TokenUsage, TokenStatsManager


class TestTokenUsage:
    def test_init_default_values(self):
        usage = TokenUsage()
        assert usage.total_input_tokens == 0
        assert usage.total_output_tokens == 0
        assert usage.total_calls == 0

    def test_total_tokens_property(self):
        usage = TokenUsage(
            total_input_tokens=100, total_output_tokens=50, total_calls=1
        )
        assert usage.total_tokens == 150

    def test_to_dict(self):
        usage = TokenUsage(
            total_input_tokens=100, total_output_tokens=50, total_calls=1
        )
        data = usage.to_dict()
        assert data == {
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "total_calls": 1,
        }

    def test_from_dict(self):
        data = {"total_input_tokens": 100, "total_output_tokens": 50, "total_calls": 1}
        usage = TokenUsage.from_dict(data)
        assert usage.total_input_tokens == 100
        assert usage.total_output_tokens == 50
        assert usage.total_calls == 1

    def test_from_dict_partial(self):
        data = {"total_input_tokens": 100}
        usage = TokenUsage.from_dict(data)
        assert usage.total_input_tokens == 100
        assert usage.total_output_tokens == 0
        assert usage.total_calls == 0


class TestTokenStatsManager:
    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.get_kv_data = AsyncMock(return_value={})
        storage.put_kv_data = AsyncMock()
        storage.delete_kv_data = AsyncMock()
        return storage

    @pytest.fixture
    def manager(self, mock_storage):
        return TokenStatsManager(mock_storage)

    @pytest.mark.asyncio
    async def test_init(self, manager):
        assert manager._storage is not None
        assert manager._cache is not None

    @pytest.mark.asyncio
    async def test_record_usage_module(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)

        module_stats = manager._cache["l1_summarizer"]
        assert module_stats.total_input_tokens == 100
        assert module_stats.total_output_tokens == 50
        assert module_stats.total_calls == 1

        global_stats = manager._cache["global"]
        assert global_stats.total_input_tokens == 100
        assert global_stats.total_output_tokens == 50
        assert global_stats.total_calls == 1

    @pytest.mark.asyncio
    async def test_record_usage_multiple_times(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)
        await manager.record_usage("l1_summarizer", 200, 100)

        module_stats = manager._cache["l1_summarizer"]
        assert module_stats.total_input_tokens == 300
        assert module_stats.total_output_tokens == 150
        assert module_stats.total_calls == 2

    @pytest.mark.asyncio
    async def test_record_usage_different_modules(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)
        await manager.record_usage("l3_kg_extraction", 200, 100)

        l1_stats = manager._cache["l1_summarizer"]
        assert l1_stats.total_tokens == 150

        l3_stats = manager._cache["l3_kg_extraction"]
        assert l3_stats.total_tokens == 300

        global_stats = manager._cache["global"]
        assert global_stats.total_tokens == 450
        assert global_stats.total_calls == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)

        stats = await manager.get_stats("l1_summarizer")
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 50
        assert stats.total_calls == 1

    @pytest.mark.asyncio
    async def test_get_stats_global(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)

        stats = await manager.get_stats("global")
        assert stats.total_tokens == 150

    @pytest.mark.asyncio
    async def test_reset_stats(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)

        await manager.reset_stats("l1_summarizer")

        stats = await manager.get_stats("l1_summarizer")
        assert stats.total_tokens == 0
        assert stats.total_calls == 0

    @pytest.mark.asyncio
    async def test_get_all_stats(self, manager):
        await manager.record_usage("l1_summarizer", 100, 50)
        await manager.record_usage("l3_kg_extraction", 200, 100)

        all_stats = await manager.get_all_stats()

        assert "l1_summarizer" in all_stats
        assert "l3_kg_extraction" in all_stats
        assert "global" in all_stats

    @pytest.mark.asyncio
    async def test_kv_storage_persistence(self, manager, mock_storage):
        await manager.record_usage("l1_summarizer", 100, 50)

        assert mock_storage.put_kv_data.called

        keys = [call[0][0] for call in mock_storage.put_kv_data.call_args_list]
        assert "token_stats:module:l1_summarizer" in keys
        assert "token_stats:global" in keys

    @pytest.mark.asyncio
    async def test_record_usage_loads_from_kv_before_accumulating(self, mock_storage):
        """回归：record_usage 重启后必须先 _load_from_kv 回读历史累计

        历史 bug：_cache 从 0 起，record_usage 在 += 前从不 _load_from_kv
        回读历史累计；唯一加载路径只在 get_stats 触发。重启后首次调用
        基于 0 累加，_save_to_kv 用本次会话小值覆盖历史总量。
        """
        # 模拟 KV 中已有历史数据
        stored_module = {
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_calls": 10,
        }
        stored_global = {
            "total_input_tokens": 2000,
            "total_output_tokens": 1000,
            "total_calls": 20,
        }

        def get_kv_side_effect(key, default=None):
            if key == "token_stats:module:l1_summarizer":
                return stored_module
            if key == "token_stats:global":
                return stored_global
            return {}

        mock_storage.get_kv_data = AsyncMock(side_effect=get_kv_side_effect)

        # 模拟重启：新建 manager（_cache 为空 defaultdict）
        manager = TokenStatsManager(mock_storage)

        # 重启后先调 record_usage（不先调 get_stats）
        await manager.record_usage("l1_summarizer", 100, 50)

        module_stats = manager._cache["l1_summarizer"]
        # 应在历史 1000 的基础上累加，不是从 0 累加
        assert module_stats.total_input_tokens == 1100, (
            "重启后应加载历史累计 1000 再 +100，不得从 0 覆盖"
        )
        assert module_stats.total_output_tokens == 550
        assert module_stats.total_calls == 11

        global_stats = manager._cache["global"]
        assert global_stats.total_input_tokens == 2100, (
            "global 同样应加载历史 2000 再 +100"
        )
        assert global_stats.total_output_tokens == 1050
        assert global_stats.total_calls == 21

    @pytest.mark.asyncio
    async def test_record_usage_global_directly(self, mock_storage):
        """直接对 global 调 record_usage 也不覆盖历史"""
        stored_global = {
            "total_input_tokens": 5000,
            "total_output_tokens": 2000,
            "total_calls": 50,
        }
        mock_storage.get_kv_data = AsyncMock(
            side_effect=lambda key, default=None: (
                stored_global if key == "token_stats:global" else {}
            )
        )

        manager = TokenStatsManager(mock_storage)
        await manager.record_usage("global", 100, 50)

        global_stats = manager._cache["global"]
        assert global_stats.total_input_tokens == 5100
        assert global_stats.total_calls == 51

    @pytest.mark.asyncio
    async def test_get_all_stats_loads_from_kv_after_restart(self):
        """回归：get_all_stats 重启后应从 KV 加载，而非仅返回空内存缓存

        历史 bug：get_all_stats 直接返回 ``dict(self._cache)``，重启后 _cache
        为空 defaultdict，返回空字典，丢失所有持久化统计。修复后遍历
        ``_known_modules`` 从 KV 回读；``_known_modules`` 由 record_usage
        登记模块名（KV 存储无 list_keys 接口）。
        """
        # 使用真实持久化的 dict-backed KV 存储
        store: dict = {}
        storage = MagicMock()

        async def _get_kv_data(key, default=None):
            return store.get(key, default if default is not None else {})

        async def _put_kv_data(key, value):
            store[key] = value

        storage.get_kv_data = _get_kv_data
        storage.put_kv_data = _put_kv_data

        # 会话1：记录 usage，持久化到 KV 并登记模块名
        manager1 = TokenStatsManager(storage)
        await manager1.record_usage("test_module", 100, 50)

        # record_usage 应将模块名登记到 _known_modules
        assert "test_module" in manager1._known_modules
        assert "global" in manager1._known_modules

        # 模拟重启：新建 manager（_cache 为空，_known_modules 重置为 {"global"}）
        manager2 = TokenStatsManager(storage)

        all_stats = await manager2.get_all_stats()

        # 不应返回空缓存——应从 KV 加载已持久化的 global 统计
        # （bug 版本会返回空 dict，"global" 不在结果中）
        assert "global" in all_stats
        assert all_stats["global"].total_input_tokens == 100
        assert all_stats["global"].total_output_tokens == 50
        assert all_stats["global"].total_calls == 1
