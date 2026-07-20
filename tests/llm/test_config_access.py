"""LLM 配置访问测试

验证修复的问题：
- 隐藏配置键访问（移除 "hidden." 前缀）
"""

import pytest
from unittest.mock import MagicMock, patch

from iris_memory.llm.manager import LLMManager


class TestLLMConfigAccess:
    @pytest.fixture
    def mock_context(self):
        context = MagicMock()

        mock_response = MagicMock()
        mock_response.completion_text = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.input_other = 80
        mock_response.usage.input_cached = 20
        mock_response.usage.output = 50
        context.llm_generate = MagicMock(return_value=mock_response)

        context.get_kv_data = MagicMock(return_value={})
        context.put_kv_data = MagicMock()

        return context

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.get_kv_data = MagicMock(return_value={})
        storage.put_kv_data = MagicMock()
        return storage

    @pytest.mark.asyncio
    async def test_hidden_config_access_without_prefix(
        self, mock_context, mock_storage
    ):
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            config = MagicMock()

            def config_get(key, default=None):
                if key == "call_log_max_entries":
                    return 200
                return default

            config.get = MagicMock(side_effect=config_get)
            mock_get_config.return_value = config

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            assert manager._call_logs.maxlen == 200

            config.get.assert_called()
            called_keys = [call[0][0] for call in config.get.call_args_list]
            assert "call_log_max_entries" in called_keys
            assert "hidden.call_log_max_entries" not in called_keys

    @pytest.mark.asyncio
    async def test_default_value_when_config_missing(self, mock_context, mock_storage):
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            config = MagicMock()

            config.get = MagicMock(side_effect=lambda key, default=None: default)
            mock_get_config.return_value = config

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            assert manager._call_logs.maxlen == 100

    @pytest.mark.asyncio
    async def test_config_access_with_custom_limit(self, mock_context, mock_storage):
        with patch("iris_memory.llm.manager.get_config") as mock_get_config:
            config = MagicMock()

            def config_get(key, default=None):
                if key == "call_log_max_entries":
                    return 500
                return default

            config.get = MagicMock(side_effect=config_get)
            mock_get_config.return_value = config

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            assert manager._call_logs.maxlen == 500

            for i in range(150):
                manager._call_logs.append({"test": i})

            assert len(manager._call_logs) == 150
