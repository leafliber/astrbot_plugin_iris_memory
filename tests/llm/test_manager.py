"""
LLM 管理器测试
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from iris_memory.llm.manager import LLMManager


class TestLLMManager:
    @pytest.fixture
    def mock_context(self):
        context = MagicMock()

        mock_response = MagicMock()
        mock_response.completion_text = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.input_other = 80
        mock_response.usage.input_cached = 20
        mock_response.usage.output = 50
        context.llm_generate = AsyncMock(return_value=mock_response)

        return context

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        storage.get_kv_data = AsyncMock(return_value={})
        storage.put_kv_data = AsyncMock()
        storage.delete_kv_data = AsyncMock()
        return storage

    @pytest.fixture
    def mock_config(self):
        with patch("iris_memory.llm.manager.get_config") as mock:
            config = MagicMock()
            config.get = MagicMock(return_value=100)
            mock.return_value = config
            yield config

    @pytest.mark.asyncio
    async def test_init(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        assert manager.is_available is True
        assert manager._token_stats is not None
        assert manager._call_logs is not None

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()
        await manager.shutdown()

        assert manager.is_available is False

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        response = await manager.generate(prompt="Hello", module="l1_summarizer")

        assert response == "Test response"
        assert mock_context.llm_generate.called

        logs = manager.get_recent_call_logs()
        assert len(logs) == 1
        assert logs[0]["success"] is True
        assert logs[0]["module"] == "l1_summarizer"
        assert logs[0]["input_tokens"] == 100
        assert logs[0]["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_with_provider(
        self, mock_context, mock_storage, mock_config
    ):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate(
            prompt="Hello", module="l1_summarizer", provider_id="gpt-4o"
        )

        call_args = mock_context.llm_generate.call_args
        assert call_args[1]["chat_provider_id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_failure(self, mock_context, mock_storage, mock_config):
        mock_context.llm_generate = AsyncMock(side_effect=Exception("API Error"))

        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        with pytest.raises(Exception, match="API Error"):
            await manager.generate(prompt="Hello", module="l1_summarizer")

        logs = manager.get_recent_call_logs()
        assert len(logs) == 1
        assert logs[0]["success"] is False
        assert "API Error" in logs[0]["error_message"]

    @pytest.mark.asyncio
    async def test_call_protocol(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        response = await manager.call("Hello", provider="gpt-4o")

        assert response == "Test response"

    @pytest.mark.asyncio
    async def test_get_token_stats(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate("Hello", module="l1_summarizer")

        stats = await manager.get_token_stats("l1_summarizer")

        assert stats["module"] == "l1_summarizer"
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert stats["total_calls"] == 1

    @pytest.mark.asyncio
    async def test_get_all_token_stats(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate("Hello", module="l1_summarizer")
        await manager.generate("World", module="l3_kg_extraction")

        all_stats = await manager.get_all_token_stats()

        assert "l1_summarizer" in all_stats
        assert "l3_kg_extraction" in all_stats
        assert "global" in all_stats

    @pytest.mark.asyncio
    async def test_reset_token_stats(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate("Hello", module="l1_summarizer")

        await manager.reset_token_stats("l1_summarizer")

        stats = await manager.get_token_stats("l1_summarizer")
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0

    @pytest.mark.asyncio
    async def test_get_recent_call_logs(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate("Hello", module="l1_summarizer")
        await manager.generate("World", module="l3_kg_extraction")

        logs = manager.get_recent_call_logs(limit=10)

        assert len(logs) == 2
        assert logs[0]["module"] == "l1_summarizer"
        assert logs[1]["module"] == "l3_kg_extraction"

    @pytest.mark.asyncio
    async def test_provider_resolution_priority(
        self, mock_context, mock_storage, mock_config
    ):
        def config_get(key, default=None):
            if key == "call_log_max_entries":
                return 100
            return "gpt-4o-mini"

        mock_config.get = MagicMock(side_effect=config_get)

        manager = LLMManager(mock_context, mock_storage)
        await manager.initialize()

        await manager.generate(
            prompt="Hello", module="l1_summarizer", provider_id="gpt-4o"
        )

        call_args = mock_context.llm_generate.call_args
        assert call_args[1]["chat_provider_id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, mock_context, mock_storage, mock_config):
        manager = LLMManager(mock_context, mock_storage)

        with pytest.raises(RuntimeError, match="LLMManager 未初始化"):
            await manager.generate("Hello", module="test")

    def _timeout_config(self):
        """返回带短超时的 config.get side_effect"""
        return lambda key, default=None: {
            "call_log_max_entries": 100,
            "llm_call_timeout_ms": 100,
        }.get(key, default)

    @pytest.mark.asyncio
    async def test_generate_direct_timeout(self, mock_context, mock_storage):
        """provider 卡住时，generate_direct 在配置超时后抛 TimeoutError"""
        with patch("iris_memory.llm.manager.get_config") as mock_cfg:
            mock_cfg.return_value.get.side_effect = self._timeout_config()

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            async def slow_text_chat(**kw):
                await asyncio.sleep(10)

            slow_provider = MagicMock()
            slow_provider.text_chat = slow_text_chat
            manager._get_provider_instance = MagicMock(return_value=slow_provider)

            with pytest.raises(asyncio.TimeoutError):
                await manager.generate_direct(prompt="Hi", module="test")

            logs = manager.get_recent_call_logs()
            assert len(logs) == 1
            assert logs[0]["success"] is False
            assert "超时" in logs[0]["error_message"]

    @pytest.mark.asyncio
    async def test_generate_timeout(self, mock_context, mock_storage):
        """provider 卡住时，generate 同样在配置超时后抛 TimeoutError"""
        with patch("iris_memory.llm.manager.get_config") as mock_cfg:
            mock_cfg.return_value.get.side_effect = self._timeout_config()

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            async def slow_llm_generate(**kw):
                await asyncio.sleep(10)

            mock_context.llm_generate = slow_llm_generate

            with pytest.raises(asyncio.TimeoutError):
                await manager.generate(prompt="Hi", module="test")

    @pytest.mark.asyncio
    async def test_generate_direct_timeout_disabled(self, mock_context, mock_storage):
        """timeout=0（显式禁用）时不超时，调用正常完成"""
        with patch("iris_memory.llm.manager.get_config") as mock_cfg:
            mock_cfg.return_value.get.side_effect = self._timeout_config()

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            async def fast_text_chat(**kw):
                await asyncio.sleep(0.3)
                resp = MagicMock()
                resp.completion_text = "Test response"
                resp.usage = None
                return resp

            fast_provider = MagicMock()
            fast_provider.text_chat = fast_text_chat
            manager._get_provider_instance = MagicMock(return_value=fast_provider)

            response = await manager.generate_direct(
                prompt="Hi", module="test", timeout=0
            )
            assert response == "Test response"

    @pytest.mark.asyncio
    async def test_generate_direct_explicit_timeout_override(
        self, mock_context, mock_storage
    ):
        """显式 timeout 参数覆盖全局配置"""
        with patch("iris_memory.llm.manager.get_config") as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda key, default=None: {
                "call_log_max_entries": 100,
                "llm_call_timeout_ms": 60000,
            }.get(key, default)

            manager = LLMManager(mock_context, mock_storage)
            await manager.initialize()

            async def slow_text_chat(**kw):
                await asyncio.sleep(10)

            slow_provider = MagicMock()
            slow_provider.text_chat = slow_text_chat
            manager._get_provider_instance = MagicMock(return_value=slow_provider)

            with pytest.raises(asyncio.TimeoutError):
                await manager.generate_direct(prompt="Hi", module="test", timeout=0.1)
