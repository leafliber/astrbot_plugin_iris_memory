"""L2 等待 Embedding Provider 就绪的测试"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import (
    EmbeddingProviderNotReadyError,
    L2MemoryAdapter,
)

ADAPTER_CONFIG = "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config"


class FakeProvider:
    model_name = "embed-model"
    provider_config = {"id": "embed"}

    def get_dim(self) -> int:
        return 4

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _config(tmp_path, configured: str = "") -> Mock:
    config = Mock()
    config.data_dir = tmp_path
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "l2_memory.enable": True,
            "l2_memory.embedding_source": "provider",
            "l2_memory.embedding_provider": configured,
        }.get(key, default)
    )
    return config


def _adapter(providers: list) -> L2MemoryAdapter:
    pm = Mock()
    pm.embedding_provider_insts = providers
    pm.inst_map = {}
    context = Mock()
    context.provider_manager = pm
    context.get_provider_by_id = Mock(return_value=None)
    return L2MemoryAdapter(context=context)


@pytest.mark.asyncio
async def test_missing_provider_waits_then_requests_retry(tmp_path):
    adapter = _adapter([])

    with (
        patch(ADAPTER_CONFIG, return_value=_config(tmp_path)),
        patch("asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        await adapter.initialize()

    assert sleep.await_count == 4
    assert not adapter.is_available
    assert adapter.should_retry_init
    assert "未找到可用的 AstrBot Embedding Provider" in adapter.init_error


@pytest.mark.asyncio
async def test_retry_round_probes_once(tmp_path):
    adapter = _adapter([])
    adapter._waiting_for_provider = True

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(EmbeddingProviderNotReadyError):
            await adapter._init_provider_embedding(_config(tmp_path))

    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovers_when_provider_appears(tmp_path):
    providers: list = []
    adapter = _adapter(providers)

    with (
        patch(ADAPTER_CONFIG, return_value=_config(tmp_path)),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        await adapter.initialize()
        assert adapter.should_retry_init

        providers.append(FakeProvider())
        await adapter.initialize()

    try:
        assert adapter.is_available
        assert not adapter.should_retry_init
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_configured_provider_falls_back_to_first_available(tmp_path):
    provider = FakeProvider()
    adapter = _adapter([provider])

    await adapter._init_provider_embedding(_config(tmp_path, "missing"))

    assert not adapter.should_retry_init
    assert adapter._embedding_provider is provider
