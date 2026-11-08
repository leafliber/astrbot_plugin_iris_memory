"""L2 Embedding 请求分批测试"""

from unittest.mock import Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import L2MemoryAdapter


class LimitedProvider:
    """超过批次上限时像 DashScope 一样拒绝请求的 Provider。"""

    def __init__(self, limit: int = 25):
        self.limit = limit
        self.batches: list[int] = []

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > self.limit:
            raise ValueError(f"batch size is invalid: {len(texts)}")
        self.batches.append(len(texts))
        return [[float(text), 1.0] for text in texts]


def _adapter(provider, batch_size: int = 25) -> tuple[L2MemoryAdapter, Mock]:
    adapter = L2MemoryAdapter()
    adapter._embedding_source = "provider"
    adapter._embedding_provider = provider
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "l2_embedding_batch_size": batch_size
        }.get(key, default)
    )
    return adapter, config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,expected",
    [(1, [1]), (25, [25]), (26, [25, 1]), (60, [25, 25, 10])],
)
async def test_embed_splits_requests_at_provider_limit(count, expected):
    provider = LimitedProvider()
    adapter, config = _adapter(provider)

    with patch(
        "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config",
        return_value=config,
    ):
        vectors = await adapter._embed([str(i) for i in range(count)])

    assert provider.batches == expected
    assert len(vectors) == count
    assert adapter.get_embedding_stats() == {
        "requests": len(expected),
        "texts": count,
    }


@pytest.mark.asyncio
async def test_embed_keeps_input_order_across_batches():
    adapter, config = _adapter(LimitedProvider())

    with patch(
        "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config",
        return_value=config,
    ):
        vectors = await adapter._embed([str(i) for i in range(1, 61)])

    # 归一化不改变分量比例，v[0] / v[1] 还原输入值
    restored = [round(vector[0] / vector[1]) for vector in vectors]
    assert restored == list(range(1, 61))


@pytest.mark.asyncio
async def test_embed_honours_configured_batch_size():
    provider = LimitedProvider(limit=10)
    adapter, config = _adapter(provider, batch_size=10)

    with patch(
        "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config",
        return_value=config,
    ):
        await adapter._embed([str(i) for i in range(30)])

    assert provider.batches == [10, 10, 10]


@pytest.mark.asyncio
async def test_embed_surfaces_provider_errors():
    provider = LimitedProvider(limit=5)
    adapter, config = _adapter(provider, batch_size=25)

    with patch(
        "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config",
        return_value=config,
    ):
        with pytest.raises(ValueError, match="batch size is invalid"):
            await adapter._embed([str(i) for i in range(6)])
