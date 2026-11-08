"""L2 嵌入模型签名测试"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import L2MemoryAdapter

MODEL = "qwen3.7-text-embedding-flash"
ADAPTER_CONFIG = "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config"


class FakeProvider:
    """DashScope 型：通过 set_model 暴露 model_name。"""

    model_name = MODEL
    provider_config = {"id": "dashscope_embedding"}

    def get_dim(self) -> int:
        return 4

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class OpenAICompatibleProvider(FakeProvider):
    """openai_embedding 型：只设置 model，model_name 为空串。"""

    model_name = ""
    model = MODEL


def _config(tmp_path, configured_provider: str = "") -> Mock:
    config = Mock()
    config.data_dir = tmp_path
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "l2_memory.enable": True,
            "l2_memory.embedding_source": "provider",
            "l2_memory.embedding_provider": configured_provider,
        }.get(key, default)
    )
    return config


def _adapter(provider_cls=FakeProvider) -> L2MemoryAdapter:
    provider = provider_cls()
    pm = Mock()
    pm.embedding_provider_insts = [provider]
    pm.inst_map = {"dashscope_embedding": provider}
    context = Mock()
    context.provider_manager = pm
    context.get_provider_by_id = Mock(return_value=provider)
    return L2MemoryAdapter(context=context)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["", "dashscope_embedding"])
async def test_signature_uses_resolved_provider(tmp_path, configured):
    adapter = _adapter()

    await adapter._init_provider_embedding(_config(tmp_path, configured))

    assert (
        adapter._actual_embedding_model == f"provider:dashscope_embedding/{MODEL}"
    )


@pytest.mark.asyncio
async def test_signature_records_model_of_openai_compatible_provider(tmp_path):
    adapter = _adapter(OpenAICompatibleProvider)

    await adapter._init_provider_embedding(_config(tmp_path))

    assert (
        adapter._actual_embedding_model == f"provider:dashscope_embedding/{MODEL}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        "provider:/",
        f"provider:/{MODEL}",
        "provider:dashscope_embedding/dashscope_embedding",
    ],
)
@pytest.mark.parametrize("provider_cls", [FakeProvider, OpenAICompatibleProvider])
async def test_legacy_signature_upgrades_without_reembedding(
    tmp_path, stored, provider_cls
):
    """旧版写入的各种残缺签名升级后都不重嵌入，只把 meta 升级为新签名。"""
    adapter = _adapter(provider_cls)
    persist_dir = tmp_path / "faiss" / "memory_default"
    persist_dir.mkdir(parents=True)
    meta_path = persist_dir / "index_meta.json"
    meta_path.write_text(
        json.dumps({"embedding_model": stored, "embedding_dimensions": 4}),
        encoding="utf-8",
    )

    with (
        patch(ADAPTER_CONFIG, return_value=_config(tmp_path)),
        patch.object(
            adapter, "_migrate_on_model_change", new=AsyncMock(return_value=True)
        ) as migrate,
    ):
        await adapter.initialize()

    try:
        assert adapter.is_available
        migrate.assert_not_awaited()
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        assert saved["embedding_model"] == f"provider:dashscope_embedding/{MODEL}"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        "provider:/other",
        "provider:dashscope_embedding/other",
        "provider:other/other",
        "local-model",
    ],
)
async def test_real_model_change_still_migrates(tmp_path, stored):
    adapter = _adapter(OpenAICompatibleProvider)
    persist_dir = tmp_path / "faiss" / "memory_default"
    persist_dir.mkdir(parents=True)
    (persist_dir / "index_meta.json").write_text(
        json.dumps({"embedding_model": stored, "embedding_dimensions": 4}),
        encoding="utf-8",
    )

    with (
        patch(ADAPTER_CONFIG, return_value=_config(tmp_path)),
        patch.object(
            adapter, "_migrate_on_model_change", new=AsyncMock(return_value=False)
        ) as migrate,
    ):
        await adapter.initialize()

    migrate.assert_awaited_once()
