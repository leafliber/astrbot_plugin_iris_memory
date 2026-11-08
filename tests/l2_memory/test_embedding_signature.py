"""L2 嵌入模型签名测试"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter import L2MemoryAdapter

MODEL = "qwen3.7-text-embedding-flash"
ADAPTER_CONFIG = "astrbot_plugin_iris_memory.iris_memory.l2_memory.adapter.get_config"


class FakeProvider:
    model_name = MODEL
    provider_config = {"id": "dashscope_embedding"}

    def get_dim(self) -> int:
        return 4

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


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


def _adapter() -> L2MemoryAdapter:
    provider = FakeProvider()
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


@pytest.mark.parametrize(
    "stored,actual,expected",
    [
        ("provider:a/m", "provider:a/m", True),
        ("provider:/m", "provider:a/m", True),
        ("provider:/BAAI/bge-m3", "provider:sf/BAAI/bge-m3", True),
        ("provider:/m1", "provider:a/m2", False),
        ("provider:a/m", "provider:b/m", False),
        ("local-model", "provider:a/local-model", False),
    ],
)
def test_is_same_embedding_model(stored, actual, expected):
    assert L2MemoryAdapter._is_same_embedding_model(stored, actual) is expected


@pytest.mark.asyncio
async def test_legacy_signature_upgrades_without_reembedding(tmp_path):
    """默认配置（留空）升级后不重嵌入，只把 meta 升级为新签名。"""
    adapter = _adapter()
    persist_dir = tmp_path / "faiss" / "memory_default"
    persist_dir.mkdir(parents=True)
    meta_path = persist_dir / "index_meta.json"
    meta_path.write_text(
        json.dumps({"embedding_model": f"provider:/{MODEL}", "embedding_dimensions": 4}),
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
async def test_real_model_change_still_migrates(tmp_path):
    adapter = _adapter()
    persist_dir = tmp_path / "faiss" / "memory_default"
    persist_dir.mkdir(parents=True)
    (persist_dir / "index_meta.json").write_text(
        json.dumps({"embedding_model": "provider:/other", "embedding_dimensions": 4}),
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
