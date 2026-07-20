"""config_migrator 测试"""

import pytest

from iris_memory.config.config import reset_config
from iris_memory.legacy_migration.config_migrator import (
    HIDDEN_SUGGESTIONS_KEY,
    migrate_config,
)

from .conftest import FakeUserConfig


@pytest.fixture(autouse=True)
def _reset_iris_config():
    yield
    reset_config()


def _old_config(extra=None, save_raises=False):
    raw = FakeUserConfig(
        {
            "persona": {"enabled": False},
            "embedding": {
                "source": "local",
                "local_model": "moka-ai/m3e-small",
                "astrbot_provider_id": "emb_provider_1",
            },
            "knowledge_graph": {"enabled": False},
            "proactive_reply": {"enable": True},
            "llm_providers": {
                "knowledge_graph_provider_id": "kg_llm",
                "persona_provider_id": "persona_llm",
            },
            "error_friendly": {"enable": False},
        },
        save_raises=save_raises,
    )
    if extra:
        raw.update(extra)
    return raw


class TestDirectWrite:
    """直写 AstrBot 用户配置"""

    def test_full_mapping_written(self):
        raw = _old_config()
        stats = migrate_config(raw)

        assert stats["status"] == "ok"
        assert raw.save_calls == 1
        assert raw["profile"]["enable"] is False
        assert raw["l2_memory"]["embedding_source"] == "local"
        assert raw["l2_memory"]["embedding_model"] == "moka-ai/m3e-small"
        assert raw["l2_memory"]["embedding_provider"] == "emb_provider_1"
        assert raw["l3_kg"]["enable"] is False
        assert raw["l3_kg"]["extraction_provider"] == "kg_llm"
        assert raw["profile"]["analysis_provider"] == "persona_llm"
        assert raw["proactive"]["enabled"] is True

        migrated_keys = {new_key for _, new_key, _ in stats["migrated"]}
        assert "l3_kg.enable" in migrated_keys
        assert "profile.enable" in migrated_keys
        # 同名键（新旧 schema 段名一致）无需迁移，不列入映射
        assert "error_friendly.enable" not in migrated_keys

    def test_same_name_sections_not_detected(self):
        """新旧同名的配置段不视为旧配置键（加载时自动沿用）"""
        from iris_memory.legacy_migration.detector import detect_legacy_config_keys

        raw = {"error_friendly": {"enable": False}, "markdown_stripper": {"enable": False}}
        assert detect_legacy_config_keys(raw) == []

    def test_embedding_source_transform(self):
        raw = FakeUserConfig({"embedding": {"source": "auto"}})
        migrate_config(raw)
        assert raw["l2_memory"]["embedding_source"] == "provider"

        raw2 = FakeUserConfig({"embedding": {"source": "astrbot"}})
        migrate_config(raw2)
        assert raw2["l2_memory"]["embedding_source"] == "provider"

        raw3 = FakeUserConfig({"embedding": {"source": "unknown_value"}})
        stats3 = migrate_config(raw3)
        assert "l2_memory" not in raw3
        assert stats3["migrated"] == []

    def test_skip_when_user_customized(self):
        raw = _old_config(
            {
                # 用户已在新版配置中显式关闭 L3（与默认值 True 不同）
                "l3_kg": {"enable": False},
                "knowledge_graph": {"enabled": True},
            }
        )
        stats = migrate_config(raw)

        assert "l3_kg.enable" in stats["skipped_non_default"]
        assert raw["l3_kg"]["enable"] is False  # 未被覆盖

    def test_no_overwrite_when_same_as_default(self):
        # 新键当前值等于默认值 → 视为未自定义，允许写入
        raw = FakeUserConfig(
            {
                "l3_kg": {"enable": True},  # 与 schema 默认值相同
                "knowledge_graph": {"enabled": False},
            }
        )
        stats = migrate_config(raw)
        assert raw["l3_kg"]["enable"] is False
        assert any(k == "l3_kg.enable" for _, k, _ in stats["migrated"])

    def test_empty_string_provider_skipped(self):
        raw = FakeUserConfig(
            {
                "embedding": {"astrbot_provider_id": ""},
                "llm_providers": {"persona_provider_id": ""},
            }
        )
        stats = migrate_config(raw)
        assert stats["migrated"] == []
        assert raw.save_calls == 0

    def test_no_old_keys(self):
        raw = FakeUserConfig({"l2_memory": {"enable": True}})
        stats = migrate_config(raw)
        assert stats["status"] == "ok"
        assert stats["migrated"] == []
        assert raw.save_calls == 0

    def test_no_config(self):
        stats = migrate_config(None)
        assert stats["status"] == "skipped_no_config"


class TestHiddenFallback:
    """持久化失败 → hidden_config.json + 用户可见建议"""

    def test_save_failure_fallback(self, tmp_path):
        from iris_memory.config import get_config, init_config

        raw = _old_config(save_raises=True)
        init_config(raw, tmp_path)

        stats = migrate_config(raw)
        assert stats["status"] == "hidden_fallback"

        suggestions = get_config().get(HIDDEN_SUGGESTIONS_KEY)
        assert isinstance(suggestions, dict)
        assert suggestions["l3_kg.enable"]["value"] is False
        assert suggestions["l3_kg.enable"]["from"] == "knowledge_graph.enabled"
        assert suggestions["profile.enable"]["value"] is False
        assert suggestions["l2_memory.embedding_model"]["value"] == "moka-ai/m3e-small"

    def test_plain_dict_fallback(self, tmp_path):
        """普通 dict 无 save_config → 同样走 hidden 回退"""
        from iris_memory.config import get_config, init_config

        init_config(FakeUserConfig(), tmp_path)

        raw = {"persona": {"enabled": False}}
        stats = migrate_config(raw)
        assert stats["status"] == "hidden_fallback"
        # dict 内已写入（内存生效），但因无法持久化而给出建议
        assert raw["profile"]["enable"] is False

        suggestions = get_config().get(HIDDEN_SUGGESTIONS_KEY)
        assert suggestions["profile.enable"]["value"] is False
