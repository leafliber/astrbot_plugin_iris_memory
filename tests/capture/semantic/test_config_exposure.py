"""
语义提取配置与上下文注入测试

验证：
- _conf_schema.json 不暴露 semantic_extraction 配置（隐性配置）
- DEFAULTS 中有完整的语义提取默认值
- lifecycle_manager 使用 DEFAULTS 作为默认语义提取参数
- set_astrbot_context 正确注入上下文到 lifecycle_manager 和 SemanticExtractor
- _init_semantic_extractor 传递 context/provider_id 到 SemanticExtractor
- 配置覆盖流程 lifecycle_manager → SemanticClustering
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from iris_memory.config import get_store


_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "_conf_schema.json"


class TestConfigSchemaHidden:
    """验证 semantic_extraction 不暴露在用户配置 schema 中"""

    def test_schema_does_not_expose_semantic_extraction(self):
        """_conf_schema.json 不应包含 semantic_extraction 区块"""
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)
        assert "semantic_extraction" not in schema

    def test_defaults_have_semantic_extraction(self):
        """ConfigStore 中有完整的语义提取默认值"""
        cfg = get_store()
        assert cfg.get("semantic_extraction.enabled") is not None
        assert cfg.get("semantic_extraction.min_age_days") is not None
        assert cfg.get("semantic_extraction.min_cluster_size") is not None
        assert cfg.get("semantic_extraction.min_confidence") is not None
        assert cfg.get("semantic_extraction.extraction_interval") is not None

    def test_defaults_values_reasonable(self):
        """默认值合理"""
        cfg = get_store()
        assert cfg.get("semantic_extraction.enabled") is True
        assert cfg.get("semantic_extraction.min_age_days") >= 1
        assert cfg.get("semantic_extraction.min_cluster_size") >= 2
        assert 0.0 <= cfg.get("semantic_extraction.min_confidence") <= 1.0
        assert cfg.get("semantic_extraction.extraction_interval") >= 60


class TestAstrBotContextInjection:
    """验证 AstrBot 上下文注入链"""

    def test_lifecycle_manager_stores_context(self):
        """set_astrbot_context 应保存 context 和 provider_id"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mock_ctx = MagicMock()
        mgr.set_astrbot_context(mock_ctx, "test_provider")

        assert mgr._astrbot_context is mock_ctx
        assert mgr._semantic_provider_id == "test_provider"

    def test_lifecycle_manager_default_context_is_none(self):
        """lifecycle_manager 默认 context 为 None"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        assert mgr._astrbot_context is None
        assert mgr._semantic_provider_id == ""

    def test_set_context_propagates_to_existing_extractor(self):
        """set_astrbot_context 同步更新已初始化的 extractor"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mgr._init_semantic_extractor()
        assert mgr._semantic_extractor.astrbot_context is None

        mock_ctx = MagicMock()
        mgr.set_astrbot_context(mock_ctx, "later_provider")

        assert mgr._semantic_extractor.astrbot_context is mock_ctx
        assert mgr._semantic_extractor._provider_id == "later_provider"

    def test_init_extractor_receives_context(self):
        """_init_semantic_extractor 将已存储的 context 传递给 SemanticExtractor"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mock_ctx = MagicMock()
        mgr.set_astrbot_context(mock_ctx, "my_provider")
        mgr._init_semantic_extractor()

        assert mgr._semantic_extractor is not None
        assert mgr._semantic_extractor.astrbot_context is mock_ctx
        assert mgr._semantic_extractor._provider_id == "my_provider"


class TestConfigWiring:
    """配置从 lifecycle_manager → SemanticClustering 传递"""

    def test_lifecycle_manager_defaults(self):
        """lifecycle_manager 默认语义提取配置与 DEFAULTS 一致"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        assert mgr._semantic_extraction_enabled == get_store().get("semantic_extraction.enabled")
        assert mgr._semantic_extraction_interval == get_store().get("semantic_extraction.extraction_interval")
        assert mgr._semantic_extraction_config == {}

    def test_lifecycle_manager_config_override(self):
        """lifecycle_manager 配置可被内部覆盖"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mgr._semantic_extraction_enabled = False
        mgr._semantic_extraction_interval = 3600
        mgr._semantic_extraction_config = {
            "min_age_days": 7,
            "min_cluster_size": 5,
            "min_confidence": 0.6,
        }

        assert mgr._semantic_extraction_enabled is False
        assert mgr._semantic_extraction_interval == 3600
        assert mgr._semantic_extraction_config["min_age_days"] == 7

    def test_init_semantic_extractor_uses_config(self):
        """_init_semantic_extractor 将 config 传递到 SemanticClustering"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mgr._semantic_extraction_config = {
            "min_age_days": 14,
            "min_cluster_size": 4,
            "min_confidence": 0.5,
        }

        mgr._init_semantic_extractor()

        assert mgr._semantic_extractor is not None
        clustering = mgr._semantic_extractor.clustering
        assert clustering.min_age_days == 14
        assert clustering.min_cluster_size == 4
        assert clustering.min_confidence == 0.5

    def test_init_semantic_extractor_no_config(self):
        """无覆盖配置时使用 SemanticClustering 默认值"""
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager

        mgr = SessionLifecycleManager(
            session_manager=MagicMock(),
            chroma_manager=MagicMock(),
        )
        mgr._init_semantic_extractor()

        assert mgr._semantic_extractor is not None
        clustering = mgr._semantic_extractor.clustering
        assert clustering is not None
