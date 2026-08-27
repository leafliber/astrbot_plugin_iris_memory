"""配置系统测试"""

import json
from pathlib import Path
from iris_memory.config import Config, init_config, get_config
from iris_memory.config.hidden_config import HiddenConfigManager
from iris_memory.config.defaults import Defaults, HiddenConfig


class TestConfig:
    def test_get_flat_key(self, tmp_path: Path):
        astrbot_config = {
            "l1_buffer": {"enable": True, "max_tokens": 1000},
        }

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        assert config.get("l1_buffer.enable")

    def test_extras_group_defaults(self, tmp_path: Path):
        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        config = Config({}, hidden_manager, Defaults(), tmp_path)

        # pure_at_reply 默认关闭，与 _conf_schema.json 一致
        assert config.get("extras.pure_at_reply.enable") is False
        assert config.get("extras.error_friendly.enable") is True
        assert config.get("extras.markdown_stripper.enable") is True

    def test_schema_groups_auxiliary_features(self):
        schema_path = Path(__file__).resolve().parents[2] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        assert "error_friendly" not in schema
        assert "markdown_stripper" not in schema
        assert set(schema["extras"]["items"]) == {
            "pure_at_reply",
            "error_friendly",
            "markdown_stripper",
        }

    @staticmethod
    def _flatten_schema(schema: dict) -> dict:
        """把 schema 的嵌套 items 展平为 {section.key: default}。"""
        flat: dict = {}
        for section, body in schema.items():
            items = body.get("items", {}) if isinstance(body, dict) else {}
            for key, spec in items.items():
                if isinstance(spec, dict) and "default" in spec:
                    flat[f"{section}.{key}"] = spec["default"]
        return flat

    @staticmethod
    def _flatten_defaults(defaults) -> dict:
        """把 Defaults dataclass 展平为 {section.key: default}（不含 hidden）。"""
        from dataclasses import asdict

        flat: dict = {}
        for section, values in asdict(defaults).items():
            if section == "hidden" or not isinstance(values, dict):
                continue
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        return flat

    def test_schema_defaults_match_dataclass_defaults(self):
        """schema 与 Defaults 的同名键默认值必须一致，防止双源漂移。

        AstrBot 会把 schema 默认物化进用户配置，Defaults 是用户配置
        缺键时的回退——两处不一致会让不同路径拿到相反语义。
        proactive 分组由 proactive.config._DEFAULTS 单独管理，单独比对。
        """
        from iris_memory.proactive.config import _DEFAULTS as reply_defaults

        schema_path = Path(__file__).resolve().parents[2] / "_conf_schema.json"
        schema_flat = self._flatten_schema(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )
        defaults_flat = self._flatten_defaults(Defaults())

        schema_only = set(schema_flat) - set(defaults_flat) - {
            "proactive.enabled",
            "proactive.proactive_enabled",
            "proactive.stats_enabled",
            "proactive.provider_id",
        }
        assert not schema_only, f"schema 存在 Defaults 未覆盖的键：{schema_only}"

        for key, schema_value in schema_flat.items():
            if key not in defaults_flat:
                continue
            assert schema_value == defaults_flat[key], (
                f"配置默认值双源分歧：{key} schema={schema_value!r} "
                f"defaults={defaults_flat[key]!r}"
            )

        # proactive 分组与 proactive.config._DEFAULTS 对齐
        assert schema_flat.get("proactive.enabled") == reply_defaults.get("enabled")
        assert (
            schema_flat.get("proactive.proactive_enabled")
            == reply_defaults.get("proactive_enabled")
        )
        assert (
            schema_flat.get("proactive.stats_enabled")
            == reply_defaults.get("stats_enabled")
        )
        assert (
            schema_flat.get("proactive.provider_id")
            == reply_defaults.get("provider_id")
        )

    def test_legacy_dream_switches_do_not_map_to_new_stages(self, tmp_path: Path):
        astrbot_config = {
            "scheduled_tasks": {
                "dream_enable_consolidation": False,
                "dream_enable_temporal_anchor": False,
                "dream_enable_contradiction": False,
                "dream_enable_pattern_discovery": False,
                "dream_enable_knowledge_extract": False,
                "dream_enable_pruning": False,
            }
        }
        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        config = Config(astrbot_config, hidden_manager, Defaults(), tmp_path)

        new_switches = [
            "dream_stage_temporal_anchor_enabled",
            "dream_stage_reconciliation_enabled",
            "dream_stage_knowledge_induction_enabled",
            "dream_stage_l2_pruning_enabled",
            "dream_stage_l3_maintenance_enabled",
        ]
        assert all(
            config.get(f"scheduled_tasks.{switch}") is True
            for switch in new_switches
        )

    def test_get_with_default(self, tmp_path: Path):
        astrbot_config = {}

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        assert config.get("nonexistent", "default") == "default"

    def test_typed_numeric_getters(self, tmp_path: Path):
        astrbot_config = {
            "numbers": {
                "integer": "12",
                "decimal": "0.75",
                "invalid": {"unexpected": True},
            },
        }
        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        config = Config(astrbot_config, hidden_manager, Defaults(), tmp_path)

        assert config.get_int("numbers.integer", 6) == 12
        assert config.get_float("numbers.decimal", 0.5) == 0.75
        assert config.get_int("numbers.invalid", 6) == 6
        assert config.get_float("numbers.invalid", 0.5) == 0.5

    def test_set_hidden_config(self, tmp_path: Path):
        astrbot_config = {}

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        config.set_hidden("debug_mode", True)

        assert config.get("debug_mode")

    def test_config_priority(self, tmp_path: Path):
        astrbot_config = {
            "test_section": {"test_key": "user_value"},
        }

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        assert config.get("test_section.test_key") == "user_value"

        config.set_hidden("test_section.test_key", "hidden_value")
        assert config.get("test_section.test_key") == "user_value"

    def test_data_dir_property(self, tmp_path: Path):
        astrbot_config = {}

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        assert config.data_dir == tmp_path

    def test_on_config_change(self, tmp_path: Path):
        astrbot_config = {}

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        changes = []

        def on_change(key, old_value, new_value):
            changes.append((key, old_value, new_value))

        config.on_config_change(on_change)

        config.set_hidden("debug_mode", True)

        assert len(changes) == 1
        assert changes[0] == ("debug_mode", None, True)

    def test_has_nonexistent_deep_key_returns_false(self, tmp_path: Path):
        """回归：has() 对不存在的多段键应返回 False

        历史 bug：has() 使用 ``is not None`` 而非 ``is not _UNSET`` 判断用户
        配置。_get_user_config 找不到键时返回 _UNSET 哨兵对象，而
        ``_UNSET is not None`` 恒为 True，导致所有 >=2 段键的 has() 错误返回
        True。修复后使用 ``is not _UNSET`` 正确识别缺失键。
        """
        astrbot_config = {}

        hidden_manager = HiddenConfigManager(
            tmp_path / "hidden_config.json", HiddenConfig()
        )
        defaults = Defaults()

        config = Config(astrbot_config, hidden_manager, defaults, tmp_path)

        # 不存在的深层键应返回 False（此前 bug 会返回 True）
        assert config.has("nonexistent.key.deep") is False

        # 存在于用户配置中的键仍应返回 True，确保修复未破坏正向判断
        astrbot_config_with_value = {"l1_buffer": {"enable": True}}
        config_with_value = Config(
            astrbot_config_with_value, hidden_manager, defaults, tmp_path
        )
        assert config_with_value.has("l1_buffer.enable") is True


class TestHiddenConfigManager:
    def test_get_set(self, tmp_path: Path):
        manager = HiddenConfigManager(tmp_path / "hidden_config.json", HiddenConfig())

        manager.set("test_key", "test_value")

        assert manager.get("test_key") == "test_value"

    def test_persistence(self, tmp_path: Path):
        config_path = tmp_path / "hidden_config.json"
        manager1 = HiddenConfigManager(config_path, HiddenConfig())
        manager1.set("test_key", "test_value")

        manager2 = HiddenConfigManager(config_path, HiddenConfig())
        assert manager2.get("test_key") == "test_value"

    def test_default_value(self, tmp_path: Path):
        manager = HiddenConfigManager(tmp_path / "hidden_config.json", HiddenConfig())

        assert manager.get("nonexistent") is None


def test_global_config(tmp_path: Path):
    astrbot_config = {
        "l1_buffer": {"enable": True},
    }

    init_config(astrbot_config, tmp_path)

    config = get_config()
    assert config is not None
    assert config.data_dir == tmp_path
