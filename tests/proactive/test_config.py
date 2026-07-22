"""proactive.config.ConfigManager 测试

优先级：隐藏配置（hidden_get）> AstrBotConfig（平铺遗留值 / proactive 嵌套组）> 内置默认值。
"""

import pytest

from iris_memory.proactive.config import ConfigManager


class TestDefaults:
    """空配置时全部回落到内置默认值"""

    def test_defaults_with_empty_config(self, make_config):
        cm = make_config()
        assert cm.enabled is True
        assert cm.stats_enabled is False
        assert cm.provider_id == ""
        assert cm.default_n == 15
        assert cm.default_t == 30
        assert cm.window_size == 15
        assert cm.max_token == 3000
        assert cm.quality_threshold == pytest.approx(0.2)
        assert cm.mute_period == (1, 0, 7, 0)
        assert cm.proactive_enabled is False
        assert cm.proactive_max_per_day == 2
        assert cm.proactive_instruction == ""

    def test_non_dict_proactive_group_ignored(self, make_config):
        assert make_config(cfg={"proactive": None}).enabled is True
        assert make_config(cfg={"proactive": "yes"}).enabled is True


class TestAstrBotConfigLayer:
    """AstrBotConfig 层：遗留平铺值与三个面板键（proactive 嵌套组）"""

    def test_flat_key_beats_default(self, make_config):
        cm = make_config(cfg={"default_n": 30, "window_size": 20})
        assert cm.default_n == 30
        assert cm.window_size == 20

    def test_nested_proactive_group(self, make_config):
        cm = make_config(
            cfg={"proactive": {"enabled": False, "stats_enabled": True, "provider_id": "p1"}}
        )
        assert cm.enabled is False
        assert cm.stats_enabled is True
        assert cm.provider_id == "p1"

    def test_nested_none_value_falls_back_to_default(self, make_config):
        cm = make_config(cfg={"proactive": {"enabled": None}})
        assert cm.enabled is True

    def test_page_keys_not_read_from_nested_group(self, make_config):
        # 页面管理键不走嵌套组
        cm = make_config(cfg={"proactive": {"default_n": 99}})
        assert cm.default_n == 15

    def test_flat_value_clamped_by_property(self, make_config):
        cm = make_config(cfg={"proactive_quiet_minutes": 5})
        assert cm.proactive_quiet_minutes == 30  # 属性层钳制 [30, 720]

    def test_flat_mute_period_dict(self, make_config):
        cm = make_config(
            cfg={"mute_period": {"start_hour": 2, "start_minute": 30, "end_hour": 5, "end_minute": 45}}
        )
        assert cm.mute_period == (2, 30, 5, 45)


class TestHiddenLayer:
    """隐藏配置层优先级最高，键名为 HiddenConfig 字段名"""

    def test_hidden_beats_astrbot_config(self, make_config):
        cm = make_config(cfg={"default_n": 30}, hidden={"reply_default_n": 50})
        assert cm.default_n == 50

    def test_hidden_beats_default(self, make_config):
        cm = make_config(hidden={"reply_default_n": 50})
        assert cm.default_n == 50

    def test_schema_keys_not_affected_by_hidden(self, make_config):
        cm = make_config(hidden={"reply_default_n": 99})
        assert cm.enabled is True
        assert cm.provider_id == ""
        assert cm.default_n == 99

    def test_hidden_clamps_int(self, make_config):
        assert make_config(hidden={"reply_default_n": 1}).default_n == 5
        assert make_config(hidden={"reply_default_n": 999}).default_n == 120
        assert make_config(hidden={"reply_window_size": 999}).window_size == 30

    def test_hidden_clamps_float(self, make_config):
        assert make_config(hidden={"reply_quality_threshold": 9.9}).quality_threshold == 1.0
        assert make_config(hidden={"reply_quality_threshold": -1.0}).quality_threshold == 0.0

    def test_hidden_bool(self, make_config):
        assert make_config(hidden={"proactive_enabled": True}).proactive_enabled is True
        assert make_config(hidden={"proactive_enabled": False}).proactive_enabled is False

    def test_hidden_str(self, make_config):
        cm = make_config(hidden={"proactive_instruction": "多聊技术话题"})
        assert cm.proactive_instruction == "多聊技术话题"

    def test_hidden_mute_period_parts(self, make_config):
        cm = make_config(hidden={
            "reply_mute_start_hour": 2,
            "reply_mute_start_minute": 30,
            "reply_mute_end_hour": 5,
            "reply_mute_end_minute": 45,
        })
        assert cm.mute_period == (2, 30, 5, 45)

    def test_hidden_mute_part_beats_flat_dict(self, make_config):
        cm = make_config(
            cfg={"mute_period": {"start_hour": 9, "start_minute": 0, "end_hour": 10, "end_minute": 0}},
            hidden={"reply_mute_start_hour": 3},
        )
        sh, sm, eh, em = cm.mute_period
        assert sh == 3  # 隐藏配置覆盖单字段
        assert (sm, eh, em) == (0, 10, 0)  # 其余回落到平铺 dict


class TestLegacyOverridesMigration:
    """旧版 KV 页面覆写 → 隐藏配置键值翻译"""

    def test_non_dict_input_returns_empty(self):
        assert ConfigManager.legacy_overrides_to_hidden(None) == {}
        assert ConfigManager.legacy_overrides_to_hidden("junk") == {}
        assert ConfigManager.legacy_overrides_to_hidden({}) == {}

    def test_basic_keys_mapped_with_prefix(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"default_n": 42, "window_size": 20, "quality_threshold": 0.5}
        )
        assert result == {
            "reply_default_n": 42,
            "reply_window_size": 20,
            "reply_quality_threshold": 0.5,
        }

    def test_proactive_keys_keep_name(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"proactive_enabled": True, "proactive_max_per_day": 3}
        )
        assert result == {"proactive_enabled": True, "proactive_max_per_day": 3}

    def test_mute_period_split_into_four_keys(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"mute_period": {"start_hour": 2, "start_minute": 30, "end_hour": 5, "end_minute": 45}}
        )
        assert result == {
            "reply_mute_start_hour": 2,
            "reply_mute_start_minute": 30,
            "reply_mute_end_hour": 5,
            "reply_mute_end_minute": 45,
        }

    def test_schema_and_unknown_keys_skipped(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"enabled": False, "provider_id": "x", "stats_enabled": True, "no_such_key": 1}
        )
        assert result == {}

    def test_bad_values_skipped(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"default_n": "not-a-number", "mute_period": "not-a-dict", "window_size": 18}
        )
        assert result == {"reply_window_size": 18}

    def test_bool_and_str_coercion(self):
        result = ConfigManager.legacy_overrides_to_hidden(
            {"proactive_enabled": "yes", "proactive_instruction": "  多聊技术话题  "}
        )
        assert result == {"proactive_enabled": True, "proactive_instruction": "多聊技术话题"}

    def test_migrated_values_round_trip(self, make_config):
        legacy = {"default_n": 42, "mute_period": {"start_hour": 2, "start_minute": 0, "end_hour": 6, "end_minute": 30}}
        hidden = ConfigManager.legacy_overrides_to_hidden(legacy)
        cm = make_config(hidden=hidden)
        assert cm.default_n == 42
        assert cm.mute_period == (2, 0, 6, 30)
