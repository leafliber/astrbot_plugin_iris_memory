"""proactive.config.ConfigManager 测试

优先级：页面 overrides > AstrBotConfig（含 proactive 嵌套组）> 内置默认值。
"""

import pytest


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
    """AstrBotConfig 层：页面管理键读平铺值，三个面板键读 proactive 嵌套组"""

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
        # 页面管理键不走嵌套组，只读平铺值
        cm = make_config(cfg={"proactive": {"default_n": 99}})
        assert cm.default_n == 15

    def test_flat_value_clamped_by_property(self, make_config):
        cm = make_config(cfg={"proactive_quiet_minutes": 5})
        assert cm.proactive_quiet_minutes == 30  # 属性层钳制 [30, 720]


class TestOverridesLayer:
    """overrides 层优先级最高，但只接受 21 个页面管理键"""

    def test_override_beats_astrbot_config(self, make_config):
        cm = make_config(cfg={"default_n": 30}, overrides={"default_n": 50})
        assert cm.default_n == 50

    def test_override_beats_default(self, make_config):
        cm = make_config(overrides={"default_n": 50})
        assert cm.default_n == 50

    def test_load_overrides_filters_schema_keys(self, make_config):
        cm = make_config(overrides={"enabled": False, "provider_id": "x", "default_n": 99})
        # enabled / provider_id 非页面管理键，overrides 不生效
        assert cm.enabled is True
        assert cm.provider_id == ""
        assert cm.default_n == 99

    def test_load_overrides_ignores_unknown_keys(self, make_config):
        cm = make_config(overrides={"no_such_key": 1})
        assert cm.get_overrides() == {}

    def test_load_overrides_non_dict_ignored(self, make_config):
        cm = make_config()
        cm.load_overrides(None)
        cm.load_overrides("junk")
        assert cm.get_overrides() == {}

    def test_set_override_ignored_for_schema_keys(self, make_config):
        cm = make_config()
        cm.set_override("enabled", False)
        cm.set_override("provider_id", "abc")
        assert cm.enabled is True
        assert cm.provider_id == ""

    def test_page_managed_keys_are_22(self, make_config):
        # 25 个默认键 - 3 个 schema 面板键（enabled/stats_enabled/provider_id）
        all_cfg = make_config().get_all_page_config()
        assert len(all_cfg) == 22
        for key in ("enabled", "stats_enabled", "provider_id"):
            assert key not in all_cfg

    def test_set_override_clamps_int(self, make_config):
        cm = make_config()
        cm.set_override("default_n", 1)
        assert cm.default_n == 5
        cm.set_override("default_n", 999)
        assert cm.default_n == 120
        cm.set_override("window_size", 999)
        assert cm.window_size == 30

    def test_set_override_clamps_float(self, make_config):
        cm = make_config()
        cm.set_override("quality_threshold", 9.9)
        assert cm.quality_threshold == 1.0
        cm.set_override("quality_threshold", -1.0)
        assert cm.quality_threshold == 0.0

    def test_set_override_bool_coercion(self, make_config):
        cm = make_config()
        cm.set_override("proactive_enabled", "yes")
        assert cm.proactive_enabled is True
        cm.set_override("proactive_enabled", False)
        assert cm.proactive_enabled is False

    def test_set_override_str_trimmed(self, make_config):
        cm = make_config()
        cm.set_override("proactive_instruction", "  多聊技术话题  ")
        assert cm.proactive_instruction == "多聊技术话题"

    def test_set_override_mute_period(self, make_config):
        cm = make_config()
        cm.set_override(
            "mute_period",
            {"start_hour": 2, "start_minute": 30, "end_hour": 5, "end_minute": 45},
        )
        assert cm.mute_period == (2, 30, 5, 45)
        # 非法输入被忽略，保持原值
        cm.set_override("mute_period", "not-a-dict")
        assert cm.mute_period == (2, 30, 5, 45)

    def test_get_overrides_round_trip(self, make_config):
        cm = make_config()
        cm.set_override("default_n", 42)
        data = cm.get_overrides()
        assert data == {"default_n": 42}
        cm2 = make_config()
        cm2.load_overrides(data)
        assert cm2.default_n == 42
