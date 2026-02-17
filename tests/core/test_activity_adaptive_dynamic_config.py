"""群聊自适应动态配置测试"""

from unittest.mock import Mock

from iris_memory.core.activity_config import GroupActivityTracker
from iris_memory.core.config_manager import ConfigManager
from iris_memory.core.defaults import ACTIVITY_PRESETS, DEFAULTS, GroupActivityLevel


def _build_user_config_with_advanced(**overrides):
    config = Mock()
    config.advanced = overrides
    return config


class TestActivityAdaptiveDynamicConfig:
    """覆盖群聊自适应下所有动态行为配置键"""

    def test_group_uses_activity_presets_when_enabled(self, monkeypatch):
        """启用自适应时，群聊应使用活跃度预设值"""
        cfg = ConfigManager(user_config=Mock())
        tracker = GroupActivityTracker()
        cfg.init_activity_provider(tracker=tracker, enabled=True)

        monkeypatch.setattr(
            tracker,
            "get_activity_level",
            lambda group_id: GroupActivityLevel.ACTIVE,
        )

        group_id = "group-active"
        assert cfg.get_cooldown_seconds(group_id) == ACTIVITY_PRESETS.cooldown_seconds["active"]
        assert cfg.get_max_daily_replies(group_id) == ACTIVITY_PRESETS.max_daily_replies["active"]
        assert cfg.get_batch_threshold_count(group_id) == ACTIVITY_PRESETS.batch_threshold_count["active"]
        assert cfg.get_batch_threshold_interval(group_id) == ACTIVITY_PRESETS.batch_threshold_interval["active"]
        assert cfg.get_daily_analysis_budget(group_id) == ACTIVITY_PRESETS.daily_analysis_budget["active"]
        assert cfg.get_chat_context_count(group_id) == ACTIVITY_PRESETS.chat_context_count["active"]
        assert cfg.get_reply_temperature(group_id) == ACTIVITY_PRESETS.reply_temperature["active"]

    def test_private_chat_uses_advanced_overrides_even_if_enabled(self):
        """私聊（group_id=None）应走 advanced.* 覆盖值，不使用活跃度预设"""
        user_config = _build_user_config_with_advanced(
            cooldown_seconds=111,
            max_daily_replies=12,
            batch_threshold_count=28,
            batch_threshold_interval=420,
            daily_analysis_budget=77,
            chat_context_count=9,
            reply_temperature=0.66,
        )
        cfg = ConfigManager(user_config=user_config)
        cfg.init_activity_provider(tracker=GroupActivityTracker(), enabled=True)

        assert cfg.get_cooldown_seconds(None) == 111
        assert cfg.get_max_daily_replies(None) == 12
        assert cfg.get_batch_threshold_count(None) == 28
        assert cfg.get_batch_threshold_interval(None) == 420
        assert cfg.get_daily_analysis_budget(None) == 77
        assert cfg.get_chat_context_count(None) == 9
        assert cfg.get_reply_temperature(None) == 0.66

    def test_group_uses_advanced_overrides_when_adaptive_disabled(self):
        """禁用自适应时，群聊也应走 advanced.* 覆盖值"""
        user_config = _build_user_config_with_advanced(
            cooldown_seconds=95,
            max_daily_replies=33,
            batch_threshold_count=40,
            batch_threshold_interval=180,
            daily_analysis_budget=123,
            chat_context_count=14,
            reply_temperature=0.73,
        )
        cfg = ConfigManager(user_config=user_config)
        cfg.init_activity_provider(tracker=GroupActivityTracker(), enabled=False)

        group_id = "group-disabled"
        assert cfg.get_cooldown_seconds(group_id) == 95
        assert cfg.get_max_daily_replies(group_id) == 33
        assert cfg.get_batch_threshold_count(group_id) == 40
        assert cfg.get_batch_threshold_interval(group_id) == 180
        assert cfg.get_daily_analysis_budget(group_id) == 123
        assert cfg.get_chat_context_count(group_id) == 14
        assert cfg.get_reply_temperature(group_id) == 0.73

    def test_missing_advanced_values_fallback_to_defaults_when_disabled(self):
        """禁用自适应且无 advanced 覆盖时，应回退到默认值"""
        cfg = ConfigManager(user_config=None)
        cfg.init_activity_provider(tracker=GroupActivityTracker(), enabled=False)

        group_id = "group-default"
        assert cfg.get_cooldown_seconds(group_id) == DEFAULTS.proactive_reply.cooldown_seconds
        assert cfg.get_max_daily_replies(group_id) == DEFAULTS.proactive_reply.max_daily_replies
        assert cfg.get_batch_threshold_count(group_id) == DEFAULTS.message_processing.batch_threshold_count
        assert cfg.get_batch_threshold_interval(group_id) == DEFAULTS.message_processing.batch_threshold_interval
        assert cfg.get_daily_analysis_budget(group_id) == DEFAULTS.image_analysis.daily_analysis_budget
        assert cfg.get_chat_context_count(group_id) == DEFAULTS.llm_integration.chat_context_count
        assert cfg.get_reply_temperature(group_id) == DEFAULTS.proactive_reply.reply_temperature
