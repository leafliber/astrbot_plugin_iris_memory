"""
配置调用链验证测试

验证从 ConfigStore → ProactiveConfig → 各组件的配置访问是否完整可用。
"""

import pytest
from iris_memory.config import get_store, init_store, reset_store
from iris_memory.proactive.config import ProactiveConfig


@pytest.fixture(autouse=True)
def _store():
    """每个测试使用默认 Schema 值初始化 ConfigStore"""
    init_store(None)
    yield
    reset_store()


def test_store_has_proactive_reply_keys():
    """ConfigStore 中有完整的 proactive_reply 配置"""
    cfg = get_store()
    required_keys = [
        "proactive_reply.followup_after_all_replies",
        "proactive_reply.followup_window_seconds",
        "proactive_reply.max_followup_count",
        "proactive_reply.followup_short_window_seconds",
        "proactive_reply.signal_weight_direct_reply",
        "proactive_reply.signal_ttl_emotion_high",
        "proactive_reply.cooldown_seconds",
        "proactive_reply.max_daily_replies",
        "proactive_reply.quiet_hours",
    ]
    for key in required_keys:
        assert cfg.get(key) is not None, f"缺少配置键：{key}"


def test_proactive_config_creation():
    """ProactiveConfig 可从 ConfigStore 成功创建"""
    config = ProactiveConfig(get_store())
    assert config.enabled is not None
    assert config.followup_enabled is True
    assert config.followup_after_all_replies is not None


def test_proactive_config_flat_fields():
    """ProactiveConfig 所有扁平化配置字段均可访问"""
    config = ProactiveConfig(get_store())
    flat_fields = [
        "followup_window_seconds",
        "max_followup_count",
        "followup_short_window_seconds",
        "signal_weight_direct_reply",
        "signal_ttl_emotion_high",
        "cooldown_seconds",
    ]
    for field in flat_fields:
        assert hasattr(config, field), f"缺少扁平化字段：{field}"
        assert getattr(config, field) is not None


def test_proactive_config_component_access():
    """各组件所需的配置项均可从 ProactiveConfig 访问"""
    config = ProactiveConfig(get_store())
    component_attrs = [
        ("FollowUpPlanner.short_window", config.followup_short_window_seconds),
        ("FollowUpPlanner.fallback", config.followup_fallback_to_rule_on_llm_error),
        ("Manager.short_window", config.followup_short_window_seconds),
        ("SignalQueue.max_signals", config.signal_max_signals_per_group),
        ("SignalGenerator.ttl_rule", config.signal_ttl_rule_match),
        ("SignalGenerator.ttl_emotion", config.signal_ttl_emotion_high),
        ("GroupScheduler.check_interval", config.signal_check_interval_seconds),
        ("GroupScheduler.silence_timeout", config.signal_silence_timeout_seconds),
        ("GroupScheduler.min_silence", config.signal_min_silence_seconds),
        ("GroupScheduler.weight_direct", config.signal_weight_direct_reply),
        ("GroupScheduler.weight_llm", config.signal_weight_llm_confirm),
    ]
    for name, value in component_attrs:
        assert value is not None, f"{name} 为 None"


def test_proactive_config_to_dict():
    """ProactiveConfig.to_dict() 包含关键配置键"""
    config = ProactiveConfig(get_store())
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert len(config_dict) > 0

    required_keys = [
        "followup_short_window_seconds",
        "signal_weight_direct_reply",
        "max_followup_count",
    ]
    for key in required_keys:
        assert key in config_dict, f"配置字典中缺少键：{key}"
