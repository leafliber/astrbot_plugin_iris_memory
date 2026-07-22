"""tests/proactive 共享夹具

ConfigManager 对配置对象只调用 .get()，故用普通 dict 模拟 AstrBotConfig。
门控/状态类夹具默认使用永不命中的静音时段（start == end，区间为空），
保证测试结果与真实运行时刻无关。
"""

import time

import pytest

from iris_memory.proactive.config import ConfigManager
from iris_memory.proactive.perception import WindowMessage
from iris_memory.proactive.signals import SignalGate
from iris_memory.proactive.state import StateManager

# 永不命中的静音时段：start_mins == end_mins → start <= current < start 恒为 False
NEVER_MUTE = {"start_hour": 0, "start_minute": 0, "end_hour": 0, "end_minute": 0}


@pytest.fixture
def make_config():
    """ConfigManager 工厂：cfg 模拟 AstrBotConfig，hidden 模拟隐藏配置。

    overrides 兼容旧版页面键名（default_n 等），经 legacy_overrides_to_hidden
    翻译为隐藏配置键后合并进 hidden。
    """

    def _make(
        cfg: dict | None = None,
        hidden: dict | None = None,
        overrides: dict | None = None,
    ) -> ConfigManager:
        merged = dict(hidden or {})
        if overrides:
            merged.update(ConfigManager.legacy_overrides_to_hidden(overrides))
        return ConfigManager(cfg or {}, hidden_get=merged.get)

    return _make


@pytest.fixture
def nm_config(make_config):
    """never-mute 版工厂：门控/状态测试专用，消除真实时刻影响"""

    def _make(
        cfg: dict | None = None,
        hidden: dict | None = None,
        overrides: dict | None = None,
    ) -> ConfigManager:
        base = {"mute_period": dict(NEVER_MUTE)}
        if cfg:
            base.update(cfg)
        return make_config(cfg=base, hidden=hidden, overrides=overrides)

    return _make


@pytest.fixture
def config(nm_config):
    return nm_config()


@pytest.fixture
def state(config):
    return StateManager(config)


@pytest.fixture
def gate(config, state):
    return SignalGate(config, state)


@pytest.fixture
def make_msg():
    """WindowMessage 工厂"""

    def _make(
        sender_id: str = "u1",
        sender_name: str = "User1",
        content: str = "hello",
        timestamp: float | None = None,
    ) -> WindowMessage:
        return WindowMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=time.time() if timestamp is None else timestamp,
        )

    return _make
