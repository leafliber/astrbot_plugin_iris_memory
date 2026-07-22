from __future__ import annotations

from collections.abc import Callable
from typing import Any

from astrbot.api import AstrBotConfig


_DEFAULTS = {
    "enabled": True,
    "stats_enabled": False,
    "mute_period": {
        "start_hour": 1,
        "start_minute": 0,
        "end_hour": 7,
        "end_minute": 0,
    },
    "window_size": 15,
    "default_n": 15,
    "default_t": 30,
    "max_token": 3000,
    "follow_up_ttl": 10,
    "follow_up_aggregate_window": 6,
    "quality_threshold": 0.2,
    "provider_id": "",
    "trigger_min_interval": 30,
    "boost_factor": 0.6,
    "boost_duration": 15,
    "max_boosted_replies": 5,
    "proactive_enabled": False,
    "proactive_check_interval": 5,
    "proactive_quiet_minutes": 120,
    "proactive_max_per_day": 2,
    "proactive_min_interval": 360,
    "proactive_drift_delay": 15,
    "proactive_pending_timeout": 30,
    "proactive_max_streak": 2,
    "proactive_instruction": "",
    "proactive_max_message_len": 300,
}

# _conf_schema.json 中 proactive 分组内的三个键（嵌套存储，面板管理）
_SCHEMA_GROUP = "proactive"
_SCHEMA_KEYS = {"enabled", "stats_enabled", "provider_id"}

# 页面管理键 → 隐藏配置键（HiddenConfig 字段名）。
# 主动发起类键名与隐藏配置字段同名，基本参数类加 reply_ 前缀，
# mute_period 拆为四个独立整数键。
_HIDDEN_KEY_MAP = {
    "window_size": "reply_window_size",
    "default_n": "reply_default_n",
    "default_t": "reply_default_t",
    "max_token": "reply_max_token",
    "follow_up_ttl": "reply_follow_up_ttl",
    "follow_up_aggregate_window": "reply_follow_up_aggregate_window",
    "quality_threshold": "reply_quality_threshold",
    "trigger_min_interval": "reply_trigger_min_interval",
    "boost_factor": "reply_boost_factor",
    "boost_duration": "reply_boost_duration",
    "max_boosted_replies": "reply_max_boosted_replies",
    "proactive_enabled": "proactive_enabled",
    "proactive_check_interval": "proactive_check_interval",
    "proactive_quiet_minutes": "proactive_quiet_minutes",
    "proactive_max_per_day": "proactive_max_per_day",
    "proactive_min_interval": "proactive_min_interval",
    "proactive_drift_delay": "proactive_drift_delay",
    "proactive_pending_timeout": "proactive_pending_timeout",
    "proactive_max_streak": "proactive_max_streak",
    "proactive_instruction": "proactive_instruction",
    "proactive_max_message_len": "proactive_max_message_len",
}

_MUTE_HIDDEN_KEYS = {
    "start_hour": "reply_mute_start_hour",
    "start_minute": "reply_mute_start_minute",
    "end_hour": "reply_mute_end_hour",
    "end_minute": "reply_mute_end_minute",
}

HiddenGetFn = Callable[[str], Any]


def _coerce_like(default: Any, value: Any) -> Any:
    """按默认值类型强制转换迁移值，失败返回 None（调用方跳过）。"""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "on")
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value).strip()
    except (TypeError, ValueError):
        return None
    return value


class ConfigManager:
    """主动回复配置管理

    取值优先级：隐藏配置（hidden_get）> AstrBotConfig 平铺值（遗留兼容）> 内置默认值。
    enabled / stats_enabled / provider_id 三个面板键仍读 _conf_schema.json 的
    proactive 嵌套组，不经过隐藏配置。
    """

    def __init__(
        self,
        config: AstrBotConfig,
        hidden_get: HiddenGetFn | None = None,
    ) -> None:
        self._cfg = config
        self._hidden_get = hidden_get

    @staticmethod
    def legacy_overrides_to_hidden(data: dict | None) -> dict[str, Any]:
        """把旧版 KV 页面覆写（iris_reply:config_overrides）翻译为隐藏配置键值。

        返回 {隐藏配置键: 值}，非法/未知键被跳过；空输入返回空 dict。
        """
        if not isinstance(data, dict) or not data:
            return {}
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key == "mute_period":
                if not isinstance(value, dict):
                    continue
                dmp = _DEFAULTS["mute_period"]
                for sub_key, hidden_key in _MUTE_HIDDEN_KEYS.items():
                    coerced = _coerce_like(0, value.get(sub_key, dmp[sub_key]))
                    if coerced is not None:
                        result[hidden_key] = coerced
                continue
            hidden_key = _HIDDEN_KEY_MAP.get(key)
            if hidden_key is None or key not in _DEFAULTS:
                continue
            coerced = _coerce_like(_DEFAULTS[key], value)
            if coerced is not None:
                result[hidden_key] = coerced
        return result

    def _get(self, key: str, default=None):
        if key in _SCHEMA_KEYS:
            # 三个面板配置项存于 _conf_schema.json 的 proactive 分组（嵌套 dict）
            group = self._cfg.get(_SCHEMA_GROUP)
            if isinstance(group, dict):
                val = group.get(key)
                if val is not None:
                    return val
            return _DEFAULTS.get(key, default)

        hidden_key = _HIDDEN_KEY_MAP.get(key)
        if hidden_key is not None and self._hidden_get is not None:
            val = self._hidden_get(hidden_key)
            if val is not None:
                return val

        val = self._cfg.get(key)
        if val is not None:
            return val
        return _DEFAULTS.get(key, default)

    def _get_mute_part(self, part: str) -> int:
        dmp = _DEFAULTS["mute_period"]
        hidden_key = _MUTE_HIDDEN_KEYS[part]
        if self._hidden_get is not None:
            val = self._hidden_get(hidden_key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        mp = self._cfg.get("mute_period")
        if isinstance(mp, dict):
            try:
                return int(mp.get(part, dmp[part]))
            except (TypeError, ValueError):
                pass
        return dmp[part]

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled"))

    @property
    def stats_enabled(self) -> bool:
        return bool(self._get("stats_enabled"))

    @property
    def mute_period(self) -> tuple[int, int, int, int]:
        return (
            self._get_mute_part("start_hour"),
            self._get_mute_part("start_minute"),
            self._get_mute_part("end_hour"),
            self._get_mute_part("end_minute"),
        )

    @property
    def mute_start_hour(self) -> int:
        return self.mute_period[0]

    @property
    def mute_start_minute(self) -> int:
        return self.mute_period[1]

    @property
    def mute_end_hour(self) -> int:
        return self.mute_period[2]

    @property
    def mute_end_minute(self) -> int:
        return self.mute_period[3]

    @property
    def window_size(self) -> int:
        return max(5, min(30, int(self._get("window_size"))))

    @property
    def default_n(self) -> int:
        return max(5, min(120, int(self._get("default_n"))))

    @property
    def default_t(self) -> int:
        return max(5, min(180, int(self._get("default_t"))))

    @property
    def max_token(self) -> int:
        return max(1000, min(8000, int(self._get("max_token"))))

    @property
    def follow_up_ttl(self) -> int:
        return max(5, min(120, int(self._get("follow_up_ttl"))))

    @property
    def follow_up_aggregate_window(self) -> int:
        return max(3, min(30, int(self._get("follow_up_aggregate_window"))))

    @property
    def quality_threshold(self) -> float:
        return max(0.0, min(1.0, float(self._get("quality_threshold"))))

    @property
    def provider_id(self) -> str:
        return str(self._get("provider_id", ""))

    @property
    def trigger_min_interval(self) -> int:
        return max(10, min(120, int(self._get("trigger_min_interval"))))

    @property
    def boost_factor(self) -> float:
        return max(0.3, min(0.95, float(self._get("boost_factor"))))

    @property
    def boost_duration(self) -> int:
        return max(1, min(60, int(self._get("boost_duration"))))

    @property
    def max_boosted_replies(self) -> int:
        return max(2, min(10, int(self._get("max_boosted_replies"))))

    @property
    def proactive_enabled(self) -> bool:
        v = self._get("proactive_enabled")
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "1", "yes", "on")

    @property
    def proactive_check_interval(self) -> int:
        return max(1, min(30, int(self._get("proactive_check_interval"))))

    @property
    def proactive_quiet_minutes(self) -> int:
        return max(30, min(720, int(self._get("proactive_quiet_minutes"))))

    @property
    def proactive_max_per_day(self) -> int:
        return max(1, min(10, int(self._get("proactive_max_per_day"))))

    @property
    def proactive_min_interval(self) -> int:
        return max(60, min(1440, int(self._get("proactive_min_interval"))))

    @property
    def proactive_drift_delay(self) -> int:
        return max(5, min(120, int(self._get("proactive_drift_delay"))))

    @property
    def proactive_pending_timeout(self) -> int:
        return max(5, min(120, int(self._get("proactive_pending_timeout"))))

    @property
    def proactive_max_streak(self) -> int:
        return max(1, min(5, int(self._get("proactive_max_streak"))))

    @property
    def proactive_instruction(self) -> str:
        return str(self._get("proactive_instruction", ""))

    @property
    def proactive_max_message_len(self) -> int:
        return max(50, min(1000, int(self._get("proactive_max_message_len"))))
