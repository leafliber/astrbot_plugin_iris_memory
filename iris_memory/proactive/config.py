from __future__ import annotations

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

# Keys managed exclusively through the pages UI (stored in KV overrides).
# _conf_schema.json only keeps proactive.enabled / proactive.stats_enabled / proactive.provider_id.
_PAGE_MANAGED_KEYS = {
    k for k in _DEFAULTS if k not in {"enabled", "stats_enabled", "provider_id"}
}

# _conf_schema.json 中 proactive 分组内的三个键（嵌套存储，非页面管理）
_SCHEMA_GROUP = "proactive"
_SCHEMA_KEYS = {"enabled", "stats_enabled", "provider_id"}

# Metadata for the settings page UI.
_CONFIG_META = {
    "mute_period": {
        "label": "静音时段",
        "type": "object",
        "hint": "在静音时段内，Iris 不会主动触发任何回复",
        "items": {
            "start_hour": {"label": "开始小时（0-23）", "type": "int", "min": 0, "max": 23},
            "start_minute": {"label": "开始分钟（0-59）", "type": "int", "min": 0, "max": 59},
            "end_hour": {"label": "结束小时（0-23）", "type": "int", "min": 0, "max": 23},
            "end_minute": {"label": "结束分钟（0-59）", "type": "int", "min": 0, "max": 59},
        },
    },
    "window_size": {
        "label": "滑动记忆窗口大小（条数）",
        "type": "int", "min": 5, "max": 30,
        "hint": "保留最近 N 条有效发言",
    },
    "default_n": {
        "label": "默认消息计数阈值 N",
        "type": "int", "min": 5, "max": 120,
        "hint": "每收到 N 条有效消息触发一次采样",
    },
    "default_t": {
        "label": "默认时间间隔阈值 T（分钟）",
        "type": "int", "min": 5, "max": 180,
        "hint": "距上次采样超过 T 分钟且有新消息时触发",
    },
    "max_token": {
        "label": "上下文 token 上限",
        "type": "int", "min": 1000, "max": 8000,
        "hint": "提交给 LLM 的上下文最大 token 数",
    },
    "follow_up_ttl": {
        "label": "跟进锚点默认 TTL（分钟）",
        "type": "int", "min": 5, "max": 120,
        "hint": "对话锚点中关注用户/关键词的存活时长",
    },
    "follow_up_aggregate_window": {
        "label": "follow-up 消息聚合等待窗口（秒）",
        "type": "int", "min": 3, "max": 30,
        "hint": "锚点命中后等待此时间再触发 LLM 评估",
    },
    "quality_threshold": {
        "label": "消息质量评分阈值",
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
        "hint": "低于此阈值的消息不进入滑动窗口",
    },
    "trigger_min_interval": {
        "label": "触发阶段最小间隔（秒）",
        "type": "int", "min": 10, "max": 120,
        "hint": "同一群两次触发 LLM 调用之间的最小时间间隔",
    },
    "boost_factor": {
        "label": "回复后频率提升系数",
        "type": "float", "min": 0.3, "max": 0.95, "step": 0.05,
        "hint": "Iris 回复后有效阈值临时乘以此系数（<1.0 降低阈值）",
    },
    "boost_duration": {
        "label": "频率提升持续时间（分钟）",
        "type": "int", "min": 1, "max": 60,
        "hint": "回复后 boost 效果的持续时间，之后线性衰减回正常",
    },
    "max_boosted_replies": {
        "label": "最大连续回复 boost 次数",
        "type": "int", "min": 2, "max": 10,
        "hint": "连续回复不超过此次数时享受完整 boost，超出后逐渐减弱",
    },
    "proactive_enabled": {
        "label": "启用主动发起会话",
        "type": "bool",
        "hint": "开启后，Iris 会在群冷场或话题结束时主动开启话题",
    },
    "proactive_check_interval": {
        "label": "发起检查周期（分钟）",
        "type": "int", "min": 1, "max": 30,
        "hint": "每隔此时间扫描一次白名单群，评估是否满足发起条件",
    },
    "proactive_quiet_minutes": {
        "label": "冷场静默阈值（分钟）",
        "type": "int", "min": 30, "max": 720,
        "hint": "群内最后一条消息超过此时间无人说话，才考虑主动发起",
    },
    "proactive_max_per_day": {
        "label": "每日最大发起次数",
        "type": "int", "min": 1, "max": 10,
        "hint": "每个群每天最多主动发起的次数",
    },
    "proactive_min_interval": {
        "label": "两次发起最小间隔（分钟）",
        "type": "int", "min": 60, "max": 1440,
        "hint": "同一群两次主动发起之间的最小时间间隔",
    },
    "proactive_drift_delay": {
        "label": "话题结束后发起延迟（分钟）",
        "type": "int", "min": 5, "max": 120,
        "hint": "检测到话题结束（drifted）后，若持续静默此时间可提前发起新话题",
    },
    "proactive_pending_timeout": {
        "label": "发起接话等待（分钟）",
        "type": "int", "min": 5, "max": 120,
        "hint": "发起后等待群友接话的时间，超时视为无人接话",
    },
    "proactive_max_streak": {
        "label": "当日无人接话上限（次）",
        "type": "int", "min": 1, "max": 5,
        "hint": "连续发起无人接话达到此次数后，当天不再发起",
    },
    "proactive_instruction": {
        "label": "发起话题倾向（可选）",
        "type": "str",
        "hint": "自定义发起话题的偏好说明，如「多聊技术话题」，留空由 Iris 自行发挥",
    },
    "proactive_max_message_len": {
        "label": "发起消息最大长度（字符）",
        "type": "int", "min": 50, "max": 1000,
        "hint": "主动发起消息超过此长度将被截断",
    },
}


class ConfigManager:
    def __init__(self, config: AstrBotConfig) -> None:
        self._cfg = config
        self._overrides: dict[str, Any] = {}

    def load_overrides(self, data: dict | None) -> None:
        if isinstance(data, dict):
            self._overrides = {k: v for k, v in data.items() if k in _PAGE_MANAGED_KEYS}

    def get_overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def set_override(self, key: str, value: Any) -> None:
        if key not in _PAGE_MANAGED_KEYS:
            return
        if key == "mute_period":
            if not isinstance(value, dict):
                return
            dmp = _DEFAULTS["mute_period"]
            self._overrides[key] = {
                "start_hour": int(value.get("start_hour", dmp["start_hour"])),
                "start_minute": int(value.get("start_minute", dmp["start_minute"])),
                "end_hour": int(value.get("end_hour", dmp["end_hour"])),
                "end_minute": int(value.get("end_minute", dmp["end_minute"])),
            }
        else:
            meta = _CONFIG_META.get(key, {})
            v = value
            if meta.get("type") == "int":
                v = int(v)
                lo, hi = meta.get("min"), meta.get("max")
                if lo is not None: v = max(lo, v)
                if hi is not None: v = min(hi, v)
            elif meta.get("type") == "float":
                v = float(v)
                lo, hi = meta.get("min"), meta.get("max")
                if lo is not None: v = max(lo, v)
                if hi is not None: v = min(hi, v)
            elif meta.get("type") == "bool":
                if not isinstance(v, bool):
                    v = str(v).strip().lower() in ("true", "1", "yes", "on")
            elif meta.get("type") == "str":
                v = str(v).strip()
            self._overrides[key] = v

    def get_all_page_config(self) -> dict[str, Any]:
        result = {}
        for key in _PAGE_MANAGED_KEYS:
            result[key] = self._get(key)
        return result

    @staticmethod
    def get_page_config_meta() -> dict[str, Any]:
        return {k: dict(v) for k, v in _CONFIG_META.items()}

    def _get(self, key: str, default=None):
        if key in self._overrides:
            return self._overrides[key]
        if key in _SCHEMA_KEYS:
            # 三个面板配置项存于 _conf_schema.json 的 proactive 分组（嵌套 dict）
            group = self._cfg.get(_SCHEMA_GROUP)
            if isinstance(group, dict):
                val = group.get(key)
                if val is not None:
                    return val
        else:
            val = self._cfg.get(key)
            if val is not None:
                return val
        return _DEFAULTS.get(key, default)

    @property
    def enabled(self) -> bool:
        return bool(self._get("enabled"))

    @property
    def stats_enabled(self) -> bool:
        return bool(self._get("stats_enabled"))

    @property
    def mute_period(self) -> tuple[int, int, int, int]:
        mp = self._get("mute_period", {})
        dmp = _DEFAULTS["mute_period"]
        return (
            int(mp.get("start_hour", dmp["start_hour"])),
            int(mp.get("start_minute", dmp["start_minute"])),
            int(mp.get("end_hour", dmp["end_hour"])),
            int(mp.get("end_minute", dmp["end_minute"])),
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
