"""
主动回复 v3 配置

分为两层：
- 用户可见配置：通过 AstrBot 管理界面修改（_conf_schema.json）
- 高级配置：仅在 defaults.py 中设置，用户无需修改
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalQueueConfig:
    """SignalQueue 高级配置（用户无需修改）"""

    # 定时器
    check_interval_seconds: int = 30        # 群定时器检查间隔
    silence_timeout_seconds: int = 600      # 沉默超时（定时器销毁）
    min_silence_seconds: int = 60           # 最小沉默时间才触发判断

    # 信号 TTL（秒）
    ttl_emotion_high: int = 180             # 3 分钟
    ttl_rule_match: int = 300               # 5 分钟

    # 权重阈值
    weight_direct_reply: float = 0.8        # 直接回复阈值
    weight_llm_confirm: float = 0.5         # LLM 确认阈值

    # 信号队列容量
    max_signals_per_group: int = 50         # 每个群最大信号数


@dataclass
class FollowUpConfig:
    """FollowUpPlanner 高级配置（用户无需修改）"""

    # 时间窗口
    short_window_seconds: int = 10          # 短期窗口

    # LLM 判断
    llm_max_tokens: int = 500
    llm_temperature: float = 0.3

    # 降级策略
    fallback_to_rule_on_llm_error: bool = True


@dataclass
class ProactiveConfig:
    """主动回复 v3 完整配置

    聚合用户可见配置和高级配置。
    由 ProactiveModule 从 AstrBot 配置 + defaults 组合构建。
    """

    # 总开关
    enabled: bool = False

    # SignalQueue 开关
    signal_queue_enabled: bool = True

    # FollowUp 开关（完全独立）
    followup_enabled: bool = True

    # FollowUp 在所有 Bot 回复后创建期待（不仅限于主动回复）
    followup_after_all_replies: bool = False

    # 用户可见配置
    followup_window_seconds: int = 120      # FollowUp 窗口时长
    max_followup_count: int = 2             # 最大跟进次数
    max_reply_tokens: int = 150             # 最大回复 token 数
    reply_temperature: float = 0.7          # 回复温度
    cooldown_seconds: int = 60              # 回复冷却时间
    max_daily_replies: int = 20             # 每日最大回复数
    max_daily_per_user: int = 5             # 每用户每日最大回复数

    # 白名单
    group_whitelist_mode: bool = False
    group_whitelist: list = field(default_factory=list)

    # 静音时段
    quiet_hours: list = field(default_factory=lambda: [23, 7])

    # 主动回复模式（rule / hybrid）
    proactive_mode: str = "rule"

    # 高级配置
    signal_queue: SignalQueueConfig = field(default_factory=SignalQueueConfig)
    followup: FollowUpConfig = field(default_factory=FollowUpConfig)
