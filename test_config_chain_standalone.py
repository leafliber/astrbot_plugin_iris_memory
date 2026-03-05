#!/usr/bin/env python3
"""
配置调用链验证脚本 - 独立版本

直接验证 defaults.py 中的配置结构是否正确。
"""

from dataclasses import dataclass, field


@dataclass
class ProactiveReplyDefaults:
    """主动回复默认配置 - 复制自 defaults.py"""
    # ===== 用户可见配置 =====
    followup_after_all_replies: bool = False
    cooldown_seconds: int = 60
    max_daily_replies: int = 20
    max_reply_tokens: int = 150
    reply_temperature: float = 0.7
    group_whitelist: list = field(default_factory=list)
    group_whitelist_mode: bool = False
    quiet_hours: list = field(default_factory=lambda: [23, 7])
    max_daily_per_user: int = 5
    web_dashboard: bool = False

    # ===== v3 SignalQueue 高级配置 =====
    signal_check_interval_seconds: int = 30
    signal_silence_timeout_seconds: int = 600
    signal_min_silence_seconds: int = 60
    signal_ttl_emotion_high: int = 180
    signal_ttl_rule_match: int = 300
    signal_weight_direct_reply: float = 0.8
    signal_weight_llm_confirm: float = 0.5
    signal_max_signals_per_group: int = 50

    # ===== v3 FollowUp 高级配置 =====
    followup_window_seconds: int = 150
    max_followup_count: int = 3
    followup_short_window_seconds: int = 10
    followup_llm_max_tokens: int = 1000
    followup_llm_temperature: float = 0.3
    followup_fallback_to_rule: bool = True
    
    # ===== 其他配置 =====
    proactive_mode: str = "rule"
    quiet_hours_activity_exempt_minutes: int = 20
    timezone_offset: int = 8


class ProactiveConfig:
    """主动回复配置 - 复制自 proactive/config.py"""
    
    def __init__(self, defaults: ProactiveReplyDefaults) -> None:
        # 总开关
        self.enabled: bool = getattr(defaults, 'enable', False)
        self.signal_queue_enabled: bool = True
        self.followup_enabled: bool = True
        
        # 核心功能开关
        self.followup_after_all_replies: bool = defaults.followup_after_all_replies
        self.group_whitelist_mode: bool = defaults.group_whitelist_mode
        self.proactive_mode: str = getattr(defaults, 'proactive_mode', 'rule')
        
        # 时间窗口配置
        self.followup_window_seconds: int = defaults.followup_window_seconds
        self.cooldown_seconds: int = defaults.cooldown_seconds
        
        # 限制配置
        self.max_followup_count: int = defaults.max_followup_count
        self.max_daily_replies: int = defaults.max_daily_replies
        self.max_daily_per_user: int = defaults.max_daily_per_user
        self.max_reply_tokens: int = defaults.max_reply_tokens
        
        # 回复参数
        self.reply_temperature: float = defaults.reply_temperature
        
        # 白名单
        self.group_whitelist: list = list(defaults.group_whitelist)
        
        # 静音时段
        self.quiet_hours: list = list(defaults.quiet_hours)
        self.quiet_hours_activity_exempt_minutes: int = getattr(
            defaults, 'quiet_hours_activity_exempt_minutes', 20
        )
        self.timezone_offset: int = getattr(defaults, 'timezone_offset', 8)
        
        # SignalQueue 高级配置 (直接扁平化)
        self.signal_check_interval_seconds: int = defaults.signal_check_interval_seconds
        self.signal_silence_timeout_seconds: int = defaults.signal_silence_timeout_seconds
        self.signal_min_silence_seconds: int = defaults.signal_min_silence_seconds
        self.signal_ttl_emotion_high: int = defaults.signal_ttl_emotion_high
        self.signal_ttl_rule_match: int = defaults.signal_ttl_rule_match
        self.signal_weight_direct_reply: float = defaults.signal_weight_direct_reply
        self.signal_weight_llm_confirm: float = defaults.signal_weight_llm_confirm
        self.signal_max_signals_per_group: int = getattr(
            defaults, 'signal_max_signals_per_group', 50
        )
        
        # FollowUp 高级配置 (直接扁平化)
        self.followup_short_window_seconds: int = defaults.followup_short_window_seconds
        self.followup_llm_max_tokens: int = defaults.followup_llm_max_tokens
        self.followup_llm_temperature: float = defaults.followup_llm_temperature
        self.followup_fallback_to_rule_on_llm_error: bool = defaults.followup_fallback_to_rule
    
    def to_dict(self) -> dict:
        """导出配置为字典"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


def test_config_chain():
    """测试完整的配置调用链"""
    print("=" * 70)
    print("配置调用链验证测试")
    print("=" * 70)
    print()
    
    # 1. 创建 defaults
    print("1. 创建 ProactiveReplyDefaults...")
    defaults = ProactiveReplyDefaults()
    print(f"   ✓ followup_window_seconds: {defaults.followup_window_seconds}")
    print(f"   ✓ max_followup_count: {defaults.max_followup_count}")
    print(f"   ✓ followup_short_window_seconds: {defaults.followup_short_window_seconds}")
    print(f"   ✓ signal_weight_direct_reply: {defaults.signal_weight_direct_reply}")
    print()
    
    # 2. 创建 ProactiveConfig
    print("2. 创建 ProactiveConfig...")
    config = ProactiveConfig(defaults)
    print(f"   ✓ enabled: {config.enabled}")
    print(f"   ✓ followup_enabled: {config.followup_enabled}")
    print(f"   ✓ followup_after_all_replies: {config.followup_after_all_replies}")
    print()
    
    # 3. 测试组件配置访问
    print("3. 测试组件配置访问...")
    tests = [
        ("FollowUpPlanner: short_window", config.followup_short_window_seconds),
        ("FollowUpPlanner: fallback", config.followup_fallback_to_rule_on_llm_error),
        ("Manager: cooldown", config.cooldown_seconds),
        ("Manager: max_daily", config.max_daily_replies),
        ("SignalQueue: max_signals", config.signal_max_signals_per_group),
        ("SignalGenerator: ttl_rule", config.signal_ttl_rule_match),
        ("SignalGenerator: ttl_emotion", config.signal_ttl_emotion_high),
        ("GroupScheduler: check_interval", config.signal_check_interval_seconds),
        ("GroupScheduler: silence_timeout", config.signal_silence_timeout_seconds),
        ("GroupScheduler: min_silence", config.signal_min_silence_seconds),
        ("GroupScheduler: weight_direct", config.signal_weight_direct_reply),
        ("GroupScheduler: weight_llm", config.signal_weight_llm_confirm),
    ]
    
    all_passed = True
    for name, value in tests:
        if value is not None:
            print(f"   ✓ {name}: {value}")
        else:
            print(f"   ✗ {name}: None")
            all_passed = False
    
    print()
    
    # 4. 测试配置导出
    print("4. 测试配置导出为字典...")
    config_dict = config.to_dict()
    print(f"   ✓ 导出配置项数量：{len(config_dict)}")
    print(f"   ✓ 包含关键配置：")
    for key in ['followup_short_window_seconds', 'signal_weight_direct_reply', 'max_followup_count']:
        if key in config_dict:
            print(f"      - {key}: {config_dict[key]}")
        else:
            print(f"      ✗ {key}: 缺失")
            all_passed = False
    
    print()
    print("=" * 70)
    if all_passed:
        print("✓ 配置调用链完整可用！")
        print("=" * 70)
        return True
    else:
        print("✗ 配置调用链存在问题")
        print("=" * 70)
        return False


if __name__ == "__main__":
    import sys
    success = test_config_chain()
    sys.exit(0 if success else 1)
