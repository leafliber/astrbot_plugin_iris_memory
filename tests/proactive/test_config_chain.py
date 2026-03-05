#!/usr/bin/env python3
"""
配置调用链验证脚本

验证从 defaults.py → ProactiveConfig → 各组件的配置访问是否完整可用。
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_defaults_structure():
    """测试 defaults.py 中的配置结构"""
    print("=" * 60)
    print("测试 1: defaults.py 配置结构")
    print("=" * 60)
    
    from iris_memory.core.defaults import DEFAULTS
    pr = DEFAULTS.proactive_reply
    
    # 检查关键配置项
    required_fields = [
        'followup_after_all_replies',
        'followup_window_seconds',
        'max_followup_count',
        'followup_short_window_seconds',
        'signal_weight_direct_reply',
        'signal_ttl_emotion_high',
        'cooldown_seconds',
        'max_daily_replies',
        'quiet_hours',
    ]
    
    missing = []
    for field in required_fields:
        if not hasattr(pr, field):
            missing.append(field)
        else:
            value = getattr(pr, field)
            print(f"  ✓ {field}: {value}")
    
    if missing:
        print(f"  ✗ 缺少字段：{missing}")
        return False
    
    print("  ✓ 所有必需字段存在\n")
    return True


def test_proactive_config():
    """测试 ProactiveConfig 的创建和属性"""
    print("=" * 60)
    print("测试 2: ProactiveConfig 创建")
    print("=" * 60)
    
    from iris_memory.core.defaults import DEFAULTS
    from iris_memory.proactive.config import ProactiveConfig
    
    try:
        config = ProactiveConfig(DEFAULTS.proactive_reply)
        print(f"  ✓ ProactiveConfig 创建成功")
        print(f"    - enabled: {config.enabled}")
        print(f"    - followup_enabled: {config.followup_enabled}")
        print(f"    - followup_after_all_replies: {config.followup_after_all_replies}")
        
        # 检查扁平化配置
        flat_fields = [
            'followup_window_seconds',
            'max_followup_count',
            'followup_short_window_seconds',
            'signal_weight_direct_reply',
            'signal_ttl_emotion_high',
            'cooldown_seconds',
        ]
        
        for field in flat_fields:
            if hasattr(config, field):
                print(f"  ✓ {field}: {getattr(config, field)}")
            else:
                print(f"  ✗ 缺少扁平化字段：{field}")
                return False
        
        print("  ✓ 所有扁平化配置正确\n")
        return True
        
    except Exception as e:
        print(f"  ✗ 创建失败：{e}")
        return False


def test_config_access():
    """测试各组件的配置访问"""
    print("=" * 60)
    print("测试 3: 组件配置访问")
    print("=" * 60)
    
    from iris_memory.core.defaults import DEFAULTS
    from iris_memory.proactive.config import ProactiveConfig
    
    config = ProactiveConfig(DEFAULTS.proactive_reply)
    
    # 模拟各组件的配置访问
    tests = [
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
    
    all_passed = True
    for name, value in tests:
        if value is not None:
            print(f"  ✓ {name}: {value}")
        else:
            print(f"  ✗ {name}: None")
            all_passed = False
    
    if all_passed:
        print("  ✓ 所有组件配置访问正确\n")
    
    return all_passed


def test_config_dict():
    """测试配置导出为字典"""
    print("=" * 60)
    print("测试 4: 配置导出为字典")
    print("=" * 60)
    
    from iris_memory.core.defaults import DEFAULTS
    from iris_memory.proactive.config import ProactiveConfig
    
    config = ProactiveConfig(DEFAULTS.proactive_reply)
    config_dict = config.to_dict()
    
    print(f"  ✓ 导出配置项数量：{len(config_dict)}")
    print(f"  ✓ 配置键列表：{sorted(config_dict.keys())[:10]}...")
    
    # 检查关键配置是否在字典中
    required_keys = [
        'followup_short_window_seconds',
        'signal_weight_direct_reply',
        'max_followup_count',
    ]
    
    for key in required_keys:
        if key in config_dict:
            print(f"  ✓ {key}: {config_dict[key]}")
        else:
            print(f"  ✗ 缺少键：{key}")
            return False
    
    print("  ✓ 配置导出正确\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("配置调用链完整性验证")
    print("=" * 60 + "\n")
    
    tests = [
        test_defaults_structure,
        test_proactive_config,
        test_config_access,
        test_config_dict,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ 测试异常：{e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"测试结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ 配置调用链完整可用！\n")
        return 0
    else:
        print(f"\n✗ 配置调用链存在问题，请检查上述错误\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
