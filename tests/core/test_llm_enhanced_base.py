"""
LLMEnhancedBase - 统一 Hybrid 框架增强测试

测试新增的冷却追踪器、扩展统计指标等功能。
"""

import pytest

from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedBase,
    LLMEnhancedDetector,
)
from iris_memory.core.detection.base_result import BaseDetectionResult


# ==============================================================
# 冷却追踪器测试
# ==============================================================

class TestCooldownSupport:
    """LLMEnhancedBase 冷却支持"""

    def test_no_cooldown_by_default(self):
        """默认不启用冷却"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.RULE)
        assert detector._cooldown_tracker is None

    def test_cooldown_enabled(self):
        """启用冷却追踪器"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.HYBRID, cooldown_seconds=60)
        assert detector._cooldown_tracker is not None
        assert detector._cooldown_tracker.cooldown_seconds == 60

    def test_check_cooldown_no_tracker(self):
        """无追踪器时始终返回 True"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.RULE)
        assert detector._check_cooldown("test_key") is True

    def test_check_cooldown_with_tracker(self):
        """有追踪器时正确检查冷却"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.HYBRID, cooldown_seconds=60)
        assert detector._check_cooldown("key1") is True
        detector._record_cooldown("key1")
        assert detector._check_cooldown("key1") is False  # 刚记录，还在冷却中
        assert detector._check_cooldown("key2") is True   # 不同 key 不受影响

    def test_record_cooldown_no_tracker(self):
        """无追踪器时 record 不报错"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.RULE)
        detector._record_cooldown("key1")  # 不应抛出异常


# ==============================================================
# 扩展统计指标测试
# ==============================================================

class TestExpandedStats:
    """扩展统计指标"""

    def test_stats_include_hybrid_fields(self):
        """统计包含 hybrid 决策字段"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.HYBRID)
        stats = detector.get_stats()
        assert "rule_only_decisions" in stats
        assert "llm_triggered_decisions" in stats
        assert "llm_skipped_cooldown" in stats
        assert "llm_skipped_limit" in stats
        assert stats["rule_only_decisions"] == 0
        assert stats["llm_triggered_decisions"] == 0

    def test_stats_include_cooldown_flag(self):
        """统计包含冷却标志"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector_no_cd = DummyDetector(mode=DetectionMode.RULE)
        assert detector_no_cd.get_stats()["has_cooldown"] is False

        detector_cd = DummyDetector(mode=DetectionMode.HYBRID, cooldown_seconds=30)
        assert detector_cd.get_stats()["has_cooldown"] is True

    def test_stats_backward_compatible(self):
        """原有统计字段仍存在"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector()
        stats = detector.get_stats()
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "total_tokens" in stats
        assert "daily_limit" in stats
        assert "remaining_calls" in stats
        assert "mode" in stats


# ==============================================================
# 模式切换测试
# ==============================================================

class TestModeManagement:
    """模式管理"""

    def test_mode_string_to_enum(self):
        """字符串模式自动转换为枚举"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode="hybrid")
        assert detector.mode == DetectionMode.HYBRID

    def test_mode_setter(self):
        """模式可修改"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        detector = DummyDetector(mode=DetectionMode.RULE)
        assert detector.mode == DetectionMode.RULE
        detector.mode = DetectionMode.HYBRID
        assert detector.mode == DetectionMode.HYBRID
        detector.mode = "llm"
        assert detector.mode == DetectionMode.LLM

    def test_is_llm_enabled(self):
        """LLM 启用状态"""
        class DummyDetector(LLMEnhancedBase):
            async def detect(self, *args, **kwargs):
                pass
            def _rule_detect(self, *args, **kwargs):
                pass

        assert DummyDetector(mode=DetectionMode.RULE).is_llm_enabled is False
        assert DummyDetector(mode=DetectionMode.LLM).is_llm_enabled is True
        assert DummyDetector(mode=DetectionMode.HYBRID).is_llm_enabled is True
