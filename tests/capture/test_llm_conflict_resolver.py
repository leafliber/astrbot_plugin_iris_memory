"""
LLM 冲突解决器 - 微妙冲突检测单元测试

测试增强后的 _might_have_subtle_conflict 多维度检测能力。
"""

import pytest

from iris_memory.capture.conflict.llm_conflict_resolver import (
    LLMConflictResolver,
    _SUBTLE_CONFLICT_INDICATORS,
)
from iris_memory.core.detection.llm_enhanced_base import DetectionMode


@pytest.fixture
def resolver():
    """创建仅用于规则检测的 LLMConflictResolver"""
    return LLMConflictResolver(mode=DetectionMode.RULE)


# ==============================================================
# _might_have_subtle_conflict 词对指示器测试
# ==============================================================

class TestSubtleConflictWordPairs:
    """词对指示器检测"""

    def test_preference_contradiction(self, resolver):
        """偏好矛盾：喜欢 vs 讨厌"""
        assert resolver._might_have_subtle_conflict(
            "我喜欢喝咖啡", "我讨厌喝咖啡"
        ) is True

    def test_emotion_contradiction(self, resolver):
        """情感矛盾：开心 vs 难过"""
        assert resolver._might_have_subtle_conflict(
            "今天很开心", "今天很难过"
        ) is True

    def test_state_contradiction(self, resolver):
        """状态矛盾：有 vs 没有"""
        assert resolver._might_have_subtle_conflict(
            "我有养猫", "我没有宠物"
        ) is True

    def test_temporal_signal(self, resolver):
        """时间变化信号：以前 vs 现在"""
        assert resolver._might_have_subtle_conflict(
            "以前我很胖", "现在我很瘦"
        ) is True

    def test_frequency_contradiction(self, resolver):
        """频率矛盾：经常 vs 很少"""
        assert resolver._might_have_subtle_conflict(
            "我经常运动", "我很少锻炼"
        ) is True

    def test_english_contradiction(self, resolver):
        """英文矛盾检测"""
        assert resolver._might_have_subtle_conflict(
            "I love coffee", "I hate tea"
        ) is True

    def test_no_contradiction(self, resolver):
        """无矛盾的文本"""
        assert resolver._might_have_subtle_conflict(
            "天气真好", "阳光明媚"
        ) is False

    def test_bidirectional_detection(self, resolver):
        """双向检测：word1 在 text1 或 text2 都可以"""
        assert resolver._might_have_subtle_conflict(
            "不喜欢", "喜欢这个"
        ) is True
        assert resolver._might_have_subtle_conflict(
            "喜欢这个", "不喜欢"
        ) is True


# ==============================================================
# 数值差异检测测试
# ==============================================================

class TestNumericConflict:
    """数值差异检测"""

    def test_different_numbers_similar_context(self, resolver):
        """相同描述框架但数值不同"""
        assert resolver._might_have_subtle_conflict(
            "我有3个苹果", "我有5个苹果"
        ) is True

    def test_same_numbers_no_conflict(self, resolver):
        """相同数值不冲突"""
        assert resolver._might_have_subtle_conflict(
            "我有3个苹果", "我有3个苹果"
        ) is False

    def test_different_numbers_different_context(self, resolver):
        """不同描述不同数值不冲突"""
        assert resolver._might_have_subtle_conflict(
            "那个房间有23平米", "周五有5节课"
        ) is False

    def test_numbers_with_overlapping_context(self, resolver):
        """有字符重叠的数值差异"""
        assert resolver._might_have_subtle_conflict(
            "我今年28岁", "我今年32岁"
        ) is True


# ==============================================================
# 时间矛盾检测测试
# ==============================================================

class TestTimeConflict:
    """时间矛盾检测"""

    def test_different_time_markers(self, resolver):
        """不同时间标记"""
        assert resolver._might_have_subtle_conflict(
            "昨天去了公司", "今天去了公司"
        ) is True

    def test_same_time_no_conflict(self, resolver):
        """相同时间标记不冲突"""
        assert resolver._might_have_subtle_conflict(
            "今天很开心", "今天天气好"
        ) is False

    def test_no_time_markers(self, resolver):
        """无时间标记"""
        assert resolver._might_have_subtle_conflict(
            "我喜欢跑步", "跑步很健康"
        ) is False


# ==============================================================
# 综合测试
# ==============================================================

class TestComprehensive:
    """综合场景测试"""

    def test_multiple_signal_types(self, resolver):
        """多种信号类型同时存在"""
        # 同时有偏好矛盾和时间变化
        assert resolver._might_have_subtle_conflict(
            "以前喜欢喝咖啡", "现在不喝咖啡了"
        ) is True

    def test_empty_texts(self, resolver):
        """空文本不触发"""
        assert resolver._might_have_subtle_conflict("", "") is False

    def test_very_short_texts(self, resolver):
        """极短文本"""
        assert resolver._might_have_subtle_conflict("好", "坏") is False

    def test_indicators_expanded(self):
        """确认指示器已扩展"""
        # 扩展后应该有多种类别
        assert len(_SUBTLE_CONFLICT_INDICATORS) > 10
        # 检查各类别都在
        has_emotion = any("开心" in pair for pair in _SUBTLE_CONFLICT_INDICATORS)
        has_temporal = any("以前" in pair for pair in _SUBTLE_CONFLICT_INDICATORS)
        has_frequency = any("经常" in pair for pair in _SUBTLE_CONFLICT_INDICATORS)
        has_english = any("love" in pair for pair in _SUBTLE_CONFLICT_INDICATORS)
        assert all([has_emotion, has_temporal, has_frequency, has_english])
