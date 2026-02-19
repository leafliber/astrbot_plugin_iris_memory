"""
触发器检测器单元测试
测试TriggerDetector的所有功能
"""

import pytest

from iris_memory.capture.detector.trigger_detector import TriggerDetector
from iris_memory.core.types import TriggerType


class TestTriggerDetector:
    """TriggerDetector单元测试"""

    @pytest.fixture
    def detector(self):
        """创建TriggerDetector实例"""
        return TriggerDetector()

    # ========== 初始化测试 ==========

    def test_detector_initialization(self, detector):
        """测试触发器检测器初始化"""
        assert detector is not None
        assert TriggerType.EXPLICIT in detector.triggers
        assert TriggerType.PREFERENCE in detector.triggers
        assert TriggerType.EMOTION in detector.triggers
        assert TriggerType.RELATIONSHIP in detector.triggers
        assert TriggerType.FACT in detector.triggers
        assert TriggerType.BOUNDARY in detector.triggers
        assert len(detector.negative_patterns) > 0

    # ========== 显式触发器测试 ==========

    def test_detect_explicit_trigger_chinese(self, detector):
        """测试检测中文显式触发器"""
        text = "记住，明天下午3点开会"
        triggers = detector.detect_triggers(text)

        assert len(triggers) > 0
        assert any(t.type == TriggerType.EXPLICIT for t in triggers)
        assert any("记住" in t.pattern for t in triggers)

    def test_detect_explicit_trigger_important(self, detector):
        """测试检测'重要'触发器"""
        text = "这个信息很重要，要记住"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.EXPLICIT for t in triggers)
        assert any("重要" in t.pattern for t in triggers)

    def test_detect_explicit_trigger_english(self, detector):
        """测试检测英文显式触发器"""
        text = "Remember this important date"
        triggers = detector.detect_triggers(text)

        assert len(triggers) > 0
        assert any(t.type == TriggerType.EXPLICIT for t in triggers)

    # ========== 偏好触发器测试 ==========

    def test_detect_preference_like(self, detector):
        """测试检测'喜欢'触发器"""
        text = "我喜欢吃苹果"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.PREFERENCE for t in triggers)
        assert any("喜欢" in t.pattern for t in triggers)

    def test_detect_preference_hate(self, detector):
        """测试检测'讨厌'触发器"""
        text = "我讨厌下雨天"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.PREFERENCE for t in triggers)

    def test_detect_preference_english(self, detector):
        """测试检测英文偏好触发器"""
        text = "I love reading books"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.PREFERENCE for t in triggers)

    # ========== 情感触发器测试 ==========

    def test_detect_emotion_feel(self, detector):
        """测试检测'觉得'触发器"""
        text = "我觉得今天心情不错"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.EMOTION for t in triggers)
        assert any("觉得" in t.pattern for t in triggers)

    def test_detect_emotion_mood(self, detector):
        """测试检测'心情'触发器"""
        text = "我现在心情很好"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.EMOTION for t in triggers)

    def test_detect_emotion_english(self, detector):
        """测试检测英文情感触发器"""
        text = "I feel very happy today"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.EMOTION for t in triggers)

    # ========== 关系触发器测试 ==========

    def test_detect_relationship_friend(self, detector):
        """测试检测'我们是朋友'触发器"""
        text = "我们是很好的朋友"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.RELATIONSHIP for t in triggers)

    def test_detect_relationship_english(self, detector):
        """测试检测英文关系触发器"""
        text = "You're like a brother to me"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.RELATIONSHIP for t in triggers)

    # ========== 事实触发器测试 ==========

    def test_detect_fact_i_am(self, detector):
        """测试检测'我是'触发器"""
        text = "我是软件工程师"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.FACT for t in triggers)
        assert any("我是" in t.pattern for t in triggers)

    def test_detect_fact_i_have(self, detector):
        """测试检测'我有'触发器"""
        text = "我有两只猫"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.FACT for t in triggers)

    def test_detect_fact_english(self, detector):
        """测试检测英文事实触发器"""
        text = "I work as a teacher"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.FACT for t in triggers)

    # ========== 边界触发器测试 ==========

    def test_detect_boundary_dont(self, detector):
        """测试检测'不要'触发器"""
        text = "不要问我的年龄"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.BOUNDARY for t in triggers)
        assert any("不要" in t.pattern for t in triggers)

    def test_detect_boundary_private(self, detector):
        """测试检测'隐私'触发器"""
        text = "这是我的隐私，别问"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.BOUNDARY for t in triggers)

    def test_detect_boundary_english(self, detector):
        """测试检测英文边界触发器"""
        text = "This is private, don't ask"
        triggers = detector.detect_triggers(text)

        assert any(t.type == TriggerType.BOUNDARY for t in triggers)

    # ========== 负样本测试 ==========

    def test_negative_sample_weather(self, detector):
        """测试负样本：天气"""
        text = "天气怎么样？"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    def test_negative_sample_hello(self, detector):
        """测试负样本：你好"""
        text = "你好"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    def test_negative_sample_short_confirmation(self, detector):
        """测试负样本：短确认"""
        text = "嗯"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    def test_negative_sample_too_short(self, detector):
        """测试负样本：太短"""
        text = "好的"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    def test_negative_sample_laugh(self, detector):
        """测试负样本：笑声"""
        text = "哈哈"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    def test_negative_sample_thanks(self, detector):
        """测试负样本：感谢"""
        text = "谢谢"
        triggers = detector.detect_triggers(text)

        assert len(triggers) == 0

    # ========== 多触发器测试 ==========

    def test_multiple_triggers(self, detector):
        """测试多个触发器"""
        text = "记住，我喜欢吃苹果"
        triggers = detector.detect_triggers(text)

        # 应该检测到EXPLICIT和PREFERENCE两个触发器
        assert len(triggers) >= 2
        trigger_types = [t.type for t in triggers]
        assert TriggerType.EXPLICIT in trigger_types
        assert TriggerType.PREFERENCE in trigger_types

    def test_multiple_same_type_triggers(self, detector):
        """测试同一类型的多个触发器"""
        text = "我喜欢苹果，也喜欢橙子"
        triggers = detector.detect_triggers(text)

        # 应该检测到多个PREFERENCE触发器
        preference_triggers = [t for t in triggers if t.type == TriggerType.PREFERENCE]
        assert len(preference_triggers) >= 1

    # ========== 置信度测试 ==========

    def test_explicit_confidence(self, detector):
        """测试显式触发器置信度"""
        text = "记住这个信息"
        triggers = detector.detect_triggers(text)

        explicit_triggers = [t for t in triggers if t.type == TriggerType.EXPLICIT]
        assert len(explicit_triggers) > 0
        # 显式触发器置信度应该是0.95
        assert abs(explicit_triggers[0].confidence - 0.95) < 0.01

    def test_boundary_confidence(self, detector):
        """测试边界触发器置信度"""
        text = "不要问这个问题"
        triggers = detector.detect_triggers(text)

        boundary_triggers = [t for t in triggers if t.type == TriggerType.BOUNDARY]
        assert len(boundary_triggers) > 0
        # 边界触发器置信度应该是0.9
        assert abs(boundary_triggers[0].confidence - 0.9) < 0.01

    def test_fact_confidence(self, detector):
        """测试事实触发器置信度"""
        text = "我是程序员"
        triggers = detector.detect_triggers(text)

        fact_triggers = [t for t in triggers if t.type == TriggerType.FACT]
        assert len(fact_triggers) > 0
        # 事实触发器置信度应该是0.8
        assert abs(fact_triggers[0].confidence - 0.8) < 0.01

    def test_emotion_confidence(self, detector):
        """测试情感触发器置信度"""
        text = "我感到很开心"
        triggers = detector.detect_triggers(text)

        emotion_triggers = [t for t in triggers if t.type == TriggerType.EMOTION]
        assert len(emotion_triggers) > 0
        # 情感触发器置信度应该是0.7
        assert abs(emotion_triggers[0].confidence - 0.7) < 0.01

    # ========== 位置信息测试 ==========

    def test_trigger_position(self, detector):
        """测试触发器位置"""
        text = "记住，我喜欢吃苹果"
        triggers = detector.detect_triggers(text)

        # 找到"记住"触发器的位置
        remember_triggers = [t for t in triggers if "记住" in t.pattern]
        if remember_triggers:
            # 应该在文本开头
            assert remember_triggers[0].position >= 0
            assert remember_triggers[0].position < len(text)

    # ========== 辅助方法测试 ==========

    def test_has_trigger_true(self, detector):
        """测试has_trigger方法 - 有触发器"""
        text = "我喜欢吃苹果"
        assert detector.has_trigger(text) is True

    def test_has_trigger_false(self, detector):
        """测试has_trigger方法 - 无触发器"""
        text = "天气怎么样？"
        assert detector.has_trigger(text) is False

    def test_has_trigger_empty(self, detector):
        """测试has_trigger方法 - 空文本"""
        text = ""
        assert detector.has_trigger(text) is False

    def test_get_trigger_types(self, detector):
        """测试获取触发器类型"""
        text = "记住，我喜欢吃苹果"
        trigger_types = detector.get_trigger_types(text)

        assert TriggerType.EXPLICIT in trigger_types
        assert TriggerType.PREFERENCE in trigger_types

    def test_get_trigger_types_empty(self, detector):
        """测试获取触发器类型 - 无触发器"""
        text = "天气怎么样？"
        trigger_types = detector.get_trigger_types(text)

        assert len(trigger_types) == 0

    def test_get_highest_confidence_trigger(self, detector):
        """测试获取最高置信度触发器"""
        text = "记住，我喜欢吃苹果"
        highest = detector.get_highest_confidence_trigger(text)

        assert highest is not None
        # 显式触发器置信度最高
        assert highest.type == TriggerType.EXPLICIT

    def test_get_highest_confidence_trigger_none(self, detector):
        """测试获取最高置信度触发器 - 无触发器"""
        text = "天气怎么样？"
        highest = detector.get_highest_confidence_trigger(text)

        assert highest is None

    # ========== 边界情况测试 ==========

    def test_empty_text(self, detector):
        """测试空文本"""
        triggers = detector.detect_triggers("")
        assert triggers == []

    def test_whitespace_only(self, detector):
        """测试只有空白字符"""
        text = "   "
        triggers = detector.detect_triggers(text)

        assert triggers == []

    def test_case_insensitive(self, detector):
        """测试大小写不敏感"""
        text1 = "Remember this"
        text2 = "remember this"

        triggers1 = detector.detect_triggers(text1)
        triggers2 = detector.detect_triggers(text2)

        assert len(triggers1) == len(triggers2)

    def test_unicode_text(self, detector):
        """测试Unicode文本"""
        text = "我喜欢🍎和🍊"
        triggers = detector.detect_triggers(text)

        assert len(triggers) > 0

    def test_very_long_text(self, detector):
        """测试超长文本"""
        text = "我喜欢" + "苹果" * 1000
        triggers = detector.detect_triggers(text)

        assert len(triggers) > 0

    # ========== 特殊字符测试 ==========

    def test_text_with_punctuation(self, detector):
        """测试带标点符号的文本"""
        text = "记住！我喜欢苹果、橙子。"
        triggers = detector.detect_triggers(text)

        assert len(triggers) > 0

    def test_text_with_numbers(self, detector):
        """测试带数字的文本"""
        text = "我出生于1990年"
        triggers = detector.detect_triggers(text)

        # 应该检测到FACT触发器
        assert any(t.type == TriggerType.FACT for t in triggers)

    def test_text_with_special_chars(self, detector):
        """测试带特殊字符的文本"""
        text = "记住！@#$%^&*()我喜欢"
        triggers = detector.detect_triggers(text)

        # 应该仍然能检测到触发器
        assert len(triggers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
