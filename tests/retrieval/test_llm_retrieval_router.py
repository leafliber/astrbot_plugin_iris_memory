"""
LLM 检索路由器 - 复杂度检测增强单元测试

测试增强后的 _might_be_complex 多层复杂度检测、
扩展的指示词列表和高价值查询模式。
"""

import pytest

from iris_memory.retrieval.llm_retrieval_router import (
    LLMRetrievalRouter,
    _COMPLEX_INDICATORS,
    _HIGH_VALUE_QUERY_PATTERNS,
)
from iris_memory.core.detection.llm_enhanced_base import DetectionMode


@pytest.fixture
def router():
    """创建仅用于规则检测的 LLMRetrievalRouter"""
    return LLMRetrievalRouter(mode=DetectionMode.RULE)


# ==============================================================
# _might_be_complex 增强测试
# ==============================================================

class TestMightBeComplex:
    """复杂度检测测试"""

    def test_single_indicator_not_complex(self, router):
        """单个指示词且短文本不认为复杂"""
        assert router._might_be_complex("和") is False

    def test_two_indicators_complex(self, router):
        """两个指示词判定为复杂"""
        assert router._might_be_complex("为什么之前不一样") is True

    def test_single_indicator_long_text(self, router):
        """单个指示词 + 较长文本认为复杂"""
        assert router._might_be_complex("我想知道关于机器学习的相关内容和应用场景") is True

    def test_no_indicators(self, router):
        """无指示词"""
        assert router._might_be_complex("你好") is False

    def test_high_value_pattern_what_and(self, router):
        """高价值模式：什么和什么"""
        assert router._might_be_complex("什么方法和技术最有效") is True

    def test_high_value_pattern_why_not(self, router):
        """高价值模式：为什么不"""
        assert router._might_be_complex("为什么不能这样做") is True

    def test_high_value_pattern_how_change(self, router):
        """高价值模式：怎么变化"""
        assert router._might_be_complex("最近怎么变化了") is True

    def test_high_value_pattern_who_between(self, router):
        """高价值模式：谁之间"""
        assert router._might_be_complex("谁和小明之间有什么关系") is True

    def test_english_high_value(self, router):
        """英文高价值模式"""
        assert router._might_be_complex("how did things change over time") is True

    def test_multiple_quoted_entities(self, router):
        """多个引号实体"""
        assert router._might_be_complex("「小明」和「小红」") is True

    def test_causation_indicators(self, router):
        """因果关系指示词"""
        assert router._might_be_complex("因为工作压力导致失眠") is True

    def test_comparison_indicators(self, router):
        """对比指示词"""
        assert router._might_be_complex("相比以前现在如何") is True


# ==============================================================
# 指示词列表完整性测试
# ==============================================================

class TestIndicatorsExpanded:
    """指示词列表扩展验证"""

    def test_has_connector_words(self):
        """包含连接词"""
        assert "以及" in _COMPLEX_INDICATORS
        assert "还有" in _COMPLEX_INDICATORS

    def test_has_causation_words(self):
        """包含因果词"""
        assert "因为" in _COMPLEX_INDICATORS
        assert "所以" in _COMPLEX_INDICATORS
        assert "导致" in _COMPLEX_INDICATORS

    def test_has_condition_words(self):
        """包含条件/对比词"""
        assert "如果" in _COMPLEX_INDICATORS
        assert "相比" in _COMPLEX_INDICATORS
        assert "区别" in _COMPLEX_INDICATORS

    def test_has_multi_hop_words(self):
        """包含多跳推理词"""
        assert "关于" in _COMPLEX_INDICATORS
        assert "涉及" in _COMPLEX_INDICATORS

    def test_has_english_words(self):
        """包含英文指示词"""
        assert "why" in _COMPLEX_INDICATORS
        assert "because" in _COMPLEX_INDICATORS

    def test_indicators_count_expanded(self):
        """指示词数量显著扩展"""
        assert len(_COMPLEX_INDICATORS) >= 20

    def test_high_value_patterns_exist(self):
        """高价值模式列表存在且非空"""
        assert len(_HIGH_VALUE_QUERY_PATTERNS) >= 5
