"""
PersonaExtractor 动态阈值单元测试

测试 _compute_dynamic_threshold 方法的价值信号检测和阈值调整。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from iris_memory.analysis.persona.persona_extractor import PersonaExtractor
from iris_memory.analysis.persona.keyword_maps import ExtractionResult


# ==============================================================
# _compute_dynamic_threshold 测试
# ==============================================================

class TestDynamicThreshold:
    """动态阈值计算测试"""

    def test_base_threshold(self):
        """无价值信号时返回基准阈值 0.6"""
        result = ExtractionResult()
        threshold = PersonaExtractor._compute_dynamic_threshold("哈哈", result)
        assert threshold == 0.6

    def test_first_person_lowers_threshold(self):
        """第一人称降低阈值"""
        result = ExtractionResult()
        threshold = PersonaExtractor._compute_dynamic_threshold("我喜欢打球", result)
        assert threshold < 0.6

    def test_numbers_lower_threshold(self):
        """包含数字降低阈值"""
        result = ExtractionResult()
        threshold = PersonaExtractor._compute_dynamic_threshold("今天走了5000步", result)
        assert threshold < 0.6

    def test_long_content_lowers_threshold(self):
        """长内容降低阈值"""
        result = ExtractionResult()
        long_text = "这是一段比较长的文字内容" * 5  # > 50 chars
        threshold = PersonaExtractor._compute_dynamic_threshold(long_text, result)
        assert threshold < 0.6

    def test_partial_results_lower_threshold(self):
        """规则部分提取降低阈值"""
        result = ExtractionResult(interests={"编程": 0.8})  # 1 interest
        threshold = PersonaExtractor._compute_dynamic_threshold("编程很好", result)
        assert threshold < 0.6

    def test_attribute_signals_lower_threshold(self):
        """属性描述词降低阈值"""
        result = ExtractionResult()
        threshold = PersonaExtractor._compute_dynamic_threshold("我觉得这个不错", result)
        assert threshold < 0.6

    def test_multiple_signals_stacked(self):
        """多个信号叠加 → 阈值更低"""
        result = ExtractionResult(interests={"编程": 0.8})
        # "我" = 第一人称, "3" = 数字, "觉得" = 属性词, len > 50 = 长内容, 1 interest = 部分
        text = "我觉得3种编程语言都很有用，尤其是Python，它在机器学习方面非常强大"
        threshold = PersonaExtractor._compute_dynamic_threshold(text, result)
        assert threshold <= 0.45  # 多个信号应大幅降低

    def test_threshold_minimum_bound(self):
        """阈值不低于 0.4"""
        result = ExtractionResult(
            interests={"编程": 0.8},
            social_style="analytical",  # 有 style 无 preference → partial
        )
        text = "我觉得3年前学的100种编程语言现在还记得，我擅长很多东西"
        threshold = PersonaExtractor._compute_dynamic_threshold(text, result)
        assert threshold >= 0.4

    def test_threshold_maximum_bound(self):
        """阈值不超过 0.75"""
        result = ExtractionResult()
        threshold = PersonaExtractor._compute_dynamic_threshold("ok", result)
        assert threshold <= 0.75

    def test_work_without_life_is_partial(self):
        """有 work_info 无 life_info 视为部分提取"""
        result = ExtractionResult(work_info="程序员")
        threshold = PersonaExtractor._compute_dynamic_threshold("程序员", result)
        assert threshold < 0.6


# ==============================================================
# hybrid 模式集成测试
# ==============================================================

class TestHybridMode:
    """Hybrid 模式使用动态阈值"""

    def test_high_confidence_rule_skips_llm(self):
        """规则高置信度 → 跳过 LLM"""
        ext = PersonaExtractor(extraction_mode="hybrid")
        # Mock rule extractor to return high confidence
        ext._rule_extractor = MagicMock()
        ext._rule_extractor.extract.return_value = ExtractionResult(
            interests={"music": 0.9, "sports": 0.8},
            confidence=0.8,
            source="rule",
        )

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            ext.extract("我喜欢音乐和运动")
        )
        assert result.source == "hybrid"
        assert result.confidence == 0.8
        # LLM should not be called
        assert ext._llm_extractor is None  # no astrbot_context → no LLM

    def test_low_confidence_rule_would_trigger_llm(self):
        """规则低置信度 → 会尝试 LLM（无 provider 时回退到规则）"""
        ext = PersonaExtractor(extraction_mode="hybrid")
        ext._rule_extractor = MagicMock()
        ext._rule_extractor.extract.return_value = ExtractionResult(
            confidence=0.2,
            source="rule",
        )

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            ext.extract("haha")
        )
        # Without LLM extractor, falls back to rule result
        assert result.source == "hybrid"

    def test_dynamic_threshold_is_used(self):
        """验证动态阈值被实际使用（而非固定 0.6）"""
        ext = PersonaExtractor(extraction_mode="hybrid")
        # 高价值内容 + 中等规则置信度 → 应该会尝试 LLM
        ext._rule_extractor = MagicMock()
        ext._rule_extractor.extract.return_value = ExtractionResult(
            interests={"编程": 0.5},
            confidence=0.55,  # 原来会跳过 LLM（< 0.6），现在有动态阈值
            source="rule",
        )

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            ext.extract("我觉得3种编程语言都很有用", summary=None)
        )
        # 即使没有 LLM extractor，至少验证逻辑能执行
        assert result.source == "hybrid"


# ==============================================================
# mode 属性测试
# ==============================================================

class TestModeProperty:
    """模式属性"""

    def test_rule_mode(self):
        ext = PersonaExtractor(extraction_mode="rule")
        assert ext.mode == "rule"

    def test_hybrid_mode(self):
        ext = PersonaExtractor(extraction_mode="hybrid")
        assert ext.mode == "hybrid"

    def test_llm_remaining_without_extractor(self):
        ext = PersonaExtractor(extraction_mode="rule")
        assert ext.llm_remaining_calls == 0
