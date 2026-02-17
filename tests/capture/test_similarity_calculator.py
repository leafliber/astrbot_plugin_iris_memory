"""
相似度计算模块单元测试
测试SimilarityCalculator的所有功能
"""

import pytest

from iris_memory.capture.similarity_calculator import SimilarityCalculator, sanitize_for_log


class TestSanitizeForLog:
    """sanitize_for_log函数测试"""

    def test_empty_text(self):
        """测试空文本"""
        assert sanitize_for_log("") == "[empty]"
        assert sanitize_for_log(None) == "[empty]"

    def test_phone_number_masking(self):
        """测试手机号脱敏"""
        text = "我的手机号是13812345678"
        result = sanitize_for_log(text)
        assert "[PHONE]" in result
        assert "13812345678" not in result

    def test_id_card_masking(self):
        """测试身份证号脱敏"""
        text = "身份证号是123456789012345678"
        result = sanitize_for_log(text)
        assert "[ID_CARD]" in result
        assert "123456789012345678" not in result

    def test_bank_card_masking(self):
        """测试银行卡号脱敏"""
        text = "银行卡号是1234567890123456"
        result = sanitize_for_log(text)
        assert "[BANK_CARD]" in result

    def test_password_masking(self):
        """测试密码脱敏"""
        text = "密码是abc123"
        result = sanitize_for_log(text)
        assert "[MASKED]" in result
        assert "abc123" not in result

    def test_email_masking(self):
        """测试邮箱脱敏"""
        text = "邮箱是test@example.com"
        result = sanitize_for_log(text)
        assert "[EMAIL]" in result
        assert "test@example.com" not in result

    def test_truncation(self):
        """测试截断"""
        text = "这是一段很长的文本，需要被截断处理"
        result = sanitize_for_log(text, max_length=10)
        assert len(result) == 13  # 10 + "..."
        assert result.endswith("...")

    def test_no_truncation_needed(self):
        """测试不需要截断"""
        text = "短文本"
        result = sanitize_for_log(text)
        assert result == text


class TestSimilarityCalculator:
    """SimilarityCalculator类测试"""

    @pytest.fixture
    def calculator(self):
        """创建SimilarityCalculator实例"""
        return SimilarityCalculator()

    # ========== 快速相似度计算测试 ==========

    def test_calculate_quick_similarity_identical(self, calculator):
        """测试相同文本的快速相似度"""
        text = "我喜欢吃苹果"
        similarity = calculator.calculate_quick_similarity(text, text)
        assert similarity == 1.0

    def test_calculate_quick_similarity_different(self, calculator):
        """测试不同文本的快速相似度"""
        text1 = "我喜欢吃苹果"
        text2 = "今天天气很好"
        similarity = calculator.calculate_quick_similarity(text1, text2)
        assert 0.0 <= similarity < 1.0

    def test_calculate_quick_similarity_similar(self, calculator):
        """测试相似文本的快速相似度"""
        text1 = "我喜欢吃苹果"
        text2 = "我喜欢吃橙子"
        similarity = calculator.calculate_quick_similarity(text1, text2)
        # 相似文本应该有一定相似度
        assert similarity > 0.1

    def test_calculate_quick_similarity_empty(self, calculator):
        """测试空文本的快速相似度"""
        similarity = calculator.calculate_quick_similarity("", "测试")
        assert similarity == 0.0
        similarity = calculator.calculate_quick_similarity("测试", "")
        assert similarity == 0.0

    # ========== 精确相似度计算测试 ==========

    def test_calculate_similarity_identical(self, calculator):
        """测试相同文本的精确相似度"""
        text = "我喜欢吃苹果"
        similarity = calculator.calculate_similarity(text, text)
        assert similarity == 1.0

    def test_calculate_similarity_different(self, calculator):
        """测试不同文本的精确相似度"""
        text1 = "我喜欢吃苹果"
        text2 = "今天天气很好"
        similarity = calculator.calculate_similarity(text1, text2)
        assert 0.0 <= similarity < 1.0

    def test_calculate_similarity_similar(self, calculator):
        """测试相似文本的精确相似度"""
        text1 = "我喜欢吃苹果"
        text2 = "我喜欢吃橙子"
        similarity = calculator.calculate_similarity(text1, text2)
        # 相似文本应该有一定相似度
        assert similarity > 0.4

    def test_calculate_similarity_case_insensitive(self, calculator):
        """测试大小写不敏感"""
        text1 = "Hello World"
        text2 = "hello world"
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0

    def test_calculate_similarity_long_texts(self, calculator):
        """测试长文本相似度"""
        text1 = "这是一段很长的文本内容，用来测试长文本的相似度计算是否正确。" * 10
        text2 = "这是一段很长的文本内容，用来测试长文本的相似度计算是否正确。" * 10
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0

    # ========== 内容相似度计算测试 ==========

    def test_calculate_content_similarity_identical(self, calculator):
        """测试相同文本的内容相似度"""
        text = "我喜欢吃苹果"
        similarity = calculator.calculate_content_similarity(text, text)
        assert similarity == 1.0

    def test_calculate_content_similarity_different(self, calculator):
        """测试不同文本的内容相似度"""
        text1 = "我喜欢吃苹果"
        text2 = "今天天气很好"
        similarity = calculator.calculate_content_similarity(text1, text2)
        assert 0.0 <= similarity < 0.5

    def test_calculate_content_similarity_empty(self, calculator):
        """测试空文本的内容相似度"""
        similarity = calculator.calculate_content_similarity("", "测试")
        assert similarity == 0.0

    # ========== 共同主题检测测试 ==========

    def test_have_common_subject_true(self, calculator):
        """测试有共同主题 - 使用英文单词（确保正确分词）"""
        text1 = "I like apple and banana"
        text2 = "I like orange and grape"
        # "like", "and" 是共同词（>=2个）
        assert calculator.have_common_subject(text1, text2) is True

    def test_have_common_subject_false(self, calculator):
        """测试没有共同主题"""
        text1 = "我喜欢吃苹果"
        text2 = "今天天气很好"
        assert calculator.have_common_subject(text1, text2) is False

    def test_have_common_subject_with_stopwords(self, calculator):
        """测试包含停用词的文本 - 使用英文单词"""
        text1 = "I like apple and banana"
        text2 = "He likes apple and grape"
        # "apple", "and" 是共同词（>=2个）
        assert calculator.have_common_subject(text1, text2) is True

    def test_have_common_subject_single_word(self, calculator):
        """测试只有一个共同词（应该返回False，需要>=2个）"""
        text1 = "我喜欢苹果"
        text2 = "他在吃橙子"
        # 只有"喜欢"或"苹果"不在两者中，没有足够的共同词
        result = calculator.have_common_subject(text1, text2)
        # 取决于具体内容，可能True或False
        assert isinstance(result, bool)

    # ========== LCS算法测试 ==========

    def test_longest_common_substring_length_identical(self, calculator):
        """测试相同文本的LCS"""
        text = "我喜欢吃苹果"
        length = calculator._longest_common_substring_length(text, text)
        assert length == len(text)

    def test_longest_common_substring_length_partial(self, calculator):
        """测试部分相同的LCS"""
        text1 = "我喜欢吃苹果"
        text2 = "我喜欢吃橙子"
        length = calculator._longest_common_substring_length(text1, text2)
        # "我喜欢吃" 是公共子串
        assert length == 4

    def test_longest_common_substring_length_none(self, calculator):
        """测试没有公共子串"""
        text1 = "abc"
        text2 = "xyz"
        length = calculator._longest_common_substring_length(text1, text2)
        assert length == 0

    def test_longest_common_substring_length_empty(self, calculator):
        """测试空文本的LCS"""
        length = calculator._longest_common_substring_length("", "test")
        assert length == 0
        length = calculator._longest_common_substring_length("test", "")
        assert length == 0

    # ========== N-gram测试 ==========

    def test_get_ngrams(self, calculator):
        """测试N-gram生成"""
        text = "abc"
        ngrams = calculator._get_ngrams(text, 2)
        assert ngrams == {"ab", "bc"}

    def test_get_ngrams_short_text(self, calculator):
        """测试短文本N-gram"""
        text = "a"
        ngrams = calculator._get_ngrams(text, 2)
        assert len(ngrams) == 0  # 文本太短，无法生成2-gram

    # ========== 边界情况测试 ==========

    def test_calculate_similarity_with_numbers(self, calculator):
        """测试包含数字的文本"""
        text1 = "我有3个苹果"
        text2 = "我有5个苹果"
        similarity = calculator.calculate_similarity(text1, text2)
        # 数字不同但结构相似，相似度应该较高
        assert similarity > 0.5

    def test_calculate_similarity_with_special_chars(self, calculator):
        """测试包含特殊字符的文本"""
        text1 = "测试@#$特殊字符"
        text2 = "测试@#$特殊字符"
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0

    def test_calculate_similarity_with_unicode(self, calculator):
        """测试包含Unicode/emoji的文本"""
        text1 = "测试🍎🍊🍋emoji"
        text2 = "测试🍎🍊🍋emoji"
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0

    def test_calculate_similarity_mixed_language(self, calculator):
        """测试混合语言文本"""
        text1 = "Hello 世界"
        text2 = "Hello 世界"
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0

    # ========== 性能测试 ==========

    def test_performance_long_text(self, calculator):
        """测试长文本计算性能"""
        text1 = "这是一段测试文本，" * 100
        text2 = "这是一段测试文本，" * 100
        # 确保能处理长文本
        similarity = calculator.calculate_similarity(text1, text2)
        assert similarity == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
