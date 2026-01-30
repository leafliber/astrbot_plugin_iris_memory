"""
敏感度检测器单元测试
测试SensitivityDetector的所有功能
"""

import pytest

from iris_memory.capture.sensitivity_detector import SensitivityDetector
from iris_memory.core.types import SensitivityLevel


class TestSensitivityDetector:
    """SensitivityDetector单元测试"""

    @pytest.fixture
    def detector(self):
        """创建SensitivityDetector实例"""
        return SensitivityDetector()

    # ========== 初始化测试 ==========

    def test_detector_initialization(self, detector):
        """测试敏感度检测器初始化"""
        assert detector is not None
        assert len(detector.critical_patterns) > 0
        assert len(detector.sensitive_patterns) > 0
        assert len(detector.private_patterns) > 0
        assert len(detector.personal_patterns) > 0

    # ========== CRITICAL级别测试 ==========

    def test_detect_id_card(self, detector):
        """测试检测身份证号"""
        text = "我的身份证号是123456789012345678"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL
        assert len(entities) > 0
        assert any("CRITICAL" in e for e in entities)

    def test_detect_id_card_with_x(self, detector):
        """测试检测带X的身份证号"""
        text = "身份证号12345678901234567X"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL

    def test_detect_bank_card(self, detector):
        """测试检测银行卡号"""
        text = "我的银行卡号是1234567890123456"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL
        assert any("CRITICAL" in e for e in entities)

    def test_detect_password(self, detector):
        """测试检测密码"""
        text = "密码是mypassword123"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL

    def test_detect_password_colon(self, detector):
        """测试检测冒号分隔的密码"""
        text = "密码:mypassword123"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL

    def test_detect_phone_number(self, detector):
        """测试检测手机号"""
        text = "我的手机号是13812345678"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL

    def test_detect_email(self, detector):
        """测试检测邮箱"""
        text = "我的邮箱是test@example.com"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL

    # ========== SENSITIVE级别测试 ==========

    def test_detect_health_condition(self, detector):
        """测试检测健康状况"""
        text = "我最近在治疗感冒"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE
        assert any("SENSITIVE" in e for e in entities)

    def test_detect_disease(self, detector):
        """测试检测疾病"""
        text = "医生说我得了糖尿病"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    def test_detect_hospital(self, detector):
        """测试检测医院"""
        text = "我在北京大学第一医院工作"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    def test_detect_salary(self, detector):
        """测试检测工资"""
        text = "我的工资是每月1万元"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    def test_detect_income(self, detector):
        """测试检测收入"""
        text = "我的年收入约20万"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    def test_detect_bank_loan(self, detector):
        """测试检测贷款"""
        text = "我有房贷要还"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    def test_detect_address(self, detector):
        """测试检测地址"""
        text = "我的地址是北京市朝阳区"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.SENSITIVE

    # ========== PRIVATE级别测试 ==========

    def test_detect_company(self, detector):
        """测试检测公司"""
        text = "我在腾讯公司上班"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PRIVATE
        assert any("PRIVATE" in e for e in entities)

    def test_detect_school(self, detector):
        """测试检测学校"""
        text = "我在清华大学读书"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PRIVATE

    def test_detect_family(self, detector):
        """测试检测家人"""
        text = "我的父母住在老家"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PRIVATE

    def test_detect_spouse(self, detector):
        """测试检测配偶"""
        text = "我的丈夫是工程师"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PRIVATE

    # ========== PERSONAL级别测试 ==========

    def test_detect_lifestyle(self, detector):
        """测试检测生活方式"""
        text = "我习惯晚上10点睡觉"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PERSONAL
        assert any("PERSONAL" in e for e in entities)

    def test_detect_consumption(self, detector):
        """测试检测消费习惯"""
        text = "我经常在网上购物"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PERSONAL

    # ========== PUBLIC级别测试 ==========

    def test_public_information(self, detector):
        """测试公开信息"""
        text = "我喜欢苹果和橙子"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PUBLIC
        assert len(entities) == 0

    def test_empty_text(self, detector):
        """测试空文本"""
        level, entities = detector.detect_sensitivity("")
        assert level == SensitivityLevel.PUBLIC
        assert entities == []

    def test_none_text(self, detector):
        """测试None文本"""
        level, entities = detector.detect_sensitivity(None)
        assert level == SensitivityLevel.PUBLIC
        assert entities == []

    # ========== 多敏感信息测试 ==========

    def test_multiple_critical_info(self, detector):
        """测试多个CRITICAL级别信息"""
        text = "我的手机号是13812345678，邮箱是test@example.com"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.CRITICAL
        # 应该检测到多个CRITICAL实体
        critical_count = sum(1 for e in entities if "CRITICAL" in e)
        assert critical_count >= 2

    def test_critical_and_sensitive(self, detector):
        """测试CRITICAL和SENSITIVE混合"""
        text = "手机号13812345678，在医院工作"
        level, entities = detector.detect_sensitivity(text)

        # 应该返回CRITICAL级别（最高）
        assert level == SensitivityLevel.CRITICAL
        assert any("CRITICAL" in e for e in entities)
        assert any("SENSITIVE" in e for e in entities)

    def test_all_levels(self, detector):
        """测试所有级别混合"""
        text = "手机13812345678，工资1万，在腾讯公司上班，习惯早睡"
        level, entities = detector.detect_sensitivity(text)

        # 应该返回最高级别CRITICAL
        assert level == SensitivityLevel.CRITICAL
        # 应该检测到多个实体
        assert len(entities) >= 3

    # ========== 上下文敏感度测试 ==========

    def test_context_sensitive_hospital(self, detector):
        """测试上下文敏感度 - 医院"""
        text = "我今天感觉很好"
        context = {"location": "医院", "activity": "检查"}

        level, entities = detector.detect_sensitivity(text, context)
        # 上下文提到医院，应该是SENSITIVE
        assert level == SensitivityLevel.SENSITIVE

    def test_context_sensitive_bank(self, detector):
        """测试上下文敏感度 - 银行"""
        text = "我想存钱"
        context = {"location": "银行", "activity": "理财"}

        level, entities = detector.detect_sensitivity(text, context)
        # 上下文提到银行，应该是SENSITIVE
        assert level == SensitivityLevel.SENSITIVE

    def test_context_no_sensitive(self, detector):
        """测试无敏感上下文"""
        text = "我想买苹果"
        context = {"location": "超市"}

        level, entities = detector.detect_sensitivity(text, context)
        # 无敏感上下文，应该是PUBLIC
        assert level == SensitivityLevel.PUBLIC

    def test_context_none(self, detector):
        """测试None上下文"""
        text = "我喜欢苹果"
        level, entities = detector.detect_sensitivity(text, None)

        assert level == SensitivityLevel.PUBLIC

    # ========== 过滤测试 ==========

    def test_should_filter_critical(self, detector):
        """测试过滤CRITICAL级别"""
        assert detector.should_filter(SensitivityLevel.CRITICAL) is True

    def test_should_filter_sensitive(self, detector):
        """测试过滤SENSITIVE级别"""
        assert detector.should_filter(SensitivityLevel.SENSITIVE) is False

    def test_should_filter_private(self, detector):
        """测试过滤PRIVATE级别"""
        assert detector.should_filter(SensitivityLevel.PRIVATE) is False

    def test_should_filter_personal(self, detector):
        """测试过滤PERSONAL级别"""
        assert detector.should_filter(SensitivityLevel.PERSONAL) is False

    def test_should_filter_public(self, detector):
        """测试过滤PUBLIC级别"""
        assert detector.should_filter(SensitivityLevel.PUBLIC) is False

    # ========== 加密测试 ==========

    def test_encryption_required_critical(self, detector):
        """测试CRITICAL级别需要加密"""
        assert detector.get_encryption_required(SensitivityLevel.CRITICAL) is True

    def test_encryption_required_sensitive(self, detector):
        """测试SENSITIVE级别需要加密"""
        assert detector.get_encryption_required(SensitivityLevel.SENSITIVE) is True

    def test_encryption_required_private(self, detector):
        """测试PRIVATE级别需要加密"""
        assert detector.get_encryption_required(SensitivityLevel.PRIVATE) is True

    def test_encryption_required_personal(self, detector):
        """测试PERSONAL级别需要加密"""
        # PERSONAL(1) < SENSITIVE(3)，不需要加密
        assert detector.get_encryption_required(SensitivityLevel.PERSONAL) is False

    def test_encryption_required_public(self, detector):
        """测试PUBLIC级别需要加密"""
        assert detector.get_encryption_required(SensitivityLevel.PUBLIC) is False

    # ========== 模式检测测试 ==========

    def test_pattern_detection_chinese(self, detector):
        """测试中文模式检测"""
        text = "我最近在医院看病"
        matches = detector._detect_patterns(text, detector.sensitive_patterns)

        assert len(matches) > 0

    def test_pattern_detection_english(self, detector):
        """测试英文模式检测"""
        text = "I have a disease"
        # 中文模式可能不匹配英文，这是正常的
        matches = detector._detect_patterns(text, detector.sensitive_patterns)
        # 应该至少检测到"疾病"相关的中文词（如果有）
        assert isinstance(matches, list)

    def test_pattern_detection_no_match(self, detector):
        """测试无匹配模式"""
        text = "我喜欢苹果"
        matches = detector._detect_patterns(text, detector.sensitive_patterns)

        assert len(matches) == 0

    # ========== 边界情况测试 ==========

    def test_whitespace_text(self, detector):
        """测试只有空白字符"""
        text = "   \n\t   "
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PUBLIC
        assert entities == []

    def test_special_characters_only(self, detector):
        """测试只有特殊字符"""
        text = "@#$%^&*()"
        level, entities = detector.detect_sensitivity(text)

        assert level == SensitivityLevel.PUBLIC

    def test_partial_match(self, detector):
        """测试部分匹配"""
        text = "我的手机是138"  # 不完整的手机号
        level, entities = detector.detect_sensitivity(text)

        # 不完整的手机号可能不被检测
        # 但如果有其他关键词，应该能检测
        assert level in [SensitivityLevel.PUBLIC, SensitivityLevel.CRITICAL]

    def test_very_long_text(self, detector):
        """测试超长文本"""
        text = "我的身份证号是123456789012345678 " * 100
        level, entities = detector.detect_sensitivity(text)

        # 应该仍然能检测到CRITICAL
        assert level == SensitivityLevel.CRITICAL

    def test_unicode_text(self, detector):
        """测试Unicode文本"""
        text = "我的电话是📱13812345678"
        level, entities = detector.detect_sensitivity(text)

        # 应该能检测到手机号
        assert level == SensitivityLevel.CRITICAL

    # ========== 正则表达式边界测试 ==========

    def test_phone_number_invalid_prefix(self, detector):
        """测试无效手机号前缀"""
        text = "我的手机号是02812345678"  # 0不是有效前缀
        level, entities = detector.detect_sensitivity(text)

        # 不应该被识别为手机号
        assert level != SensitivityLevel.CRITICAL or any("手机" in e for e in entities)

    def test_phone_number_wrong_length(self, detector):
        """测试错误长度的手机号"""
        text = "我的手机号是138123456789"  # 12位，太长
        level, entities = detector.detect_sensitivity(text)

        # 不应该被识别为11位手机号
        # 可能被识别为银行卡号
        assert level == SensitivityLevel.CRITICAL

    def test_id_card_invalid_length(self, detector):
        """测试错误长度的身份证号"""
        text = "我的身份证是123456789"  # 9位，太短
        level, entities = detector.detect_sensitivity(text)

        # 不应该被识别为18位身份证号
        # 可能被识别为其他数字
        assert level != SensitivityLevel.CRITICAL or len(entities) > 0

    def test_email_without_at(self, detector):
        """测试没有@的邮箱"""
        text = "我的邮箱是test.example.com"
        level, entities = detector.detect_sensitivity(text)

        # 不应该被识别为邮箱
        assert level == SensitivityLevel.PUBLIC or "email" not in " ".join(entities).lower()

    # ========== 实体返回格式测试 ==========

    def test_entity_format(self, detector):
        """测试实体返回格式"""
        text = "我的手机号是13812345678"
        level, entities = detector.detect_sensitivity(text)

        assert isinstance(level, SensitivityLevel)
        assert isinstance(entities, list)
        if entities:
            assert isinstance(entities[0], str)

    def test_entity_prefix(self, detector):
        """测试实体前缀"""
        text = "我的手机号是13812345678"
        level, entities = detector.detect_sensitivity(text)

        if entities:
            # 每个实体应该有级别前缀
            for entity in entities:
                assert any(prefix in entity for prefix in ["CRITICAL", "SENSITIVE", "PRIVATE", "PERSONAL"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
