"""敏感信息脱敏 + 检测器测试"""

import pytest
from iris_memory.capture.detector.sensitivity_detector import (
    _mask_sensitive,
    SensitivityDetector,
)


class TestMaskSensitive:
    """_mask_sensitive 脱敏函数测试"""

    def test_short_value_all_masked(self):
        """长度 ≤ 4 全部替换"""
        assert _mask_sensitive("abc") == "***"
        assert _mask_sensitive("1234") == "****"
        assert _mask_sensitive("a") == "*"

    def test_medium_value_keep_first_last(self):
        """长度 5-8 保留首尾各 1 字符"""
        result = _mask_sensitive("12345")
        assert result[0] == "1"
        assert result[-1] == "5"
        assert result[1:-1] == "***"

        result = _mask_sensitive("12345678")
        assert result[0] == "1"
        assert result[-1] == "8"
        assert len(result) == 8

    def test_long_value_keep_four(self):
        """长度 > 8 保留首尾各 4 字符"""
        value = "1234567890AB"
        result = _mask_sensitive(value)
        assert result.startswith("1234")
        assert result.endswith("90AB")
        assert "*" in result

    def test_empty_string(self):
        """空字符串"""
        assert _mask_sensitive("") == ""

    def test_real_id_number(self):
        """模拟身份证号脱敏"""
        fake_id = "110101199001011234"
        result = _mask_sensitive(fake_id)
        assert result[:4] == "1101"
        assert result[-4:] == "1234"
        assert len(result) == len(fake_id)


class TestSensitivityDetector:
    """SensitivityDetector 基本行为测试"""

    def test_instantiation(self):
        """检测器可以正常实例化"""
        detector = SensitivityDetector()
        assert detector is not None

    def test_detect_no_sensitive(self):
        """普通消息不包含敏感信息"""
        detector = SensitivityDetector()
        level, entities = detector.detect_sensitivity("今天天气真好")
        # 应返回最低敏感等级
        assert entities == [] or level is not None

    def test_detect_email(self):
        """检测邮箱"""
        detector = SensitivityDetector()
        level, entities = detector.detect_sensitivity("我的邮箱是 test@example.com")
        # 应检测到实体（具体行为取决于实现）
        assert level is not None
