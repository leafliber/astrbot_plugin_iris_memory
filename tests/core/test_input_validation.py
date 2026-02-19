"""输入验证常量测试"""

import pytest
from iris_memory.core.constants import InputValidationConfig


class TestInputValidationConfig:
    """输入验证配置测试"""

    def test_max_message_length_positive(self):
        assert InputValidationConfig.MAX_MESSAGE_LENGTH > 0

    def test_max_query_length_positive(self):
        assert InputValidationConfig.MAX_QUERY_LENGTH > 0

    def test_max_save_content_length_positive(self):
        assert InputValidationConfig.MAX_SAVE_CONTENT_LENGTH > 0

    def test_query_shorter_than_message(self):
        """查询长度限制应小于消息长度限制"""
        assert InputValidationConfig.MAX_QUERY_LENGTH <= InputValidationConfig.MAX_MESSAGE_LENGTH

    def test_save_shorter_than_message(self):
        """保存长度限制应小于等于消息长度限制"""
        assert InputValidationConfig.MAX_SAVE_CONTENT_LENGTH <= InputValidationConfig.MAX_MESSAGE_LENGTH
