"""日志模块测试"""

import logging
from unittest.mock import patch

from astrbot_plugin_iris_memory.iris_memory.core.logger import (
    IrisMemoryLoggerAdapter,
    _get_base_logger,
    get_logger,
)


class TestLogger:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test_module")

        assert isinstance(logger, IrisMemoryLoggerAdapter)

    def test_logger_name_format(self):
        logger = get_logger("test_module")

        assert logger.extra["prefix"] == "[iris-memory:test_module]"

    def test_different_modules(self):
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert "module1" in logger1.extra["prefix"]
        assert "module2" in logger2.extra["prefix"]

    def test_same_module_returns_same_logger(self):
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2

    def test_logger_follows_plugin_log_level(self):
        """面板把插件日志级别调高后，INFO 不再输出"""
        logger = get_logger("level_follow_test")
        plugin_logger = logging.getLogger("astrbot.plugin.astrbot_plugin_iris_memory")
        original = plugin_logger.level
        try:
            plugin_logger.setLevel(logging.ERROR)
            assert not logger.isEnabledFor(logging.INFO)
            assert logger.isEnabledFor(logging.ERROR)
        finally:
            plugin_logger.setLevel(original)

    def test_falls_back_to_global_logger_on_old_astrbot(self):
        with patch("astrbot.core.log.LogManager", spec=[]):
            assert _get_base_logger().name == "astrbot"

    def test_logger_has_handler(self):
        logger = get_logger("test_module")

        assert isinstance(logger, IrisMemoryLoggerAdapter)
