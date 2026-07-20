"""日志模块测试"""

from iris_memory.core.logger import get_logger, IrisMemoryLoggerAdapter


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

    def test_logger_has_handler(self):
        logger = get_logger("test_module")

        assert isinstance(logger, IrisMemoryLoggerAdapter)
