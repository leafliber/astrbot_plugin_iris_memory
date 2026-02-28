import logging
from unittest.mock import Mock

from iris_memory.utils import logger as logger_module


def setup_function() -> None:
    logger_module._loggers.clear()


def test_get_logger_uses_flat_name_and_cache() -> None:
    logger = logger_module.get_logger("memory_service.business")

    assert logger.name == "iris_memory__memory_service_business"
    assert logger_module.get_logger("memory_service.business") is logger


def test_get_logger_flat_name_has_no_hierarchy_dots() -> None:
    logger = logger_module.get_logger("a.b.c")

    assert logger.name == "iris_memory__a_b_c"
    assert "." not in logger.name.removeprefix("iris_memory__")


def test_astrbot_handler_formats_flat_module_name(monkeypatch) -> None:
    mock_astrbot_logger = Mock()
    monkeypatch.setattr(logger_module, "_ASTRBOT_LOGGER_AVAILABLE", True)
    monkeypatch.setattr(logger_module, "astrbot_logger", mock_astrbot_logger)

    handler = logger_module.AstrBotLogHandler()
    record = logging.LogRecord(
        name="iris_memory__memory_service_business",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="Message classified",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    mock_astrbot_logger.debug.assert_called_once()
    call_args = mock_astrbot_logger.debug.call_args[0][0]
    assert "[memory_service_business]" in call_args
    assert "Message classified" in call_args


def test_astrbot_handler_compat_with_dotted_name(monkeypatch) -> None:
    mock_astrbot_logger = Mock()
    monkeypatch.setattr(logger_module, "_ASTRBOT_LOGGER_AVAILABLE", True)
    monkeypatch.setattr(logger_module, "astrbot_logger", mock_astrbot_logger)

    handler = logger_module.AstrBotLogHandler()
    record = logging.LogRecord(
        name="iris_memory.memory_service.business",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="legacy",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    mock_astrbot_logger.info.assert_called_once()
    call_args = mock_astrbot_logger.info.call_args[0][0]
    assert "[memory_service_business]" in call_args
    assert "legacy" in call_args
