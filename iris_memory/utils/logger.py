"""
Logger工具模块

使用示例:
    from iris_memory.utils.logger import get_logger
    
    logger = get_logger("chroma_manager")
    logger.info("普通信息")
    logger.error("错误信息")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 尝试导入 AstrBot 的 logger
try:
    from astrbot.api import logger as astrbot_logger
    _ASTRBOT_LOGGER_AVAILABLE = True
except ImportError:
    astrbot_logger = None
    _ASTRBOT_LOGGER_AVAILABLE = False

# 全局配置
_LOG_CONFIG = {
    "level": "INFO",  # 默认改为 INFO，减少 DEBUG 输出
    "log_dir": None,
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# 已创建的logger缓存
_loggers: Dict[str, logging.Logger] = {}


def _to_flat_logger_name(name: str) -> str:
    """将模块名扁平化为单层 logger 名，避免形成父子 logger 层级。"""
    normalized = name.strip().replace(" ", "_")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    normalized = normalized.strip(".")
    flat = normalized.replace(".", "_")
    return f"iris_memory__{flat}" if flat else "iris_memory"


def _extract_module_name(logger_name: str) -> str:
    """从 logger 名中提取用于展示的模块名。"""
    if logger_name.startswith("iris_memory__"):
        return logger_name[len("iris_memory__"):]
    if logger_name.startswith("iris_memory."):
        return logger_name[len("iris_memory."):].replace(".", "_")
    return logger_name.replace(".", "_")


class AstrBotLogHandler(logging.Handler):
    """将日志转发到 AstrBot 控制台"""

    def emit(self, record: logging.LogRecord) -> None:
        if not _ASTRBOT_LOGGER_AVAILABLE or astrbot_logger is None:
            return

        module_name = _extract_module_name(record.name)

        # 构建消息：模块名: 消息内容
        msg = f"[{module_name}] {record.getMessage()}"

        level = record.levelno
        if level >= logging.ERROR:
            astrbot_logger.error(msg)
        elif level >= logging.WARNING:
            astrbot_logger.warning(msg)
        elif level >= logging.INFO:
            astrbot_logger.info(msg)
        else:
            astrbot_logger.debug(msg)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "DEBUG",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """配置日志系统"""
    global _LOG_CONFIG
    
    _LOG_CONFIG["level"] = level.upper()
    _LOG_CONFIG["max_bytes"] = max_bytes
    _LOG_CONFIG["backup_count"] = backup_count
    
    # 设置日志目录
    if log_dir is None:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            log_dir = Path(get_astrbot_data_path()) / "plugin_data" / "iris_memory" / "logs"
        except ImportError:
            log_dir = Path.cwd() / "logs"
    
    _LOG_CONFIG["log_dir"] = Path(log_dir)
    _LOG_CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    
    # 更新所有已存在的logger
    for name, logger in _loggers.items():
        _configure_logger(logger, name)


def _configure_logger(logger: logging.Logger, name: str) -> None:
    """配置单个logger"""
    logger.handlers = []
    logger.setLevel(getattr(logging, _LOG_CONFIG["level"]))
    
    formatter = logging.Formatter(
        _LOG_CONFIG["format"],
        datefmt=_LOG_CONFIG["date_format"]
    )
    
    # 1. AstrBot 输出
    if _ASTRBOT_LOGGER_AVAILABLE:
        astrbot_handler = AstrBotLogHandler()
        logger.addHandler(astrbot_handler)
    
    # 2. 文件输出（仅在 AstrBot 不可用时输出到控制台）
    if _LOG_CONFIG["log_dir"]:
        # 统一日志文件
        unified_log = _LOG_CONFIG["log_dir"] / "iris_memory.log"
        file_handler = logging.handlers.RotatingFileHandler(
            unified_log,
            maxBytes=_LOG_CONFIG["max_bytes"],
            backupCount=_LOG_CONFIG["backup_count"],
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 3. 测试环境：AstrBot 不可用时输出到控制台
    if not _ASTRBOT_LOGGER_AVAILABLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """获取logger实例"""
    global _loggers
    
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(_to_flat_logger_name(name))
    _configure_logger(logger, name)
    _loggers[name] = logger
    
    return logger


def init_logging_from_config(config: Any, plugin_data_path: Path) -> None:
    """从配置初始化日志系统"""
    from iris_memory.core.defaults import DEFAULTS
    from iris_memory.core.config_manager import get_config_manager
    
    cfg = get_config_manager()
    level = cfg.log_level
    
    log_dir = plugin_data_path / "logs"
    max_bytes = DEFAULTS.log.max_file_size * 1024 * 1024
    backup_count = DEFAULTS.log.backup_count
    
    setup_logging(log_dir=log_dir, level=level, max_bytes=max_bytes, backup_count=backup_count)


__all__ = ["get_logger", "setup_logging", "init_logging_from_config"]
