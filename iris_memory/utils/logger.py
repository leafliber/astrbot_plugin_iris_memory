"""
Logger工具模块 - 完整的日志管理系统

提供统一的日志管理功能：
- 支持文件日志输出（带自动轮转）
- 支持多模块独立的logger实例
- 可配置的日志级别和格式
- 与AstrBot logger无缝集成
- 支持控制台和文件同时输出

使用示例:
    from iris_memory.utils.logger import get_logger
    
    logger = get_logger("chroma_manager")
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import os

# 全局配置
_LOG_CONFIG = {
    "level": "DEBUG",
    "log_to_file": True,
    "log_dir": None,  # 将在初始化时设置
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "console_output": True,
    "file_output": True,
}

# 已创建的logger缓存
_loggers: Dict[str, logging.Logger] = {}


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "DEBUG",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    配置日志系统
    
    Args:
        log_dir: 日志文件目录，如果为None则使用默认路径
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        format_string: 自定义日志格式
    """
    global _LOG_CONFIG
    
    # 更新配置
    _LOG_CONFIG["level"] = level.upper()
    _LOG_CONFIG["max_bytes"] = max_bytes
    _LOG_CONFIG["backup_count"] = backup_count
    _LOG_CONFIG["console_output"] = console_output
    _LOG_CONFIG["file_output"] = file_output
    if format_string:
        _LOG_CONFIG["format"] = format_string
    
    # 设置日志目录
    if log_dir is None:
        # 默认使用插件数据目录下的logs文件夹
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            log_dir = Path(get_astrbot_data_path()) / "plugin_data" / "iris_memory" / "logs"
        except ImportError:
            # 测试环境使用当前目录
            log_dir = Path.cwd() / "logs"
    
    _LOG_CONFIG["log_dir"] = Path(log_dir)
    
    # 创建日志目录
    if file_output and _LOG_CONFIG["log_dir"]:
        _LOG_CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    
    # 更新所有已存在的logger
    for name, logger in _loggers.items():
        _configure_logger(logger, name)
    
    # 记录配置信息
    root_logger = get_logger("logger_setup")
    root_logger.info(f"Logging system configured: level={level}, log_dir={log_dir}")


def _configure_logger(logger: logging.Logger, name: str) -> None:
    """
    配置单个logger实例
    
    Args:
        logger: 要配置的logger实例
        name: logger名称
    """
    # 清除现有handlers
    logger.handlers = []
    logger.setLevel(getattr(logging, _LOG_CONFIG["level"]))
    
    # 创建格式化器
    formatter = logging.Formatter(
        _LOG_CONFIG["format"],
        datefmt=_LOG_CONFIG["date_format"]
    )
    
    # 控制台输出
    if _LOG_CONFIG["console_output"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if _LOG_CONFIG["file_output"] and _LOG_CONFIG["log_dir"]:
        log_file = _LOG_CONFIG["log_dir"] / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_LOG_CONFIG["max_bytes"],
            backupCount=_LOG_CONFIG["backup_count"],
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 同时写入到统一日志文件
        unified_log = _LOG_CONFIG["log_dir"] / "iris_memory.log"
        unified_handler = logging.handlers.RotatingFileHandler(
            unified_log,
            maxBytes=_LOG_CONFIG["max_bytes"],
            backupCount=_LOG_CONFIG["backup_count"],
            encoding="utf-8"
        )
        unified_handler.setFormatter(formatter)
        logger.addHandler(unified_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取或创建一个logger实例
    
    Args:
        name: logger名称，建议使用模块名（如 "chroma_manager", "capture_engine"）
    
    Returns:
        配置好的logger实例
    """
    global _loggers
    
    # 如果logger已存在，直接返回
    if name in _loggers:
        return _loggers[name]
    
    # 创建新的logger
    logger = logging.getLogger(f"iris_memory.{name}")
    logger.setLevel(getattr(logging, _LOG_CONFIG["level"]))
    
    # 配置logger
    _configure_logger(logger, name)
    
    # 缓存logger
    _loggers[name] = logger
    
    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    根据模块名获取logger（简化版）
    
    Args:
        module_name: 模块名称
        
    Returns:
        logger实例
    """
    return get_logger(module_name)


class DebugLogger:
    """
    调试日志装饰器/上下文管理器
    
    用于记录函数调用和性能统计：
    
    使用示例:
        @DebugLogger("chroma_manager")
        async def my_function():
            pass
            
        或者:
        with DebugLogger("operation_name"):
            # 执行操作
            pass
    """
    
    def __init__(self, name: str, log_args: bool = True, log_result: bool = False):
        """
        Args:
            name: 操作名称
            log_args: 是否记录参数
            log_result: 是否记录返回值
        """
        self.name = name
        self.log_args = log_args
        self.log_result = log_result
        self.logger = get_logger("debug")
        self.start_time = None
    
    def __call__(self, func):
        """作为装饰器使用"""
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self.start_time = datetime.now()
            
            # 记录开始
            if self.log_args:
                args_str = ", ".join([str(a) for a in args[1:]])  # 排除self
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                self.logger.debug(f"[START] {self.name}({all_args})")
            else:
                self.logger.debug(f"[START] {self.name}")
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录完成
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if self.log_result:
                    self.logger.debug(f"[END] {self.name} - {elapsed:.3f}s - Result: {result}")
                else:
                    self.logger.debug(f"[END] {self.name} - {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                self.logger.error(f"[ERROR] {self.name} - {elapsed:.3f}s - {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            self.start_time = datetime.now()
            
            # 记录开始
            if self.log_args:
                args_str = ", ".join([str(a) for a in args[1:]])  # 排除self
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                self.logger.debug(f"[START] {self.name}({all_args})")
            else:
                self.logger.debug(f"[START] {self.name}")
            
            try:
                result = func(*args, **kwargs)
                
                # 记录完成
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if self.log_result:
                    self.logger.debug(f"[END] {self.name} - {elapsed:.3f}s - Result: {result}")
                else:
                    self.logger.debug(f"[END] {self.name} - {elapsed:.3f}s")
                
                return result
            except Exception as e:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                self.logger.error(f"[ERROR] {self.name} - {elapsed:.3f}s - {e}")
                raise
        
        # 根据函数类型返回适当的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    def __enter__(self):
        """作为上下文管理器使用"""
        self.start_time = datetime.now()
        self.logger.debug(f"[START] {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"[ERROR] {self.name} - {elapsed:.3f}s - {exc_val}")
        else:
            self.logger.debug(f"[END] {self.name} - {elapsed:.3f}s")


def log_method_call(logger_name: str, log_args: bool = True, log_result: bool = False):
    """
    方法调用日志装饰器
    
    使用示例:
        @log_method_call("chroma_manager")
        async def add_memory(self, memory):
            pass
    """
    return DebugLogger(logger_name, log_args=log_args, log_result=log_result)


def init_logging_from_config(config: Any, plugin_data_path: Path) -> None:
    """
    从配置初始化日志系统
    
    Args:
        config: 配置对象（支持字典或AstrBotConfig）
        plugin_data_path: 插件数据目录
    """
    # 导入默认配置
    from iris_memory.core.defaults import DEFAULTS
    from iris_memory.core.config_manager import get_config_manager
    
    # 尝试使用配置管理器
    cfg = get_config_manager()
    if cfg._user_config is not None:
        level = cfg.log_level
    else:
        # 回退到直接读取配置
        if hasattr(config, 'get'):
            log_config = config.get("log_config", {}) or {}
        else:
            log_config = getattr(config, 'log_config', {}) or {}
            if hasattr(log_config, '__dict__'):
                log_config = log_config.__dict__
        level = log_config.get("level", DEFAULTS.log.level)
    
    log_dir = plugin_data_path / "logs"
    max_bytes = DEFAULTS.log.max_file_size * 1024 * 1024  # MB to bytes
    backup_count = DEFAULTS.log.backup_count
    console_output = DEFAULTS.log.console_output
    file_output = DEFAULTS.log.file_output
    
    setup_logging(
        log_dir=log_dir,
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_output=console_output,
        file_output=file_output
    )
    
    logger = get_logger("init")
    logger.info(f"Logging initialized from config: level={level}, log_dir={log_dir}")


# 导出公共接口
__all__ = [
    "get_logger",
    "get_module_logger",
    "setup_logging",
    "init_logging_from_config",
    "DebugLogger",
    "log_method_call",
]
