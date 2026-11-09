"""
Iris Chat Memory - 日志模块

集成 AstrBot 日志系统，提供模块化日志输出功能。

特性：
- 自动添加 [iris-memory:{submodule}] 前缀
- 线程安全的日志适配器缓存
"""

import logging
import threading
from typing import Dict


class IrisMemoryLoggerAdapter(logging.LoggerAdapter):
    """自动添加 [iris-memory:{submodule}] 前缀的日志适配器

    通过继承 logging.LoggerAdapter，自动为所有日志消息添加模块前缀。

    Examples:
        >>> logger = get_logger("config")
        >>> logger.info("配置加载完成")
        # 输出: [iris-memory:config] 配置加载完成
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """处理日志消息，添加前缀

        Args:
            msg: 原始日志消息
            kwargs: 日志关键字参数

        Returns:
            (添加前缀后的消息, 原始kwargs)
        """
        extra = self.extra or {}
        prefix = extra.get("prefix", "[iris-memory:unknown]")
        return f"{prefix} {msg}", kwargs


# ============================================================================
# 模块级缓存
# ============================================================================

_logger_cache: Dict[str, IrisMemoryLoggerAdapter] = {}
_cache_lock = threading.Lock()

_PLUGIN_NAME = "astrbot_plugin_iris_memory"


def _get_base_logger() -> logging.Logger:
    """取得插件的底层日志器

    AstrBot 面板的「日志级别」只作用于按插件分配的日志器，写到全局
    ``astrbot`` 日志器的输出不受其控制。旧版 AstrBot 没有该接口时回退到全局日志器。
    """
    try:
        from astrbot.core.log import LogManager

        return LogManager.get_plugin_logger(_PLUGIN_NAME)
    except (ImportError, AttributeError):
        return logging.getLogger("astrbot")


def get_logger(submodule: str) -> IrisMemoryLoggerAdapter:
    """获取带模块前缀的日志器

    使用模块级缓存避免重复创建日志适配器，支持线程安全访问。

    Args:
        submodule: 子模块名称，如 "config", "context_buffer", "summarizer"

    Returns:
        带 [iris-memory:{submodule}] 前缀的日志适配器

    Raises:
        ValueError: submodule 为空字符串

    Examples:
        >>> logger = get_logger("config")
        >>> logger.info("配置加载完成")
        # 输出: [iris-memory:config] 配置加载完成

    Notes:
        - 内部使用 AstrBot 为本插件分配的日志器，遵循面板上的插件日志级别
        - 推荐在各模块顶部调用一次，避免重复获取
    """
    if not submodule:
        raise ValueError("submodule 参数不能为空")

    # 检查缓存
    with _cache_lock:
        if submodule in _logger_cache:
            return _logger_cache[submodule]

        # 创建新的日志适配器
        base_logger = _get_base_logger()
        prefix = f"[iris-memory:{submodule}]"
        adapter = IrisMemoryLoggerAdapter(base_logger, {"prefix": prefix})

        # 缓存适配器
        _logger_cache[submodule] = adapter

        return adapter
