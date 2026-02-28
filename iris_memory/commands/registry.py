"""
命令注册表模块

提供命令名称到处理器方法的映射，支持动态注册和查找。

使用 dict dispatch 模式替代 if-elif 链。
"""

from typing import Callable, Awaitable, Any, Dict, Optional

from astrbot.api.event import AstrMessageEvent

from iris_memory.commands.handlers import CommandHandlers


class CommandRegistry:
    """
    命令注册表

    管理命令名称到处理器方法的映射，提供统一的命令分发接口。

    使用方式：
        registry = CommandRegistry(handlers)
        handler = registry.get_handler("memory_save")
        result = await handler(event, *args)
    """

    def __init__(self, handlers: CommandHandlers) -> None:
        """
        初始化命令注册表

        Args:
            handlers: CommandHandlers 实例
        """
        self._handlers = handlers
        self._registry: Dict[str, Callable[..., Awaitable[str]]] = {}
        self._register_commands()

    def _register_commands(self) -> None:
        """注册所有命令处理器"""
        self._registry = {
            "memory_save": self._handlers.handle_save_memory,
            "memory_search": self._handlers.handle_search_memory,
            "memory_clear": self._handlers.handle_clear_memory,
            "memory_stats": self._handlers.handle_memory_stats,
            "memory_delete": None,  # 需要额外参数，特殊处理
            "proactive_reply": None,  # 需要额外参数，特殊处理
            "activity_status": self._handlers.handle_activity_status,
            "iris_reset": None,  # 需要额外参数，特殊处理
        }

    def get_handler(
        self,
        command_name: str
    ) -> Optional[Callable[..., Awaitable[str]]]:
        """
        获取命令处理器

        Args:
            command_name: 命令名称

        Returns:
            Optional[Callable]: 处理器方法，不存在返回 None
        """
        return self._registry.get(command_name)

    def has_handler(self, command_name: str) -> bool:
        """
        检查命令是否已注册

        Args:
            command_name: 命令名称

        Returns:
            bool: 是否已注册
        """
        return command_name in self._registry

    def requires_kv_operations(self, command_name: str) -> bool:
        """
        检查命令是否需要 KV 操作函数

        Args:
            command_name: 命令名称

        Returns:
            bool: 是否需要 KV 操作
        """
        return command_name in {
            "memory_delete",
            "proactive_reply",
            "iris_reset",
        }

    def get_registered_commands(self) -> list[str]:
        """
        获取所有已注册的命令名称

        Returns:
            list[str]: 命令名称列表
        """
        return list(self._registry.keys())
