"""
命令注册表模块

提供命令元信息查询，支持参数化子命令模式。
"""

from typing import Callable, Awaitable, Any, Dict, Optional

from iris_memory.commands.handlers import CommandHandlers


class CommandRegistry:
    """命令注册表

    管理命令元信息，提供命令查询接口。
    实际命令分发由 CommandHandlers 的统一入口方法处理。
    """

    def __init__(self, handlers: CommandHandlers) -> None:
        self._handlers = handlers
        self._memory_subcommands = {
            "save", "search", "clear", "stats", "delete",
            "review", "approve", "reject",
        }
        self._iris_subcommands = {
            "proactive", "activity", "reset", "cooldown", "persona",
        }
        self._kv_required_commands = {
            "memory.delete",
            "iris.proactive",
            "iris.reset",
            "iris.persona",
        }

    def get_memory_handler(self, sub_command: str) -> Optional[Callable[..., Awaitable[str]]]:
        """获取 memory 子命令处理器"""
        if sub_command in self._memory_subcommands:
            return self._handlers.handle_memory_command
        return None

    def get_iris_handler(self, sub_command: str) -> Optional[Callable[..., Awaitable[str]]]:
        """获取 iris 子命令处理器"""
        if sub_command in self._iris_subcommands:
            return self._handlers.handle_iris_command
        return None

    def has_memory_handler(self, sub_command: str) -> bool:
        """检查 memory 子命令是否已注册"""
        return sub_command in self._memory_subcommands

    def has_iris_handler(self, sub_command: str) -> bool:
        """检查 iris 子命令是否已注册"""
        return sub_command in self._iris_subcommands

    def requires_kv_operations(self, main_command: str, sub_command: str) -> bool:
        """检查命令是否需要 KV 操作函数"""
        return f"{main_command}.{sub_command}" in self._kv_required_commands

    def get_all_commands(self) -> Dict[str, list]:
        """获取所有已注册的命令"""
        return {
            "memory": list(self._memory_subcommands),
            "iris": list(self._iris_subcommands),
        }
