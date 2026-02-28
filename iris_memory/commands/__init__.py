"""
命令处理模块

提供 AstrBot 插件命令的处理器、权限检查和注册表。
"""

from iris_memory.commands.permissions import PermissionChecker
from iris_memory.commands.handlers import CommandHandlers
from iris_memory.commands.registry import CommandRegistry

__all__ = ["PermissionChecker", "CommandHandlers", "CommandRegistry"]
