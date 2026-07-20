"""
Iris Chat Memory - 指令模块

提供管理员指令接口，支持：
- L1/L2/L3 记忆管理
- 画像管理
- 批量删除操作
"""

from .base import (
    CommandHandler,
    CommandResult,
    DeleteScope,
    ParsedArgs,
)
from .parser import CommandParser
from .registry import CommandRegistry, register_handler, get_registry
from .l1_handler import L1CommandHandler
from .l2_handler import L2CommandHandler
from .l3_handler import L3CommandHandler
from .profile_handler import ProfileCommandHandler
from .all_handler import AllCommandHandler
from .executor import execute_command

__all__ = [
    "CommandHandler",
    "CommandResult",
    "DeleteScope",
    "ParsedArgs",
    "CommandParser",
    "CommandRegistry",
    "register_handler",
    "get_registry",
    "L1CommandHandler",
    "L2CommandHandler",
    "L3CommandHandler",
    "ProfileCommandHandler",
    "AllCommandHandler",
    "execute_command",
]
