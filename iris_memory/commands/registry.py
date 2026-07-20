"""
Iris Chat Memory - 指令注册中心

管理所有指令处理器的注册和分发。
"""

from typing import Dict, Optional, TYPE_CHECKING

from iris_memory.core import get_logger
from .base import CommandHandler

if TYPE_CHECKING:
    pass

logger = get_logger("commands.registry")


class CommandRegistry:
    """指令注册中心

    管理所有指令处理器的注册和分发。
    """

    _instance: Optional["CommandRegistry"] = None

    def __init__(self):
        self._handlers: Dict[str, CommandHandler] = {}

    @classmethod
    def get_instance(cls) -> "CommandRegistry":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, handler: CommandHandler) -> None:
        """注册指令处理器

        Args:
            handler: 指令处理器实例
        """
        name = handler.name.lower()
        if name in self._handlers:
            logger.warning(f"指令处理器 {name} 已存在，将被覆盖")

        self._handlers[name] = handler
        logger.info(f"已注册指令处理器: {name}")

    def get_handler(self, name: str) -> Optional[CommandHandler]:
        """获取指令处理器

        Args:
            name: 指令名称

        Returns:
            指令处理器实例，不存在则返回 None
        """
        return self._handlers.get(name.lower())

    def get_all_handlers(self) -> Dict[str, CommandHandler]:
        """获取所有指令处理器"""
        return self._handlers.copy()

    def get_help_text(self) -> str:
        """获取所有指令的帮助文本"""
        lines = [
            "📚 Iris Chat Memory 指令帮助",
            "=" * 40,
            "",
            "用法: iris_mem <模块> <子指令> [参数]",
            "",
            "可用模块:",
        ]

        for name, handler in self._handlers.items():
            lines.append(f"  {name}: {handler.description}")

        lines.extend(
            [
                "",
                "范围参数:",
                "  @用户     - 指定用户",
                "  --group   - 当前群聊所有用户",
                "  --all     - 所有用户",
                "",
                "示例:",
                "  iris_mem l1 clear              # 清空当前用户 L1",
                "  iris_mem l2 clear @张三        # 清空张三的 L2",
                "  iris_mem l3 clear --group      # 清空当前群 L3",
                "  iris_mem profile reset         # 重置当前用户画像",
                "  iris_mem all clear --all       # 清空所有记忆",
                "",
                "输入 'iris_mem <模块> help' 查看详细帮助",
            ]
        )

        return "\n".join(lines)


def get_registry() -> CommandRegistry:
    """获取指令注册中心实例"""
    return CommandRegistry.get_instance()


def register_handler(handler: CommandHandler) -> None:
    """注册指令处理器（便捷函数）"""
    get_registry().register(handler)
