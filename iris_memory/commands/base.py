"""
Iris Chat Memory - 指令基础类

定义指令处理器的基类和通用数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


class DeleteScope(Enum):
    """删除范围枚举"""

    CURRENT_USER = "current_user"
    SPECIFIED_USER = "specified_user"
    GROUP = "group"
    ALL = "all"


@dataclass
class ParsedArgs:
    """解析后的指令参数"""

    target_user_id: Optional[str] = None
    target_user_name: Optional[str] = None
    is_group_scope: bool = False
    is_all_scope: bool = False
    raw_args: list[str] = field(default_factory=list)

    @property
    def scope(self) -> DeleteScope:
        """获取删除范围"""
        if self.is_all_scope:
            return DeleteScope.ALL
        if self.is_group_scope:
            return DeleteScope.GROUP
        if self.target_user_id:
            return DeleteScope.SPECIFIED_USER
        return DeleteScope.CURRENT_USER

    def get_scope_description(self) -> str:
        """获取范围描述文本"""
        scope_map = {
            DeleteScope.CURRENT_USER: "当前用户",
            DeleteScope.SPECIFIED_USER: f"用户 {self.target_user_name or self.target_user_id}",
            DeleteScope.GROUP: "当前群聊所有用户",
            DeleteScope.ALL: "所有用户",
        }
        return scope_map[self.scope]


@dataclass
class CommandResult:
    """指令执行结果"""

    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.message


class CommandHandler(ABC):
    """指令处理器基类

    所有指令处理器都需要继承此类并实现 handle 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """指令名称（如 l1, l2, l3, profile, all）"""
        pass

    @property
    def description(self) -> str:
        """指令描述"""
        return ""

    @property
    def sub_commands(self) -> Dict[str, str]:
        """子指令及其描述"""
        return {}

    @abstractmethod
    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理指令

        Args:
            event: AstrBot 消息事件
            args: 解析后的参数
            sub_command: 子指令（如 clear, reset, stats）

        Returns:
            执行结果
        """
        pass

    def get_help_text(self) -> str:
        """获取帮助文本"""
        lines = [f"iris_mem {self.name} - {self.description}"]

        if self.sub_commands:
            lines.append("\n子指令:")
            for cmd, desc in self.sub_commands.items():
                lines.append(f"  {cmd}: {desc}")

        lines.append("\n范围参数:")
        lines.append("  @用户: 指定用户")
        lines.append("  --group / -g: 当前群聊所有用户")
        lines.append("  --all / -a: 所有用户")

        return "\n".join(lines)
