"""
Iris Chat Memory - 指令解析器

解析用户输入的指令文本，提取参数和子指令。
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

from .base import ParsedArgs

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


@dataclass
class ParsedCommand:
    """解析后的完整指令"""

    module: str
    sub_command: Optional[str] = None
    args: ParsedArgs = field(default_factory=ParsedArgs)
    is_valid: bool = True
    error_message: Optional[str] = None


class CommandParser:
    """指令解析器

    解析格式: iris_mem <module> [sub_command] [args]

    Examples:
        iris_mem l1 clear
        iris_mem l2 clear @张三
        iris_mem l3 clear --group
        iris_mem profile reset --all
        iris_mem all clear @李四
    """

    PREFIX = "iris_mem"

    SCOPE_FLAGS = {
        "--group": ("group", True),
        "-g": ("group", True),
        "--all": ("all", True),
        "-a": ("all", True),
        "—group": ("group", True),
        "—all": ("all", True),
    }

    @classmethod
    def parse(cls, text: str) -> ParsedCommand:
        """解析指令文本

        Args:
            text: 用户输入的文本

        Returns:
            解析后的指令对象
        """
        text = text.strip()

        # 用单词边界匹配，避免正文出现 "iris_memory" 等子串被误判为指令
        match = re.search(r"\b" + re.escape(cls.PREFIX) + r"\b", text.lower())
        if not match:
            return ParsedCommand(
                module="", is_valid=False, error_message="不是有效的 iris_mem 指令"
            )
        iris_mem_index = match.start()

        text_after_prefix = text[iris_mem_index + len(cls.PREFIX) :].strip()
        parts = text_after_prefix.split()

        if not parts:
            return ParsedCommand(module="", sub_command="help", is_valid=True)

        module = parts[0].lower()
        remaining = parts[1:]

        sub_command = None
        args = ParsedArgs()
        args.raw_args = remaining

        if remaining:
            first_arg = remaining[0].lower()

            if (
                not first_arg.startswith("-")
                and not first_arg.startswith("—")
                and not first_arg.startswith("@")
            ):
                sub_command = first_arg
                remaining = remaining[1:]

        for arg in remaining:
            arg_lower = arg.lower()

            if arg_lower in cls.SCOPE_FLAGS:
                flag_type, _ = cls.SCOPE_FLAGS[arg_lower]
                if flag_type == "group":
                    args.is_group_scope = True
                elif flag_type == "all":
                    args.is_all_scope = True

            elif arg.startswith("@"):
                args.target_user_name = arg[1:]

        if args.is_group_scope and args.is_all_scope:
            return ParsedCommand(
                module=module,
                sub_command=sub_command,
                args=args,
                is_valid=False,
                error_message="--group 和 --all 不能同时使用",
            )

        if args.target_user_name and (args.is_group_scope or args.is_all_scope):
            return ParsedCommand(
                module=module,
                sub_command=sub_command,
                args=args,
                is_valid=False,
                error_message="指定用户与 --group/--all 不能同时使用",
            )

        return ParsedCommand(
            module=module, sub_command=sub_command, args=args, is_valid=True
        )

    @classmethod
    def is_iris_mem_command(cls, text: str) -> bool:
        """检查是否为 iris_mem 指令

        Args:
            text: 用户输入的文本

        Returns:
            是否为 iris_mem 指令
        """
        return text.strip().lower().startswith(cls.PREFIX)

    @classmethod
    async def extract_target_user_id(
        cls, event: "AstrMessageEvent", args: ParsedArgs
    ) -> Tuple[Optional[str], Optional[str]]:
        """提取目标用户 ID

        如果指定了 @用户，需要从消息中解析用户 ID。

        Args:
            event: AstrBot 消息事件
            args: 解析后的参数

        Returns:
            (user_id, error_message)
        """
        from iris_memory.platform import get_adapter

        if args.target_user_name:
            adapter = get_adapter(event)
            mentioned_users = adapter.get_mentioned_users(event)

            if mentioned_users:
                for user_id, user_name in mentioned_users:
                    if user_name == args.target_user_name:
                        return user_id, None

            return None, f"未找到用户 @{args.target_user_name}"

        # 仅对 specified_user 解析目标用户 ID；current_user 不应返回 user_id，
        # 否则 executor 会把它写入 target_user_id，使 scope 实际变为
        # specified_user，各 handler 的 current-user 分支成为死代码。
        if args.scope.value == "specified_user":
            if not args.target_user_id:
                adapter = get_adapter(event)
                user_id = adapter.get_user_id(event)
                return user_id, None

        return None, None
