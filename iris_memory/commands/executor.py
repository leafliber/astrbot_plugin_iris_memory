"""
Iris Chat Memory - 指令执行器

处理 iris_mem 指令的解析和执行。
"""

from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger
from .parser import CommandParser
from .registry import get_registry
from .base import CommandResult

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.executor")


async def execute_command(event: "AstrMessageEvent") -> Optional[str]:
    """执行 iris_mem 指令

    解析指令并分发到对应的处理器执行。

    Args:
        event: AstrBot 消息事件对象

    Returns:
        要发送的消息文本，None 表示无消息
    """
    message_text = event.get_message_outline()
    parsed = CommandParser.parse(message_text)

    if not parsed.is_valid:
        return f"❌ {parsed.error_message}"

    if parsed.module == "help" or parsed.module == "":
        return get_registry().get_help_text()

    registry = get_registry()
    handler = registry.get_handler(parsed.module)

    if not handler:
        return f"❌ 未知的模块: {parsed.module}\n{registry.get_help_text()}"

    if parsed.sub_command == "help":
        return handler.get_help_text()

    target_user_id, error = await CommandParser.extract_target_user_id(
        event, parsed.args
    )
    if error:
        return f"❌ {error}"

    if target_user_id:
        parsed.args.target_user_id = target_user_id

    try:
        result: CommandResult = await handler.handle(
            event, parsed.args, parsed.sub_command
        )
        return result.message
    except Exception as e:
        logger.error(f"执行指令失败: {e}", exc_info=True)
        return "❌ 执行指令失败，详见服务日志"
