"""
事件工具模块 - 提供事件处理的通用工具函数
"""

from typing import Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("event_utils")


def get_group_id(event) -> Optional[str]:
    """安全地获取群组ID

    从 AstrMessageEvent 中获取群组ID。
    在群聊场景中，event.message_obj.group_id 会返回群组ID；私聊场景返回 None 或空值。

    Args:
        event: 消息事件对象（AstrMessageEvent）

    Returns:
        Optional[str]: 群组ID，如果不存在（私聊场景）则返回 None

    Examples:
        >>> # 在私聊场景中
        >>> group_id = get_group_id(event)
        >>> print(group_id)  # None

        >>> # 在群聊场景中
        >>> group_id = get_group_id(event)
        >>> print(group_id)  # "123456789"
    """
    try:
        if hasattr(event, 'message_obj') and hasattr(event.message_obj, 'group_id'):
            group_id = event.message_obj.group_id
            if group_id:
                return str(group_id)
        return None
    except Exception as e:
        logger.debug(f"Failed to get group_id from event: {e}")
        return None


def get_user_id(event) -> Optional[str]:
    """安全地获取用户ID

    支持不同类型的事件。

    Args:
        event: 消息事件对象

    Returns:
        Optional[str]: 用户ID，如果不存在则返回 None
    """
    try:
        if hasattr(event, 'get_sender_id'):
            return event.get_sender_id()
        return None
    except (AttributeError, Exception):
        return None


def get_sender_name(event) -> Optional[str]:
    """安全地获取发送者显示名称

    尝试多种方式获取发送者的显示名称（昵称/群名片），
    用于在群聊记忆中标注消息来源。

    Args:
        event: 消息事件对象（AstrMessageEvent）

    Returns:
        Optional[str]: 发送者显示名称，如果不存在则返回 None
    """
    try:
        # 方式1: message_obj.sender 对象
        if hasattr(event, 'message_obj'):
            msg_obj = event.message_obj
            # 尝试 sender.nickname
            if hasattr(msg_obj, 'sender') and msg_obj.sender:
                sender = msg_obj.sender
                # 优先使用群名片（card），其次使用昵称
                for attr in ['card', 'nickname', 'user_name', 'name', 'display_name']:
                    name = getattr(sender, attr, None)
                    if name and str(name).strip():
                        return str(name).strip()
                # 如果 sender 是 dict
                if isinstance(sender, dict):
                    for key in ['card', 'nickname', 'user_name', 'name', 'display_name']:
                        name = sender.get(key)
                        if name and str(name).strip():
                            return str(name).strip()
        # 方式2: get_sender_name 方法
        if hasattr(event, 'get_sender_name'):
            name = event.get_sender_name()
            if name and str(name).strip():
                return str(name).strip()
        # 方式3: nickname 属性
        if hasattr(event, 'nickname'):
            name = event.nickname
            if name and str(name).strip():
                return str(name).strip()
        return None
    except Exception as e:
        logger.debug(f"Failed to get sender_name from event: {e}")
        return None


def get_message_content(event) -> Optional[str]:
    """安全地获取消息内容

    Args:
        event: 消息事件对象

    Returns:
        Optional[str]: 消息内容，如果不存在则返回 None
    """
    try:
        if hasattr(event, 'message_str'):
            return event.message_str
        return None
    except (AttributeError, Exception):
        return None
