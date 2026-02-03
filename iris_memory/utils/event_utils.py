"""
事件工具模块 - 提供事件处理的通用工具函数
"""

from typing import Optional


def get_group_id(event) -> Optional[str]:
    """安全地获取群组ID

    支持不同类型的事件，包括 WebChatMessageEvent 和 AstrMessageEvent。
    用于处理各种消息事件场景，避免 AttributeError。

    Args:
        event: 消息事件对象（支持多种类型）

    Returns:
        Optional[str]: 群组ID，如果不存在或事件不支持则返回 None

    Examples:
        >>> # 在私聊场景中
        >>> group_id = get_group_id(event)
        >>> print(group_id)  # None

        >>> # 在群聊场景中
        >>> group_id = get_group_id(event)
        >>> print(group_id)  # "123456789"
    """
    try:
        # 检查事件是否支持 get_sender_group_id 方法
        if hasattr(event, 'get_sender_group_id'):
            return event.get_sender_group_id()
        # 事件不支持群组ID（如 WebChat 或私聊）
        return None
    except (AttributeError, Exception):
        # 发生任何异常，返回 None（表示私聊或未知类型）
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
