"""
事件工具模块 - 提供事件处理的通用工具函数
"""

from typing import Optional


def get_group_id(event) -> Optional[str]:
    """安全地获取群组ID

    根据 AstrBot 官方文档，从 AstrMessageEvent 中获取群组ID。
    在群聊场景中，event.group_id 会返回群组ID；私聊场景返回 None 或空字符串。

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
        # 首先尝试从 event.group_id 获取
        if hasattr(event, 'group_id'):
            group_id = event.group_id
            # 如果是有效的群组ID（非空、非None），直接返回
            if group_id:
                return str(group_id) if group_id else None
        
        # 如果 event.group_id 无效，尝试从 unified_msg_origin 解析
        # unified_msg_origin 格式: platform:chat_type:chat_id (如 qq:group:123456 或 qq:private:123456)
        if hasattr(event, 'unified_msg_origin'):
            umo = event.unified_msg_origin
            if umo and isinstance(umo, str):
                parts = umo.split(':')
                if len(parts) >= 3 and parts[1] == 'group':
                    return parts[2]
        
        return None
    except Exception:
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
