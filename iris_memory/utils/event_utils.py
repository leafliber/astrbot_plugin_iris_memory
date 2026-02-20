"""
事件工具模块 - 提供事件处理的通用工具函数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any

from iris_memory.utils.logger import get_logger

logger = get_logger("event_utils")


@dataclass
class ReplyInfo:
    """从 AstrBot Reply 组件中提取的引用消息信息

    Attributes:
        message_id: 被引用消息的 ID
        sender_id: 被引用消息发送者 ID
        sender_nickname: 被引用消息发送者昵称
        content: 被引用消息的纯文本内容
        timestamp: 被引用消息的发送时间戳（Unix 秒）
    """
    message_id: str
    sender_id: Optional[str] = None
    sender_nickname: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[int] = None

    def format_for_prompt(self, max_length: int = 200) -> str:
        """格式化为可注入 LLM 系统提示词的文本

        Args:
            max_length: 引用内容最大截取长度

        Returns:
            格式化后的引用描述文本
        """
        sender = self.sender_nickname or self.sender_id or "某人"
        content = self.content or "（内容不可用）"
        if len(content) > max_length:
            content = content[:max_length] + "..."
        return f"[引用 {sender} 的消息]: {content}"

    def format_for_buffer(self, max_length: int = 150) -> str:
        """格式化为聊天缓冲区的简短标记

        Args:
            max_length: 引用内容最大截取长度

        Returns:
            格式化后的简短引用标记
        """
        sender = self.sender_nickname or self.sender_id or "某人"
        content = self.content or ""
        if len(content) > max_length:
            content = content[:max_length] + "..."
        if content:
            return f"↩️回复{sender}「{content}」"
        return f"↩️回复{sender}的消息"


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


def extract_reply_info(event) -> Optional[ReplyInfo]:
    """从 AstrBot 消息事件中提取引用消息信息

    遍历 ``event.message_obj.message`` 消息链，查找 ``Reply`` 组件并
    提取被引用消息的发送者、内容等信息。

    AstrBot Reply 组件结构::

        class Reply(BaseMessageComponent):
            id: str | int                                    # 被引用消息 ID
            chain: list[BaseMessageComponent] | None = []    # 被引用消息段列表
            sender_id: int | None | str = 0                 # 发送者 ID
            sender_nickname: str | None = ""                 # 发送者昵称
            time: int | None = 0                             # 发送时间
            message_str: str | None = ""                     # 纯文本字符串

    Args:
        event: AstrMessageEvent 消息事件对象

    Returns:
        Optional[ReplyInfo]: 提取到的引用消息信息，无引用则返回 None

    Examples:
        >>> reply = extract_reply_info(event)
        >>> if reply:
        ...     print(reply.format_for_prompt())
        [引用 张三 的消息]: 今天天气真好
    """
    try:
        if not hasattr(event, 'message_obj') or not event.message_obj:
            return None
        message_chain = getattr(event.message_obj, 'message', None)
        if not message_chain or not isinstance(message_chain, (list, tuple)):
            return None

        for component in message_chain:
            # 通过类名判断是否为 Reply 组件（最可靠）
            if type(component).__name__ == 'Reply':
                pass  # 匹配成功，继续处理
            else:
                # 备用判断：通过 type 属性的枚举名
                comp_type = getattr(component, 'type', None)
                if comp_type is None:
                    continue
                type_name = (
                    getattr(comp_type, 'name', '')
                    or getattr(comp_type, 'value', '')
                    or ''
                )
                if type_name.lower() != 'reply':
                    continue

            # 找到 Reply 组件 —— 提取字段
            msg_id = getattr(component, 'id', '') or ''
            sender_id_raw = getattr(component, 'sender_id', None)
            sender_nickname = getattr(component, 'sender_nickname', None) or None
            time_val = getattr(component, 'time', None)
            message_str = getattr(component, 'message_str', None) or None

            # 如果 message_str 为空，尝试从 chain 拼接纯文本
            if not message_str:
                chain = getattr(component, 'chain', None)
                if chain and isinstance(chain, (list, tuple)):
                    text_parts: List[str] = []
                    for seg in chain:
                        seg_type_name = type(seg).__name__
                        if seg_type_name == 'Plain':
                            text = getattr(seg, 'text', '')
                            if text:
                                text_parts.append(str(text))
                        elif seg_type_name == 'Image':
                            text_parts.append('[图片]')
                        elif seg_type_name == 'At':
                            at_name = getattr(seg, 'name', '') or getattr(seg, 'qq', '')
                            text_parts.append(f'@{at_name}')
                        elif seg_type_name == 'Face':
                            text_parts.append('[表情]')
                    if text_parts:
                        message_str = ''.join(text_parts)

            # 处理 sender_id
            sender_id_str: Optional[str] = None
            if sender_id_raw and sender_id_raw != 0:
                sender_id_str = str(sender_id_raw)

            # 清理空昵称
            if sender_nickname and not sender_nickname.strip():
                sender_nickname = None
            elif sender_nickname:
                sender_nickname = sender_nickname.strip()

            # 清理空内容
            if message_str and not message_str.strip():
                message_str = None
            elif message_str:
                message_str = message_str.strip()

            # 时间戳
            timestamp: Optional[int] = None
            if time_val and isinstance(time_val, (int, float)) and time_val > 0:
                timestamp = int(time_val)

            return ReplyInfo(
                message_id=str(msg_id),
                sender_id=sender_id_str,
                sender_nickname=sender_nickname,
                content=message_str,
                timestamp=timestamp,
            )

        return None
    except Exception as e:
        logger.debug(f"Failed to extract reply info from event: {e}")
        return None


def extract_reply_info(event) -> Optional[ReplyInfo]:
    """从 AstrBot 消息事件中提取引用消息信息

    遍历 ``event.message_obj.message`` 消息链，查找 ``Reply`` 组件并
    提取被引用消息的发送者、内容等信息。

    AstrBot Reply 组件结构::

        class Reply(BaseMessageComponent):
            id: str | int                                    # 被引用消息 ID
            chain: list[BaseMessageComponent] | None = []    # 被引用消息段列表
            sender_id: int | None | str = 0                 # 发送者 ID
            sender_nickname: str | None = ""                 # 发送者昵称
            time: int | None = 0                             # 发送时间
            message_str: str | None = ""                     # 纯文本字符串

    Args:
        event: AstrMessageEvent 消息事件对象

    Returns:
        Optional[ReplyInfo]: 提取到的引用消息信息，无引用则返回 None

    Examples:
        >>> reply = extract_reply_info(event)
        >>> if reply:
        ...     print(reply.format_for_prompt())
        [引用 张三 的消息]: 今天天气真好
    """
    try:
        if not hasattr(event, 'message_obj') or not event.message_obj:
            return None
        message_chain = getattr(event.message_obj, 'message', None)
        if not message_chain or not isinstance(message_chain, (list, tuple)):
            return None

        for component in message_chain:
            comp_type = getattr(component, 'type', None)
            if comp_type is None:
                continue
            # 兼容多种判断方式：枚举值名称 / 类名
            type_name = (
                getattr(comp_type, 'name', '')
                or getattr(comp_type, 'value', '')
                or type(component).__name__
            )
            if type_name.lower() not in ('reply', 'Reply'):
                if type(component).__name__ != 'Reply':
                    continue

            # 找到 Reply 组件 —— 提取字段
            msg_id = getattr(component, 'id', '') or ''
            sender_id_raw = getattr(component, 'sender_id', None)
            sender_nickname = getattr(component, 'sender_nickname', None) or None
            time_val = getattr(component, 'time', None)
            message_str = getattr(component, 'message_str', None) or None

            # 如果 message_str 为空，尝试从 chain 拼接纯文本
            if not message_str:
                chain = getattr(component, 'chain', None)
                if chain and isinstance(chain, (list, tuple)):
                    text_parts: List[str] = []
                    for seg in chain:
                        seg_type_name = type(seg).__name__
                        if seg_type_name == 'Plain':
                            text = getattr(seg, 'text', '')
                            if text:
                                text_parts.append(str(text))
                        elif seg_type_name == 'Image':
                            text_parts.append('[图片]')
                        elif seg_type_name == 'At':
                            at_name = getattr(seg, 'name', '') or getattr(seg, 'qq', '')
                            text_parts.append(f'@{at_name}')
                        elif seg_type_name == 'Face':
                            text_parts.append('[表情]')
                    if text_parts:
                        message_str = ''.join(text_parts)

            # 处理 sender_id
            sender_id_str: Optional[str] = None
            if sender_id_raw and sender_id_raw != 0:
                sender_id_str = str(sender_id_raw)

            # 清理空昵称
            if sender_nickname and not sender_nickname.strip():
                sender_nickname = None
            elif sender_nickname:
                sender_nickname = sender_nickname.strip()

            # 清理空内容
            if message_str and not message_str.strip():
                message_str = None
            elif message_str:
                message_str = message_str.strip()

            # 时间戳
            timestamp: Optional[int] = None
            if time_val and isinstance(time_val, (int, float)) and time_val > 0:
                timestamp = int(time_val)

            return ReplyInfo(
                message_id=str(msg_id),
                sender_id=sender_id_str,
                sender_nickname=sender_nickname,
                content=message_str,
                timestamp=timestamp,
            )

        return None
    except Exception as e:
        logger.debug(f"Failed to extract reply info from event: {e}")
        return None
