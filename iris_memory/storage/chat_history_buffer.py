"""
聊天记录缓冲区 - 维护每个会话的最近N条聊天消息

用于在LLM请求时注入最近的群聊/私聊上下文，
让AI了解当前正在进行的对话话题。

与工作记忆(WorkingMemory)的区别：
- 工作记忆：捕获的高价值"记忆"对象，经过触发器、情感分析等筛选
- 聊天缓冲区：原始消息流的滑动窗口，不做筛选，保持对话连续性
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any

from iris_memory.utils.logger import get_logger
from iris_memory.utils.member_utils import format_member_tag

logger = get_logger("chat_history_buffer")


@dataclass
class ChatMessage:
    """单条聊天消息"""
    sender_id: str
    sender_name: Optional[str]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    group_id: Optional[str] = None
    is_bot: bool = False  # 是否为Bot自己的回复

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "group_id": self.group_id,
            "is_bot": self.is_bot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()
        return cls(
            sender_id=data.get("sender_id", ""),
            sender_name=data.get("sender_name"),
            content=data.get("content", ""),
            timestamp=ts,
            group_id=data.get("group_id"),
            is_bot=data.get("is_bot", False),
        )


class ChatHistoryBuffer:
    """聊天记录缓冲区

    为每个会话(session_key)维护一个固定大小的消息滑动窗口。
    群聊按 group_id 为key，私聊按 user_id:private 为key。

    与 AstrBot 内置的 req.contexts 的关系：
    - req.contexts 只包含"Bot参与的对话"（用户@Bot的消息和Bot的回复）
    - 本缓冲区记录"所有消息"，包括群里其他人的发言
    - 在群聊场景中，本缓冲区提供了Bot未参与的对话上下文
    """

    def __init__(self, max_messages: int = 10):
        """初始化缓冲区

        Args:
            max_messages: 每个会话保留的最大消息数量
        """
        self.max_messages = max_messages
        # session_key -> deque[ChatMessage]
        self._buffers: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    def _get_session_key(self, user_id: str, group_id: Optional[str]) -> str:
        """生成会话键

        群聊：以group_id为维度（同一群的消息共享缓冲区）
        私聊：以user_id为维度
        """
        if group_id:
            return f"group:{group_id}"
        return f"private:{user_id}"

    async def add_message(
        self,
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        group_id: Optional[str] = None,
        is_bot: bool = False,
        session_user_id: Optional[str] = None,
    ) -> None:
        """添加一条消息到缓冲区

        Args:
            sender_id: 发送者ID
            sender_name: 发送者昵称
            content: 消息内容
            group_id: 群组ID（私聊为None）
            is_bot: 是否为Bot的回复
            session_user_id: 用于定位缓冲区的用户ID（私聊时Bot的回复
                需要归入对话用户的缓冲区，传入对话用户的ID）
        """
        if not content or not content.strip():
            return

        # 群聊按group_id，私聊按session_user_id或sender_id
        buffer_user_id = session_user_id or sender_id
        session_key = self._get_session_key(buffer_user_id, group_id)
        msg = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content.strip(),
            group_id=group_id,
            is_bot=is_bot,
        )

        async with self._lock:
            if session_key not in self._buffers:
                self._buffers[session_key] = deque(maxlen=self.max_messages)
            self._buffers[session_key].append(msg)

    async def get_recent_messages(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """获取最近的聊天消息

        Args:
            user_id: 用户ID（私聊时用来定位缓冲区）
            group_id: 群组ID
            limit: 返回数量限制，None则返回全部缓冲区内容

        Returns:
            List[ChatMessage]: 按时间正序排列的消息列表
        """
        session_key = self._get_session_key(user_id, group_id)

        async with self._lock:
            buf = self._buffers.get(session_key)
            if not buf:
                return []
            messages = list(buf)

        if limit and limit < len(messages):
            messages = messages[-limit:]

        return messages

    def format_for_llm(
        self,
        messages: List[ChatMessage],
        group_id: Optional[str] = None,
        bot_name: str = "Bot",
    ) -> str:
        """将聊天记录格式化为LLM可理解的上下文

        Args:
            messages: 消息列表
            group_id: 群组ID
            bot_name: Bot的名称

        Returns:
            str: 格式化的聊天记录文本
        """
        if not messages:
            return ""

        lines = []
        if group_id:
            lines.append("【近期群聊记录】")
            lines.append("以下是群里最近的对话，帮助你了解当前话题：")
        else:
            lines.append("【近期对话记录】")
            lines.append("以下是你们最近的对话：")

        for msg in messages:
            time_str = msg.timestamp.strftime("%H:%M")
            if msg.is_bot:
                sender = bot_name
            elif group_id:
                sender = format_member_tag(msg.sender_name, msg.sender_id, group_id)
            else:
                sender = msg.sender_name or "对方"

            # 截断过长消息
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."

            lines.append(f"[{time_str}] {sender}: {content}")

        return "\n".join(lines)

    def set_max_messages(self, max_messages: int) -> None:
        """更新最大消息数

        Args:
            max_messages: 新的最大消息数
        """
        self.max_messages = max(1, max_messages)
        # 更新已有缓冲区的大小
        for key in self._buffers:
            old = self._buffers[key]
            new_deque = deque(old, maxlen=self.max_messages)
            self._buffers[key] = new_deque

    def clear_session(self, user_id: str, group_id: Optional[str] = None) -> None:
        """清除指定会话的缓冲区"""
        session_key = self._get_session_key(user_id, group_id)
        self._buffers.pop(session_key, None)

    def clear_all(self) -> None:
        """清除所有缓冲区"""
        self._buffers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        total_messages = sum(len(buf) for buf in self._buffers.values())
        return {
            "session_count": len(self._buffers),
            "total_messages": total_messages,
            "max_messages_per_session": self.max_messages,
        }

    async def serialize(self) -> Dict[str, Any]:
        """序列化为可持久化的字典"""
        async with self._lock:
            return {
                "max_messages": self.max_messages,
                "buffers": {
                    key: [msg.to_dict() for msg in buf]
                    for key, buf in self._buffers.items()
                },
            }

    async def deserialize(self, data: Dict[str, Any]) -> None:
        """从字典反序列化"""
        async with self._lock:
            self.max_messages = data.get("max_messages", self.max_messages)
            buffers_data = data.get("buffers", {})
            self._buffers = {}
            for key, msgs in buffers_data.items():
                buf = deque(maxlen=self.max_messages)
                for msg_data in msgs:
                    try:
                        buf.append(ChatMessage.from_dict(msg_data))
                    except Exception as e:
                        logger.warning(f"Skipping malformed chat message during deserialization: {e}")
                        continue
                self._buffers[key] = buf
            logger.info(
                f"Chat history deserialized: {len(self._buffers)} sessions"
            )
