"""
聊天记录缓冲区（ChatHistoryBuffer）单元测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from iris_memory.storage.chat_history_buffer import (
    ChatHistoryBuffer,
    ChatMessage,
)


class TestChatMessage:
    """ChatMessage数据类测试"""

    def test_create_message(self):
        msg = ChatMessage(
            sender_id="user1",
            sender_name="Alice",
            content="Hello!",
            group_id="group1",
        )
        assert msg.sender_id == "user1"
        assert msg.sender_name == "Alice"
        assert msg.content == "Hello!"
        assert msg.group_id == "group1"
        assert msg.is_bot is False
        assert isinstance(msg.timestamp, datetime)

    def test_bot_message(self):
        msg = ChatMessage(
            sender_id="bot",
            sender_name=None,
            content="Hi there!",
            is_bot=True,
        )
        assert msg.is_bot is True
        assert msg.sender_name is None

    def test_serialize_deserialize(self):
        msg = ChatMessage(
            sender_id="user1",
            sender_name="Alice",
            content="Test message",
            group_id="group1",
            is_bot=False,
        )
        d = msg.to_dict()
        restored = ChatMessage.from_dict(d)
        assert restored.sender_id == msg.sender_id
        assert restored.sender_name == msg.sender_name
        assert restored.content == msg.content
        assert restored.group_id == msg.group_id
        assert restored.is_bot == msg.is_bot

    def test_from_dict_missing_fields(self):
        msg = ChatMessage.from_dict({"content": "hi"})
        assert msg.sender_id == ""
        assert msg.sender_name is None
        assert msg.content == "hi"


class TestChatHistoryBuffer:
    """ChatHistoryBuffer核心功能测试"""

    @pytest.fixture
    def buffer(self):
        return ChatHistoryBuffer(max_messages=5)

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, buffer):
        await buffer.add_message("u1", "Alice", "msg1", group_id="g1")
        await buffer.add_message("u2", "Bob", "msg2", group_id="g1")
        await buffer.add_message("u1", "Alice", "msg3", group_id="g1")

        messages = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(messages) == 3
        assert messages[0].content == "msg1"
        assert messages[2].content == "msg3"

    @pytest.mark.asyncio
    async def test_max_messages_limit(self, buffer):
        """超过max_messages时自动淘汰旧消息"""
        for i in range(8):
            await buffer.add_message("u1", "A", f"msg{i}", group_id="g1")

        messages = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(messages) == 5
        # 应保留最新的5条
        assert messages[0].content == "msg3"
        assert messages[-1].content == "msg7"

    @pytest.mark.asyncio
    async def test_group_session_isolation(self, buffer):
        """不同群聊的消息相互隔离"""
        await buffer.add_message("u1", "A", "g1 msg", group_id="group1")
        await buffer.add_message("u2", "B", "g2 msg", group_id="group2")

        g1_msgs = await buffer.get_recent_messages("u1", group_id="group1")
        g2_msgs = await buffer.get_recent_messages("u2", group_id="group2")
        assert len(g1_msgs) == 1
        assert len(g2_msgs) == 1
        assert g1_msgs[0].content == "g1 msg"
        assert g2_msgs[0].content == "g2 msg"

    @pytest.mark.asyncio
    async def test_group_shared_buffer(self, buffer):
        """同一群内不同用户共享缓冲区"""
        await buffer.add_message("u1", "Alice", "hello", group_id="g1")
        await buffer.add_message("u2", "Bob", "hi", group_id="g1")

        # 任一用户都能看到群内所有消息
        msgs_u1 = await buffer.get_recent_messages("u1", group_id="g1")
        msgs_u2 = await buffer.get_recent_messages("u2", group_id="g1")
        assert len(msgs_u1) == 2
        assert len(msgs_u2) == 2

    @pytest.mark.asyncio
    async def test_private_chat_isolation(self, buffer):
        """私聊按用户隔离"""
        await buffer.add_message("u1", "Alice", "private msg 1")
        await buffer.add_message("u2", "Bob", "private msg 2")

        msgs_u1 = await buffer.get_recent_messages("u1")
        msgs_u2 = await buffer.get_recent_messages("u2")
        assert len(msgs_u1) == 1
        assert len(msgs_u2) == 1
        assert msgs_u1[0].content == "private msg 1"

    @pytest.mark.asyncio
    async def test_get_with_limit(self, buffer):
        for i in range(5):
            await buffer.add_message("u1", "A", f"msg{i}", group_id="g1")

        msgs = await buffer.get_recent_messages("u1", group_id="g1", limit=3)
        assert len(msgs) == 3
        # 返回最新的3条
        assert msgs[0].content == "msg2"
        assert msgs[2].content == "msg4"

    @pytest.mark.asyncio
    async def test_empty_content_filtered(self, buffer):
        """空消息不应被记录"""
        await buffer.add_message("u1", "A", "", group_id="g1")
        await buffer.add_message("u1", "A", "  ", group_id="g1")
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_clear_session(self, buffer):
        await buffer.add_message("u1", "A", "msg1", group_id="g1")
        await buffer.add_message("u1", "A", "msg2", group_id="g1")
        buffer.clear_session("u1", group_id="g1")
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_clear_all(self, buffer):
        await buffer.add_message("u1", "A", "msg1", group_id="g1")
        await buffer.add_message("u2", "B", "msg2")
        buffer.clear_all()
        assert len(await buffer.get_recent_messages("u1", group_id="g1")) == 0
        assert len(await buffer.get_recent_messages("u2")) == 0


class TestFormatForLLM:
    """format_for_llm格式化测试"""

    @pytest.fixture
    def buffer(self):
        return ChatHistoryBuffer(max_messages=10)

    @pytest.mark.asyncio
    async def test_group_format(self, buffer):
        await buffer.add_message("u1", "Alice", "大家好", group_id="g1")
        await buffer.add_message("u2", "Bob", "你好", group_id="g1")
        await buffer.add_message("bot", None, "有什么可以帮忙的？", group_id="g1", is_bot=True)

        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        formatted = buffer.format_for_llm(msgs, group_id="g1", bot_name="Chito")

        assert "【近期群聊记录】" in formatted
        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "Chito" in formatted
        assert "大家好" in formatted

    @pytest.mark.asyncio
    async def test_private_format(self, buffer):
        await buffer.add_message("u1", "Alice", "你好")
        await buffer.add_message("bot", None, "你好呀", is_bot=True, session_user_id="u1")

        msgs = await buffer.get_recent_messages("u1")
        formatted = buffer.format_for_llm(msgs, bot_name="Bot")

        assert "【近期对话记录】" in formatted
        assert "Bot" in formatted
        assert "你好" in formatted
        assert "你好呀" in formatted

    @pytest.mark.asyncio
    async def test_long_message_truncation(self, buffer):
        long_msg = "a" * 300
        await buffer.add_message("u1", "A", long_msg, group_id="g1")
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        formatted = buffer.format_for_llm(msgs, group_id="g1")
        # 应被截断到200+...
        assert "..." in formatted

    def test_empty_messages(self, buffer):
        formatted = buffer.format_for_llm([], group_id="g1")
        assert formatted == ""


class TestSetMaxMessages:
    """动态调整max_messages测试"""

    @pytest.mark.asyncio
    async def test_resize_shrink(self):
        buffer = ChatHistoryBuffer(max_messages=10)
        for i in range(10):
            await buffer.add_message("u1", "A", f"msg{i}", group_id="g1")

        buffer.set_max_messages(3)
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 3
        assert msgs[-1].content == "msg9"

    @pytest.mark.asyncio
    async def test_resize_expand(self):
        buffer = ChatHistoryBuffer(max_messages=3)
        for i in range(3):
            await buffer.add_message("u1", "A", f"msg{i}", group_id="g1")

        buffer.set_max_messages(10)
        # 已有的消息不变
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 3

        # 现在可以容纳更多
        for i in range(3, 10):
            await buffer.add_message("u1", "A", f"msg{i}", group_id="g1")
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 10


class TestSerializationPersistence:
    """序列化/反序列化持久化测试"""

    @pytest.mark.asyncio
    async def test_serialize_deserialize_roundtrip(self):
        buf1 = ChatHistoryBuffer(max_messages=5)
        await buf1.add_message("u1", "Alice", "msg1", group_id="g1")
        await buf1.add_message("u2", "Bob", "msg2", group_id="g1")
        await buf1.add_message("u1", "Alice", "private msg")

        data = await buf1.serialize()

        buf2 = ChatHistoryBuffer(max_messages=5)
        await buf2.deserialize(data)

        g1_msgs = await buf2.get_recent_messages("u1", group_id="g1")
        assert len(g1_msgs) == 2
        assert g1_msgs[0].content == "msg1"

        priv_msgs = await buf2.get_recent_messages("u1")
        assert len(priv_msgs) == 1
        assert priv_msgs[0].content == "private msg"

    @pytest.mark.asyncio
    async def test_deserialize_empty(self):
        buf = ChatHistoryBuffer(max_messages=5)
        await buf.deserialize({})
        msgs = await buf.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 0


class TestStats:
    """统计信息测试"""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        buf = ChatHistoryBuffer(max_messages=10)
        await buf.add_message("u1", "A", "msg1", group_id="g1")
        await buf.add_message("u2", "B", "msg2", group_id="g2")
        await buf.add_message("u1", "A", "msg3")

        stats = buf.get_stats()
        assert stats["session_count"] == 3
        assert stats["total_messages"] == 3
        assert stats["max_messages_per_session"] == 10
