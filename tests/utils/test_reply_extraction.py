"""
引用消息提取 (extract_reply_info) 单元测试

测试从 AstrBot 消息链的 Reply 组件中提取被引用消息信息的功能。
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum

from iris_memory.utils.event_utils import extract_reply_info, ReplyInfo


# ── 模拟 AstrBot 组件 ──


class ComponentType(Enum):
    """模拟 AstrBot 的 ComponentType 枚举"""
    Plain = "Plain"
    Image = "Image"
    At = "At"
    Reply = "Reply"
    Face = "Face"


@dataclass
class Plain:
    text: str = ""
    type: ComponentType = ComponentType.Plain


@dataclass
class Image:
    url: str = ""
    file: str = ""
    type: ComponentType = ComponentType.Image


@dataclass
class At:
    qq: str = ""
    name: str = ""
    type: ComponentType = ComponentType.At


@dataclass
class Face:
    id: int = 0
    type: ComponentType = ComponentType.Face


@dataclass
class Reply:
    """模拟 AstrBot Reply 组件"""
    id: str = ""
    chain: Optional[List[Any]] = field(default_factory=list)
    sender_id: Any = 0
    sender_nickname: str = ""
    time: int = 0
    message_str: str = ""
    type: ComponentType = ComponentType.Reply


@dataclass
class MessageObj:
    message: List[Any] = field(default_factory=list)
    group_id: Optional[str] = None


@dataclass
class MockEvent:
    """模拟 AstrMessageEvent"""
    message_obj: Optional[MessageObj] = None
    message_str: str = ""


# ── 测试 ──


class TestReplyInfo:
    """ReplyInfo 数据类测试"""

    def test_basic_creation(self):
        info = ReplyInfo(message_id="123", sender_nickname="张三", content="你好")
        assert info.message_id == "123"
        assert info.sender_nickname == "张三"
        assert info.content == "你好"
        assert info.sender_id is None
        assert info.timestamp is None

    def test_format_for_prompt(self):
        info = ReplyInfo(message_id="1", sender_nickname="张三", content="今天天气真好")
        result = info.format_for_prompt()
        assert "[引用 张三 的消息]" in result
        assert "今天天气真好" in result

    def test_format_for_prompt_no_nickname(self):
        info = ReplyInfo(message_id="1", sender_id="12345", content="Hello")
        result = info.format_for_prompt()
        assert "[引用 12345 的消息]" in result
        assert "Hello" in result

    def test_format_for_prompt_no_sender(self):
        info = ReplyInfo(message_id="1", content="Test")
        result = info.format_for_prompt()
        assert "[引用 某人 的消息]" in result

    def test_format_for_prompt_no_content(self):
        info = ReplyInfo(message_id="1", sender_nickname="张三")
        result = info.format_for_prompt()
        assert "（内容不可用）" in result

    def test_format_for_prompt_truncation(self):
        long_text = "a" * 300
        info = ReplyInfo(message_id="1", sender_nickname="张三", content=long_text)
        result = info.format_for_prompt(max_length=50)
        assert "..." in result
        assert len(result) < 300

    def test_format_for_buffer(self):
        info = ReplyInfo(message_id="1", sender_nickname="张三", content="你好吗")
        result = info.format_for_buffer()
        assert "↩️回复张三" in result
        assert "「你好吗」" in result

    def test_format_for_buffer_no_content(self):
        info = ReplyInfo(message_id="1", sender_nickname="张三")
        result = info.format_for_buffer()
        assert "↩️回复张三的消息" in result

    def test_format_for_buffer_truncation(self):
        long_text = "b" * 200
        info = ReplyInfo(message_id="1", sender_nickname="A", content=long_text)
        result = info.format_for_buffer(max_length=50)
        assert "..." in result


class TestExtractReplyInfo:
    """extract_reply_info 函数测试"""

    def test_no_event(self):
        """无事件对象"""
        assert extract_reply_info(None) is None

    def test_no_message_obj(self):
        """事件无 message_obj"""
        event = MockEvent(message_obj=None)
        assert extract_reply_info(event) is None

    def test_empty_message_chain(self):
        """空消息链"""
        event = MockEvent(message_obj=MessageObj(message=[]))
        assert extract_reply_info(event) is None

    def test_no_reply_in_chain(self):
        """消息链中无 Reply 组件"""
        event = MockEvent(
            message_obj=MessageObj(message=[
                Plain(text="Hello"),
                Image(url="https://example.com/img.jpg"),
            ])
        )
        assert extract_reply_info(event) is None

    def test_basic_reply_extraction(self):
        """基本引用消息提取"""
        reply = Reply(
            id="msg_100",
            sender_id=12345,
            sender_nickname="张三",
            message_str="今天吃什么",
            time=1700000000,
        )
        event = MockEvent(
            message_obj=MessageObj(message=[
                reply,
                Plain(text="吃火锅吧"),
            ])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.message_id == "msg_100"
        assert info.sender_id == "12345"
        assert info.sender_nickname == "张三"
        assert info.content == "今天吃什么"
        assert info.timestamp == 1700000000

    def test_reply_with_empty_message_str(self):
        """message_str 为空时从 chain 拼接"""
        reply = Reply(
            id="200",
            sender_id="user_a",
            sender_nickname="李四",
            message_str="",
            chain=[
                Plain(text="看看这个"),
                Image(url="https://example.com/img.jpg"),
                Plain(text="好不好看"),
            ],
        )
        event = MockEvent(
            message_obj=MessageObj(message=[reply, Plain(text="好看！")])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.content == "看看这个[图片]好不好看"

    def test_reply_chain_with_at_and_face(self):
        """chain 中包含 At 和 Face 组件"""
        reply = Reply(
            id="300",
            sender_id="user_b",
            sender_nickname="王五",
            message_str="",
            chain=[
                At(qq="999", name="小明"),
                Plain(text=" 你来看看"),
                Face(id=21),
            ],
        )
        event = MockEvent(
            message_obj=MessageObj(message=[reply])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert "@小明" in info.content
        assert "你来看看" in info.content
        assert "[表情]" in info.content

    def test_sender_id_zero_treated_as_none(self):
        """sender_id 为 0 时应视为 None"""
        reply = Reply(id="400", sender_id=0, sender_nickname="Test", message_str="hi")
        event = MockEvent(
            message_obj=MessageObj(message=[reply])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.sender_id is None

    def test_empty_nickname_treated_as_none(self):
        """空昵称应视为 None"""
        reply = Reply(id="500", sender_id=111, sender_nickname="  ", message_str="test")
        event = MockEvent(
            message_obj=MessageObj(message=[reply])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.sender_nickname is None

    def test_empty_content_treated_as_none(self):
        """空内容应视为 None"""
        reply = Reply(id="600", sender_id=111, sender_nickname="A", message_str="  ")
        event = MockEvent(
            message_obj=MessageObj(message=[reply])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.content is None

    def test_reply_not_first_in_chain(self):
        """Reply 组件不在消息链第一位"""
        reply = Reply(id="700", sender_nickname="Bob", message_str="hello")
        event = MockEvent(
            message_obj=MessageObj(message=[
                Plain(text="prefix"),
                reply,
                Plain(text="suffix"),
            ])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.sender_nickname == "Bob"
        assert info.content == "hello"

    def test_invalid_time_value(self):
        """无效时间戳（0或负数）"""
        reply = Reply(id="800", sender_nickname="C", message_str="x", time=0)
        event = MockEvent(message_obj=MessageObj(message=[reply]))
        info = extract_reply_info(event)
        assert info is not None
        assert info.timestamp is None

        reply2 = Reply(id="801", sender_nickname="C", message_str="x", time=-1)
        event2 = MockEvent(message_obj=MessageObj(message=[reply2]))
        info2 = extract_reply_info(event2)
        assert info2.timestamp is None

    def test_only_first_reply_extracted(self):
        """如果有多个 Reply 组件，只取第一个"""
        reply1 = Reply(id="901", sender_nickname="First", message_str="first msg")
        reply2 = Reply(id="902", sender_nickname="Second", message_str="second msg")
        event = MockEvent(
            message_obj=MessageObj(message=[reply1, reply2, Plain(text="test")])
        )
        info = extract_reply_info(event)
        assert info is not None
        assert info.sender_nickname == "First"
        assert info.content == "first msg"

    def test_message_chain_not_list(self):
        """message 属性不是列表"""
        msg_obj = MessageObj()
        msg_obj.message = "not a list"  # type: ignore
        event = MockEvent(message_obj=msg_obj)
        assert extract_reply_info(event) is None

    def test_exception_safety(self):
        """组件属性访问异常时不崩溃"""

        class BrokenComponent:
            @property
            def type(self):
                raise RuntimeError("boom")

        event = MockEvent(
            message_obj=MessageObj(message=[BrokenComponent()])
        )
        # 应该返回 None 而不是抛出异常
        result = extract_reply_info(event)
        assert result is None


class TestChatBufferReplyIntegration:
    """聊天缓冲区引用消息集成测试"""

    @pytest.fixture
    def buffer(self):
        from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer
        return ChatHistoryBuffer(max_messages=10)

    @pytest.mark.asyncio
    async def test_add_message_with_reply(self, buffer):
        """带引用信息的消息正确存储"""
        await buffer.add_message(
            sender_id="u1",
            sender_name="Alice",
            content="我觉得也是",
            group_id="g1",
            reply_sender_name="Bob",
            reply_sender_id="u2",
            reply_content="今天天气不错",
        )
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg.has_reply is True
        assert msg.reply_sender_name == "Bob"
        assert msg.reply_sender_id == "u2"
        assert msg.reply_content == "今天天气不错"

    @pytest.mark.asyncio
    async def test_add_message_without_reply(self, buffer):
        """无引用的消息 has_reply 为 False"""
        await buffer.add_message(
            sender_id="u1",
            sender_name="Alice",
            content="Hello",
            group_id="g1",
        )
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 1
        assert msgs[0].has_reply is False

    @pytest.mark.asyncio
    async def test_format_for_llm_with_reply(self, buffer):
        """格式化时正确显示引用标记"""
        await buffer.add_message(
            sender_id="u1",
            sender_name="Alice",
            content="同意",
            group_id="g1",
            reply_sender_name="Bob",
            reply_content="我们去吃火锅吧",
        )
        await buffer.add_message(
            sender_id="u2",
            sender_name="Bob",
            content="那走",
            group_id="g1",
        )
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        formatted = buffer.format_for_llm(msgs, group_id="g1")
        assert "↩️回复Bob" in formatted
        assert "「我们去吃火锅吧」" in formatted
        assert "同意" in formatted
        # 无引用的消息不应有回复标记
        assert formatted.count("↩️") == 1

    @pytest.mark.asyncio
    async def test_format_for_llm_reply_content_truncated(self, buffer):
        """格式化时引用内容超长被截断"""
        long_ref = "x" * 200
        await buffer.add_message(
            sender_id="u1",
            sender_name="Alice",
            content="ok",
            group_id="g1",
            reply_sender_name="Bob",
            reply_content=long_ref,
        )
        msgs = await buffer.get_recent_messages("u1", group_id="g1")
        formatted = buffer.format_for_llm(msgs, group_id="g1")
        assert "..." in formatted

    @pytest.mark.asyncio
    async def test_serialize_deserialize_with_reply(self, buffer):
        """带引用信息的消息序列化和反序列化"""
        await buffer.add_message(
            sender_id="u1",
            sender_name="Alice",
            content="是的",
            group_id="g1",
            reply_sender_name="Bob",
            reply_sender_id="u2",
            reply_content="对不对？",
        )
        data = await buffer.serialize()

        from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer
        buffer2 = ChatHistoryBuffer(max_messages=10)
        await buffer2.deserialize(data)

        msgs = await buffer2.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg.has_reply is True
        assert msg.reply_sender_name == "Bob"
        assert msg.reply_sender_id == "u2"
        assert msg.reply_content == "对不对？"

    @pytest.mark.asyncio
    async def test_serialize_backward_compat(self, buffer):
        """旧格式（无reply字段）的反序列化保持兼容"""
        old_data = {
            "max_messages": 10,
            "buffers": {
                "group:g1": [
                    {
                        "sender_id": "u1",
                        "sender_name": "Alice",
                        "content": "old message",
                        "timestamp": "2025-01-01T12:00:00",
                        "group_id": "g1",
                        "is_bot": False,
                    }
                ]
            },
        }
        from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer
        buffer2 = ChatHistoryBuffer(max_messages=10)
        await buffer2.deserialize(old_data)
        msgs = await buffer2.get_recent_messages("u1", group_id="g1")
        assert len(msgs) == 1
        assert msgs[0].has_reply is False
        assert msgs[0].reply_sender_name is None
