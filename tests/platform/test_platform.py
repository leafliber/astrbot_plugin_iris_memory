"""平台适配器测试"""

import pytest
from unittest.mock import Mock, AsyncMock
from iris_memory.platform.base import (
    PlatformAdapter,
    ReplyInfo,
    UnsupportedPlatformError,
)
from iris_memory.platform.factory import get_adapter
from iris_memory.platform.generic import GenericAdapter
from iris_memory.platform.qq import OneBot11Adapter


class TestReplyInfo:
    """ReplyInfo 数据类测试"""

    def test_default_empty(self):
        """测试默认空值"""
        info = ReplyInfo()
        assert info.message_id == ""
        assert info.user_id == ""
        assert info.user_name == ""
        assert info.content == ""
        assert info.has_reply is False

    def test_has_reply_with_message_id(self):
        """测试有 message_id 时 has_reply 为 True"""
        info = ReplyInfo(message_id="6283")
        assert info.has_reply is True

    def test_has_reply_without_message_id(self):
        """测试无 message_id 时 has_reply 为 False"""
        info = ReplyInfo(user_id="123")
        assert info.has_reply is False

    def test_full_reply_info(self):
        """测试完整的回复信息"""
        info = ReplyInfo(
            message_id="6283", user_id="1234567", user_name="张三", content="你好"
        )
        assert info.message_id == "6283"
        assert info.user_id == "1234567"
        assert info.user_name == "张三"
        assert info.content == "你好"
        assert info.has_reply is True


class TestUnsupportedPlatformError:
    """UnsupportedPlatformError 测试"""

    def test_error_message(self):
        """测试错误消息"""
        error = UnsupportedPlatformError("wechat", "当前仅支持 QQ 平台")

        assert error.platform_type == "wechat"
        assert error.message == "当前仅支持 QQ 平台"
        assert str(error) == "当前仅支持 QQ 平台"

    def test_default_message(self):
        """测试默认消息"""
        error = UnsupportedPlatformError("wechat")

        assert error.platform_type == "wechat"
        assert "wechat" in error.message


class TestOneBot11Adapter:
    """OneBot11Adapter 测试"""

    def test_get_user_id(self):
        """测试获取用户ID"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = "12345"

        user_id = adapter.get_user_id(event)

        assert user_id == "12345"

    def test_get_group_id(self):
        """测试获取群ID"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.group_id = "group_123"
        event.message_obj.sender = Mock()

        group_id = adapter.get_group_id(event)

        assert group_id == "group_123"

    def test_get_username(self):
        """测试获取用户名"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.nickname = "测试用户"

        username = adapter.get_user_name(event)

        assert username == "测试用户"

    def test_is_group_message_true(self):
        """测试群聊判断 - 群聊"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.group_id = "group_123"
        event.message_obj.sender = Mock()

        assert adapter.is_group_message(event)

    def test_is_group_message_false(self):
        """测试群聊判断 - 私聊"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()

        assert not adapter.is_group_message(event)

    def test_get_reply_info_with_reply_segment(self):
        """测试从数组格式消息段提取回复信息"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": [
                {"type": "reply", "data": {"id": "6283"}},
                {"type": "text", "data": {"text": "我也觉得"}},
            ],
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is True
        assert reply_info.message_id == "6283"

    def test_get_reply_info_with_full_reply_data(self):
        """测试提取完整的回复信息（go-cqhttp 扩展格式）"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": [
                {
                    "type": "reply",
                    "data": {
                        "id": "6283",
                        "user_id": "1234567",
                        "sender": {"nickname": "张三"},
                        "content": [{"type": "text", "data": {"text": "你好啊"}}],
                    },
                },
                {"type": "text", "data": {"text": "我也觉得"}},
            ],
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is True
        assert reply_info.message_id == "6283"
        assert reply_info.user_id == "1234567"
        assert reply_info.user_name == "张三"
        assert reply_info.content == "你好啊"

    def test_get_reply_info_with_string_content(self):
        """测试回复消息内容为字符串格式"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": [
                {"type": "reply", "data": {"id": "6283", "content": "你好啊"}},
                {"type": "text", "data": {"text": "我也觉得"}},
            ],
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is True
        assert reply_info.content == "你好啊"

    def test_get_reply_info_no_reply(self):
        """测试非回复消息返回空 ReplyInfo"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": [{"type": "text", "data": {"text": "你好"}}],
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is False

    def test_get_reply_info_cq_code_format(self):
        """测试从 CQ 码格式提取回复信息"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": "[CQ:reply,id=6283]我也觉得",
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is True
        assert reply_info.message_id == "6283"

    def test_get_reply_info_empty_raw_message(self):
        """测试原始消息为空时返回空 ReplyInfo"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = None

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is False


class TestGetAdapter:
    """get_adapter 工厂方法测试"""

    def test_get_onebot11_adapter_via_get_platform_name(self):
        """测试获取 OneBot11 适配器 - 通过 event.get_platform_name()"""
        event = Mock()
        event.get_platform_name = Mock(return_value="qq")

        adapter = get_adapter(event)

        assert isinstance(adapter, OneBot11Adapter)

    def test_get_onebot11_adapter_custom_instance_name(self):
        """测试用户自定义平台实例名仍能正确识别协议类型"""
        event = Mock()
        # 用户在 AstrBot 中将实例命名为 "yuki"，但协议类型是 aiocqhttp
        event.get_platform_name = Mock(return_value="aiocqhttp")

        adapter = get_adapter(event)

        assert isinstance(adapter, OneBot11Adapter)

    def test_unsupported_platform_returns_generic_adapter(self):
        """测试未支持的平台返回通用适配器"""
        event = Mock()
        event.get_platform_name = Mock(return_value="wechat")

        adapter = get_adapter(event)

        assert isinstance(adapter, GenericAdapter)

    def test_adapter_is_singleton(self):
        """测试适配器是单例"""
        event1 = Mock()
        event1.get_platform_name = Mock(return_value="qq")

        event2 = Mock()
        event2.get_platform_name = Mock(return_value="qq")

        adapter1 = get_adapter(event1)
        adapter2 = get_adapter(event2)

        assert adapter1 is adapter2


class TestGetMsgById:
    """OneBot11Adapter.get_msg_by_id 测试"""

    @pytest.mark.asyncio
    async def test_get_msg_by_id_success(self):
        """测试成功获取消息内容"""
        adapter = OneBot11Adapter()

        bot = Mock()
        bot.call_action = AsyncMock(
            return_value={
                "message_id": 6283,
                "sender": {
                    "user_id": 1234567,
                    "nickname": "张三",
                    "card": "",
                },
                "message": [{"type": "text", "data": {"text": "你好啊"}}],
            }
        )

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.has_reply is True
        assert result.message_id == "6283"
        assert result.content == "你好啊"
        assert result.user_name == "张三"
        assert result.user_id == "1234567"
        bot.call_action.assert_called_once_with("get_msg", message_id=6283)

    @pytest.mark.asyncio
    async def test_get_msg_by_id_with_card(self):
        """测试群名片优先于昵称"""
        adapter = OneBot11Adapter()

        bot = Mock()
        bot.call_action = AsyncMock(
            return_value={
                "message_id": 6283,
                "sender": {
                    "user_id": 1234567,
                    "nickname": "张三",
                    "card": "三哥",
                },
                "message": [{"type": "text", "data": {"text": "你好"}}],
            }
        )

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.user_name == "三哥"

    @pytest.mark.asyncio
    async def test_get_msg_by_id_raw_message_fallback(self):
        """测试 message 为空时回退到 raw_message"""
        adapter = OneBot11Adapter()

        bot = Mock()
        bot.call_action = AsyncMock(
            return_value={
                "message_id": 6283,
                "sender": {
                    "user_id": 1234567,
                    "nickname": "张三",
                },
                "message": [],
                "raw_message": "你好啊",
            }
        )

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.content == "你好啊"

    @pytest.mark.asyncio
    async def test_get_msg_by_id_no_bot(self):
        """测试 event 无 bot 属性时返回空"""
        adapter = OneBot11Adapter()

        event = Mock(spec=[])
        delattr(event, "bot") if hasattr(event, "bot") else None

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.has_reply is False

    @pytest.mark.asyncio
    async def test_get_msg_by_id_api_error(self):
        """测试 API 调用失败时返回空"""
        adapter = OneBot11Adapter()

        bot = Mock()
        bot.call_action = AsyncMock(side_effect=Exception("API_NOT_FOUND"))

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.has_reply is False

    @pytest.mark.asyncio
    async def test_get_msg_by_id_empty_message_id(self):
        """测试空 message_id 返回空"""
        adapter = OneBot11Adapter()

        event = Mock()

        result = await adapter.get_msg_by_id(event, "")

        assert result.has_reply is False

    @pytest.mark.asyncio
    async def test_get_msg_by_id_timeout(self):
        """测试 API 超时返回空"""
        import asyncio

        adapter = OneBot11Adapter()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        bot = Mock()
        bot.call_action = slow_call

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.has_reply is False

    @pytest.mark.asyncio
    async def test_get_msg_by_id_empty_result(self):
        """测试 API 返回空结果"""
        adapter = OneBot11Adapter()

        bot = Mock()
        bot.call_action = AsyncMock(return_value=None)

        event = Mock()
        event.bot = bot

        result = await adapter.get_msg_by_id(event, "6283")

        assert result.has_reply is False


class TestBaseAdapterGetMsgById:
    """PlatformAdapter 基类 get_msg_by_id 默认实现测试"""

    @pytest.mark.asyncio
    async def test_default_returns_empty(self):
        """测试基类默认实现返回空 ReplyInfo"""

        class DummyAdapter(PlatformAdapter):
            def get_user_id(self, event):
                return ""

            def get_user_name(self, event):
                return ""

            def get_user_nickname(self, event):
                return ""

            def get_group_id(self, event):
                return ""

            def get_group_name(self, event):
                return ""

            def get_user_role(self, event):
                return ""

            def get_raw_message(self, event):
                return {}

            def is_group_message(self, event):
                return False

            def get_images(self, event):
                return []

            def get_reply_info(self, event):
                return ReplyInfo()

        adapter = DummyAdapter()
        result = await adapter.get_msg_by_id(Mock(), "123")

        assert result.has_reply is False


class TestGetMentionedUsers:
    """get_mentioned_users 测试（@用户定向功能回归）"""

    def test_factory_degrades_unimplemented_platform(self):
        """回归：已注册未实现的平台降级到 GenericAdapter，不抛异常

        历史 bug：qqofficial/gewechat 已注册但 adapter_class=None，
        get_adapter 抛 UnsupportedPlatformError，钩子链无 try/except 兜底，
        每条消息崩溃。
        """
        event = Mock()
        event.platform_meta = Mock()
        event.platform_meta.name = "qqofficial"
        event.platform_meta.id = "test_bot"

        adapter = get_adapter(event)
        assert isinstance(adapter, GenericAdapter), (
            "已注册未实现的平台应降级到 GenericAdapter，不抛异常"
        )

    def test_base_default_returns_empty(self):
        """基类默认实现返回空列表"""

        class DummyAdapter(PlatformAdapter):
            def get_user_id(self, event):
                return ""

            def get_user_name(self, event):
                return ""

            def get_user_nickname(self, event):
                return ""

            def get_group_id(self, event):
                return ""

            def get_group_name(self, event):
                return ""

            def get_user_role(self, event):
                return ""

            def get_raw_message(self, event):
                return {}

            def is_group_message(self, event):
                return False

            def get_images(self, event):
                return []

            def get_reply_info(self, event):
                return ReplyInfo()

        adapter = DummyAdapter()
        result = adapter.get_mentioned_users(Mock())
        assert result == []

    def test_onebot11_segment_format(self):
        """OneBot11 段列表格式提取 @用户"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.raw_message = {
            "message": [
                {"type": "text", "data": {"text": "你好 "}},
                {"type": "at", "data": {"qq": "123456", "name": "张三"}},
                {"type": "at", "data": {"qq": "789", "name": "李四"}},
            ]
        }

        result = adapter.get_mentioned_users(event)
        assert len(result) == 2
        assert result[0] == ("123456", "张三")
        assert result[1] == ("789", "李四")

    def test_onebot11_cq_code_format(self):
        """OneBot11 CQ 码字符串格式提取 @用户"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.raw_message = {
            "message": "[CQ:at,qq=123456,name=张三] 你好 [CQ:at,qq=789,name=李四]"
        }

        result = adapter.get_mentioned_users(event)
        assert len(result) == 2
        assert result[0] == ("123456", "张三")
        assert result[1] == ("789", "李四")

    def test_onebot11_skip_at_all(self):
        """@全体成员应被跳过"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.raw_message = {
            "message": [
                {"type": "at", "data": {"qq": "all"}},
                {"type": "at", "data": {"qq": "123456", "name": "张三"}},
            ]
        }

        result = adapter.get_mentioned_users(event)
        assert len(result) == 1
        assert result[0] == ("123456", "张三")

    def test_onebot11_no_at_returns_empty(self):
        """无 @ 段时返回空列表"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.raw_message = {
            "message": [{"type": "text", "data": {"text": "你好"}}]
        }

        result = adapter.get_mentioned_users(event)
        assert result == []


class TestGetSessionId:
    """get_session_id 会话键测试（私聊 L1 队列隔离修复）

    私聊事件 group_id 为空字符串，L1 缓冲等按会话隔离的组件
    使用 private:{user_id} 作为会话键，避免不同私聊用户共用队列。
    """

    def _make_event(self, group_id: str, user_id: str):
        event = Mock()
        event.message_obj = Mock()
        event.message_obj.group_id = group_id
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = user_id
        return event

    def test_group_message_returns_group_id(self):
        """群聊会话键即群号"""
        adapter = OneBot11Adapter()
        event = self._make_event("987654321", "12345")

        assert adapter.get_session_id(event) == "987654321"

    def test_private_message_returns_private_key(self):
        """私聊会话键为 private:{user_id}，不同用户键不同"""
        adapter = OneBot11Adapter()

        assert adapter.get_session_id(self._make_event("", "111")) == "private:111"
        assert adapter.get_session_id(self._make_event("", "222")) == "private:222"

    def test_private_message_generic_adapter(self):
        """通用适配器私聊同样返回 private:{user_id}"""
        adapter = GenericAdapter()
        event = self._make_event("", "12345")

        assert adapter.get_session_id(event) == "private:12345"

    def test_event_structure_broken_returns_empty(self):
        """事件结构异常（无 message_obj）时返回空字符串而非抛异常"""
        adapter = OneBot11Adapter()
        event = Mock(spec=[])

        assert adapter.get_session_id(event) == ""
