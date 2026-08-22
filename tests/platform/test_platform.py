"""平台适配器测试"""

import pytest
from unittest.mock import Mock, AsyncMock

from astrbot.api.message_components import At, AtAll, Plain, Reply

from tests.platform.fakes import make_qq_event
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

    def test_get_reply_info_string_message_returns_empty(self):
        """string/CQ 码格式消息段返回空

        AstrBot 4.x 上游对非 array 格式消息直接丢弃，该格式不应到达适配器；
        若意外到达，安全返回空 ReplyInfo。
        """
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.raw_message = {
            "message_id": "999",
            "message": "[CQ:reply,id=6283]我也觉得",
        }

        reply_info = adapter.get_reply_info(event)

        assert reply_info.has_reply is False

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
        event.get_platform_name = Mock(return_value="aiocqhttp")

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
        event1.get_platform_name = Mock(return_value="aiocqhttp")

        event2 = Mock()
        event2.get_platform_name = Mock(return_value="aiocqhttp")

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

        历史 bug：待实现平台 adapter_class=None，get_adapter 抛
        UnsupportedPlatformError，钩子链无 try/except 兜底，每条消息崩溃。
        """
        event = Mock()
        event.platform_meta = Mock()
        event.platform_meta.name = "qq_official"
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

    def test_onebot11_string_message_returns_empty(self):
        """string/CQ 码格式消息返回空列表（AstrBot 上游仅放行 array 格式）"""
        adapter = OneBot11Adapter()

        event = Mock()
        event.message_obj = Mock()
        event.message_obj.raw_message = {
            "message": "[CQ:at,qq=123456,name=张三] 你好 [CQ:at,qq=789,name=李四]"
        }

        result = adapter.get_mentioned_users(event)
        assert result == []

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


class TestOneBot11DataSources:
    """raw sender / 结构化群名字段数据源测试

    夹具按 AstrBot 4.27.2 真实形状构造（MessageMember 仅 user_id/nickname），
    防止适配器重新依赖 AstrBot 上不存在的 sender.card/sender.role 字段。
    """

    def test_group_card_preferred_over_nickname(self):
        """群聊显示名优先群名片（读取 raw sender.card）"""
        event = make_qq_event(card="三哥", nickname="张三", group_id="987654321")
        assert OneBot11Adapter().get_user_name(event) == "三哥"

    def test_group_name_without_card_falls_to_nickname(self):
        """无群名片时显示名退化为昵称"""
        event = make_qq_event(nickname="张三", group_id="987654321")
        assert OneBot11Adapter().get_user_name(event) == "张三"

    def test_nickname_is_raw_not_card(self):
        """原始昵称不受群名片影响（读取 raw sender.nickname）"""
        event = make_qq_event(card="三哥", nickname="张三", group_id="987654321")
        assert OneBot11Adapter().get_user_nickname(event) == "张三"

    def test_role_read_from_raw_sender(self):
        """群角色读取 raw sender.role（MessageMember 上不存在该字段）"""
        event = make_qq_event(role="admin", group_id="987654321")
        assert OneBot11Adapter().get_user_role(event) == "admin"

    def test_role_owner(self):
        event = make_qq_event(role="owner", group_id="987654321")
        assert OneBot11Adapter().get_user_role(event) == "owner"

    def test_role_private_chat(self):
        event = make_qq_event()
        assert OneBot11Adapter().get_user_role(event) == "private"

    def test_role_defaults_member_without_raw(self):
        """raw 载荷缺失时角色回退 member 而非报错"""
        event = make_qq_event(group_id="987654321")
        event.message_obj.raw_message = {}
        assert OneBot11Adapter().get_user_role(event) == "member"

    def test_group_name_from_structured_field(self):
        """群名优先读取结构化 message_obj.group.group_name"""
        event = make_qq_event(group_id="987654321", group_name="技术交流群")
        assert OneBot11Adapter().get_group_name(event) == "技术交流群"

    def test_group_name_maps_na_sentinel_to_empty(self):
        """aiocqhttp 以 "N/A" 作为缺省哨兵，应映射为空字符串"""
        event = make_qq_event(group_id="987654321", group_name="N/A")
        assert OneBot11Adapter().get_group_name(event) == ""

    def test_group_name_falls_back_to_raw_payload(self):
        """结构化字段缺失时回退 raw 载荷的 group_name"""
        event = make_qq_event(group_id="987654321", group_name=None)
        event.message_obj.raw_message["group_name"] = "后备群名"
        assert OneBot11Adapter().get_group_name(event) == "后备群名"


class TestChainFirstReply:
    """回复信息消息链优先策略测试

    AstrBot 转换消息时已为 reply 段调过 get_msg 并把结果放进消息链的
    Reply 组件，适配器直接读取、不再重复调用平台 API。
    """

    def test_reply_from_chain_component(self):
        """链上完整 Reply 组件一次性提供全部回复元数据"""
        event = make_qq_event(
            chain=[
                Reply(
                    id=6283,
                    sender_id=1234567,
                    sender_nickname="张三",
                    message_str="你好啊",
                ),
                Plain(text="我也觉得"),
            ]
        )

        info = OneBot11Adapter().get_reply_info(event)

        assert info.has_reply is True
        assert info.message_id == "6283"
        assert info.user_id == "1234567"
        assert info.user_name == "张三"
        assert info.content == "你好啊"

    def test_chain_reply_content_from_nested_chain(self):
        """message_str 为空时从 Reply.chain 的 Plain 组件提取文本"""
        event = make_qq_event(
            chain=[
                Reply(
                    id="6283",
                    sender_id=1234567,
                    sender_nickname="张三",
                    chain=[Plain(text="你好啊")],
                ),
                Plain(text="嗯"),
            ]
        )

        info = OneBot11Adapter().get_reply_info(event)

        assert info.content == "你好啊"

    def test_raw_fallback_when_chain_has_no_reply(self):
        """链上无 Reply 组件时回退 raw reply 段解析"""
        event = make_qq_event(
            chain=[Plain(text="嗯")],
            raw_segments=[{"type": "reply", "data": {"id": "6283"}}],
        )

        info = OneBot11Adapter().get_reply_info(event)

        assert info.has_reply is True
        assert info.message_id == "6283"

    def test_bare_chain_reply_without_sender(self):
        """AstrBot 转换时 get_msg 失败的场景：链上只有裸 Reply(id=...)"""
        event = make_qq_event(chain=[Reply(id="6283"), Plain(text="嗯")])

        info = OneBot11Adapter().get_reply_info(event)

        assert info.has_reply is True
        assert info.message_id == "6283"
        assert info.user_id == ""  # sender_id 默认值 0 应清洗为空


class TestChainFirstMentions:
    """@提及消息链优先策略测试

    At 组件的 name 由 AstrBot 调 get_group_member_info 解析，raw at 段的
    data.name 在多数协议端实现中不存在。
    """

    def test_mentions_from_chain_components(self):
        """链上 At 组件提供已解析的名称，raw 段无 name 也不再丢失"""
        event = make_qq_event(
            group_id="987654321",
            chain=[
                Plain(text="hi"),
                At(qq="123456", name="张三"),
                At(qq="789", name="李四"),
            ],
            raw_segments=[{"type": "at", "data": {"qq": "123456"}}],
        )

        result = OneBot11Adapter().get_mentioned_users(event)

        assert result == [("123456", "张三"), ("789", "李四")]

    def test_at_all_in_chain_skipped(self):
        """@全体成员（AtAll 是 At 子类）应被跳过"""
        event = make_qq_event(
            group_id="987654321",
            chain=[AtAll(), At(qq="123456", name="张三")],
        )

        result = OneBot11Adapter().get_mentioned_users(event)

        assert result == [("123456", "张三")]

    def test_raw_fallback_when_chain_empty(self):
        """链为空时回退 raw at 段解析"""
        event = make_qq_event(
            chain=[],
            raw_segments=[{"type": "at", "data": {"qq": "123456", "name": "张三"}}],
        )

        result = OneBot11Adapter().get_mentioned_users(event)

        assert result == [("123456", "张三")]


class TestSelfIdRouting:
    """bot API 多账号路由测试

    aiocqhttp 多反向 WS 连接下，call_action 只有携带 self_id 才能路由到
    事件所属协议端（对齐 AstrBot 核心的 routing_params 写法）。
    """

    @pytest.mark.asyncio
    async def test_get_msg_by_id_passes_self_id(self):
        bot = Mock()
        bot.call_action = AsyncMock(return_value=None)

        event = make_qq_event(self_id="10000", bot=bot)

        await OneBot11Adapter().get_msg_by_id(event, "6283")

        bot.call_action.assert_called_once_with(
            "get_msg", message_id=6283, self_id="10000"
        )

    @pytest.mark.asyncio
    async def test_get_forward_msg_passes_self_id(self):
        bot = Mock()
        bot.call_action = AsyncMock(return_value=None)

        event = make_qq_event(
            self_id="10000",
            bot=bot,
            chain=[],
            raw_segments=[{"type": "forward", "data": {"id": "resid123"}}],
        )

        await OneBot11Adapter().get_forward_messages(event)

        bot.call_action.assert_called_once_with(
            "get_forward_msg", message_id="resid123", self_id="10000"
        )

    @pytest.mark.asyncio
    async def test_no_self_id_when_missing(self):
        """self_id 缺失时不传路由参数（单连接部署仍可工作）"""
        bot = Mock()
        bot.call_action = AsyncMock(return_value=None)

        event = make_qq_event(bot=bot)
        event.message_obj.self_id = ""

        await OneBot11Adapter().get_msg_by_id(event, "6283")

        bot.call_action.assert_called_once_with("get_msg", message_id=6283)
