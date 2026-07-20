"""Cron 定时任务平台适配器测试

验证 CronAdapter 能从 event.session 正确恢复群聊/私聊归属，
修复 CronMessageEvent 将群号误识别为 user_id 的问题。
"""

import pytest
from unittest.mock import Mock
from iris_memory.platform.base import ReplyInfo
from iris_memory.platform.cron import CronAdapter
from iris_memory.platform.factory import get_adapter
from iris_memory.platform.generic import GenericAdapter


class TestCronAdapterGroupSession:
    """群聊定时任务场景测试

    场景：AstrBot 定时任务投递目标为 QQ 群，session.message_type 为 GROUP_MESSAGE，
    session.session_id 为群号。message_obj.sender.user_id 也被填为群号（误导性），
    message_obj.group_id 为空。
    """

    def setup_method(self):
        self.adapter = CronAdapter()

    def _make_group_cron_event(
        self,
        session_id="433685042",
        self_id="10000",
        sender_user_id="433685042",
    ):
        """构造群聊 CronMessageEvent mock

        关键特征：
        - session.message_type == GROUP_MESSAGE
        - session.session_id == 群号
        - message_obj.sender.user_id == 群号（被误导性填充）
        - message_obj.group_id == ""（空）
        - message_obj.self_id == 机器人 QQ
        """
        event = Mock()
        event.session = Mock()
        event.session.message_type = "GROUP_MESSAGE"
        event.session.session_id = session_id
        event.message_obj = Mock()
        event.message_obj.self_id = self_id
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = sender_user_id
        event.message_obj.sender.nickname = ""
        return event

    def test_group_cron_group_id_recovered_from_session(self):
        """群聊定时任务：group_id 从 session.session_id 恢复为群号"""
        event = self._make_group_cron_event(session_id="433685042")
        assert self.adapter.get_group_id(event) == "433685042"

    def test_group_cron_user_id_uses_self_id(self):
        """群聊定时任务：user_id 使用机器人自身 ID，而非群号"""
        event = self._make_group_cron_event(self_id="10000", sender_user_id="433685042")
        # 不应返回 sender.user_id（群号），应返回 self_id（机器人 QQ）
        assert self.adapter.get_user_id(event) == "10000"
        assert self.adapter.get_user_id(event) != "433685042"

    def test_group_cron_user_id_fallback_when_no_self_id(self):
        """群聊定时任务：self_id 不可用时 user_id 降级为 'cron'"""
        event = self._make_group_cron_event(self_id="", sender_user_id="433685042")
        assert self.adapter.get_user_id(event) == "cron"

    def test_group_cron_is_group_message_true(self):
        """群聊定时任务：is_group_message 返回 True"""
        event = self._make_group_cron_event()
        assert self.adapter.is_group_message(event) is True

    def test_group_cron_user_role_member(self):
        """群聊定时任务：user_role 返回 member"""
        event = self._make_group_cron_event()
        assert self.adapter.get_user_role(event) == "member"


class TestCronAdapterPrivateSession:
    """私聊定时任务场景测试

    场景：AstrBot 定时任务投递目标为私聊，session.message_type 为 PRIVATE_MESSAGE，
    session.session_id 为目标用户 ID。
    """

    def setup_method(self):
        self.adapter = CronAdapter()

    def _make_private_cron_event(self, session_id="123456789", self_id="10000"):
        event = Mock()
        event.session = Mock()
        event.session.message_type = "PRIVATE_MESSAGE"
        event.session.session_id = session_id
        event.message_obj = Mock()
        event.message_obj.self_id = self_id
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = session_id
        event.message_obj.sender.nickname = ""
        return event

    def test_private_cron_user_id_recovered_from_session(self):
        """私聊定时任务：user_id 从 session.session_id 恢复为目标用户 ID"""
        event = self._make_private_cron_event(session_id="123456789")
        assert self.adapter.get_user_id(event) == "123456789"

    def test_private_cron_group_id_empty(self):
        """私聊定时任务：group_id 为空"""
        event = self._make_private_cron_event()
        assert self.adapter.get_group_id(event) == ""

    def test_private_cron_is_group_message_false(self):
        """私聊定时任务：is_group_message 返回 False"""
        event = self._make_private_cron_event()
        assert self.adapter.is_group_message(event) is False

    def test_private_cron_user_role_private(self):
        """私聊定时任务：user_role 返回 private"""
        event = self._make_private_cron_event()
        assert self.adapter.get_user_role(event) == "private"


class TestCronAdapterMessageTypeFormats:
    """message_type 不同表示形式兼容性测试

    EventMessageType 可能为枚举成员、字符串等不同形式，
    CronAdapter 应能兼容各种表示。
    """

    def setup_method(self):
        self.adapter = CronAdapter()

    def _make_event_with_message_type(self, message_type):
        event = Mock()
        event.session = Mock()
        event.session.message_type = message_type
        event.session.session_id = "433685042"
        event.message_obj = Mock()
        event.message_obj.self_id = "10000"
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = "433685042"
        event.message_obj.sender.nickname = ""
        return event

    def test_message_type_string_group_message(self):
        event = self._make_event_with_message_type("GROUP_MESSAGE")
        assert self.adapter.is_group_message(event) is True
        assert self.adapter.get_group_id(event) == "433685042"

    def test_message_type_string_lowercase(self):
        event = self._make_event_with_message_type("group_message")
        assert self.adapter.is_group_message(event) is True

    def test_message_type_string_private_message(self):
        event = self._make_event_with_message_type("PRIVATE_MESSAGE")
        assert self.adapter.is_group_message(event) is False

    def test_message_type_enum_like_object(self):
        """模拟 EventMessageType 枚举成员（str() 包含 'GROUP'）"""

        class FakeEnum:
            def __str__(self):
                return "EventMessageType.GROUP_MESSAGE"

        event = self._make_event_with_message_type(FakeEnum())
        assert self.adapter.is_group_message(event) is True


class TestCronAdapterFallback:
    """session 不可用时的回退测试

    当 event.session 不存在或字段缺失时，CronAdapter 应回退到
    GenericAdapter 一致的行为（读取 message_obj.sender.user_id / group_id），
    而非崩溃。
    """

    def setup_method(self):
        self.adapter = CronAdapter()

    def _make_event_without_session(self, user_id="12345", group_id="", self_id="10000"):
        event = Mock()
        # 不设置 session 属性
        del event.session
        event.message_obj = Mock()
        event.message_obj.self_id = self_id
        event.message_obj.group_id = group_id
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = user_id
        event.message_obj.sender.nickname = "测试用户"
        return event

    def test_fallback_get_user_id(self):
        """无 session 时回退到 sender.user_id"""
        event = self._make_event_without_session(user_id="67890")
        assert self.adapter.get_user_id(event) == "67890"

    def test_fallback_get_group_id(self):
        """无 session 时回退到 message_obj.group_id"""
        event = self._make_event_without_session(group_id="group_123")
        assert self.adapter.get_group_id(event) == "group_123"

    def test_fallback_is_group_message(self):
        """无 session 时回退到 group_id 是否非空判断"""
        event = self._make_event_without_session(group_id="group_123")
        assert self.adapter.is_group_message(event) is True

    def test_fallback_is_group_message_false(self):
        event = self._make_event_without_session(group_id="")
        assert self.adapter.is_group_message(event) is False

    def test_session_with_empty_message_type(self):
        """session 存在但 message_type 为空时回退"""
        event = Mock()
        event.session = Mock()
        event.session.message_type = ""
        event.session.session_id = "433685042"
        event.message_obj = Mock()
        event.message_obj.self_id = "10000"
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = "433685042"
        event.message_obj.sender.nickname = ""

        # message_type 为空，回退到 group_id 判断
        assert self.adapter.is_group_message(event) is False
        assert self.adapter.get_group_id(event) == ""


class TestCronAdapterDegraded:
    """CronAdapter 降级方法测试"""

    def setup_method(self):
        self.adapter = CronAdapter()

    def _make_event(self, message_type="GROUP_MESSAGE", session_id="433685042"):
        event = Mock()
        event.session = Mock()
        event.session.message_type = message_type
        event.session.session_id = session_id
        event.message_obj = Mock()
        event.message_obj.self_id = "10000"
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = session_id
        event.message_obj.sender.nickname = ""
        return event

    def test_get_group_name_returns_empty(self):
        event = self._make_event()
        assert self.adapter.get_group_name(event) == ""

    def test_get_user_name_fallback_to_nickname(self):
        event = self._make_event()
        event.message_obj.sender.nickname = "定时机器人"
        assert self.adapter.get_user_name(event) == "定时机器人"

    def test_get_user_name_default_when_no_nickname(self):
        event = self._make_event()
        event.message_obj.sender.nickname = ""
        assert self.adapter.get_user_name(event) == "定时任务"

    def test_get_user_nickname_fallback(self):
        event = self._make_event()
        event.message_obj.sender.nickname = "小定时"
        assert self.adapter.get_user_nickname(event) == "小定时"

    def test_get_user_nickname_default(self):
        event = self._make_event()
        event.message_obj.sender.nickname = ""
        assert self.adapter.get_user_nickname(event) == "定时任务"

    def test_get_raw_message_empty(self):
        """无 raw_message 时返回空字典"""
        event = self._make_event()
        del event.message_obj.raw_message
        assert self.adapter.get_raw_message(event) == {}

    def test_get_raw_message_with_dict(self):
        event = self._make_event()
        event.message_obj.raw_message = {"key": "value"}
        assert self.adapter.get_raw_message(event) == {"key": "value"}

    def test_get_images_returns_empty(self):
        event = self._make_event()
        assert self.adapter.get_images(event) == []

    def test_get_reply_info_returns_empty(self):
        event = self._make_event()
        result = self.adapter.get_reply_info(event)
        assert isinstance(result, ReplyInfo)
        assert result.has_reply is False

    @pytest.mark.asyncio
    async def test_get_msg_by_id_returns_empty(self):
        event = self._make_event()
        result = await self.adapter.get_msg_by_id(event, "123")
        assert isinstance(result, ReplyInfo)
        assert result.has_reply is False


class TestCronAdapterFactoryRouting:
    """工厂路由测试：cron 平台应路由到 CronAdapter，且不输出警告"""

    def test_cron_platform_returns_cron_adapter(self):
        """cron 平台应返回 CronAdapter 实例，而非 GenericAdapter"""
        event = Mock()
        event.get_platform_name = Mock(return_value="cron")

        adapter = get_adapter(event)

        assert isinstance(adapter, CronAdapter)
        assert not isinstance(adapter, GenericAdapter)

    def test_cron_adapter_is_singleton(self):
        """cron 适配器应为单例"""
        event1 = Mock()
        event1.get_platform_name = Mock(return_value="cron")

        event2 = Mock()
        event2.get_platform_name = Mock(return_value="cron")

        adapter1 = get_adapter(event1)
        adapter2 = get_adapter(event2)

        assert adapter1 is adapter2


class TestCronAdapterRegressionScenario:
    """回归测试：模拟用户报告的完整场景

    重现用户报告的 bug：
    - 定时任务投递到 QQ 群 433685042
    - CronMessageEvent 将群号放入 sender.user_id
    - 修复前：user_id=433685042, group_id=""（错误）
    - 修复后：user_id=机器人QQ, group_id=433685042（正确）
    """

    def setup_method(self):
        self.adapter = CronAdapter()

    def test_group_cron_attribution_corrected(self):
        """群聊定时任务的 user_id/group_id 归属应正确"""
        # 模拟 CronMessageEvent：群号 433685042 被放入 sender.user_id
        event = Mock()
        event.session = Mock()
        event.session.message_type = "GROUP_MESSAGE"
        event.session.session_id = "433685042"
        event.message_obj = Mock()
        event.message_obj.self_id = "10000"  # 机器人 QQ
        event.message_obj.group_id = ""  # 空（bug 根源）
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = "433685042"  # 群号被误放入
        event.message_obj.sender.nickname = ""

        # 修复后：group_id 应为群号，user_id 不应为群号
        group_id = self.adapter.get_group_id(event)
        user_id = self.adapter.get_user_id(event)

        assert group_id == "433685042", "群聊定时任务 group_id 应为群号"
        assert user_id != "433685042", "群聊定时任务 user_id 不应为群号"
        assert user_id == "10000", "群聊定时任务 user_id 应为机器人自身 ID"

    def test_private_cron_attribution_corrected(self):
        """私聊定时任务的 user_id/group_id 归属应正确"""
        event = Mock()
        event.session = Mock()
        event.session.message_type = "PRIVATE_MESSAGE"
        event.session.session_id = "123456789"  # 目标用户 QQ
        event.message_obj = Mock()
        event.message_obj.self_id = "10000"
        event.message_obj.group_id = ""
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = "123456789"
        event.message_obj.sender.nickname = ""

        user_id = self.adapter.get_user_id(event)
        group_id = self.adapter.get_group_id(event)

        assert user_id == "123456789", "私聊定时任务 user_id 应为目标用户 ID"
        assert group_id == "", "私聊定时任务 group_id 应为空"
