"""通用平台适配器测试"""

import pytest
from unittest.mock import Mock
from iris_memory.platform.generic import GenericAdapter
from iris_memory.platform.base import ReplyInfo


class TestGenericAdapterCore:
    """GenericAdapter 核心信息提取测试"""

    def setup_method(self):
        self.adapter = GenericAdapter()

    def _make_event(self, user_id="12345", nickname="测试用户", group_id=""):
        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = user_id
        event.message_obj.sender.nickname = nickname
        event.message_obj.group_id = group_id
        return event

    def test_get_user_id(self):
        event = self._make_event(user_id="67890")
        assert self.adapter.get_user_id(event) == "67890"

    def test_get_user_nickname(self):
        event = self._make_event(nickname="小明")
        assert self.adapter.get_user_nickname(event) == "小明"

    def test_get_user_name_returns_nickname(self):
        """通用适配器无群名片概念，get_user_name 退化为 nickname"""
        event = self._make_event(nickname="小明", group_id="group_123")
        assert self.adapter.get_user_name(event) == "小明"

    def test_get_group_id_with_group(self):
        event = self._make_event(group_id="group_456")
        assert self.adapter.get_group_id(event) == "group_456"

    def test_get_group_id_private_chat(self):
        event = self._make_event(group_id="")
        assert self.adapter.get_group_id(event) == ""

    def test_is_group_message_true(self):
        event = self._make_event(group_id="group_123")
        assert self.adapter.is_group_message(event) is True

    def test_is_group_message_false(self):
        event = self._make_event(group_id="")
        assert self.adapter.is_group_message(event) is False


class TestGenericAdapterDegraded:
    """GenericAdapter 降级方法测试"""

    def setup_method(self):
        self.adapter = GenericAdapter()

    def _make_event(self, **kwargs):
        event = Mock()
        event.message_obj = Mock()
        event.message_obj.sender = Mock()
        event.message_obj.sender.user_id = kwargs.get("user_id", "12345")
        event.message_obj.sender.nickname = kwargs.get("nickname", "测试用户")
        event.message_obj.group_id = kwargs.get("group_id", "")
        return event

    def test_get_group_name_returns_empty(self):
        event = self._make_event(group_id="group_123")
        assert self.adapter.get_group_name(event) == ""

    def test_get_user_role_group(self):
        """群聊时默认返回 member"""
        event = self._make_event(group_id="group_123")
        assert self.adapter.get_user_role(event) == "member"

    def test_get_user_role_private(self):
        """私聊时返回 private"""
        event = self._make_event(group_id="")
        assert self.adapter.get_user_role(event) == "private"

    def test_get_raw_message_empty(self):
        """无 raw_message 时返回空字典"""
        event = self._make_event()
        del event.message_obj.raw_message
        assert self.adapter.get_raw_message(event) == {}

    def test_get_raw_message_with_dict(self):
        """有 raw_message 字典时正常返回"""
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


class TestGenericAdapterErrorHandling:
    """GenericAdapter 异常处理测试"""

    def setup_method(self):
        self.adapter = GenericAdapter()

    def test_get_user_id_raises_on_missing_sender(self):
        event = Mock(spec=[])
        with pytest.raises(AttributeError):
            self.adapter.get_user_id(event)

    def test_is_group_message_returns_false_on_error(self):
        event = Mock(spec=[])
        assert self.adapter.is_group_message(event) is False

    def test_get_user_role_returns_private_on_error(self):
        """无法判断群聊时退化为 private"""
        event = Mock(spec=[])
        assert self.adapter.get_user_role(event) == "private"

    def test_get_raw_message_returns_empty_on_error(self):
        event = Mock(spec=[])
        assert self.adapter.get_raw_message(event) == {}
