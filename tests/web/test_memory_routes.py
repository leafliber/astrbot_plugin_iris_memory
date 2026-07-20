"""Web 记忆路由辅助函数测试"""

from datetime import datetime
from unittest.mock import Mock

from iris_memory.l1_buffer import ContextMessage
from iris_memory.web.routes.memory import get_private_queue_display


def _make_msg(role: str, content: str, metadata: dict | None = None) -> ContextMessage:
    return ContextMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        token_count=10,
        source="12345" if role == "user" else "assistant",
        metadata=metadata or {},
    )


def _make_buffer(messages: list) -> Mock:
    buffer = Mock()
    buffer.get_context = Mock(return_value=messages)
    return buffer


class TestGetPrivateQueueDisplay:
    """私聊队列展示信息提取（WebUI 私聊队列显示昵称与用户 ID）"""

    def test_extracts_user_id_and_latest_nickname(self):
        """从最近一条携带昵称的用户消息元数据中提取昵称"""
        messages = [
            _make_msg("user", "早", {"user_name": "旧昵称"}),
            _make_msg("assistant", "早啊"),
            _make_msg("user", "在吗", {"user_name": "新昵称"}),
        ]
        buffer = _make_buffer(messages)

        display = get_private_queue_display(buffer, "private:12345")

        assert display == {
            "is_private": True,
            "user_id": "12345",
            "group_name": "新昵称",
        }

    def test_no_nickname_returns_empty(self):
        """队列中没有昵称元数据时昵称返回空字符串"""
        buffer = _make_buffer([_make_msg("assistant", "你好")])

        display = get_private_queue_display(buffer, "private:12345")

        assert display["user_id"] == "12345"
        assert display["group_name"] == ""

    def test_empty_queue(self):
        """空队列返回用户 ID 与空昵称"""
        buffer = _make_buffer([])

        display = get_private_queue_display(buffer, "private:999")

        assert display == {"is_private": True, "user_id": "999", "group_name": ""}
