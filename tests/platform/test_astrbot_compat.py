"""AstrBot 兼容性契约测试

直接针对运行环境中安装的 AstrBot 断言适配器依赖的字段与行为。
AstrBot 升级导致形状变化时，这些测试会先于线上行为漂移而失败，
此时需同步 iris_memory/platform/ 的数据源策略。
"""

import dataclasses

import pytest

astrbot = pytest.importorskip("astrbot")


def test_message_member_only_has_user_id_and_nickname():
    """适配器假定 MessageMember 仅有 user_id/nickname

    群名片（card）与群角色（role）不在其中——若 AstrBot 未来扩充字段，
    可考虑迁移 iris_memory/platform/qq.py 的 raw sender 读取逻辑。
    """
    from astrbot.core.platform.astrbot_message import MessageMember

    fields = {f.name for f in dataclasses.fields(MessageMember)}
    assert fields == {"user_id", "nickname"}


def test_message_member_nickname_is_merged_card_or_nickname():
    """aiocqhttp 适配器将 card or nickname 合并进 MessageMember.nickname"""
    from astrbot.core.platform.astrbot_message import MessageMember

    member = MessageMember(user_id="10001", nickname="三哥")
    assert member.nickname == "三哥"
    assert not hasattr(member, "card")
    assert not hasattr(member, "role")


def test_at_component_shape():
    from astrbot.api.message_components import At

    at = At(qq="123456", name="张三")
    assert str(at.qq) == "123456"
    assert at.name == "张三"

    # qq 为 int 时适配器的 _clean_id 应能转字符串
    at_int = At(qq=123456)
    assert str(at_int.qq) == "123456"


def test_reply_component_shape():
    from astrbot.api.message_components import Reply

    reply = Reply(
        id=6283,
        sender_id=1234567,
        sender_nickname="张三",
        message_str="你好啊",
    )
    assert str(reply.id) == "6283"
    assert str(reply.sender_id) == "1234567"
    assert reply.sender_nickname == "张三"
    assert reply.message_str == "你好啊"

    # 裸 Reply（AstrBot 转换时 get_msg 失败的降级形态）：默认值需清洗为空
    bare = Reply(id="6283")
    assert bare.sender_id == 0
    assert not bare.message_str
    assert str(bare.id) == "6283"


def test_reply_chain_carries_components():
    from astrbot.api.message_components import Plain, Reply

    reply = Reply(id="6283", chain=[Plain(text="你好啊")])
    assert reply.chain and str(reply.chain[0].text) == "你好啊"


def test_at_all_is_at_subclass_with_all_qq():
    from astrbot.api.message_components import At, AtAll

    at_all = AtAll()
    assert isinstance(at_all, At)
    assert str(at_all.qq) == "all"


def test_astrbot_message_group_id_property():
    from astrbot.core.platform.astrbot_message import AstrBotMessage

    msg = AstrBotMessage()
    assert msg.group_id == ""
    assert msg.group is None

    msg.group_id = "987654321"
    assert msg.group is not None
    assert msg.group.group_id == "987654321"

    msg.group_id = ""
    assert msg.group is None


def test_astrbot_message_group_name_field():
    from astrbot.core.platform.astrbot_message import Group

    group = Group(group_id="987654321")
    assert group.group_name is None


def test_message_session_str_format():
    """CronAdapter 依赖 session.message_type/session_id 的字符串形态"""
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.platform.message_type import MessageType

    session = MessageSession("yuki", MessageType.GROUP_MESSAGE, "987654321")
    assert str(session) == "yuki:GroupMessage:987654321"
    assert "group" in str(MessageType.GROUP_MESSAGE).lower()
    assert "group" not in str(MessageType.FRIEND_MESSAGE).lower()


def test_cron_message_event_shape():
    """CronAdapter 的全部假设：平台名 cron、sender.user_id=session_id、
    self_id 为占位 sender_id、group_id 为空"""
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.platform.message_type import MessageType

    session = MessageSession("cron_inst", MessageType.GROUP_MESSAGE, "987654321")
    event = CronMessageEvent(context=None, session=session, message="该活跃了")

    assert event.get_platform_name() == "cron"
    assert event.message_obj.sender.user_id == "987654321"
    assert event.message_obj.self_id == "astrbot"
    assert event.message_obj.group_id == ""
    assert event.session.message_type == MessageType.GROUP_MESSAGE
    assert event.session.session_id == "987654321"


def test_aiocqhttp_event_is_dict_with_array_message():
    """raw_message 是 aiocqhttp Event（dict 子类），message 为段列表"""
    from aiocqhttp import Event

    event = Event(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 10000,
            "user_id": 10001,
            "message": [{"type": "text", "data": {"text": "你好"}}],
            "sender": {"user_id": 10001, "nickname": "张三", "role": "admin"},
        }
    )
    assert isinstance(event, dict)
    segments = event.get("message")
    assert isinstance(segments, list)
    assert isinstance(segments[0], dict)
    assert segments[0]["type"] == "text"
    assert event.get("sender", {}).get("role") == "admin"
