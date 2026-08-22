"""按 AstrBot 4.27.2 真实形状构造的测试事件夹具。

这些夹具是"形状漂移探针"：

- FakeMessageMember 刻意只含 user_id/nickname，与真实 MessageMember 一致
  （没有 card/role/group_name）。若适配器错误依赖 AstrBot 上不存在的
  字段，使用这些夹具的测试会立即失败。
- 消息链使用真实 astrbot 组件（Plain/At/Reply），使适配器的 isinstance
  分支被真实执行。

相比之下，unittest.mock.Mock 会自动伪造任意属性，无法提供这种保护——
此前 sender.card/sender.role 读取失效正是 Mock 测试漏掉的。
"""

from dataclasses import dataclass
from typing import Any, Optional

from astrbot.api.message_components import Plain


@dataclass
class FakeMessageMember:
    """对齐 astrbot.core.platform.astrbot_message.MessageMember（仅两个字段）"""

    user_id: str = ""
    nickname: Optional[str] = None


@dataclass
class FakeGroup:
    """对齐 astrbot.core.platform.astrbot_message.Group（用到字段的子集）"""

    group_id: str = ""
    group_name: Optional[str] = None


class FakeAstrBotMessage:
    """对齐 astrbot.core.platform.astrbot_message.AstrBotMessage 的形状子集"""

    def __init__(
        self,
        sender: Optional[FakeMessageMember] = None,
        group: Optional[FakeGroup] = None,
        raw_message: Any = None,
        message: Optional[list] = None,
        self_id: str = "",
        message_id: Any = "",
    ) -> None:
        self.sender = sender or FakeMessageMember()
        self.group = group
        self.raw_message = raw_message
        self.message = message if message is not None else []
        self.self_id = self_id
        self.message_id = message_id

    @property
    def group_id(self) -> str:
        """与真实 AstrBotMessage 一致：无群对象时返回空字符串"""
        return self.group.group_id if self.group else ""


class FakeEvent:
    """对齐 AstrMessageEvent 的形状子集（aiocqhttp 场景）"""

    def __init__(
        self,
        message_obj: Optional[FakeAstrBotMessage] = None,
        platform: str = "aiocqhttp",
        bot: Any = None,
    ) -> None:
        self.message_obj = message_obj or FakeAstrBotMessage()
        self._platform = platform
        self.bot = bot

    def get_platform_name(self) -> str:
        return self._platform

    def get_messages(self) -> list:
        return self.message_obj.message


def make_qq_event(
    user_id: str = "10001",
    nickname: str = "张三",
    card: str = "",
    role: str = "",
    group_id: str = "",
    group_name: Optional[str] = None,
    self_id: str = "10000",
    message_id: Any = 6283,
    chain: Optional[list] = None,
    raw_segments: Optional[list] = None,
    bot: Any = None,
    platform: str = "aiocqhttp",
) -> FakeEvent:
    """构造一个 aiocqhttp 形状的群/私聊消息事件

    raw OneBot 载荷按 array 格式填充（AstrBot 上游仅放行该格式）；
    message_obj.sender.nickname 按 AstrBot 的方式合并为 card or nickname；
    消息链默认放一个 Plain 组件，可用 chain 覆盖。
    """
    uid_raw = int(user_id) if str(user_id).isdigit() else user_id
    raw_message: dict[str, Any] = {
        "post_type": "message",
        "message_type": "group" if group_id else "private",
        "self_id": int(self_id) if str(self_id).isdigit() else self_id,
        "user_id": uid_raw,
        "message_id": message_id,
        "message": raw_segments if raw_segments is not None else [],
        "sender": {"user_id": uid_raw, "nickname": nickname},
    }
    if group_id:
        raw_message["group_id"] = (
            int(group_id) if str(group_id).isdigit() else group_id
        )
    if card:
        raw_message["sender"]["card"] = card
    if role:
        raw_message["sender"]["role"] = role

    message_obj = FakeAstrBotMessage(
        sender=FakeMessageMember(user_id=user_id, nickname=card or nickname),
        group=FakeGroup(group_id=group_id, group_name=group_name) if group_id else None,
        raw_message=raw_message,
        message=chain if chain is not None else [Plain(text="你好")],
        self_id=self_id,
        message_id=message_id,
    )
    return FakeEvent(message_obj=message_obj, platform=platform, bot=bot)
