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

from astrbot.api.message_components import At, Image, Plain, Reply


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


class FakeBotpyRawMessage:
    """对齐 AstrBot Patched botpy 消息对象形状的探针。

    botpy 消息类全部使用 __slots__（实例无 __dict__），AstrBot 的
    Patched*Message 在其上挂载 raw_data/message_type/msg_elements。
    本类刻意同样只用 __slots__——若适配器错误依赖 __dict__ 回退，
    使用本夹具的测试会立即失败。
    """

    __slots__ = ("raw_data", "message_type", "msg_elements")

    def __init__(
        self,
        raw_data: dict[str, Any],
        message_type: Any = None,
        msg_elements: Optional[list] = None,
    ) -> None:
        self.raw_data = raw_data
        self.message_type = message_type
        self.msg_elements = msg_elements if msg_elements is not None else []


# qq_official 四种消息场景标识
QQ_OFFICIAL_SCENE_GROUP = "group"
QQ_OFFICIAL_SCENE_C2C = "c2c"
QQ_OFFICIAL_SCENE_GUILD_CHANNEL = "guild_channel"
QQ_OFFICIAL_SCENE_GUILD_DM = "guild_dm"


def make_qq_official_event(
    scene: str = QQ_OFFICIAL_SCENE_GROUP,
    message_id: str = "QQOFFMSG0001",
    member_openid: str = "m3a1f2c3d4e5f6a7b8c9d0e1f2a3b4c5",
    user_openid: str = "u5b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    author_id: str = "U789GUILD",
    username: str = "频道用户",
    roles: Optional[list] = None,
    group_openid: str = "g7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
    channel_id: str = "C1234567890",
    guild_id: str = "G4567890123",
    self_id: str = "B999OFFICIAL",
    quote_id: str = "",
    quote_content: str = "",
    quote_image_url: str = "",
    extra_mentions: Optional[list] = None,
    attachments: Optional[list] = None,
    chain: Optional[list] = None,
    bot: Any = None,
    platform: str = "qq_official",
) -> FakeEvent:
    """构造一个 qq_official 形状的消息事件（群/单聊/频道/频道私信）。

    对齐 AstrBot qqofficial 解析产物的形状：
    - raw_message 为 FakeBotpyRawMessage（__slots__ 探针，无 __dict__）
    - 群/单聊场景 sender.nickname 为空（平台不提供昵称）
    - 消息链按 AstrBot 的组装顺序：Reply（引用）→ 标记 At → Plain →
      附件 Image（标记 At 的 qq 为群场景 self_id 或字面量 "qq_official"）
    - 附件 URL 已按 AstrBot 的方式归一化为 https 前缀（raw 载荷中为 // 前缀）
    """
    raw_data: dict[str, Any] = {
        "id": message_id,
        "content": "你好",
        "attachments": [],
    }
    marker_at: At
    sender: FakeMessageMember
    group: Optional[FakeGroup]

    if scene == QQ_OFFICIAL_SCENE_GROUP:
        raw_data["group_openid"] = group_openid
        raw_data["author"] = {"member_openid": member_openid}
        # 平台隐私设计：mentions 只下发机器人自身
        raw_data["mentions"] = [
            {
                "id": self_id,
                "username": "IrisBot",
                "bot": True,
                "is_you": True,
            }
        ]
        sender = FakeMessageMember(user_id=member_openid, nickname="")
        group = FakeGroup(group_id=group_openid, group_name=None)
        marker_at = At(qq=self_id, name="IrisBot")
    elif scene == QQ_OFFICIAL_SCENE_C2C:
        raw_data["author"] = {"user_openid": user_openid}
        sender = FakeMessageMember(user_id=user_openid, nickname="")
        group = None
        marker_at = At(qq="qq_official")
    elif scene == QQ_OFFICIAL_SCENE_GUILD_CHANNEL:
        raw_data["channel_id"] = channel_id
        raw_data["guild_id"] = guild_id
        raw_data["author"] = {"id": author_id, "username": username, "bot": False}
        raw_data["member"] = {"roles": roles if roles is not None else ["5"]}
        raw_data["mentions"] = [
            {"id": self_id, "username": "IrisBot", "bot": True, "is_you": True}
        ] + (extra_mentions or [])
        sender = FakeMessageMember(user_id=author_id, nickname=username)
        group = FakeGroup(group_id=channel_id, group_name=None)
        marker_at = At(qq="qq_official")
    elif scene == QQ_OFFICIAL_SCENE_GUILD_DM:
        raw_data["channel_id"] = channel_id
        raw_data["guild_id"] = guild_id
        raw_data["direct_message"] = True
        raw_data["author"] = {"id": author_id, "username": username, "bot": False}
        sender = FakeMessageMember(user_id=author_id, nickname=username)
        group = None
        marker_at = At(qq="qq_official")
    else:
        raise ValueError(f"未知 qq_official 场景: {scene}")

    if attachments is not None:
        raw_data["attachments"] = attachments

    # 默认消息链按 AstrBot 组装顺序构造；引用消息在最前
    if chain is None:
        chain = []
        if quote_id:
            quote_chain: list = [Plain(text=quote_content)]
            if quote_image_url:
                quote_chain.append(Image.fromURL(quote_image_url))
            chain.append(
                Reply(
                    id=quote_id,
                    chain=quote_chain if quote_content or quote_image_url else [],
                    message_str=quote_content,
                )
            )
        chain.append(marker_at)
        chain.append(Plain(text="你好"))
        if attachments:
            for attachment in attachments:
                url = str(attachment.get("url") or "")
                if attachment.get("content_type", "").startswith("image") and url:
                    normalized = (
                        "https://" + url[2:] if url.startswith("//") else url
                    )
                    chain.append(Image.fromURL(normalized))

    message_obj = FakeAstrBotMessage(
        sender=sender,
        group=group,
        raw_message=FakeBotpyRawMessage(
            raw_data=raw_data,
            message_type=103 if quote_id else 0,
        ),
        message=chain,
        self_id=self_id,
        message_id=message_id,
    )
    return FakeEvent(message_obj=message_obj, platform=platform, bot=bot)
