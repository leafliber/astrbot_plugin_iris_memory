"""
Iris Chat Memory - 平台适配器抽象基类

定义平台适配器的统一接口，用于屏蔽不同消息平台（QQ、微信等）的差异。

设计原则：
- 无状态适配器：每次调用传入事件对象，避免生命周期问题
- 统一异常处理：定义平台相关的异常类型
- 平台无关接口：上层模块通过抽象接口访问平台信息
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.image.models import ImageInfo


@dataclass
class ReplyInfo:
    """回复消息信息数据类

    存储从平台消息中提取的回复/引用消息的关联信息。

    Attributes:
        message_id: 被回复消息的ID
        user_id: 被回复消息的发送者ID
        user_name: 被回复消息的发送者名称（如果可获取）
        content: 被回复消息的文本内容（如果可获取）

    Examples:
        >>> info = ReplyInfo(message_id="6283", user_id="1234567")
        >>> info.message_id
        '6283'
    """

    message_id: str = ""
    user_id: str = ""
    user_name: str = ""
    content: str = ""

    @property
    def has_reply(self) -> bool:
        """是否存在回复信息

        Returns:
            存在有效的 message_id 时返回 True
        """
        return bool(self.message_id)


@dataclass
class ForwardMessage:
    """合并转发子消息数据类

    存储从平台合并转发消息中提取的单条子消息信息。

    Attributes:
        user_id: 子消息发送者ID
        user_name: 子消息发送者显示名称（群名片或昵称）
        content: 子消息文本内容
        timestamp: 子消息时间戳（Unix 秒，0 表示不可用）
        message_id: 子消息ID

    Examples:
        >>> fwd = ForwardMessage(user_id="123", user_name="张三", content="你好")
    """

    user_id: str = ""
    user_name: str = ""
    content: str = ""
    timestamp: int = 0
    message_id: str = ""


# 私聊会话键前缀：私聊事件没有群 ID，按会话隔离的组件（如 L1 缓冲）
# 使用 f"{PRIVATE_SESSION_PREFIX}{user_id}" 作为会话键，
# 避免所有私聊用户混入同一个空字符串队列。
PRIVATE_SESSION_PREFIX = "private:"


class UnsupportedPlatformError(Exception):
    """不支持的平台类型异常

    当遇到未实现的平台适配器类型时抛出。

    Examples:
        >>> raise UnsupportedPlatformError("ge微信", "当前仅支持 QQ 平台")
    """

    def __init__(self, platform_type: str, message: str = ""):
        self.platform_type = platform_type
        self.message = message or f"不支持的平台类型: {platform_type}"
        super().__init__(self.message)


class PlatformAdapter(ABC):
    """平台适配器抽象基类

    定义统一的平台信息访问接口，用于获取用户ID、群ID等平台相关信息。
    各平台适配器（如 OneBot11Adapter）需要实现具体逻辑。

    设计要点：
    - 无状态设计：不持有事件引用，每次调用传入事件对象
    - 线程安全：无实例状态，所有方法可安全并发调用
    - 平台无关：上层模块通过抽象接口访问，不依赖具体平台

    Examples:
        >>> from iris_memory.platform import get_adapter
        >>>
        >>> adapter = get_adapter(event)
        >>> user_id = adapter.get_user_id(event)
        >>> group_id = adapter.get_group_id(event)
    """

    @abstractmethod
    def get_user_id(self, event: "AstrMessageEvent") -> str:
        """获取用户ID（平台原始ID）

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            用户ID字符串

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> user_id = adapter.get_user_id(event)  # "123456789"
        """
        pass

    @abstractmethod
    def get_user_name(self, event: "AstrMessageEvent") -> str:
        """获取用户显示名称

        在群聊场景下，优先返回群名片（如果有），否则返回昵称。
        在私聊场景下，直接返回用户昵称。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            用户显示名称字符串

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> name = adapter.get_user_name(event)  # "张三" 或群名片
        """
        pass

    @abstractmethod
    def get_user_nickname(self, event: "AstrMessageEvent") -> str:
        """获取用户原始昵称

        不考虑群名片，始终返回用户的原始昵称。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            用户昵称字符串

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> nickname = adapter.get_user_nickname(event)  # "张三"
        """
        pass

    @abstractmethod
    def get_group_id(self, event: "AstrMessageEvent") -> str:
        """获取群聊ID

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            群聊ID字符串，私聊时返回空字符串 ""

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> group_id = adapter.get_group_id(event)  # "987654321" 或 ""
        """
        pass

    @abstractmethod
    def get_group_name(self, event: "AstrMessageEvent") -> str:
        """获取群聊名称

        从原始消息中提取群名称信息（如果可用）。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            群聊名称字符串，无法获取时返回空字符串 ""

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> group_name = adapter.get_group_name(event)  # "技术交流群" 或 ""
        """
        pass

    @abstractmethod
    def get_user_role(self, event: "AstrMessageEvent") -> str:
        """获取用户在群聊中的角色

        常见角色：owner(群主)、admin(管理员)、member(普通成员)。
        私聊时返回 "private"。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            角色字符串：owner、admin、member、private

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> role = adapter.get_user_role(event)  # "admin"
        """
        pass

    @abstractmethod
    def get_raw_message(self, event: "AstrMessageEvent") -> dict[str, object]:
        """获取平台原始消息对象

        返回消息平台适配器的原始消息对象（转为字典），
        用于访问平台特定的高级信息。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            原始消息字典，解析失败时返回空字典 {}

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> raw = adapter.get_raw_message(event)
            >>> print(raw.get("user_id"))  # 平台特定字段

        Notes:
            - 不同平台的原始消息结构不同，需查阅平台文档
            - OneBot11 原始消息结构参考：https://github.com/botuniverse/onebot-11
        """
        pass

    @abstractmethod
    def is_group_message(self, event: "AstrMessageEvent") -> bool:
        """判断是否为群聊消息

        通过群ID是否为空来判断消息类型。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            True 表示群聊消息，False 表示私聊消息

        Examples:
            >>> if adapter.is_group_message(event):
            ...     print("这是群聊消息")
        """
        pass

    @abstractmethod
    def get_images(self, event: "AstrMessageEvent") -> list["ImageInfo"]:
        """获取消息中的图片列表

        从消息事件中提取所有图片信息（包括当前消息和引用消息）。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            图片信息列表，无图片时返回空列表

        Raises:
            AttributeError: event 结构不符合预期

        Examples:
            >>> images = adapter.get_images(event)
            >>> for img in images:
            ...     print(f"图片URL: {img.url}")

        Notes:
            - 不同平台的图片提取方式不同
            - OneBot11：从消息段中提取 [CQ:image,...]
            - 返回的 ImageInfo 可能只包含 url 或 file_path，需要后续验证
        """
        pass

    @abstractmethod
    def get_reply_info(self, event: "AstrMessageEvent") -> ReplyInfo:
        """获取回复/引用消息的关联信息

        从消息事件中提取回复消息的元数据，包括被回复消息的ID、
        发送者ID、发送者名称和内容等。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            ReplyInfo 实例，非回复消息时返回空 ReplyInfo（has_reply 为 False）

        Examples:
            >>> reply = adapter.get_reply_info(event)
            >>> if reply.has_reply:
            ...     print(f"回复消息ID: {reply.message_id}")
            ...     print(f"回复发送者: {reply.user_name}")

        Notes:
            - 不同平台的回复消息格式不同
            - OneBot11：从消息段中提取 [CQ:reply,id=xxx]
            - go-cqhttp 等实现可能包含 content 字段（被回复消息的完整内容）
        """
        pass

    def get_session_id(self, event: "AstrMessageEvent") -> str:
        """获取会话 ID（用于按会话隔离的组件，如 L1 缓冲）

        群聊返回群号；私聊返回 f"private:{user_id}"，确保每个私聊用户
        拥有独立的会话键，不会与其他私聊用户混入同一个队列。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)

        Returns:
            会话 ID 字符串；群 ID 与用户 ID 均不可获取时返回空字符串 ""

        Examples:
            >>> session_id = adapter.get_session_id(event)
            >>> print(session_id)  # "987654321" 或 "private:123456789"
        """
        try:
            group_id = self.get_group_id(event)
        except AttributeError:
            group_id = ""
        if group_id:
            return group_id
        try:
            user_id = self.get_user_id(event)
        except AttributeError:
            return ""
        return f"{PRIVATE_SESSION_PREFIX}{user_id}" if user_id else ""

    def get_mentioned_users(self, event: "AstrMessageEvent") -> list[tuple[str, str]]:
        """获取消息中 @提及的用户列表

        从消息事件中提取所有被 @提及的用户，返回 (user_id, user_name) 列表。
        用于 @用户 定向查询功能。子类应覆盖此方法以支持具体平台的 @ 语法。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            (user_id, user_name) 元组列表，无提及时返回空列表

        Examples:
            >>> users = adapter.get_mentioned_users(event)
            >>> for uid, name in users:
            ...     print(f"被@用户: {name} ({uid})")

        Notes:
            - 默认实现返回空列表（不解析 @）
            - OneBot11：从消息段中提取 [CQ:at,qq=xxx]
            - 其他平台需子类覆盖
        """
        return []

    async def get_msg_by_id(
        self, event: "AstrMessageEvent", message_id: str
    ) -> ReplyInfo:
        """通过消息ID获取消息内容

        调用平台 API 根据消息 ID 获取历史消息的内容和发送者信息。
        默认实现返回空 ReplyInfo，子类可覆盖以支持具体平台。

        Args:
            event: AstrBot 消息事件对象 (AstrMessageEvent)，用于获取平台客户端
            message_id: 消息ID

        Returns:
            ReplyInfo 实例，包含消息内容和发送者信息；
            获取失败时返回空 ReplyInfo（has_reply 为 False）

        Examples:
            >>> reply = await adapter.get_msg_by_id(event, "6283")
            >>> if reply.has_reply:
            ...     print(f"消息内容: {reply.content}")
            ...     print(f"发送者: {reply.user_name}")

        Notes:
            - 此方法为异步方法，需要调用平台 API
            - OneBot11 使用 get_msg API 获取消息
            - 不同平台实现可能不支持此 API，将返回空 ReplyInfo
        """
        return ReplyInfo()

    async def get_forward_messages(
        self, event: "AstrMessageEvent"
    ) -> list[ForwardMessage]:
        """获取合并转发消息的子消息列表

        当消息为合并转发（OneBot11 的 forward 消息段）时，
        调用平台 API 拉取子消息列表并提取文本、发送者等信息。

        默认实现返回空列表，子类可覆盖以支持具体平台。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            合并转发子消息列表；非合并转发消息或平台不支持时返回空列表

        Examples:
            >>> forwards = await adapter.get_forward_messages(event)
            >>> for fwd in forwards:
            ...     print(f"[{fwd.user_name}]: {fwd.content}")

        Notes:
            - 此方法为异步方法，需要调用平台 API
            - OneBot11 使用 get_forward_msg API 拉取子消息
            - 不同平台实现可能不支持合并转发，将返回空列表
        """
        return []
