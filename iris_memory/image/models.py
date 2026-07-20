"""
Iris Chat Memory - 图片解析数据模型

定义图片解析相关的数据结构，包括图片信息、解析结果和配额状态。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class ImageParseStatus(Enum):
    """图片解析状态枚举

    Attributes:
        PENDING: 待解析
        SUCCESS: 解析成功
        FAILED: 解析失败
    """

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


# ============================================================================
# 图片信息
# ============================================================================


@dataclass
class ImageInfo:
    """图片信息数据类

    存储单张图片的完整信息，包括URL、文件路径、格式和大小。

    Attributes:
        url: 图片URL（网络图片）
        file_path: 图片本地文件路径（本地图片）
        format: 图片格式（jpg/jpeg/png/gif/webp）
        size_kb: 图片大小（KB）
        source: 图片来源（user/upload/forward）
        message_id: 关联的消息ID

    Examples:
        >>> info = ImageInfo(
        ...     url="https://example.com/image.jpg",
        ...     format="jpg",
        ...     size_kb=256
        ... )
        >>> info.format
        'jpg'
    """

    url: Optional[str] = None
    file_path: Optional[str] = None
    format: str = ""
    size_kb: int = 0
    source: str = "user"  # user/upload/forward
    message_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "url": self.url,
            "file_path": self.file_path,
            "format": self.format,
            "size_kb": self.size_kb,
            "source": self.source,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageInfo":
        """从字典创建实例

        Args:
            data: 包含图片信息的字典

        Returns:
            ImageInfo 实例
        """
        return cls(
            url=data.get("url"),
            file_path=data.get("file_path"),
            format=data.get("format", ""),
            size_kb=data.get("size_kb", 0),
            source=data.get("source", "user"),
            message_id=data.get("message_id", ""),
        )

    @property
    def has_url(self) -> bool:
        """是否有URL

        Returns:
            是否有URL
        """
        return self.url is not None and len(self.url) > 0

    @property
    def has_file_path(self) -> bool:
        """是否有文件路径

        Returns:
            是否有文件路径
        """
        return self.file_path is not None and len(self.file_path) > 0


# ============================================================================
# 解析结果
# ============================================================================


@dataclass
class ParseResult:
    """图片解析结果数据类

    存储单张图片的解析结果，包括解析后的文本内容、Token数和时间戳。

    Attributes:
        image_info: 图片信息
        content: 解析后的文本描述
        input_tokens: 输入Token数
        output_tokens: 输出Token数
        timestamp: 解析时间戳
        success: 是否成功
        error_message: 错误信息（失败时）

    Examples:
        >>> result = ParseResult(
        ...     content="这是一张风景图片，显示蓝天白云",
        ...     input_tokens=150,
        ...     output_tokens=30,
        ...     success=True
        ... )
        >>> result.success
        True
    """

    image_info: Optional[ImageInfo] = None
    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "image_info": self.image_info.to_dict() if self.image_info else None,
            "content": self.content,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParseResult":
        """从字典创建实例

        Args:
            data: 包含解析结果的字典

        Returns:
            ParseResult 实例
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        image_info = None
        if data.get("image_info"):
            image_info = ImageInfo.from_dict(data["image_info"])

        return cls(
            image_info=image_info,
            content=data.get("content", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            timestamp=timestamp,
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
        )

    @property
    def total_tokens(self) -> int:
        """获取总Token数

        Returns:
            总Token数
        """
        return self.input_tokens + self.output_tokens


# ============================================================================
# 配额状态
# ============================================================================


@dataclass
class QuotaStatus:
    """图片解析配额状态数据类

    存储全局图片解析配额的使用情况。

    Attributes:
        date: 当前日期（YYYY-MM-DD）
        used: 已使用次数
        total: 总配额
        last_reset_time: 上次重置时间

    Examples:
        >>> status = QuotaStatus(
        ...     date="2026-03-29",
        ...     used=15,
        ...     total=200
        ... )
        >>> status.remaining
        185
    """

    date: str = ""
    used: int = 0
    total: int = 200
    last_reset_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "date": self.date,
            "used": self.used,
            "total": self.total,
            "last_reset_time": self.last_reset_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuotaStatus":
        """从字典创建实例

        Args:
            data: 包含配额状态的字典

        Returns:
            QuotaStatus 实例
        """
        last_reset_time = data.get("last_reset_time")
        if isinstance(last_reset_time, str):
            last_reset_time = datetime.fromisoformat(last_reset_time)
        else:
            last_reset_time = datetime.now()

        return cls(
            date=data.get("date", ""),
            used=data.get("used", 0),
            total=data.get("total", 200),
            last_reset_time=last_reset_time,
        )

    @property
    def remaining(self) -> int:
        """获取剩余配额

        Returns:
            剩余配额
        """
        return max(0, self.total - self.used)

    @property
    def is_exhausted(self) -> bool:
        """配额是否已耗尽

        Returns:
            是否已耗尽
        """
        return self.used >= self.total

    def use(self, count: int = 1) -> bool:
        """使用配额

        Args:
            count: 使用数量

        Returns:
            是否成功使用
        """
        if self.used + count > self.total:
            return False
        self.used += count
        return True

    def release(self, count: int = 1) -> int:
        """退还配额（解析失败/跳过时回补，避免预扣导致静默耗尽）

        Returns:
            实际退还数量（不超过已用额度）
        """
        if count <= 0:
            return 0
        actual = min(count, self.used)
        self.used -= actual
        return actual

    def reset(self, date: str, total: int) -> None:
        """重置配额

        Args:
            date: 新日期
            total: 新总配额
        """
        self.date = date
        self.used = 0
        self.total = total
        self.last_reset_time = datetime.now()


# ============================================================================
# 消息图片集合
# ============================================================================


@dataclass
class MessageImages:
    """消息图片集合数据类

    存储单条消息中的所有图片信息，包括当前消息的图片和引用消息的图片。

    Attributes:
        message_id: 消息ID
        current_images: 当前消息中的图片列表
        reply_images: 引用/回复消息中的图片列表
        is_llm_trigger: 是否触发LLM调用

    Examples:
        >>> images = MessageImages(message_id="msg_123")
        >>> images.current_images.append(ImageInfo(url="https://example.com/1.jpg"))
        >>> len(images.current_images)
        1
    """

    message_id: str = ""
    current_images: List[ImageInfo] = field(default_factory=list)
    reply_images: List[ImageInfo] = field(default_factory=list)
    is_llm_trigger: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "message_id": self.message_id,
            "current_images": [img.to_dict() for img in self.current_images],
            "reply_images": [img.to_dict() for img in self.reply_images],
            "is_llm_trigger": self.is_llm_trigger,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageImages":
        """从字典创建实例

        Args:
            data: 包含消息图片的字典

        Returns:
            MessageImages 实例
        """
        current_images = [
            ImageInfo.from_dict(img) for img in data.get("current_images", [])
        ]
        reply_images = [
            ImageInfo.from_dict(img) for img in data.get("reply_images", [])
        ]

        return cls(
            message_id=data.get("message_id", ""),
            current_images=current_images,
            reply_images=reply_images,
            is_llm_trigger=data.get("is_llm_trigger", False),
        )

    @property
    def all_images(self) -> List[ImageInfo]:
        """获取所有图片（当前+引用）

        Returns:
            所有图片列表
        """
        return self.current_images + self.reply_images

    @property
    def has_images(self) -> bool:
        """是否有图片

        Returns:
            是否有图片
        """
        return len(self.current_images) > 0 or len(self.reply_images) > 0

    @property
    def total_count(self) -> int:
        """获取图片总数

        Returns:
            图片总数
        """
        return len(self.current_images) + len(self.reply_images)


# ============================================================================
# 图片队列项
# ============================================================================


@dataclass
class ImageQueueItem:
    """图片队列项数据类

    存储在 L1 Buffer 图片队列中的图片元数据。

    Attributes:
        image_hash: 图片 hash（pHash 感知哈希或 URL MD5 hash）
        image_url: 图片 URL
        image_info: 图片完整信息
        message_id: 关联消息 ID
        group_id: 群聊 ID
        user_id: 用户 ID
        timestamp: 入队时间
        status: 解析状态

    Examples:
        >>> item = ImageQueueItem(
        ...     image_hash="abc123",
        ...     image_url="https://example.com/image.jpg",
        ...     message_id="msg_001",
        ...     group_id="group_001"
        ... )
        >>> item.status
        <ImageParseStatus.PENDING: 'pending'>
    """

    image_hash: str = ""
    image_url: str = ""
    image_info: Optional[ImageInfo] = None
    message_id: str = ""
    group_id: str = ""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: ImageParseStatus = ImageParseStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "image_hash": self.image_hash,
            "image_url": self.image_url,
            "image_info": self.image_info.to_dict() if self.image_info else None,
            "message_id": self.message_id,
            "group_id": self.group_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageQueueItem":
        """从字典创建实例

        Args:
            data: 包含队列项的字典

        Returns:
            ImageQueueItem 实例
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        image_info = None
        if data.get("image_info"):
            image_info = ImageInfo.from_dict(data["image_info"])

        status_str = data.get("status", "pending")
        status = (
            ImageParseStatus(status_str)
            if status_str in [s.value for s in ImageParseStatus]
            else ImageParseStatus.PENDING
        )

        return cls(
            image_hash=data.get("image_hash", ""),
            image_url=data.get("image_url", ""),
            image_info=image_info,
            message_id=data.get("message_id", ""),
            group_id=data.get("group_id", ""),
            user_id=data.get("user_id", ""),
            timestamp=timestamp,
            status=status,
        )


# ============================================================================
# 图片解析缓存
# ============================================================================


@dataclass
class ImageParseCache:
    """图片解析缓存数据类

    存储图片解析结果缓存，用于避免重复解析相同图片。

    Attributes:
        image_hash: 图片 hash
        content: 解析结果文本
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        created_at: 创建时间
        last_accessed: 最后访问时间
        access_count: 访问次数

    Examples:
        >>> cache = ImageParseCache(
        ...     image_hash="abc123",
        ...     content="这是一张风景图片",
        ...     input_tokens=150,
        ...     output_tokens=30
        ... )
        >>> cache.access_count
        1
    """

    image_hash: str = ""
    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典
        """
        return {
            "image_hash": self.image_hash,
            "content": self.content,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageParseCache":
        """从字典创建实例

        Args:
            data: 包含缓存数据的字典

        Returns:
            ImageParseCache 实例
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        else:
            last_accessed = datetime.now()

        return cls(
            image_hash=data.get("image_hash", ""),
            content=data.get("content", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data.get("access_count", 1),
        )

    def touch(self) -> None:
        """更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @property
    def total_tokens(self) -> int:
        """获取总 token 数

        Returns:
            总 token 数
        """
        return self.input_tokens + self.output_tokens
