"""
Iris Chat Memory - 通用平台适配器

为未明确支持的平台提供基于 AstrBot 标准 API 的降级适配器。

设计原则：
- 仅依赖 AstrBot 标准 event 属性（message_obj.sender、message_obj.group_id）
- 平台特有功能（raw_message 解析、reply 提取、bot API）安全降级
- 保证核心功能（user_id、group_id、user_name）可用
"""

from typing import Any, List, TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.platform.base import PlatformAdapter, ReplyInfo

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.image.models import ImageInfo

logger = get_logger("platform.generic")


class GenericAdapter(PlatformAdapter):
    """通用平台适配器（降级方案）

    使用 AstrBot 标准 API 提取基本信息，平台特有功能安全降级。

    适用场景：
    - 平台类型不在 _ADAPTER_REGISTRY 中
    - 新平台尚未编写专用适配器

    降级行为：
    - get_user_id/get_group_id/get_user_nickname: 使用 AstrBot 标准 API，正常工作
    - get_user_name: 退化为 nickname（无群名片概念）
    - get_group_name: 返回空字符串
    - get_user_role: 无法获取实际角色，群聊默认 "member"
    - get_raw_message: 尝试获取 raw_message 属性，失败返回空字典
    - get_images/get_reply_info/get_msg_by_id: 返回空值
    """

    def get_user_id(self, event: Any) -> str:
        try:
            return str(event.message_obj.sender.user_id)
        except AttributeError:
            logger.error("无法获取用户ID：event.message_obj.sender.user_id 不存在")
            raise

    def get_user_name(self, event: Any) -> str:
        try:
            return str(event.message_obj.sender.nickname)
        except AttributeError:
            logger.error("无法获取用户名称：event.message_obj.sender.nickname 不存在")
            raise

    def get_user_nickname(self, event: Any) -> str:
        try:
            return str(event.message_obj.sender.nickname)
        except AttributeError:
            logger.error("无法获取用户昵称：event.message_obj.sender.nickname 不存在")
            raise

    def get_group_id(self, event: Any) -> str:
        try:
            group_id = getattr(event.message_obj, "group_id", "")
            return str(group_id) if group_id else ""
        except AttributeError:
            logger.error("无法获取群ID：event.message_obj 结构异常")
            raise

    def get_group_name(self, event: Any) -> str:
        return ""

    def get_user_role(self, event: Any) -> str:
        try:
            if not self.is_group_message(event):
                return "private"
            return "member"
        except Exception:
            return "member"

    def get_raw_message(self, event: Any) -> dict[str, Any]:
        try:
            raw_msg = getattr(event.message_obj, "raw_message", None)
            if raw_msg is None:
                return {}
            if isinstance(raw_msg, dict):
                return raw_msg
            if hasattr(raw_msg, "__dict__"):
                return raw_msg.__dict__
            return {}
        except Exception:
            return {}

    def is_group_message(self, event: "AstrMessageEvent") -> bool:
        try:
            group_id = self.get_group_id(event)
            return bool(group_id)
        except Exception:
            return False

    def get_images(self, event: Any) -> List["ImageInfo"]:
        return []

    def get_reply_info(self, event: Any) -> ReplyInfo:
        return ReplyInfo()
