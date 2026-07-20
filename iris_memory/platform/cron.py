"""
Iris Chat Memory - Cron 定时任务平台适配器

AstrBot 内置定时任务触发主 Agent 时，会创建 CronMessageEvent。
该合成事件的平台名为 "cron"，且不具备普通 QQ 群消息的 message_obj.group_id 结构：
投递会话的 session_id（群号或用户 ID）被放入 message_obj.sender.user_id，
而 message_obj.group_id 为空，导致 GenericAdapter 将群号误识别为用户 ID。

本适配器通过 event.session.message_type 和 event.session.session_id 恢复正确的
群聊/私聊归属，使定时任务通过 search_memory、save_memory 等工具访问记忆时
能正确应用群聊隔离。

设计要点：
- 群聊定时任务：group_id = session.session_id，user_id = message_obj.self_id（机器人自身）
- 私聊定时任务：user_id = session.session_id，group_id = ""
- session 信息不可用时，安全回退到标准 AstrBot API（与 GenericAdapter 一致）
- 平台特有功能（raw_message、reply、图片、bot API）安全降级
"""

from typing import Any, List, TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.platform.base import PlatformAdapter, ReplyInfo

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from iris_memory.image.models import ImageInfo

logger = get_logger("platform.cron")


class CronAdapter(PlatformAdapter):
    """Cron 定时任务平台适配器

    AstrBot 内置定时任务（CronMessageEvent）的适配器，从 event.session 恢复
    投递目标的群聊/私聊归属。

    背景：
        CronMessageEvent 的 message_obj.sender.user_id 被填充为投递会话的
        session_id（群号或用户 ID），而 message_obj.group_id 为空。
        GenericAdapter 会将群号误识别为用户 ID，导致群聊隔离失效。

    恢复策略：
        - 通过 event.session.message_type 判断投递目标是群聊还是私聊
        - 群聊：group_id 取自 session.session_id，user_id 取自 message_obj.self_id
          （定时任务由机器人自身触发，无真实用户）
        - 私聊：user_id 取自 session.session_id，group_id 为空

    降级行为：
        - get_user_name/get_user_nickname: 退化为 sender.nickname 或 "定时任务"
        - get_group_name: 返回空字符串
        - get_user_role: 群聊默认 "member"，私聊返回 "private"
        - get_raw_message: 尝试获取 raw_message 属性，失败返回空字典
        - get_images/get_reply_info/get_msg_by_id: 返回空值
    """

    def _get_session_info(self, event: Any) -> tuple[str, str]:
        """从 event.session 获取 message_type 和 session_id

        Args:
            event: AstrBot 消息事件对象

        Returns:
            (message_type_str, session_id) 元组，无法获取时返回 ("", "")
        """
        session = getattr(event, "session", None)
        if session is None:
            return "", ""

        message_type = getattr(session, "message_type", "")
        session_id = getattr(session, "session_id", "")

        mt_str = str(message_type) if message_type is not None else ""
        sid_str = str(session_id) if session_id else ""

        return mt_str, sid_str

    def _is_group_session(self, message_type: str) -> bool:
        """判断是否为群聊会话

        EventMessageType.GROUP_MESSAGE 可能为枚举成员或字符串，
        通过检查字符串表示中是否包含 "group" 来兼容各种表示形式。

        Args:
            message_type: 会话类型（枚举或字符串）

        Returns:
            True 表示群聊会话
        """
        if not message_type:
            return False
        return "group" in message_type.lower()

    def get_user_id(self, event: Any) -> str:
        """获取用户ID

        群聊定时任务：返回机器人自身 ID（message_obj.self_id），
            因为定时任务由机器人触发，无真实用户；self_id 不可用时返回 "cron"。
        私聊定时任务：返回 session.session_id（目标用户 ID）。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户ID字符串
        """
        message_type, session_id = self._get_session_info(event)

        if message_type:
            if self._is_group_session(message_type):
                # 群聊定时任务：由机器人触发，使用机器人自身 ID
                self_id = getattr(event.message_obj, "self_id", "")
                if self_id:
                    return str(self_id)
                logger.debug("群聊定时任务缺少 message_obj.self_id，user_id 降级为 'cron'")
                return "cron"
            else:
                # 私聊定时任务：session_id 即为目标用户 ID
                return session_id

        # 回退：session 不可用时使用标准 API（与 GenericAdapter 一致）
        try:
            return str(event.message_obj.sender.user_id)
        except AttributeError:
            logger.error("无法获取用户ID：event.message_obj.sender.user_id 不存在")
            raise

    def get_user_name(self, event: Any) -> str:
        """获取用户显示名称

        定时任务无真实用户名称，优先使用 sender.nickname，否则返回 "定时任务"。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户显示名称
        """
        nickname = getattr(event.message_obj.sender, "nickname", "")
        if nickname:
            return str(nickname)
        return "定时任务"

    def get_user_nickname(self, event: Any) -> str:
        """获取用户原始昵称

        Args:
            event: AstrBot 消息事件对象

        Returns:
            用户昵称字符串
        """
        nickname = getattr(event.message_obj.sender, "nickname", "")
        if nickname:
            return str(nickname)
        return "定时任务"

    def get_group_id(self, event: Any) -> str:
        """获取群聊ID

        群聊定时任务：返回 session.session_id（群号）。
        私聊定时任务：返回空字符串。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            群聊ID字符串，私聊时返回空字符串
        """
        message_type, session_id = self._get_session_info(event)

        if message_type:
            if self._is_group_session(message_type):
                # 群聊定时任务：session_id 即为群号
                return session_id
            else:
                # 私聊定时任务：无群ID
                return ""

        # 回退：session 不可用时使用标准 API（与 GenericAdapter 一致）
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
        """判断是否为群聊消息

        优先通过 event.session.message_type 判断，
        session 不可用时回退到 group_id 是否非空。

        Args:
            event: AstrBot 消息事件对象

        Returns:
            True 表示群聊消息
        """
        message_type, _ = self._get_session_info(event)
        if message_type:
            return self._is_group_session(message_type)
        # 回退：与 GenericAdapter 一致
        try:
            group_id = self.get_group_id(event)
            return bool(group_id)
        except Exception:
            return False

    def get_images(self, event: Any) -> List["ImageInfo"]:
        return []

    def get_reply_info(self, event: Any) -> ReplyInfo:
        return ReplyInfo()
