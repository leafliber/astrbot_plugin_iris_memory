"""
权限检查模块

提供命令权限验证功能，隔离权限逻辑与业务逻辑。
"""

from typing import Optional

from astrbot.api.event import AstrMessageEvent

from iris_memory.utils.event_utils import get_group_id


class PermissionChecker:
    """
    权限检查器

    封装所有权限相关的检查逻辑，提供清晰的权限验证接口。

    权限层级：
    - 普通用户：可操作自己的记忆
    - 群管理员：可管理群聊记忆
    - 超级管理员：可执行全局操作
    """

    def __init__(self) -> None:
        """初始化权限检查器"""
        pass

    def is_admin(self, event: AstrMessageEvent) -> bool:
        """
        检查用户是否为管理员（群管理员或超级管理员）

        Args:
            event: 消息事件对象

        Returns:
            bool: 是否为管理员
        """
        return event.is_admin()

    def is_super_admin(self, event: AstrMessageEvent) -> bool:
        """
        检查用户是否为超级管理员

        Args:
            event: 消息事件对象

        Returns:
            bool: 是否为超级管理员
        """
        return event.is_admin()

    def check_private_only(self, event: AstrMessageEvent) -> bool:
        """
        检查是否在私聊场景

        Args:
            event: 消息事件对象

        Returns:
            bool: 是否为私聊场景
        """
        return get_group_id(event) is None

    def check_group_only(self, event: AstrMessageEvent) -> Optional[str]:
        """
        检查是否在群聊场景

        Args:
            event: 消息事件对象

        Returns:
            Optional[str]: 群聊ID，私聊返回None
        """
        return get_group_id(event)

    def require_admin(self, event: AstrMessageEvent) -> tuple[bool, Optional[str]]:
        """
        要求管理员权限，返回检查结果和错误消息

        Args:
            event: 消息事件对象

        Returns:
            tuple[bool, Optional[str]]: (是否通过, 错误消息)
        """
        if not self.is_admin(event):
            return False, "此操作需要管理员权限"
        return True, None

    def require_group(self, event: AstrMessageEvent) -> tuple[bool, Optional[str], Optional[str]]:
        """
        要求群聊场景，返回检查结果和群ID

        Args:
            event: 消息事件对象

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (是否通过, 错误消息, 群ID)
        """
        group_id = self.check_group_only(event)
        if not group_id:
            return False, "此操作仅限群聊使用", None
        return True, None, group_id

    def require_admin_in_group(
        self, event: AstrMessageEvent
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        要求群聊场景 + 管理员权限

        Args:
            event: 消息事件对象

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (是否通过, 错误消息, 群ID)
        """
        passed, error, group_id = self.require_group(event)
        if not passed:
            return False, error, None

        passed, error = self.require_admin(event)
        if not passed:
            return False, error, group_id

        return True, None, group_id
