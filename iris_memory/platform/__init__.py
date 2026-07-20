"""
Iris Chat Memory - 平台接口统一管理模块

提供平台无关的消息信息访问接口，屏蔽不同消息平台（QQ、微信等）的差异。

核心功能：
- 获取用户ID、用户名、群ID、群名称等平台信息
- 自动识别平台类型并返回对应适配器
- 支持扩展新的平台适配器

使用示例：
    >>> from iris_memory.platform import get_adapter
    >>>
    >>> # 获取平台适配器
    >>> adapter = get_adapter(event)
    >>>
    >>> # 获取用户信息
    >>> user_id = adapter.get_user_id(event)
    >>> user_name = adapter.get_user_name(event)
    >>>
    >>> # 获取群聊信息
    >>> group_id = adapter.get_group_id(event)
    >>> group_name = adapter.get_group_name(event)
    >>> is_group = adapter.is_group_message(event)
    >>>
    >>> # 获取用户角色
    >>> role = adapter.get_user_role(event)  # owner/admin/member/private

扩展新平台：
    >>> from iris_memory.platform import PlatformAdapter, register_adapter
    >>>
    >>> class FeishuAdapter(PlatformAdapter):
    ...     def get_user_id(self, event):
    ...         return event.sender.open_id
    ...     # ... 实现其他方法
    >>>
    >>> register_adapter("feishu", FeishuAdapter)

支持的平台：
- aiocqhttp: QQ 个人号（OneBot11 协议）
- cron: AstrBot 内置定时任务（CronMessageEvent）
- qqofficial: QQ 官方机器人（待实现）
- gewechat: 个微（待实现）
"""

# 导出公共 API
from iris_memory.platform.base import (
    ForwardMessage,
    PlatformAdapter,
    ReplyInfo,
    UnsupportedPlatformError,
)
from iris_memory.platform.cron import CronAdapter
from iris_memory.platform.factory import (
    get_adapter,
    get_supported_platforms,
    register_adapter,
)
from iris_memory.platform.generic import GenericAdapter
from iris_memory.platform.qq import OneBot11Adapter


__all__ = [
    # 核心接口
    "get_adapter",
    "PlatformAdapter",
    "ReplyInfo",
    "ForwardMessage",
    "UnsupportedPlatformError",
    # 平台适配器
    "OneBot11Adapter",
    "CronAdapter",
    "GenericAdapter",
    # 扩展接口
    "register_adapter",
    "get_supported_platforms",
]
