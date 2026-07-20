"""
Iris Chat Memory - 平台适配器工厂

根据消息事件的平台类型，返回对应的平台适配器实例。

支持的平台：
- qq/aiocqhttp: QQ 个人号（OneBot11 协议）
- cron: AstrBot 内置定时任务（CronMessageEvent）
- qqofficial: QQ 官方机器人（待实现）
- gewechat: 个微（待实现）

设计要点：
- 单例模式：每种平台适配器只创建一个实例
- 工厂模式：根据平台类型返回对应适配器
- 扩展性：新增平台只需注册新的适配器类
"""

import threading
from typing import TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.platform.base import PlatformAdapter
from iris_memory.platform.cron import CronAdapter
from iris_memory.platform.generic import GenericAdapter
from iris_memory.platform.qq import OneBot11Adapter

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


logger = get_logger("platform.factory")


# ============================================================================
# 平台适配器注册表
# ============================================================================

# 平台类型枚举值 -> 适配器类的映射（None 表示待实现）
_ADAPTER_REGISTRY: dict[str, type[PlatformAdapter] | None] = {
    "aiocqhttp": OneBot11Adapter,  # QQ 个人号（OneBot11）
    "qq": OneBot11Adapter,  # QQ（AstrBot v4.x 统一命名）
    "cron": CronAdapter,  # AstrBot 内置定时任务（CronMessageEvent）
    "qqofficial": None,  # QQ 官方机器人（待实现）
    "gewechat": None,  # 个微（待实现）
}

# 适配器实例缓存（单例）
_ADAPTER_INSTANCES: dict[str, PlatformAdapter] = {}
_ADAPTER_LOCK = threading.Lock()


# ============================================================================
# 工厂函数
# ============================================================================


def get_adapter(event: "AstrMessageEvent") -> PlatformAdapter:
    """获取平台适配器实例

    根据消息事件的平台类型，返回对应的平台适配器实例。
    使用单例模式，每种平台只创建一个适配器实例。

    Args:
        event: AstrBot 消息事件对象 (AstrMessageEvent)

    Returns:
        平台适配器实例

    Raises:
        UnsupportedPlatformError: 平台类型未实现或不支持
        ValueError: 无法获取平台类型

    Examples:
        >>> from iris_memory.platform import get_adapter
        >>>
        >>> adapter = get_adapter(event)
        >>> user_id = adapter.get_user_id(event)
        >>> group_id = adapter.get_group_id(event)

    Notes:
        - 线程安全：使用锁保护适配器实例创建
        - 单例模式：同一平台类型共享一个适配器实例
        - 扩展性：通过 register_adapter() 注册新的平台适配器
    """
    # 获取平台类型
    platform_type = _get_platform_type(event)

    # 转为小写字符串（枚举值可能是大写）
    platform_key = (
        platform_type.lower()
        if isinstance(platform_type, str)
        else str(platform_type).lower()
    )

    # 检查是否支持
    if platform_key not in _ADAPTER_REGISTRY:
        logger.warning(
            f"未支持的平台类型: {platform_type}，使用通用适配器降级。"
            f"当前支持: {list(_ADAPTER_REGISTRY.keys())}"
        )
        with _ADAPTER_LOCK:
            if "__generic__" not in _ADAPTER_INSTANCES:
                logger.info("创建通用平台适配器实例")
                _ADAPTER_INSTANCES["__generic__"] = GenericAdapter()
            return _ADAPTER_INSTANCES["__generic__"]

    adapter_class = _ADAPTER_REGISTRY[platform_key]
    if adapter_class is None:
        # 平台已注册但适配器尚未实现（如 qqofficial/gewechat），降级到通用适配器
        # 而非抛异常——否则钩子链无 try/except 兜底，每条消息都会崩溃。
        logger.warning(f"平台 {platform_type} 的适配器尚未实现，使用通用适配器降级。")
        with _ADAPTER_LOCK:
            if "__generic__" not in _ADAPTER_INSTANCES:
                _ADAPTER_INSTANCES["__generic__"] = GenericAdapter()
            return _ADAPTER_INSTANCES["__generic__"]

    # 获取或创建适配器实例（单例）
    with _ADAPTER_LOCK:
        if platform_key not in _ADAPTER_INSTANCES:
            logger.info(f"创建平台适配器实例: {platform_key}")
            _ADAPTER_INSTANCES[platform_key] = adapter_class()

        return _ADAPTER_INSTANCES[platform_key]


def _get_platform_type(event: "AstrMessageEvent") -> str:
    """从事件对象中获取平台类型（AstrBot v4.x）

    优先使用 event.get_platform_name() 获取协议类型（如 "aiocqhttp"），
    而非 event.session.platform_name（返回用户自定义实例名，如 "yuki"）。

    Args:
        event: AstrBot 消息事件对象

    Returns:
        平台类型字符串

    Raises:
        ValueError: 无法获取平台类型
    """
    # 优先使用 AstrBot 官方 API（返回协议类型如 "aiocqhttp"）
    if hasattr(event, "get_platform_name"):
        platform_name = event.get_platform_name()
        if platform_name is not None:
            return str(platform_name).lower()

    # 回退：直接访问 platform_meta.name
    if hasattr(event, "platform_meta") and event.platform_meta is not None:
        platform_name = getattr(event.platform_meta, "name", None)
        if platform_name is not None:
            return str(platform_name).lower()

    logger.error("无法从事件对象中获取平台类型")
    raise ValueError(
        "无法获取平台类型，event.get_platform_name() 和 event.platform_meta.name 均不可用"
    )


# ============================================================================
# 扩展接口
# ============================================================================


def register_adapter(platform_type: str, adapter_class: type[PlatformAdapter]) -> None:
    """注册新的平台适配器

    用于扩展支持新的平台类型。

    Args:
        platform_type: 平台类型标识符（如 "feishu", "dingtalk"）
        adapter_class: 适配器类（必须继承 PlatformAdapter）

    Raises:
        TypeError: adapter_class 不是 PlatformAdapter 的子类

    Examples:
        >>> class FeishuAdapter(PlatformAdapter):
        ...     # 实现所有抽象方法
        ...     pass
        >>>
        >>> register_adapter("feishu", FeishuAdapter)

    Notes:
        - 注册新适配器后，get_adapter() 即可识别该平台类型
        - 已存在的平台类型会被覆盖（用于替换默认实现）
    """
    if not issubclass(adapter_class, PlatformAdapter):  # pyright: ignore[reportUnreachable]
        raise TypeError(
            f"adapter_class 必须继承 PlatformAdapter，当前类型: {type(adapter_class)}"
        )

    platform_key = platform_type.lower()

    with _ADAPTER_LOCK:
        _ADAPTER_REGISTRY[platform_key] = adapter_class
        # 清除缓存的实例，下次获取时创建新实例
        if platform_key in _ADAPTER_INSTANCES:
            del _ADAPTER_INSTANCES[platform_key]

    logger.info(f"注册平台适配器: {platform_type} -> {adapter_class.__name__}")


def get_supported_platforms() -> list[str]:
    """获取支持的平台类型列表

    Returns:
        平台类型字符串列表

    Examples:
        >>> platforms = get_supported_platforms()
        >>> print(platforms)  # ['aiocqhttp', 'qqofficial', 'gewechat']
    """
    return list(_ADAPTER_REGISTRY.keys())
