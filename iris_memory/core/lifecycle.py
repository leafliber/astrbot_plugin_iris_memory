"""
插件生命周期管理模块

负责组件的创建、初始化和清理等生命周期管理。
"""

import asyncio
from typing import Optional, Tuple, TYPE_CHECKING
from datetime import datetime

from iris_memory.core import get_logger, ComponentManager, Component

if TYPE_CHECKING:
    from astrbot.api.star import Context, Star

logger = get_logger("lifecycle")

_component_manager: Optional[ComponentManager] = None
_start_time: Optional[datetime] = None
_background_init_task: Optional[asyncio.Task] = None


def set_component_manager(manager: ComponentManager) -> None:
    """设置全局组件管理器

    Args:
        manager: 组件管理器实例
    """
    global _component_manager, _start_time
    _component_manager = manager
    _start_time = datetime.now()
    logger.debug("已设置全局组件管理器")


def get_component_manager() -> ComponentManager:
    """获取全局组件管理器

    Returns:
        组件管理器实例

    Raises:
        RuntimeError: 如果组件管理器未初始化
    """
    if _component_manager is None:
        raise RuntimeError("组件管理器未初始化，请先调用 set_component_manager()")
    return _component_manager


def get_uptime() -> int:
    """获取运行时间（秒）

    Returns:
        运行时间（秒）
    """
    if _start_time is None:
        return 0

    delta = datetime.now() - _start_time
    return int(delta.total_seconds())


def create_components(context: "Context", star: "Star") -> Tuple[Component, ...]:
    """创建所有组件实例

    根据配置创建需要的组件实例，但暂不初始化。

    Args:
        context: AstrBot Context 对象
        star: AstrBot Star 实例（插件实例），用于 KV 存储

    Returns:
        组件元组
    """
    from iris_memory.config import get_config

    config = get_config()
    components = []

    # 阶段5: LLM 管理器（最先创建，其他组件可能依赖）
    from iris_memory.llm import LLMManager

    components.append(LLMManager(context, star))
    logger.debug("已添加 LLMManager 组件")

    # 阶段1: Persona 解析器（其他组件在运行时经 component_manager 取用）
    from iris_memory.core import PersonaResolver

    components.append(PersonaResolver(context))
    logger.debug("已添加 PersonaResolver 组件")

    # 阶段2: L1 消息缓冲
    if config.get("l1_buffer.enable"):
        from iris_memory.l1_buffer import L1Buffer

        components.append(L1Buffer())
        logger.debug("已添加 L1Buffer 组件")

    # 阶段3: L2 记忆库
    if config.get("l2_memory.enable"):
        # 延迟导入，避免循环依赖
        from iris_memory.l2_memory import L2MemoryAdapter

        # persona_id 在请求时由 PersonaResolver 解析，不再在构造期固化
        components.append(L2MemoryAdapter(context=context))
        logger.debug("已添加 L2MemoryAdapter 组件")

    # 阶段4: L3 知识图谱
    if config.get("l3_kg.enable"):
        from iris_memory.l3_kg import L3KGAdapter

        components.append(L3KGAdapter())
        logger.debug("已添加 L3KGAdapter 组件")

    # 阶段6: 定时任务调度器
    from iris_memory.tasks import TaskScheduler

    components.append(TaskScheduler())
    logger.debug("已添加 TaskScheduler 组件")

    # 阶段9: 画像存储
    if config.get("profile.enable"):
        from iris_memory.profile import ProfileStorage

        components.append(ProfileStorage(star))
        logger.debug("已添加 ProfileStorage 组件")

    # 阶段10: 图片限额管理器
    if config.get("l1_buffer.image_parsing.enable"):
        from iris_memory.image import ImageQuotaManager, ImageCacheManager

        components.append(ImageQuotaManager(star))
        components.append(ImageCacheManager(star))
        logger.debug("已添加 ImageQuotaManager 和 ImageCacheManager 组件")

    return tuple(components)


async def initialize_components(component_manager: Optional[ComponentManager]) -> bool:
    """初始化所有组件

    EAGER 组件同步初始化，BACKGROUND 组件后台异步初始化。
    注入 ComponentManager 在同步初始化完成后立即执行。
    定时任务延迟到后台组件就绪后注册。

    Args:
        component_manager: 组件管理器实例，如果为 None 则创建新的

    Returns:
        初始化是否成功

    Note:
        即使初始化失败也返回 True（已尝试初始化），避免重复尝试
    """
    global _background_init_task

    if component_manager is None:
        logger.warning("组件管理器为 None，无法初始化组件")
        return False

    try:
        await component_manager.initialize_all()

        _inject_component_manager(component_manager)

        _start_scheduled_tasks_immediate(component_manager)

        if not component_manager.all_background_init_done:
            _background_init_task = asyncio.create_task(
                _wait_and_start_deferred_tasks(component_manager),
                name="iris_background_init_wait",
            )
        else:
            await _start_scheduled_tasks_deferred(component_manager)

        return True

    except Exception as e:
        logger.error(f"组件初始化失败：{e}", exc_info=True)
        # 即使初始化抛异常，仍尝试注入引用与启动不依赖后台组件的定时任务，
        # 避免异常路径下调度任务完全不注册的静默降级；
        # 任务内部会检查所需组件可用性，不可用则自动跳过。
        try:
            _inject_component_manager(component_manager)
            _start_scheduled_tasks_immediate(component_manager)
        except Exception as task_err:
            logger.warning(f"异常路径下启动定时任务失败：{task_err}")
        return True


async def _wait_and_start_deferred_tasks(
    component_manager: ComponentManager,
) -> None:
    """等待后台组件初始化完成后启动延迟任务

    Args:
        component_manager: 组件管理器实例
    """
    try:
        await component_manager.wait_for_background_init(timeout=120)
        await _start_scheduled_tasks_deferred(component_manager)
    except Exception as e:
        logger.error(f"等待后台组件初始化失败：{e}", exc_info=True)


def _inject_component_manager(component_manager: ComponentManager) -> None:
    """注入 ComponentManager 引用到需要的组件

    某些组件需要延迟获取其他组件的引用（如 L1Buffer 需要 LLMManager），
    在组件初始化完成后注入 ComponentManager 引用。

    Args:
        component_manager: 组件管理器实例
    """
    # 注入到 L1Buffer
    l1_buffer = component_manager.get_component("l1_buffer")
    if l1_buffer and hasattr(l1_buffer, "set_component_manager"):
        l1_buffer.set_component_manager(component_manager)
        logger.debug("已注入 ComponentManager 到 L1Buffer")

    # 注入到 TaskScheduler
    scheduler = component_manager.get_component("scheduler")
    if scheduler and hasattr(scheduler, "set_component_manager"):
        scheduler.set_component_manager(component_manager)
        logger.debug("已注入 ComponentManager 到 TaskScheduler")


def _start_scheduled_tasks_immediate(component_manager: ComponentManager) -> None:
    """启动不依赖后台组件的定时任务

    在同步初始化完成后立即注册。

    Args:
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config
    from iris_memory.tasks import ImageCacheCleanupTask

    scheduler = component_manager.get_component("scheduler")
    if not scheduler or not scheduler.is_available:
        logger.warning("TaskScheduler 不可用，跳过启动定时任务")
        return

    config = get_config()

    if config.get("l1_buffer.image_parsing.enable"):
        cache_cleanup_task = ImageCacheCleanupTask(component_manager)
        interval_hours = config.get("image_cache_cleanup_interval_hours", 24)
        scheduler.register_periodic_task(
            task_name="cache_cleanup",
            coro_func=cache_cleanup_task.execute,
            interval_hours=interval_hours,
        )


async def _start_scheduled_tasks_deferred(component_manager: ComponentManager) -> None:
    """启动依赖后台组件的定时任务

    在后台组件（L2/L3）初始化完成后注册。
    如果依赖的组件不可用，则跳过对应任务。

    Args:
        component_manager: 组件管理器实例
    """
    from iris_memory.config import get_config
    from iris_memory.dream import DreamTask

    scheduler = component_manager.get_component("scheduler")
    if not scheduler or not scheduler.is_available:
        logger.warning("TaskScheduler 不可用，跳过启动延迟定时任务")
        return

    config = get_config()

    l2_available = component_manager.check_component("l2_memory") == "available"

    if l2_available and config.get("scheduled_tasks.enable_dream"):
        dream_task = DreamTask(component_manager)
        interval_hours = config.get("dream_task_interval_hours")
        scheduler.register_periodic_task(
            task_name="dream",
            coro_func=dream_task.execute,
            interval_hours=interval_hours,
        )
    elif config.get("scheduled_tasks.enable_dream") and not l2_available:
        logger.warning("L2 记忆库不可用，跳过梦境任务注册")


async def shutdown_components(component_manager: Optional[ComponentManager]) -> None:
    """关闭所有组件

    Args:
        component_manager: 组件管理器实例
    """
    global _background_init_task

    if _background_init_task and not _background_init_task.done():
        _background_init_task.cancel()
        try:
            await _background_init_task
        except asyncio.CancelledError:
            pass
        _background_init_task = None

    if not component_manager:
        return

    try:
        await component_manager.shutdown_all()
        logger.info("组件关闭完成")
    except Exception as e:
        logger.error(f"组件关闭失败：{e}", exc_info=True)
