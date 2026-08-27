"""
Iris Chat Memory - 定时任务模块

提供定时任务调度和执行功能，包括：
- TaskScheduler: 任务调度器（管理后台任务生命周期）
- ImageCacheCleanupTask: 图片缓存清理任务

注意：DreamTask 已迁移至 iris_memory.dream 顶层包。
"""

from ..core import get_logger

__all__ = [
    "TaskScheduler",
    "ImageCacheCleanupTask",
    "BoundedWorkQueue",
]


def __getattr__(name: str):
    if name == "TaskScheduler":
        from .scheduler import TaskScheduler

        return TaskScheduler
    elif name == "ImageCacheCleanupTask":
        from .cache_cleanup_task import ImageCacheCleanupTask

        return ImageCacheCleanupTask
    elif name == "BoundedWorkQueue":
        from .work_queue import BoundedWorkQueue

        return BoundedWorkQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = get_logger("tasks")
logger.debug("定时任务模块已加载")
