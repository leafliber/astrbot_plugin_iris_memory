"""
Iris Chat Memory - 图片缓存清理任务

定期清理过期的图片解析缓存。

Features:
    - 清理过期的图片解析缓存
    - 支持配置保留天数
"""

from typing import TYPE_CHECKING

from iris_memory.core import get_logger
from iris_memory.config import get_config
from iris_memory.image.cache_manager import ImageCacheManager

if TYPE_CHECKING:
    from iris_memory.core import ComponentManager

logger = get_logger("tasks.cache_cleanup")


class ImageCacheCleanupTask:
    """图片缓存清理任务

    定期清理过期的图片解析缓存。

    Attributes:
        _component_manager: 组件管理器引用

    Examples:
        >>> task = ImageCacheCleanupTask(component_manager)
        >>> await task.execute()
    """

    def __init__(self, component_manager: "ComponentManager"):
        """初始化图片缓存清理任务

        Args:
            component_manager: 组件管理器实例
        """
        self._component_manager = component_manager

    async def execute(self) -> None:
        """执行图片缓存清理任务"""
        config = get_config()

        if not config.get("l1_buffer.image_parsing.enable"):
            logger.debug("图片解析未启用，跳过缓存清理")
            return

        cache_manager = self._component_manager.get_component(
            "image_cache", ImageCacheManager
        )
        if not cache_manager or not cache_manager.is_available:
            logger.debug("图片缓存管理器不可用，跳过清理")
            return

        retention_days = config.get("image_cache_retention_days", 7)

        try:
            cleaned_count = await cache_manager.cleanup_expired(retention_days)

            if cleaned_count > 0:
                logger.info(f"图片缓存清理完成，共清理 {cleaned_count} 条缓存")
            else:
                logger.debug("无需清理的图片缓存")

        except Exception as e:
            logger.error(f"图片缓存清理失败：{e}", exc_info=True)
