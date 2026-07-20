"""
Iris Chat Memory - 图片解析缓存管理器

管理图片解析结果缓存，避免重复解析相同图片。
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Optional
import asyncio

from iris_memory.core import Component, get_logger
from iris_memory.core.storage import KVStorage
from iris_memory.config import get_config
from .models import ImageParseCache

if TYPE_CHECKING:
    pass

logger = get_logger("image")


class ImageCacheManager(Component):
    """图片解析缓存管理组件

    管理图片解析结果缓存，支持：
    - 缓存解析结果，避免重复解析
    - 访问时刷新 last_accessed 时间
    - 定期清理过期缓存

    缓存存储键格式：iris_image_cache:{image_hash}

    Attributes:
        _storage: KV 存储适配器
        _is_available: 组件是否可用
        _lock: 异步锁（防止并发竞争）
        _cache: 内存缓存（可选，加速访问）
    """

    KV_KEY_PREFIX = "iris_image_cache"
    KV_KEY_META = "iris_image_cache_meta"

    def __init__(self, storage: KVStorage):
        """初始化缓存管理器

        Args:
            storage: KV 存储适配器（实现 KVStorage 协议的对象）
        """
        super().__init__()
        self._storage = storage
        self._lock = asyncio.Lock()
        self._cache: Dict[str, ImageParseCache] = {}

    @property
    def name(self) -> str:
        """组件名称"""
        return "image_cache"

    async def initialize(self) -> None:
        """初始化缓存管理器"""
        config = get_config()

        if not config.get("l1_buffer.image_parsing.enable"):
            self._is_available = False
            logger.info("图片解析未启用，缓存管理器不初始化")
            return

        self._is_available = True
        logger.info("图片解析缓存管理器初始化完成")

    async def shutdown(self) -> None:
        """关闭缓存管理器"""
        self._cache.clear()
        self._reset_state()
        logger.info("图片解析缓存管理器已关闭")

    def _get_cache_key(self, image_hash: str) -> str:
        """获取缓存键名

        Args:
            image_hash: 图片 hash

        Returns:
            缓存键名
        """
        return f"{self.KV_KEY_PREFIX}:{image_hash}"

    async def get_cache(self, image_hash: str) -> Optional[ImageParseCache]:
        """获取缓存

        Args:
            image_hash: 图片 hash

        Returns:
            缓存对象，不存在返回 None
        """
        if not self._is_available:
            return None

        if image_hash in self._cache:
            cache = self._cache[image_hash]
            cache.touch()
            logger.debug(f"从内存缓存获取图片解析结果：{image_hash[:8]}...")
            return cache

        try:
            key = self._get_cache_key(image_hash)
            data = await self._storage.get_kv_data(key, None)

            if data:
                cache = ImageParseCache.from_dict(data)
                cache.touch()
                self._cache[image_hash] = cache
                await self._save_cache(cache)
                logger.debug(f"从 KV 存储获取图片解析结果：{image_hash[:8]}...")
                return cache

            return None

        except Exception as e:
            logger.warning(f"获取图片解析缓存失败：{e}")
            return None

    async def set_cache(self, cache: ImageParseCache) -> bool:
        """设置缓存

        Args:
            cache: 缓存对象

        Returns:
            是否成功
        """
        if not self._is_available:
            return False

        try:
            self._cache[cache.image_hash] = cache
            await self._save_cache(cache)
            await self._add_to_meta(cache.image_hash)
            logger.debug(f"保存图片解析缓存：{cache.image_hash[:8]}...")
            return True

        except Exception as e:
            logger.warning(f"保存图片解析缓存失败：{e}")
            return False

    async def _add_to_meta(self, image_hash: str) -> None:
        """添加缓存键到元数据索引

        Args:
            image_hash: 图片 hash
        """
        try:
            async with self._lock:
                meta = await self._storage.get_kv_data(self.KV_KEY_META, {})
                if "hashes" not in meta:
                    meta["hashes"] = []

                if image_hash not in meta["hashes"]:
                    meta["hashes"].append(image_hash)
                    await self._storage.put_kv_data(self.KV_KEY_META, meta)

        except Exception as e:
            logger.warning(f"更新缓存元数据失败: {e}")

    async def _remove_from_meta(self, image_hash: str) -> None:
        """从元数据索引中移除缓存键

        Args:
            image_hash: 图片 hash
        """
        try:
            async with self._lock:
                meta = await self._storage.get_kv_data(self.KV_KEY_META, {})
                if "hashes" in meta and image_hash in meta["hashes"]:
                    meta["hashes"].remove(image_hash)
                    await self._storage.put_kv_data(self.KV_KEY_META, meta)

        except Exception as e:
            logger.warning(f"更新缓存元数据失败: {e}")

    async def _save_cache(self, cache: ImageParseCache) -> None:
        """保存缓存到 KV 存储

        Args:
            cache: 缓存对象
        """
        key = self._get_cache_key(cache.image_hash)
        data = cache.to_dict()
        await self._storage.put_kv_data(key, data)

    async def delete_cache(self, image_hash: str) -> bool:
        """删除缓存

        Args:
            image_hash: 图片 hash

        Returns:
            是否成功
        """
        if not self._is_available:
            return False

        try:
            if image_hash in self._cache:
                del self._cache[image_hash]

            key = self._get_cache_key(image_hash)
            await self._storage.delete_kv_data(key)
            await self._remove_from_meta(image_hash)
            logger.debug(f"删除图片解析缓存：{image_hash[:8]}...")
            return True

        except Exception as e:
            logger.warning(f"删除图片解析缓存失败：{e}")
            return False

    async def cleanup_expired(self, retention_days: Optional[int] = None) -> int:
        """清理过期缓存

        扫描 KV 存储中的所有缓存，清理过期的缓存项。

        Args:
            retention_days: 保留天数（None 则从配置读取）

        Returns:
            清理的缓存数量
        """
        if not self._is_available:
            return 0

        config = get_config()
        if retention_days is None:
            retention_days = config.get("image_cache_retention_days", 7)

        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0

        try:
            meta = await self._storage.get_kv_data(self.KV_KEY_META, {})
            all_hashes = meta.get("hashes", [])

            expired_hashes = []

            for image_hash in all_hashes:
                cache = await self._get_cache_for_cleanup(image_hash)
                if cache and cache.last_accessed < cutoff_time:
                    expired_hashes.append(image_hash)

            for image_hash in expired_hashes:
                if await self.delete_cache(image_hash):
                    cleaned_count += 1

        except Exception as e:
            logger.warning(f"扫描 KV 存储缓存失败：{e}")

        if cleaned_count > 0:
            logger.info(f"清理过期图片解析缓存：{cleaned_count} 条")

        return cleaned_count

    async def _get_cache_for_cleanup(
        self, image_hash: str
    ) -> Optional[ImageParseCache]:
        """获取缓存用于清理检查（不更新访问时间）

        Args:
            image_hash: 图片 hash

        Returns:
            缓存对象，不存在返回 None
        """
        try:
            key = self._get_cache_key(image_hash)
            data = await self._storage.get_kv_data(key, None)

            if data:
                return ImageParseCache.from_dict(data)

            return None

        except Exception as e:
            logger.warning(f"获取缓存失败：{e}")
            return None

    async def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息

        Returns:
            统计信息字典
        """
        return {
            "memory_cache_count": len(self._cache),
        }
