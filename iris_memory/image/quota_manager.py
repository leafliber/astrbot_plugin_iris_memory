"""
Iris Chat Memory - 图片解析配额管理组件

管理全局图片解析配额，支持每日自动重置。
"""

from datetime import datetime, date
from typing import TYPE_CHECKING
import asyncio

from iris_memory.core import Component, get_logger
from iris_memory.core.storage import KVStorage
from iris_memory.config import get_config
from .models import QuotaStatus

if TYPE_CHECKING:
    pass

logger = get_logger("image")


class ImageQuotaManager(Component):
    """图片解析配额管理组件

    管理全局图片解析配额，支持每日自动重置。
    使用 AstrBot KV 存储持久化配额状态。

    配额存储键：image_parsing_quota

    数据结构：
        {
            "date": "2026-03-29",
            "used": 15,
            "total": 200
        }

    Attributes:
        _storage: KV 存储适配器
        _is_available: 组件是否可用
        _lock: 异步锁（防止并发竞争）
        _quota_status: 配额状态（内存缓存）
    """

    KV_KEY = "image_parsing_quota"

    def __init__(self, storage: KVStorage):
        """初始化配额管理器

        Args:
            storage: KV 存储适配器（实现 KVStorage 协议的对象）
        """
        super().__init__()
        self._storage = storage
        self._lock = asyncio.Lock()
        self._quota_status: QuotaStatus | None = None

    @property
    def name(self) -> str:
        """组件名称"""
        return "image_quota"

    async def initialize(self) -> None:
        """初始化配额管理器"""
        config = get_config()

        if not config.get("l1_buffer.image_parsing.enable"):
            self._is_available = False
            logger.info("图片解析未启用")
            return

        await self._load_quota_status()

        self._is_available = True
        logger.info(
            f"图片解析配额管理器初始化完成，"
            f"配额：{self._quota_status.used}/{self._quota_status.total}"
        )

    async def shutdown(self) -> None:
        """关闭配额管理器"""
        self._reset_state()
        logger.info("图片解析配额管理器已关闭")

    async def _load_quota_status(self) -> None:
        """从 KV 存储加载配额状态"""
        try:
            data = await self._storage.get_kv_data(self.KV_KEY, {})

            if data:
                self._quota_status = QuotaStatus.from_dict(data)
                logger.debug(f"从 KV 存储加载配额状态：{self._quota_status.date}")
                await self._check_and_reset_if_needed()
            else:
                await self._create_initial_status()

        except Exception as e:
            logger.warning(f"从 KV 存储加载配额状态失败：{e}")
            await self._create_initial_status()

    async def _create_initial_status(self) -> None:
        """创建初始配额状态"""
        config = get_config()
        total = config.get("l1_buffer.image_parsing.daily_quota", 200)
        today = date.today().isoformat()

        self._quota_status = QuotaStatus(
            date=today, used=0, total=total, last_reset_time=datetime.now()
        )

        await self._save_quota_status()
        logger.info(f"创建初始配额状态：{today}, total={total}")

    async def _save_quota_status(self) -> None:
        """保存配额状态到 KV 存储"""
        if not self._quota_status:
            return

        try:
            data = self._quota_status.to_dict()
            await self._storage.put_kv_data(self.KV_KEY, data)
            logger.debug(
                f"保存配额状态：{self._quota_status.date}, used={self._quota_status.used}"
            )

        except Exception as e:
            logger.warning(f"保存配额状态到 KV 存储失败：{e}")

    async def _check_and_reset_if_needed(self) -> None:
        """检查是否需要重置配额（跨天）

        调用方必须已持有 self._lock。此方法内部直接调 _reset_quota_locked
        （不加锁），避免在持锁状态下重入 asyncio.Lock 导致死锁。
        asyncio.Lock 不可重入——持锁后再次 async with self._lock 会永久挂起。
        """
        if not self._quota_status:
            return

        today = date.today().isoformat()

        if self._quota_status.date != today:
            logger.info(
                f"检测到日期变更，重置配额：{self._quota_status.date} → {today}"
            )
            await self._reset_quota_locked()

    async def _reset_quota_locked(self) -> None:
        """重置配额（不加锁版，供已持锁的调用方使用）

        调用方必须已持有 self._lock。公开方法 reset_quota() 在持锁后调用此方法，
        _check_and_reset_if_needed() 同样如此——两条路径都避免重入 asyncio.Lock。
        """
        if not self._quota_status:
            await self._create_initial_status()
            return

        config = get_config()
        today = date.today().isoformat()

        self._quota_status.reset(
            today, config.get("l1_buffer.image_parsing.daily_quota", 200)
        )
        await self._save_quota_status()
        logger.info(f"配额已重置：{today}")

    async def check_quota(self) -> bool:
        """检查配额是否充足

        Returns:
            配额是否充足
        """
        if not self._is_available:
            return False

        async with self._lock:
            await self._check_and_reset_if_needed()

            if not self._quota_status:
                return False

            return not self._quota_status.is_exhausted

    async def use_quota(self, count: int = 1) -> bool:
        """使用配额

        Args:
            count: 使用数量（默认 1）

        Returns:
            是否成功使用
        """
        if not self._is_available:
            return False

        async with self._lock:
            await self._check_and_reset_if_needed()

            if not self._quota_status:
                return False

            if self._quota_status.is_exhausted:
                logger.warning(
                    f"配额已用尽：{self._quota_status.used}/{self._quota_status.total}"
                )
                return False

            if not self._quota_status.use(count):
                return False

            await self._save_quota_status()

            logger.debug(
                f"使用配额成功：count={count}, "
                f"used={self._quota_status.used}/{self._quota_status.total}"
            )

            return True

    async def release_quota(self, count: int = 1) -> int:
        """退还配额（图片解析失败/跳过/超时时回补预扣额度）

        Args:
            count: 退还数量

        Returns:
            实际退还数量
        """
        if not self._is_available or count <= 0:
            return 0
        async with self._lock:
            if not self._quota_status:
                return 0
            actual = self._quota_status.release(count)
            if actual > 0:
                await self._save_quota_status()
                logger.debug(
                    f"退还配额：count={actual}, "
                    f"used={self._quota_status.used}/{self._quota_status.total}"
                )
            return actual

    async def reset_quota(self) -> None:
        """重置配额（公开方法，加锁后委托 _reset_quota_locked）"""
        async with self._lock:
            await self._reset_quota_locked()

    async def get_status(self) -> QuotaStatus | None:
        """获取当前配额状态

        Returns:
            配额状态对象
        """
        if not self._is_available:
            return None

        async with self._lock:
            await self._check_and_reset_if_needed()
            return self._quota_status
