"""
KV 存储适配器

封装 AstrBot KV 接口，提供统计数据的持久化存储。
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable, Dict, List, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("stats.store")


class StatsKVStore:
    """KV 存储适配器
    
    封装 AstrBot KV 接口，提供统计数据的持久化。
    支持聚合统计和最近记录的存储。
    """
    
    KEY_AGGREGATED = "llm_stats_aggregated"
    KEY_RECORDS = "llm_stats_records"
    MAX_RECORDS = 1000
    
    def __init__(self) -> None:
        self._get_kv: Optional[Callable[[str, Any], Awaitable[Any]]] = None
        self._put_kv: Optional[Callable[[str, Any], Awaitable[None]]] = None
        self._pending_saves: int = 0
        self._save_threshold: int = 10
        self._last_aggregated: Dict[str, Any] = {}
        self._last_records: List[Dict[str, Any]] = []

    def set_kv_interface(
        self,
        get_kv_data: Callable[[str, Any], Awaitable[Any]],
        put_kv_data: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """设置 KV 接口
        
        Args:
            get_kv_data: AstrBot 的 get_kv_data 方法
            put_kv_data: AstrBot 的 put_kv_data 方法
        """
        self._get_kv = get_kv_data
        self._put_kv = put_kv_data
        logger.debug("StatsKVStore: KV interface configured")

    def is_ready(self) -> bool:
        """检查 KV 接口是否已配置"""
        return self._get_kv is not None and self._put_kv is not None

    async def load_aggregated(self) -> Dict[str, Any]:
        """加载聚合统计
        
        Returns:
            聚合统计数据字典，如果不存在则返回空字典
        """
        if not self._get_kv:
            logger.debug("StatsKVStore: get_kv not configured, returning empty aggregated")
            return {}
        
        try:
            data = await self._get_kv(self.KEY_AGGREGATED, {})
            self._last_aggregated = data
            logger.debug(f"StatsKVStore: Loaded aggregated stats, total_calls={data.get('total_calls', 0)}")
            return data
        except Exception as e:
            logger.warning(f"StatsKVStore: Failed to load aggregated stats: {e}")
            return {}

    async def save_aggregated(self, data: Dict[str, Any]) -> None:
        """保存聚合统计
        
        Args:
            data: 聚合统计数据字典
        """
        if not self._put_kv:
            logger.debug("StatsKVStore: put_kv not configured, skipping save")
            return
        
        try:
            await self._put_kv(self.KEY_AGGREGATED, data)
            self._last_aggregated = data
            logger.debug(f"StatsKVStore: Saved aggregated stats, total_calls={data.get('total_calls', 0)}")
        except Exception as e:
            logger.warning(f"StatsKVStore: Failed to save aggregated stats: {e}")

    async def load_records(self) -> List[Dict[str, Any]]:
        """加载最近记录
        
        Returns:
            记录列表，如果不存在则返回空列表
        """
        if not self._get_kv:
            logger.debug("StatsKVStore: get_kv not configured, returning empty records")
            return []
        
        try:
            data = await self._get_kv(self.KEY_RECORDS, [])
            self._last_records = data if isinstance(data, list) else []
            logger.debug(f"StatsKVStore: Loaded {len(self._last_records)} records")
            return self._last_records
        except Exception as e:
            logger.warning(f"StatsKVStore: Failed to load records: {e}")
            return []

    async def save_records(self, records: List[Dict[str, Any]]) -> None:
        """保存最近记录（自动截断）
        
        Args:
            records: 记录列表
        """
        if not self._put_kv:
            logger.debug("StatsKVStore: put_kv not configured, skipping save")
            return
        
        try:
            truncated = records[-self.MAX_RECORDS:]
            await self._put_kv(self.KEY_RECORDS, truncated)
            self._last_records = truncated
            logger.debug(f"StatsKVStore: Saved {len(truncated)} records")
        except Exception as e:
            logger.warning(f"StatsKVStore: Failed to save records: {e}")

    async def save_all(
        self,
        aggregated: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> None:
        """保存所有数据
        
        Args:
            aggregated: 聚合统计
            records: 记录列表
        """
        await self.save_aggregated(aggregated)
        await self.save_records(records)

    def get_cached_aggregated(self) -> Dict[str, Any]:
        """获取缓存的聚合统计（不触发 IO）"""
        return self._last_aggregated.copy()

    def get_cached_records(self) -> List[Dict[str, Any]]:
        """获取缓存的记录列表（不触发 IO）"""
        return self._last_records.copy()

    async def clear_all(self) -> None:
        """清除所有统计数据"""
        await self.save_aggregated({})
        await self.save_records([])
        logger.info("StatsKVStore: All stats cleared")
