"""仪表盘服务

聚合多个数据源的统计信息，为仪表盘提供概览数据。
"""

from __future__ import annotations

from typing import Any, Dict, List

from iris_memory.web.data.memory_repo import MemoryRepositoryImpl
from iris_memory.web.data.session_repo import SessionRepositoryImpl
from iris_memory.utils.logger import get_logger

logger = get_logger("dashboard_service")


class DashboardService:
    """仪表盘业务服务

    聚合会话、记忆、知识图谱等多维统计数据。
    """

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service
        self._memory_repo = MemoryRepositoryImpl(memory_service)
        self._session_repo = SessionRepositoryImpl(memory_service)

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """获取仪表盘统计信息

        Returns:
            包含系统概览、记忆统计、KG 统计的字典
        """
        stats: Dict[str, Any] = {
            "system": await self._get_system_stats(),
            "memories": await self._get_memory_overview(),
            "knowledge_graph": await self._get_kg_overview(),
            "health": {},
        }

        try:
            stats["health"] = self._service.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            stats["health"] = {"status": "unknown"}

        return stats

    async def get_memory_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取记忆创建趋势数据

        Args:
            days: 回溯天数

        Returns:
            按日期分组的记忆创建数量列表
        """
        return await self._memory_repo.get_trend(days)

    async def _get_system_stats(self) -> Dict[str, Any]:
        """系统级统计"""
        result = await self._session_repo.get_session_stats()

        try:
            result["total_personas"] = len(self._service._user_personas)
        except Exception:
            result["total_personas"] = 0

        return result

    async def _get_memory_overview(self) -> Dict[str, Any]:
        """记忆总览统计"""
        result: Dict[str, Any] = {
            "total_count": 0,
            "by_layer": {"working": 0, "episodic": 0, "semantic": 0},
            "by_type": {},
        }

        # 统计工作记忆（仅存在于内存中）
        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for _key, memories in session_mgr.working_memory_cache.items():
                    working_count = len(memories) if memories else 0
                    result["by_layer"]["working"] += working_count
                    result["total_count"] += working_count
                    for mem in (memories or []):
                        mtype = getattr(mem, "type", None)
                        if mtype:
                            mtype_val = mtype.value if hasattr(mtype, "value") else str(mtype)
                            result["by_type"][mtype_val] = result["by_type"].get(mtype_val, 0) + 1
        except Exception as e:
            logger.debug(f"Working memory stats error: {e}")

        # 统计 ChromaDB 中的持久化记忆
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return result

            collection = chroma.collection
            total = collection.count()
            result["total_count"] += total

            if total == 0:
                return result

            res = collection.get(include=["metadatas"])
            if not res["ids"]:
                return result

            for meta in res["metadatas"]:
                layer = meta.get("storage_layer", "")
                if layer in result["by_layer"]:
                    result["by_layer"][layer] += 1
                mtype = meta.get("type", "")
                if mtype:
                    result["by_type"][mtype] = result["by_type"].get(mtype, 0) + 1

        except Exception as e:
            logger.warning(f"Memory overview error: {e}")

        return result

    async def _get_kg_overview(self) -> Dict[str, Any]:
        """知识图谱总览"""
        result: Dict[str, Any] = {"nodes": 0, "edges": 0, "enabled": False}

        try:
            kg = self._service.kg
            if kg and kg.enabled:
                result["enabled"] = True
                stats = await kg.get_stats()
                result["nodes"] = stats.get("nodes", 0)
                result["edges"] = stats.get("edges", 0)
        except Exception as e:
            logger.debug(f"KG overview error: {e}")

        return result
