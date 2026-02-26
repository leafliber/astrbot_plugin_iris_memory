"""Web 记忆管理服务

封装面向 Web 的记忆 CRUD 操作，委托 MemoryRepository 处理数据访问。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.web.data.memory_repo import MemoryRepositoryImpl
from iris_memory.web.service.audit import audit_log
from iris_memory.web.service.dto.converters import memory_detail_from_chroma, memory_to_web_dict
from iris_memory.utils.logger import get_logger

logger = get_logger("memory_web_service")


class MemoryWebService:
    """Web 端记忆管理服务

    通过 MemoryRepository 隔离数据访问，
    自身仅负责业务编排和审计。
    """

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service
        self._repo = MemoryRepositoryImpl(memory_service)

    async def search_memories_web(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Web 端记忆检索

        支持向量搜索和条件过滤，带分页。

        Returns:
            {items: [...], total: N, page: N, page_size: N}
        """
        page = max(1, page)
        page_size = max(1, min(page_size, 100))

        # 有搜索词时做向量搜索
        if query and user_id:
            items = await self._repo.search(
                query=query,
                user_id=user_id,
                group_id=group_id,
                storage_layer=storage_layer,
                memory_type=memory_type,
                top_k=100,
            )
            total = len(items)
            start = (page - 1) * page_size
            return {
                "items": items[start:start + page_size],
                "total": total,
                "page": page,
                "page_size": page_size,
            }

        # 无搜索词时列出所有记忆
        return await self._repo.list_all(
            user_id=user_id,
            group_id=group_id,
            storage_layer=storage_layer,
            memory_type=memory_type,
            page=page,
            page_size=page_size,
        )

    async def get_memory_detail(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆的完整详情

        Args:
            memory_id: 记忆 ID

        Returns:
            完整记忆字典，不存在返回 None
        """
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return None

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return None

            return memory_detail_from_chroma(res, 0, full=True)

        except Exception as e:
            logger.warning(f"Get memory detail error: {e}")
            return None

    async def update_memory_by_id(
        self,
        memory_id: str,
        updates: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """更新单条记忆的内容/元数据

        Args:
            memory_id: 记忆 ID
            updates: 要更新的字段字典

        Returns:
            (success, message)
        """
        success, msg = await self._repo.update(memory_id, updates)
        if success:
            audit_log("update_memory", f"id={memory_id} fields={list(updates.keys())}")
        return success, msg

    async def delete_memory_by_id(self, memory_id: str) -> Tuple[bool, str]:
        """删除单条记忆

        Returns:
            (success, message)
        """
        success, msg = await self._repo.delete(memory_id)
        if success:
            audit_log("delete_memory", f"id={memory_id}")
        return success, msg

    async def batch_delete_memories(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除记忆

        Returns:
            {success_count, fail_count, errors}
        """
        result = await self._repo.batch_delete(memory_ids)
        audit_log(
            "batch_delete_memories",
            f"total={len(memory_ids)} success={result['success_count']}",
        )
        return result
