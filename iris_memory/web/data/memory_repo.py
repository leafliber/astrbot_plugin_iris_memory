"""记忆数据仓库实现

实现 MemoryRepository 接口，封装 ChromaDB 和 SessionManager 的访问。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

logger = get_logger("memory_repo")


class MemoryRepositoryImpl:
    """记忆数据仓库实现"""

    def __init__(self, memory_service: Any) -> None:
        """初始化仓库

        Args:
            memory_service: MemoryService 实例
        """
        self._service = memory_service

    async def search(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 100,
    ) -> List[Dict[str, Any]]:
        """向量搜索记忆"""
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return []

            sl = StorageLayer(storage_layer) if storage_layer else None
            memories = await chroma.query_memories(
                query_text=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                storage_layer=sl,
            )

            items = []
            for m in memories:
                item = self._memory_to_dict(m)
                if memory_type and item.get("type") != memory_type:
                    continue
                items.append(item)
            return items
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取记忆"""
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return None

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])

            if not res["ids"]:
                return None

            return self._dict_from_chroma(res, 0)
        except Exception as e:
            logger.warning(f"Get memory by ID error: {e}")
            return None

    async def list_all(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        storage_layer: Optional[str] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """分页列出记忆"""
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        result: Dict[str, Any] = {"items": [], "total": 0, "page": page, "page_size": page_size}

        all_items: List[Dict[str, Any]] = []

        # 获取工作记忆（从内存缓存）
        if not storage_layer or storage_layer == "working":
            try:
                session_mgr = self._service.session_manager
                if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                    for session_key, memories in session_mgr.working_memory_cache.items():
                        # 解析 session_key: "user_id:group_id" 或 "user_id:private"
                        key_parts = session_key.split(":")
                        session_user_id = key_parts[0] if key_parts else ""
                        session_group_id = key_parts[1] if len(key_parts) > 1 else None

                        # 过滤条件
                        if user_id and session_user_id != user_id:
                            continue
                        if group_id and session_group_id != group_id:
                            continue

                        for memory in memories or []:
                            item = self._memory_to_dict(memory)
                            if memory_type and item.get("type") != memory_type:
                                continue
                            all_items.append(item)
            except Exception as e:
                logger.debug(f"Working memory list error: {e}")

        # 获取持久化记忆（从 ChromaDB）
        if not storage_layer or storage_layer != "working":
            try:
                chroma = self._service.chroma_manager
                if chroma and chroma.is_ready:
                    collection = chroma.collection
                    where_clause: Dict[str, Any] = {}

                    if user_id:
                        where_clause["user_id"] = user_id
                    if group_id:
                        where_clause["group_id"] = group_id
                    if storage_layer:
                        where_clause["storage_layer"] = storage_layer
                    if memory_type:
                        where_clause["type"] = memory_type

                    if where_clause:
                        built = chroma._build_where_clause(where_clause)
                        res = collection.get(where=built, include=["documents", "metadatas"])
                    else:
                        res = collection.get(include=["documents", "metadatas"])

                    if res["ids"]:
                        for i in range(len(res["ids"])):
                            item = self._dict_from_chroma(res, i)
                            all_items.append(item)
            except Exception as e:
                logger.warning(f"List memories error: {e}")

        all_items.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        result["total"] = len(all_items)
        start = (page - 1) * page_size
        result["items"] = all_items[start:start + page_size]

        return result

    async def create(self, memory_data: Dict[str, Any]) -> Tuple[bool, str]:
        """创建记忆"""
        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            memory = Memory(
                content=memory_data["content"],
                user_id=memory_data["user_id"],
                sender_name=memory_data.get("sender_name", ""),
                group_id=memory_data.get("group_id"),
                storage_layer=StorageLayer(memory_data.get("storage_layer", "episodic")),
            )

            success = await chroma.add_memory(memory)
            if success:
                return True, "创建成功"
            return False, "创建失败"

        except Exception as e:
            logger.error(f"Create memory error: {e}")
            return False, f"创建失败: {e}"

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """更新记忆"""
        allowed_keys = {"content", "type", "storage_layer", "confidence", "importance_score", "summary"}
        invalid_keys = set(updates.keys()) - allowed_keys
        if invalid_keys:
            return False, f"不允许更新的字段: {', '.join(invalid_keys)}"

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return False, "存储服务未就绪"

            collection = chroma.collection
            res = collection.get(ids=[memory_id], include=["documents", "metadatas"])
            if not res["ids"]:
                return False, "记忆不存在"

            doc = res["documents"][0] if res.get("documents") else ""
            meta = res["metadatas"][0] if res.get("metadatas") else {}

            storage_layer_val = updates.get("storage_layer", meta.get("storage_layer", "episodic"))
            try:
                sl = StorageLayer(storage_layer_val)
            except ValueError:
                return False, f"无效的 storage_layer: {storage_layer_val}"

            memory = Memory(
                id=memory_id,
                content=updates.get("content", doc),
                user_id=meta.get("user_id", ""),
                sender_name=meta.get("sender_name", ""),
                group_id=meta.get("group_id") or None,
                storage_layer=sl,
                created_time=datetime.fromisoformat(meta["created_time"]) if meta.get("created_time") else datetime.now(),
            )

            type_val = updates.get("type", meta.get("type"))
            if type_val:
                try:
                    memory.type = MemoryType(type_val)
                except ValueError:
                    return False, f"无效的 type: {type_val}"

            try:
                memory.confidence = float(updates.get("confidence", meta.get("confidence", 0.5)))
            except (ValueError, TypeError):
                return False, "confidence 必须是有效数字"
            try:
                memory.importance_score = float(updates.get("importance_score", meta.get("importance_score", 0.5)))
            except (ValueError, TypeError):
                return False, "importance_score 必须是有效数字"

            if updates.get("summary") is not None:
                memory.summary = updates["summary"]
            elif meta.get("summary"):
                memory.summary = meta["summary"]

            success = await chroma.update_memory(memory)
            if success:
                return True, "更新成功"
            return False, "更新失败"

        except Exception as e:
            logger.error(f"Update memory error: {e}")
            return False, f"更新失败: {e}"

    async def delete(self, memory_id: str) -> Tuple[bool, str]:
        """删除记忆

        优先尝试从 ChromaDB 删除持久化记忆，如果失败则尝试从工作记忆缓存中删除。
        """
        # 首先尝试从 ChromaDB 删除
        chroma = self._service.chroma_manager
        if chroma and chroma.is_ready:
            try:
                success = await chroma.delete_memory(memory_id)
                if success:
                    # 删除关联的知识图谱边
                    kg = self._service.kg
                    if kg and kg.enabled and kg.storage:
                        try:
                            edge_count = await kg.storage.delete_by_memory_id(memory_id)
                            if edge_count > 0:
                                logger.debug(f"Deleted {edge_count} KG edges for memory {memory_id}")
                        except Exception as e:
                            logger.warning(f"Failed to delete KG edges for memory {memory_id}: {e}")
                    return True, "删除成功"
            except Exception as e:
                logger.debug(f"ChromaDB delete failed for {memory_id}: {e}")

        # 如果 ChromaDB 删除失败或记忆不存在，尝试从工作记忆删除
        session_mgr = self._service.session_manager
        if session_mgr and hasattr(session_mgr, "working_memory_cache"):
            try:
                # 遍历所有会话查找并删除工作记忆
                for session_key, memories in session_mgr.working_memory_cache.items():
                    for memory in memories or []:
                        if memory.id == memory_id:
                            # 解析 session_key 获取 user_id 和 group_id
                            key_parts = session_key.split(":")
                            user_id = key_parts[0] if key_parts else ""
                            group_id = key_parts[1] if len(key_parts) > 1 else None

                            # 使用 session_manager 的方法删除工作记忆
                            removed = await session_mgr.remove_working_memory(
                                user_id=user_id,
                                group_id=group_id,
                                memory_id=memory_id
                            )
                            if removed:
                                logger.debug(f"Deleted working memory {memory_id} from session {session_key}")
                                return True, "删除成功"
            except Exception as e:
                logger.error(f"Working memory delete error: {e}")

        return False, "记忆不存在或删除失败"

    async def _delete_from_working_memory(self, memory_id: str) -> bool:
        """从工作记忆缓存中删除指定记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            是否成功删除
        """
        session_mgr = self._service.session_manager
        if not session_mgr or not hasattr(session_mgr, "working_memory_cache"):
            return False

        try:
            for session_key, memories in session_mgr.working_memory_cache.items():
                for memory in memories or []:
                    if memory.id == memory_id:
                        key_parts = session_key.split(":")
                        user_id = key_parts[0] if key_parts else ""
                        group_id = key_parts[1] if len(key_parts) > 1 else None

                        return await session_mgr.remove_working_memory(
                            user_id=user_id,
                            group_id=group_id,
                            memory_id=memory_id
                        )
        except Exception as e:
            logger.debug(f"Delete from working memory error: {e}")

        return False

    async def batch_delete(self, memory_ids: List[str]) -> Dict[str, Any]:
        """批量删除"""
        success_count = 0
        fail_count = 0
        errors: List[str] = []

        for mid in memory_ids:
            ok, msg = await self.delete(mid)
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(f"{mid}: {msg}")

        return {
            "success_count": success_count,
            "fail_count": fail_count,
            "errors": errors[:10],
        }

    async def count_by_layer(self) -> Dict[str, int]:
        """按层级统计记忆数量"""
        result: Dict[str, int] = {"working": 0, "episodic": 0, "semantic": 0, "total": 0}

        try:
            session_mgr = self._service.session_manager
            if session_mgr and hasattr(session_mgr, "working_memory_cache"):
                for memories in session_mgr.working_memory_cache.values():
                    result["working"] += len(memories or [])
                    result["total"] += len(memories or [])

            chroma = self._service.chroma_manager
            if chroma and chroma.is_ready:
                collection = chroma.collection
                total = collection.count()
                result["total"] += total

                res = collection.get(include=["metadatas"])
                for meta in res["metadatas"]:
                    layer = meta.get("storage_layer", "")
                    if layer in result:
                        result[layer] += 1

        except Exception as e:
            logger.warning(f"Count by layer error: {e}")

        return result

    async def get_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取创建趋势"""
        trend: List[Dict[str, Any]] = []

        try:
            chroma = self._service.chroma_manager
            if not chroma or not chroma.is_ready:
                return trend

            collection = chroma.collection
            results = collection.get(include=["metadatas"])

            if not results["ids"]:
                return trend

            date_counts: Dict[str, int] = {}
            cutoff = datetime.now() - timedelta(days=days)

            for meta in results["metadatas"]:
                created = meta.get("created_time", "")
                if not created:
                    continue
                try:
                    dt = datetime.fromisoformat(created)
                    if dt >= cutoff:
                        date_key = dt.strftime("%Y-%m-%d")
                        date_counts[date_key] = date_counts.get(date_key, 0) + 1
                except (ValueError, TypeError):
                    pass

            for i in range(days):
                d = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                trend.append({"date": d, "count": date_counts.get(d, 0)})

        except Exception as e:
            logger.warning(f"Get trend error: {e}")

        return trend

    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
        """Memory 模型转字典"""
        return {
            "id": memory.id,
            "content": memory.content[:500] if len(memory.content) > 500 else memory.content,
            "user_id": memory.user_id,
            "group_id": memory.group_id or "",
            "sender_name": memory.sender_name,
            "type": memory.type.value if hasattr(memory.type, "value") else str(memory.type),
            "storage_layer": memory.storage_layer.value if hasattr(memory.storage_layer, "value") else str(memory.storage_layer),
            "scope": memory.scope.value if hasattr(memory.scope, "value") else str(memory.scope),
            "confidence": memory.confidence,
            "importance_score": memory.importance_score,
            "created_time": memory.created_time.isoformat() if isinstance(memory.created_time, datetime) else str(memory.created_time),
            "summary": getattr(memory, "summary", "") or "",
        }

    def _dict_from_chroma(self, res: Dict[str, Any], index: int) -> Dict[str, Any]:
        """从 ChromaDB 结果转换"""
        return {
            "id": res["ids"][index],
            "content": res["documents"][index] if res.get("documents") else "",
            "user_id": res["metadatas"][index].get("user_id", ""),
            "group_id": res["metadatas"][index].get("group_id", ""),
            "sender_name": res["metadatas"][index].get("sender_name", ""),
            "type": res["metadatas"][index].get("type", ""),
            "storage_layer": res["metadatas"][index].get("storage_layer", ""),
            "scope": res["metadatas"][index].get("scope", ""),
            "confidence": res["metadatas"][index].get("confidence", 0),
            "importance_score": res["metadatas"][index].get("importance_score", 0),
            "created_time": res["metadatas"][index].get("created_time", ""),
            "summary": res["metadatas"][index].get("summary", ""),
        }
