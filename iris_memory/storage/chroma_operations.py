"""
Chroma 操作模块

将CRUD操作从 ChromaManager 中拆分出来，提高代码可维护性。
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from iris_memory.utils.logger import get_logger
from iris_memory.core.memory_scope import MemoryScope

logger = get_logger("chroma_operations")


class ChromaOperations:
    """Chroma CRUD 操作 Mixin
    
    职责：
    1. 记忆添加
    2. 记忆更新
    3. 记忆删除
    4. 会话删除
    5. 批量删除操作
    """

    async def add_memory(self, memory) -> Optional[str]:
        """添加记忆到Chroma
        
        Args:
            memory: 记忆对象
            
        Returns:
            Optional[str]: 记忆ID，如果嵌入生成失败则返回None
        """
        try:
            self._ensure_ready()
            
            if memory.embedding is None:
                embedding = await self._generate_embedding(memory.content)
                if embedding is None:
                    logger.error(f"Failed to generate embedding for memory {memory.id}, skipping storage")
                    return None
                memory.embedding = np.array(embedding)
            else:
                embedding = memory.embedding.tolist()
            
            metadata = self._build_memory_metadata(memory)
            metadata.update(memory.metadata)
            
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Memory added to Chroma: id={memory.id}, user={memory.user_id}, storage_layer={memory.storage_layer.value}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Failed to add memory to Chroma: id={memory.id}, error={e}", exc_info=True)
            raise

    @staticmethod
    def _safe_enum_value(value) -> Any:
        """安全获取枚举值，兼容字符串和枚举类型"""
        return value.value if hasattr(value, 'value') and not isinstance(value, str) else value

    def _build_memory_metadata(self, memory) -> Dict[str, Any]:
        """构建记忆元数据"""
        return {
            "user_id": memory.user_id,
            "sender_name": memory.sender_name if memory.sender_name else "",
            "group_id": memory.group_id if memory.group_id else "",
            "scope": self._safe_enum_value(memory.scope),
            "type": self._safe_enum_value(memory.type),
            "modality": self._safe_enum_value(memory.modality),
            "quality_level": self._safe_enum_value(memory.quality_level),
            "sensitivity_level": self._safe_enum_value(memory.sensitivity_level),
            "storage_layer": self._safe_enum_value(memory.storage_layer),
            "created_time": memory.created_time.isoformat(),
            "last_access_time": memory.last_access_time.isoformat(),
            "access_count": memory.access_count,
            "rif_score": memory.rif_score,
            "importance_score": memory.importance_score,
            "is_user_requested": memory.is_user_requested,
        }

    def _log_memory_details(self, memory, embedding: List[float]) -> None:
        """记录记忆详情（单行汇总）"""
        if not logger.isEnabledFor(10):
            return
        
        content_preview = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
        logger.debug(
            f"Memory detail: id={memory.id[:8]}... user={memory.user_id} "
            f"type={self._safe_enum_value(memory.type)} scope={self._safe_enum_value(memory.scope)} "
            f"layer={self._safe_enum_value(memory.storage_layer)} rif={memory.rif_score:.3f} "
            f"quality={self._safe_enum_value(memory.quality_level)} content='{content_preview}'"
        )

    async def update_memory(self, memory) -> bool:
        """更新记忆
        
        Args:
            memory: 更新后的记忆对象
            
        Returns:
            bool: 是否更新成功
        """
        try:
            self._ensure_ready()
            if memory.embedding is None:
                embedding = await self._generate_embedding(memory.content)
                memory.embedding = np.array(embedding)
            else:
                embedding = memory.embedding.tolist()
            
            # 复用 _build_memory_metadata 以保证字段一致性
            metadata = self._build_memory_metadata(memory)
            metadata.update(memory.metadata)
            
            self.collection.update(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Memory updated in Chroma: {memory.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False

    async def batch_update_access_stats(self, memories: list) -> int:
        """批量更新记忆的访问统计到 ChromaDB
        
        仅更新 metadata 中的 access_count、last_access_time 和 access_frequency，
        不重新生成 embedding，以保证高效。
        
        Args:
            memories: 已调用 update_access() 后的记忆对象列表
            
        Returns:
            int: 成功更新的记忆数量
        """
        if not memories:
            return 0
        
        try:
            self._ensure_ready()
        except Exception:
            return 0
        
        updated = 0
        # 收集需要更新的 id 和 metadata
        ids = []
        metadatas = []
        for memory in memories:
            try:
                ids.append(memory.id)
                # 仅更新访问相关字段，使用 collection.get 获取现有 metadata 并合并
                metadatas.append({
                    "access_count": memory.access_count,
                    "last_access_time": memory.last_access_time.isoformat(),
                })
            except Exception as e:
                logger.debug(f"Skip access stats update for {getattr(memory, 'id', '?')}: {e}")
        
        if not ids:
            return 0
        
        try:
            # 先获取现有 metadata
            existing = self.collection.get(ids=ids, include=["metadatas"])
            if existing and existing.get("ids"):
                merged_metadatas = []
                for i, mid in enumerate(existing["ids"]):
                    meta = dict(existing["metadatas"][i]) if i < len(existing["metadatas"]) else {}
                    # 找到对应的更新数据
                    idx = ids.index(mid) if mid in ids else -1
                    if idx >= 0:
                        meta.update(metadatas[idx])
                    merged_metadatas.append(meta)
                
                self.collection.update(
                    ids=existing["ids"],
                    metadatas=merged_metadatas,
                )
                updated = len(existing["ids"])
                logger.debug(f"Batch updated access stats for {updated} memories")
        except Exception as e:
            logger.warning(f"Batch access stats update failed: {e}")
        
        return updated

    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            self._ensure_ready()
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Memory deleted from Chroma: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    async def delete_session(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """删除会话的所有记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            
        Returns:
            bool: 是否删除成功
        """
        try:
            self._ensure_ready()
            where = {"user_id": user_id}
            where["group_id"] = group_id if group_id else ""

            results = self.collection.get(where=self._build_where_clause(where))
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.debug(f"Deleted {len(results['ids'])} memories for session {user_id}/{group_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session memories: {e}")
            return False

    async def delete_user_memories(self, user_id: str, in_private_only: bool = False) -> Tuple[bool, int]:
        """删除用户的所有记忆（跨群聊）
        
        Args:
            user_id: 用户ID
            in_private_only: 是否只删除私聊记忆（不包括群聊）
            
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            self._ensure_ready()
            all_ids = set()
            
            if in_private_only:
                where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
                results = self.collection.get(where=self._build_where_clause(where_user_private))
                if results['ids']:
                    all_ids.update(results['ids'])
                
                where_global = {"user_id": user_id, "scope": MemoryScope.GLOBAL.value, "group_id": ""}
                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    all_ids.update(results['ids'])
            else:
                where = {"user_id": user_id}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            
            if all_ids:
                self.collection.delete(ids=list(all_ids))
                logger.debug(f"Deleted {len(all_ids)} memories for user {user_id} (private_only={in_private_only})")
                return True, len(all_ids)
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Failed to delete user memories: {e}")
            return False, 0

    async def delete_group_memories(self, group_id: str, scope_filter: Optional[str] = None) -> Tuple[bool, int]:
        """删除群组的记忆
        
        Args:
            group_id: 群组ID
            scope_filter: 可选的scope过滤（"group_shared" 只删除共享记忆，"group_private" 只删除个人记忆）
            
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            self._ensure_ready()
            all_ids = set()
            
            if scope_filter == "group_shared":
                where = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            elif scope_filter == "group_private":
                where = {"group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            else:
                where = {"group_id": group_id}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            
            if all_ids:
                self.collection.delete(ids=list(all_ids))
                logger.debug(f"Deleted {len(all_ids)} memories for group {group_id} (scope_filter={scope_filter})")
                return True, len(all_ids)
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Failed to delete group memories: {e}")
            return False, 0

    async def delete_all_memories(self) -> Tuple[bool, int]:
        """删除所有记忆（危险操作，需要管理员权限）
        
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            self._ensure_ready()
            results = self.collection.get()
            
            if not results or not results.get('ids'):
                logger.debug("Database is empty, nothing to delete")
                return True, 0
            
            count = len(results['ids'])
            
            self._log_delete_details(results, count)
            
            if count > 0:
                self.collection.delete(ids=results['ids'])
                logger.warning(f"Deleted ALL {count} memories from database!")
            
            return True, count
            
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}", exc_info=True)
            return False, 0

    def _log_delete_details(self, results: Dict, count: int) -> None:
        """记录删除详情"""
        if not logger.isEnabledFor(10) or count <= 0:
            return
        
        logger.debug(f"Preparing to delete {count} memories")
