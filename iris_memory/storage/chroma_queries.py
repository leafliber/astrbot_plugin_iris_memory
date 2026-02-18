"""
Chroma 查询模块

将查询相关逻辑从 ChromaManager 中拆分出来，提高代码可维护性。
"""

from typing import List, Dict, Any, Optional
import numpy as np

from iris_memory.utils.logger import get_logger
from iris_memory.core.memory_scope import MemoryScope
from iris_memory.core.types import StorageLayer

logger = get_logger("chroma_queries")


class ChromaQueries:
    """Chroma 查询操作 Mixin
    
    职责：
    1. 向量相似度检索
    2. 多范围查询（群聊/私聊/全局）
    3. 结果去重和排序
    4. 结果转换为Memory对象
    """

    async def query_memories(
        self,
        query_text: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        storage_layer: Optional[StorageLayer] = None
    ) -> List:
        """查询相关记忆（支持多层级记忆）

        查询策略：
        1. 私聊场景（group_id=None）：
           - 查询所有用户私有记忆（scope=USER_PRIVATE）
           - 查询用户的全局记忆（scope=GLOBAL）
        
        2. 群聊场景（group_id有值）：
           - 查询该群组的共享记忆（scope=GROUP_SHARED）
           - 查询该用户在该群组的个人记忆（scope=GROUP_PRIVATE AND user_id=当前用户）
           - 查询全局记忆（scope=GLOBAL）

        Args:
            query_text: 查询文本
            user_id: 用户ID
            group_id: 群组ID（私聊时为None）
            top_k: 返回的最大数量
            storage_layer: 存储层过滤（可选）

        Returns:
            List[Memory]: 相关记忆列表（如果嵌入生成失败则返回空列表）
        """
        try:
            self._ensure_ready()
            logger.debug(f"Querying memories: user={user_id}, group={group_id}, top_k={top_k}, storage_layer={storage_layer.value if storage_layer else 'all'}")
            
            query_embedding = await self._generate_embedding(query_text)
            if query_embedding is None:
                logger.error("Failed to generate query embedding, returning empty results")
                return []
            logger.debug(f"Query embedding generated: dimension={len(query_embedding)}")

            all_results = []

            if group_id:
                all_results = await self._query_group_mode(
                    query_embedding, user_id, group_id, top_k, storage_layer
                )
            else:
                all_results = await self._query_private_mode(
                    query_embedding, user_id, top_k, storage_layer
                )

            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            logger.debug(f"Total results: {len(all_results)}, Unique results: {len(unique_results)}")
            
            self._log_query_results(all_results)

            if unique_results:
                unique_results.sort(key=lambda x: x['distance'] if x['distance'] is not None else float('inf'))
                unique_results = unique_results[:top_k]
                logger.debug(f"Sorted and limited to top {len(unique_results)} results")

            memories = []
            for memory_data in unique_results:
                memory_data_without_distance = {k: v for k, v in memory_data.items() if k != 'distance'}
                memory = self._result_to_memory(memory_data_without_distance)
                memories.append(memory)
            
            self._log_final_memories(memories)

            logger.info(f"Queried {len(memories)} memories for user={user_id}, group={group_id}, query='{query_text[:30]}...'")
            return memories

        except Exception as e:
            logger.error(f"Failed to query memories: user={user_id}, error={e}", exc_info=True)
            return []

    async def _query_group_mode(
        self,
        query_embedding: List[float],
        user_id: str,
        group_id: str,
        top_k: int,
        storage_layer: Optional[StorageLayer]
    ) -> List[Dict[str, Any]]:
        """群聊场景查询"""
        all_results = []
        
        where_shared = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
        if storage_layer:
            where_shared["storage_layer"] = storage_layer.value
        
        logger.debug(f"Querying GROUP_SHARED memories: where={where_shared}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where_clause(where_shared),
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        all_results.extend(self._extract_query_results(results))
        logger.debug(f"Found {len(results['ids'][0]) if results['ids'] else 0} GROUP_SHARED memories")

        where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
        if storage_layer:
            where_private["storage_layer"] = storage_layer.value

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where_clause(where_private),
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        all_results.extend(self._extract_query_results(results))

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where_clause(where_global),
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        all_results.extend(self._extract_query_results(results))
        
        return all_results

    async def _query_private_mode(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int,
        storage_layer: Optional[StorageLayer]
    ) -> List[Dict[str, Any]]:
        """私聊场景查询"""
        all_results = []
        
        where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
        if storage_layer:
            where_user_private["storage_layer"] = storage_layer.value

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where_clause(where_user_private),
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        all_results.extend(self._extract_query_results(results))

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=self._build_where_clause(where_global),
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        all_results.extend(self._extract_query_results(results))
        
        return all_results

    def _extract_query_results(self, results: Dict) -> List[Dict[str, Any]]:
        """从Chroma查询结果中提取数据"""
        extracted = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                extracted.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                    'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                    'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                })
        return extracted

    def _log_query_results(self, results: List[Dict]) -> None:
        """记录查询结果详情"""
        if not logger.isEnabledFor(10):
            return
        
        for i, result in enumerate(results[:5], 1):
            distance = result.get('distance')
            distance_str = f"{distance:.4f}" if distance is not None else "N/A"
            content = result.get('content', '')
            if len(content) > 50:
                content_str = f"content='{content[:50]}...'"
            else:
                content_str = f"content='{content}'"
            logger.debug(f"  Raw result {i}: id={result['id'][:8]}..., distance={distance_str}, {content_str}")

    def _log_final_memories(self, memories: List) -> None:
        """记录最终Memory对象"""
        if not logger.isEnabledFor(10) or not memories:
            return
        
        logger.debug(f"Final query results ({len(memories)} memories):")
        for i, memory in enumerate(memories, 1):
            logger.debug(f"  [{i}] ID={memory.id[:8]}..., Type={memory.type.value}, "
                       f"Scope={memory.scope.value}, Layer={memory.storage_layer.value}, "
                       f"RIF={memory.rif_score:.3f}, Content='{memory.content[:40]}...'")

    async def get_all_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List:
        """获取用户的所有记忆（支持多层级记忆）"""
        try:
            self._ensure_ready()
            all_memories = []

            if group_id:
                all_memories = await self._get_all_group_memories(user_id, group_id, storage_layer)
            else:
                all_memories = await self._get_all_private_memories(user_id, storage_layer)

            seen_ids = set()
            unique_memories = []
            for memory in all_memories:
                if memory.id not in seen_ids:
                    seen_ids.add(memory.id)
                    unique_memories.append(memory)

            return unique_memories

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    async def _get_all_group_memories(
        self,
        user_id: str,
        group_id: str,
        storage_layer: Optional[StorageLayer]
    ) -> List:
        """获取群聊场景的所有记忆"""
        all_memories = []
        
        where_shared = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
        if storage_layer:
            where_shared["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_shared))
        all_memories.extend(self._results_to_memories(results))

        where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
        if storage_layer:
            where_private["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_private))
        all_memories.extend(self._results_to_memories(results))

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_global))
        all_memories.extend(self._results_to_memories(results))
        
        return all_memories

    async def _get_all_private_memories(
        self,
        user_id: str,
        storage_layer: Optional[StorageLayer]
    ) -> List:
        """获取私聊场景的所有记忆"""
        all_memories = []
        
        where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
        if storage_layer:
            where_user_private["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_user_private))
        all_memories.extend(self._results_to_memories(results))

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_global))
        all_memories.extend(self._results_to_memories(results))
        
        return all_memories

    def _results_to_memories(self, results: Dict) -> List:
        """将Chroma get结果转换为Memory对象列表"""
        memories = []
        if results['ids']:
            for i in range(len(results['ids'])):
                memory_data = self._extract_memory_data(results, i)
                memory = self._result_to_memory(memory_data)
                memories.append(memory)
        return memories

    async def count_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> int:
        """统计记忆数量（支持多层级记忆）"""
        try:
            self._ensure_ready()
            all_ids = set()

            if group_id:
                all_ids = await self._count_group_memories(user_id, group_id, storage_layer)
            else:
                all_ids = await self._count_private_memories(user_id, storage_layer)

            return len(all_ids)

        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0

    async def _count_group_memories(
        self,
        user_id: str,
        group_id: str,
        storage_layer: Optional[StorageLayer]
    ) -> set:
        """统计群聊场景的记忆数量"""
        all_ids = set()
        
        where_shared = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
        if storage_layer:
            where_shared["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_shared))
        if results['ids']:
            all_ids.update(results['ids'])

        where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
        if storage_layer:
            where_private["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_private))
        if results['ids']:
            all_ids.update(results['ids'])

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_global))
        if results['ids']:
            all_ids.update(results['ids'])
        
        return all_ids

    async def _count_private_memories(
        self,
        user_id: str,
        storage_layer: Optional[StorageLayer]
    ) -> set:
        """统计私聊场景的记忆数量"""
        all_ids = set()
        
        where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
        if storage_layer:
            where_user_private["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_user_private))
        if results['ids']:
            all_ids.update(results['ids'])

        where_global = {"scope": MemoryScope.GLOBAL.value}
        if storage_layer:
            where_global["storage_layer"] = storage_layer.value

        results = self.collection.get(where=self._build_where_clause(where_global))
        if results['ids']:
            all_ids.update(results['ids'])
        
        return all_ids

    async def get_memories_by_storage_layer(
        self,
        storage_layer: StorageLayer,
        limit: int = 1000
    ) -> List:
        """获取指定存储层的所有记忆"""
        try:
            self._ensure_ready()
            where = {"storage_layer": storage_layer.value}
            
            results = self.collection.get(
                where=self._build_where_clause(where),
                include=["embeddings", "documents", "metadatas"]
            )
            
            memories = []
            if results['ids']:
                for i in range(min(len(results['ids']), limit)):
                    memory_data = self._extract_memory_data(results, i)
                    memory = self._result_to_memory(memory_data)
                    memories.append(memory)
            
            logger.debug(
                f"Retrieved {len(memories)} memories with storage_layer={storage_layer.value}"
            )
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get memories by storage layer: {e}")
            return []
