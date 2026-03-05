"""
Chroma存储管理器
管理Chroma向量数据库的CRUD操作，支持会话隔离

架构：
- 使用组合模式拆分功能模块
- chroma_queries.py: 查询操作
- chroma_operations.py: CRUD操作
"""

from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np

from iris_memory.core.memory_scope import MemoryScope

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from chromadb.errors import NotFoundError as ChromaNotFoundError
except ImportError:
    chromadb = None
    Settings = None
    embedding_functions = None
    ChromaNotFoundError = None

from iris_memory.utils.logger import get_logger
from iris_memory.models.memory import Memory
from iris_memory.core.types import QualityLevel, SensitivityLevel
from iris_memory.core.types import StorageLayer, MemoryType, ModalityType
from iris_memory.core.exceptions import StorageNotReadyError

from iris_memory.storage.chroma_queries import ChromaQueries
from iris_memory.storage.chroma_operations import ChromaOperations

logger = get_logger("chroma_manager")


class ChromaManager:
    """Chroma向量数据库管理器
    
    负责管理Chroma的CRUD操作，包括：
    - 集合创建和管理
    - 记忆的增删改查
    - 向量相似度检索
    - 会话隔离（基于user_id和group_id的元数据过滤）
    
    使用组合模式组织代码：
    - _queries: ChromaQueries 查询操作组件
    - _operations: ChromaOperations CRUD操作组件
    """
    
    def __init__(self, config, data_path: Path, plugin_context=None):
        """初始化Chroma管理器
        
        Args:
            config: 插件配置对象
            data_path: 插件数据目录路径
            plugin_context: AstrBot 插件上下文（用于嵌入API）
        """
        self.config = config
        self.data_path = data_path
        self.client = None
        self.collection = None
        self._is_ready: bool = False
        
        from iris_memory.config import get_store

        cfg = get_store()

        self.embedding_model_name = cfg.get("embedding.local_model", "bge-m3")
        self.embedding_dimension = cfg.get("embedding.local_dimension", 1024)
        self.collection_name = cfg.get("embedding.collection_name")
        self.auto_detect_dimension = cfg.get("embedding.auto_detect_dimension")
        self.reimport_on_dimension_conflict = cfg.get("embedding.reimport_on_dimension_conflict", True)

        from iris_memory.embedding.manager import EmbeddingManager
        self.embedding_manager = EmbeddingManager(config, data_path)
        if plugin_context:
            self.embedding_manager.set_plugin_context(plugin_context)
        
        self._queries: Optional[ChromaQueries] = None
        self._operations: Optional[ChromaOperations] = None
    
    @property
    def is_ready(self) -> bool:
        """检查 Chroma 管理器是否已初始化并可用"""
        return self._is_ready and self.client is not None and self.collection is not None
    
    def _ensure_ready(self) -> None:
        """确保 Chroma 已初始化，否则抛出明确异常"""
        if not self._is_ready or self.collection is None:
            raise StorageNotReadyError(
                "ChromaManager is not initialized. "
                "This may happen during hot-reload. Please wait for initialization to complete."
            )
        if self._queries is None:
            self._queries = ChromaQueries(self)
        if self._operations is None:
            self._operations = ChromaOperations(self)
    
    async def initialize(self):
        """异步初始化Chroma客户端和集合（委托给 ChromaInitializer）"""
        try:
            from iris_memory.storage.chroma_init import ChromaInitializer
            initializer = ChromaInitializer(self)
            await initializer.initialize()
        except Exception as e:
            self._is_ready = False
            logger.error(f"Failed to initialize Chroma manager: {e}", exc_info=True)
            raise

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本嵌入向量（使用策略模式的嵌入管理器）"""
        try:
            embedding = await self.embedding_manager.embed(text, self.embedding_dimension)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """构建 ChromaDB 的 where 查询条件
        
        ChromaDB 要求 where 子句只能有一个操作符，多字段需要使用 $and/$or 包装
        """
        if len(filters) <= 1:
            return filters
        
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _result_to_memory(self, memory_data: Dict[str, Any]) -> Memory:
        """将Chroma查询结果转换为Memory对象"""
        metadata = memory_data.get('metadata', {})
        
        memory = Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            user_id=metadata.get('user_id', ''),
            sender_name=metadata.get('sender_name') if metadata.get('sender_name') else None,
            group_id=metadata.get('group_id') if metadata.get('group_id') else None,
            persona_id=metadata.get('persona_id', 'default') or 'default',
            scope=MemoryScope(metadata.get('scope', MemoryScope.GROUP_PRIVATE.value)),
            type=MemoryType(metadata.get('type', 'fact')),
            modality=ModalityType(metadata.get('modality', 'text')),
            quality_level=QualityLevel(metadata.get('quality_level', 3)),
            sensitivity_level=SensitivityLevel(metadata.get('sensitivity_level', 0)),
            storage_layer=StorageLayer(metadata.get('storage_layer', 'episodic')),
            access_count=metadata.get('access_count', 0),
            confidence=metadata.get('confidence', 0.5),
            rif_score=metadata.get('rif_score', 0.5),
            importance_score=metadata.get('importance_score', 0.5),
            is_user_requested=metadata.get('is_user_requested', False),
            summarized=metadata.get('summarized', False),
            semantic_memory_id=metadata.get('semantic_memory_id'),
            source_type=metadata.get('source_type'),
            evidence_count=metadata.get('evidence_count', 0),
            review_status=metadata.get('review_status'),
        )
        
        if 'embedding' in memory_data and memory_data['embedding'] is not None:
            memory.embedding = np.array(memory_data['embedding'])
        
        self._set_memory_timestamps(memory, metadata)
        
        # 语义提取字段恢复
        evidence_ids_str = metadata.get('evidence_ids', '')
        if evidence_ids_str:
            memory.evidence_ids = [eid.strip() for eid in evidence_ids_str.split(',') if eid.strip()]
        if 'last_validated' in metadata and metadata['last_validated']:
            try:
                memory.last_validated = datetime.fromisoformat(metadata['last_validated'])
            except (ValueError, TypeError):
                pass
        
        system_keys = {
            'user_id', 'sender_name', 'group_id', 'persona_id', 'scope', 'type', 'modality',
            'quality_level', 'sensitivity_level', 'storage_layer', 'created_time',
            'last_access_time', 'access_count', 'confidence', 'rif_score',
            'importance_score', 'is_user_requested',
            'summarized', 'semantic_memory_id', 'evidence_ids', 'source_type',
            'evidence_count', 'last_validated', 'review_status',
        }
        memory.metadata = {k: v for k, v in metadata.items() if k not in system_keys}
        
        return memory

    def _set_memory_timestamps(self, memory, metadata: Dict) -> None:
        """设置记忆时间戳"""
        if 'created_time' in metadata:
            try:
                memory.created_time = datetime.fromisoformat(metadata['created_time'])
            except (ValueError, TypeError):
                pass
        
        if 'last_access_time' in metadata:
            try:
                memory.last_access_time = datetime.fromisoformat(metadata['last_access_time'])
            except (ValueError, TypeError):
                pass

    def _extract_memory_data(self, results: Dict, index: int) -> Dict[str, Any]:
        """从Chroma查询结果中提取记忆数据"""
        documents = results.get('documents', [])
        embeddings = results.get('embeddings', [])
        metadatas = results.get('metadatas', [])

        if documents and index < len(documents):
            content = documents[index]
            if isinstance(content, list) and len(content) > 0:
                content = content[0]
        else:
            content = ''

        if embeddings and index < len(embeddings):
            embedding = embeddings[index]
        else:
            embedding = None

        if metadatas and index < len(metadatas):
            metadata = metadatas[index]
            if isinstance(metadata, list) and len(metadata) > 0:
                metadata = metadata[0]
        else:
            metadata = {}

        return {
            'id': results['ids'][index],
            'content': content,
            'embedding': embedding,
            'metadata': metadata
        }
    
    async def close(self):
        """关闭 Chroma 客户端（热更新友好）"""
        self._is_ready = False
        
        if hasattr(self, 'embedding_manager') and self.embedding_manager:
            try:
                if hasattr(self.embedding_manager, 'close'):
                    await self.embedding_manager.close()
            except Exception as e:
                logger.warning(f"Error closing embedding manager: {e}")
        
        self._queries = None
        self._operations = None
        self.collection = None
        self.client = None
        
        logger.debug("[Hot-Reload] Chroma manager closed (data preserved on disk)")

    async def query_memories(
        self,
        query_text: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        storage_layer: Optional[StorageLayer] = None,
        persona_id: Optional[str] = None
    ) -> List:
        """查询相关记忆（委托到 ChromaQueries）"""
        try:
            self._ensure_ready()
            return await self._queries.query_memories(
                query_text, user_id, group_id, top_k, storage_layer, persona_id
            )
        except (RuntimeError, StorageNotReadyError):
            return []

    async def get_all_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None,
        persona_id: Optional[str] = None
    ) -> List:
        """获取用户的所有记忆（委托到 ChromaQueries）"""
        try:
            self._ensure_ready()
            return await self._queries.get_all_memories(user_id, group_id, storage_layer, persona_id)
        except (RuntimeError, StorageNotReadyError):
            return []

    async def count_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None,
        persona_id: Optional[str] = None
    ) -> int:
        """统计记忆数量（委托到 ChromaQueries）"""
        try:
            self._ensure_ready()
            return await self._queries.count_memories(user_id, group_id, storage_layer, persona_id)
        except (RuntimeError, StorageNotReadyError):
            return 0

    async def get_memories_by_storage_layer(
        self,
        storage_layer: StorageLayer,
        limit: int = 1000
    ) -> List:
        """获取指定存储层的所有记忆（委托到 ChromaQueries）"""
        try:
            self._ensure_ready()
            return await self._queries.get_memories_by_storage_layer(storage_layer, limit)
        except (RuntimeError, StorageNotReadyError):
            return []

    async def get_pending_review_memories(self, limit: int = 50) -> List:
        """获取所有待审核的语义记忆（委托到 ChromaQueries）"""
        try:
            self._ensure_ready()
            return await self._queries.get_pending_review_memories(limit)
        except (RuntimeError, StorageNotReadyError):
            return []

    async def add_memory(self, memory) -> Optional[str]:
        """添加记忆到Chroma（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.add_memory(memory)
        except (RuntimeError, StorageNotReadyError):
            return None

    async def update_memory(self, memory) -> bool:
        """更新记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.update_memory(memory)
        except (RuntimeError, StorageNotReadyError):
            return False

    async def batch_update_access_stats(self, memories: list) -> int:
        """批量更新记忆的访问统计（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.batch_update_access_stats(memories)
        except (RuntimeError, StorageNotReadyError):
            return 0

    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_memory(memory_id)
        except (RuntimeError, StorageNotReadyError):
            return False

    async def delete_session(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """删除会话的所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_session(user_id, group_id)
        except (RuntimeError, StorageNotReadyError):
            return False

    async def delete_user_memories(self, user_id: str, in_private_only: bool = False) -> Tuple[bool, int]:
        """删除用户的所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_user_memories(user_id, in_private_only)
        except (RuntimeError, StorageNotReadyError):
            return False, 0

    async def delete_group_memories(self, group_id: str, scope_filter: Optional[str] = None) -> Tuple[bool, int]:
        """删除群组的记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_group_memories(group_id, scope_filter)
        except (RuntimeError, StorageNotReadyError):
            return False, 0

    async def delete_all_memories(self) -> Tuple[bool, int]:
        """删除所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_all_memories()
        except (RuntimeError, StorageNotReadyError):
            return False, 0
