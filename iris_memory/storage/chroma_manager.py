"""
Chroma存储管理器
管理Chroma向量数据库的CRUD操作，支持会话隔离

架构：
- 使用 Mixin 模式拆分功能模块
- chroma_queries.py: 查询操作
- chroma_operations.py: CRUD操作
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

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

from iris_memory.storage.chroma_queries import ChromaQueries
from iris_memory.storage.chroma_operations import ChromaOperations

logger = get_logger("chroma_manager")


class ChromaManager(ChromaQueries, ChromaOperations):
    """Chroma向量数据库管理器
    
    负责管理Chroma的CRUD操作，包括：
    - 集合创建和管理
    - 记忆的增删改查
    - 向量相似度检索
    - 会话隔离（基于user_id和group_id的元数据过滤）
    
    使用 Mixin 模式组织代码：
    - ChromaQueries: 查询操作
    - ChromaOperations: CRUD操作
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
        
        from iris_memory.core.config_manager import get_config_manager
        from iris_memory.core.defaults import DEFAULTS
        
        cfg = get_config_manager()
        
        self.embedding_model_name = cfg.embedding_model
        self.embedding_dimension = cfg.embedding_dimension
        self.collection_name = DEFAULTS.embedding.collection_name
        self.auto_detect_dimension = DEFAULTS.embedding.auto_detect_dimension
        
        from iris_memory.embedding.manager import EmbeddingManager
        self.embedding_manager = EmbeddingManager(config, data_path)
        if plugin_context:
            self.embedding_manager.set_plugin_context(plugin_context)
    
    @property
    def is_ready(self) -> bool:
        """检查 Chroma 管理器是否已初始化并可用"""
        return self._is_ready and self.client is not None and self.collection is not None
    
    def _ensure_ready(self) -> None:
        """确保 Chroma 已初始化，否则抛出明确异常"""
        if not self._is_ready or self.collection is None:
            raise RuntimeError(
                "ChromaManager is not initialized. "
                "This may happen during hot-reload. Please wait for initialization to complete."
            )
    
    async def initialize(self):
        """异步初始化Chroma客户端和集合"""
        try:
            logger.debug("Initializing ChromaManager...")
            
            if chromadb is None:
                raise ImportError("chromadb is not installed. Please install it with: pip install chromadb")
            
            chroma_path = self.data_path / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Chroma data path: {chroma_path}")
            
            logger.debug("Creating Chroma persistent client...")
            self.client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            existing_collection = self._get_or_check_collection()
            
            detected_dimension = await self._detect_and_init_embedding(existing_collection)
            
            self._handle_dimension_conflict(existing_collection, detected_dimension)
            
            self._create_or_use_collection(existing_collection)
            
            logger.info(
                f"Chroma manager initialized successfully. "
                f"Collection: {self.collection_name}, "
                f"Model: {self.embedding_manager.get_model()}, "
                f"Dimension: {self.embedding_dimension}"
            )
            
            self._is_ready = True
            
        except Exception as e:
            self._is_ready = False
            logger.error(f"Failed to initialize Chroma manager: {e}", exc_info=True)
            raise

    def _get_or_check_collection(self):
        """获取或检查现有集合"""
        existing_collection = None
        try:
            existing_collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Found existing collection: {self.collection_name}")
        except Exception as e:
            is_not_found = (
                (ChromaNotFoundError and isinstance(e, ChromaNotFoundError)) or
                isinstance(e, ValueError) or
                "not exist" in str(e).lower() or
                "not found" in str(e).lower() or
                "does not exist" in str(e).lower()
            )
            if is_not_found:
                logger.debug(f"Collection does not exist: {self.collection_name}")
            else:
                logger.error(f"Error accessing collection {self.collection_name}: {e}")
                raise
        return existing_collection

    async def _detect_and_init_embedding(self, existing_collection) -> Optional[int]:
        """检测并初始化嵌入"""
        detected_dimension: Optional[int] = None
        if self.auto_detect_dimension and existing_collection:
            logger.debug("Auto-detecting embedding dimension from existing collection...")
            detected_dimension = await self.embedding_manager.detect_existing_dimension(existing_collection)
            if detected_dimension:
                logger.info(f"Auto-detected embedding dimension: {detected_dimension}")
                self.embedding_dimension = detected_dimension
        
        logger.debug("Initializing embedding manager...")
        await self.embedding_manager.initialize()
        
        actual_dimension = self.embedding_manager.get_dimension()
        logger.debug(f"Embedding provider dimension: {actual_dimension}, configured: {self.embedding_dimension}")
        if self.embedding_dimension != actual_dimension:
            logger.warning(f"Configured dimension ({self.embedding_dimension}) differs from provider dimension ({actual_dimension}), using provider dimension")
            self.embedding_dimension = actual_dimension
        
        return detected_dimension

    def _handle_dimension_conflict(self, existing_collection, detected_dimension: Optional[int]) -> None:
        """处理维度冲突"""
        if not existing_collection or not detected_dimension:
            return
        
        actual_dimension = self.embedding_manager.get_dimension()
        if detected_dimension == actual_dimension:
            return
        
        old_count = existing_collection.count()
        logger.warning(
            f"Embedding dimension conflict detected! "
            f"Collection has {detected_dimension}-dim vectors but provider outputs {actual_dimension}-dim. "
            f"Recreating collection (old memories count: {old_count} will be lost). "
            f"This usually happens when the embedding model/provider changes."
        )
        self.client.delete_collection(name=self.collection_name)

    def _create_or_use_collection(self, existing_collection) -> None:
        """创建或使用现有集合"""
        if existing_collection and self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            logger.debug(f"Creating new collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Iris Memory Plugin - Three-layer memory system",
                    "embedding_model": self.embedding_manager.get_model(),
                    "embedding_dimension": self.embedding_dimension
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本嵌入向量（使用策略模式的嵌入管理器）
        
        Args:
            text: 文本内容
            
        Returns:
            Optional[List[float]]: 嵌入向量，如果生成失败则返回None
        """
        try:
            embedding = await self.embedding_manager.embed(text, self.embedding_dimension)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """构建 ChromaDB 的 where 查询条件
        
        ChromaDB 要求 where 子句只能有一个操作符，多字段需要使用 $and/$or 包装
        
        Args:
            filters: 过滤条件字典
            
        Returns:
            Dict[str, Any]: ChromaDB 可用的 where 子句
        """
        if len(filters) <= 1:
            return filters
        
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _result_to_memory(self, memory_data: Dict[str, Any]) -> Memory:
        """将Chroma查询结果转换为Memory对象
        
        Args:
            memory_data: 从Chroma获取的记录数据
            
        Returns:
            Memory: 记忆对象
        """
        metadata = memory_data.get('metadata', {})
        
        memory = Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            user_id=metadata.get('user_id', ''),
            sender_name=metadata.get('sender_name') if metadata.get('sender_name') else None,
            group_id=metadata.get('group_id') if metadata.get('group_id') else None,
            scope=MemoryScope(metadata.get('scope', MemoryScope.GROUP_PRIVATE.value)),
            type=MemoryType(metadata.get('type', 'fact')),
            modality=ModalityType(metadata.get('modality', 'text')),
            quality_level=QualityLevel(metadata.get('quality_level', 3)),
            sensitivity_level=SensitivityLevel(metadata.get('sensitivity_level', 0)),
            storage_layer=StorageLayer(metadata.get('storage_layer', 'episodic')),
            access_count=metadata.get('access_count', 0),
            rif_score=metadata.get('rif_score', 0.5),
            importance_score=metadata.get('importance_score', 0.5),
            is_user_requested=metadata.get('is_user_requested', False),
        )
        
        if 'embedding' in memory_data and memory_data['embedding'] is not None:
            memory.embedding = np.array(memory_data['embedding'])
        
        self._set_memory_timestamps(memory, metadata)
        
        system_keys = {
            'user_id', 'sender_name', 'group_id', 'scope', 'type', 'modality', 'quality_level',
            'sensitivity_level', 'storage_layer', 'created_time', 'last_access_time',
            'access_count', 'rif_score', 'importance_score', 'is_user_requested'
        }
        memory.metadata = {k: v for k, v in metadata.items() if k not in system_keys}
        
        return memory

    def _set_memory_timestamps(self, memory, metadata: Dict) -> None:
        """设置记忆时间戳"""
        from datetime import datetime
        
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
        """从Chroma查询结果中提取记忆数据

        Args:
            results: Chroma查询结果
            index: 索引

        Returns:
            Dict[str, Any]: 记忆数据字典
        """
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
        """关闭 Chroma 客户端（热更新友好）
        
        不再调用 client.reset()（会清除所有数据），
        仅释放客户端引用和集合引用。
        持久化数据保留在磁盘，下次初始化时自动加载。
        """
        self._is_ready = False
        
        if hasattr(self, 'embedding_manager') and self.embedding_manager:
            try:
                if hasattr(self.embedding_manager, 'close'):
                    await self.embedding_manager.close()
            except Exception as e:
                logger.debug(f"Error closing embedding manager: {e}")
        
        self.collection = None
        self.client = None
        
        logger.info("[Hot-Reload] Chroma manager closed (data preserved on disk)")
