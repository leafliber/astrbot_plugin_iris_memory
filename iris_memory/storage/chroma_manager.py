"""
Chroma存储管理器
管理Chroma向量数据库的CRUD操作，支持会话隔离
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

# 本地导入
from iris_memory.core.memory_scope import MemoryScope

# Chroma导入
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

# 模块logger
logger = get_logger("chroma_manager")


class ChromaManager:
    """Chroma向量数据库管理器
    
    负责管理Chroma的CRUD操作，包括：
    - 集合创建和管理
    - 记忆的增删改查
    - 向量相似度检索
    - 会话隔离（基于user_id和group_id的元数据过滤）
    """
    
    def __init__(self, config, data_path: Path, plugin_context=None):
        """初始化Chroma管理器
        
        Args:
            config: 插件配置对象（保留用于接口兼容性）
            data_path: 插件数据目录路径
            plugin_context: AstrBot 插件上下文（用于嵌入API）
        """
        self.config = config
        self.data_path = data_path
        self.client = None
        self.collection = None
        
        # 导入配置管理器和默认值
        from iris_memory.core.config_manager import get_config_manager
        from iris_memory.core.defaults import DEFAULTS
        
        cfg = get_config_manager()
        
        # 从配置获取Chroma参数（优先使用传入的配置对象）
        self.embedding_model_name = self._get_config_from_object(
            config, 'chroma_config.embedding_model', cfg.embedding_model
        )
        self.embedding_dimension = self._get_config_from_object(
            config, 'chroma_config.embedding_dimension', cfg.embedding_dimension
        )
        self.collection_name = self._get_config_from_object(
            config, 'chroma_config.collection_name', DEFAULTS.embedding.collection_name
        )
        self.auto_detect_dimension = self._get_config_from_object(
            config, 'chroma_config.auto_detect_dimension', DEFAULTS.embedding.auto_detect_dimension
        )
        
        # 嵌入管理器（策略模式）
        from iris_memory.embedding.manager import EmbeddingManager
        self.embedding_manager = EmbeddingManager(config, data_path)
        if plugin_context:
            self.embedding_manager.set_plugin_context(plugin_context)
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值（兼容旧代码）
        
        Args:
            key: 配置键（支持点分隔的嵌套键）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        from iris_memory.core.config_manager import get_config_manager
        return get_config_manager().get(key, default)
    
    def _get_config_from_object(self, config: Any, key: str, default: Any = None) -> Any:
        """从配置对象获取值（支持嵌套键）
        
        Args:
            config: 配置对象（可能为None）
            key: 配置键（支持点分隔的嵌套键，如 'chroma_config.embedding_dimension'）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if config is None:
            return default
        
        try:
            keys = key.split('.')
            value = config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    value = getattr(value, k, None)
                if value is None:
                    return default
            return value if value is not None else default
        except Exception:
            return default
    
    async def initialize(self):
        """异步初始化Chroma客户端和集合"""
        try:
            logger.debug("Initializing ChromaManager...")
            
            if chromadb is None:
                raise ImportError("chromadb is not installed. Please install it with: pip install chromadb")
            
            # 创建数据目录
            chroma_path = self.data_path / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Chroma data path: {chroma_path}")
            
            # 初始化Chroma客户端（持久化存储）
            logger.debug("Creating Chroma persistent client...")
            self.client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 检查集合是否存在
            existing_collection = None
            try:
                existing_collection = self.client.get_collection(name=self.collection_name)
                logger.debug(f"Found existing collection: {self.collection_name}")
            except Exception as e:
                # 捕获所有异常（包括 ValueError, NotFoundError, 通用 Exception 等）
                # 检查是否是 NotFoundError 或通过异常消息判断集合不存在
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
                    # 其他异常（如权限问题、磁盘问题）需要记录和抛出
                    logger.error(f"Error accessing collection {self.collection_name}: {e}")
                    raise
            
            # 自动检测现有集合的维度（如果启用）
            if self.auto_detect_dimension and existing_collection:
                logger.debug("Auto-detecting embedding dimension from existing collection...")
                detected_dimension = await self.embedding_manager.detect_existing_dimension(existing_collection)
                if detected_dimension:
                    logger.info(f"Auto-detected embedding dimension: {detected_dimension}")
                    self.embedding_dimension = detected_dimension
            
            # 初始化嵌入管理器
            logger.debug("Initializing embedding manager...")
            await self.embedding_manager.initialize()
            
            # 获取实际使用的维度
            actual_dimension = self.embedding_manager.get_dimension()
            logger.debug(f"Embedding provider dimension: {actual_dimension}, configured: {self.embedding_dimension}")
            if self.embedding_dimension != actual_dimension:
                logger.warning(f"Configured dimension ({self.embedding_dimension}) differs from provider dimension ({actual_dimension}), using provider dimension")
                self.embedding_dimension = actual_dimension
            
            # 获取或创建集合
            if existing_collection:
                self.collection = existing_collection
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
            
            logger.info(
                f"Chroma manager initialized successfully. "
                f"Collection: {self.collection_name}, "
                f"Model: {self.embedding_manager.get_model()}, "
                f"Dimension: {self.embedding_dimension}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma manager: {e}", exc_info=True)
            raise
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本嵌入向量（使用策略模式的嵌入管理器）
        
        Args:
            text: 文本内容
            
        Returns:
            Optional[List[float]]: 嵌入向量，如果生成失败则返回None
        """
        try:
            # 使用嵌入管理器生成嵌入（自动降级）
            embedding = await self.embedding_manager.embed(text, self.embedding_dimension)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # 不再使用MD5哈希作为降级方案，因为它与真实语义嵌入不兼容
            # 返回None让调用者处理失败情况
            return None
    
    async def add_memory(self, memory: Memory) -> Optional[str]:
        """添加记忆到Chroma
        
        Args:
            memory: 记忆对象
            
        Returns:
            Optional[str]: 记忆ID，如果嵌入生成失败则返回None
        """
        try:
            logger.debug(f"Adding memory to Chroma: id={memory.id}, user={memory.user_id}, type={memory.type.value}")
            
            # 生成嵌入向量
            if memory.embedding is None:
                logger.debug(f"Generating embedding for memory {memory.id}...")
                embedding = await self._generate_embedding(memory.content)
                if embedding is None:
                    logger.error(f"Failed to generate embedding for memory {memory.id}, skipping storage")
                    return None
                memory.embedding = np.array(embedding)
                logger.debug(f"Embedding generated: dimension={len(embedding)}")
            else:
                embedding = memory.embedding.tolist()
                logger.debug(f"Using existing embedding: dimension={len(embedding)}")
            
            # 构建元数据
            metadata = {
                "user_id": memory.user_id,
                "group_id": memory.group_id if memory.group_id else "",  # 私聊场景用空字符串
                "scope": memory.scope.value,
                "type": memory.type.value,
                "modality": memory.modality.value,
                "quality_level": memory.quality_level.value,
                "sensitivity_level": memory.sensitivity_level.value,
                "storage_layer": memory.storage_layer.value,
                "created_time": memory.created_time.isoformat(),
                "last_access_time": memory.last_access_time.isoformat(),
                "access_count": memory.access_count,
                "rif_score": memory.rif_score,
                "importance_score": memory.importance_score,
                "is_user_requested": memory.is_user_requested,
            }

            # 添加额外元数据
            metadata.update(memory.metadata)
            
            logger.debug(f"Memory metadata: {metadata}")
            
            # 添加到Chroma
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.info(f"Memory added to Chroma: id={memory.id}, user={memory.user_id}, storage_layer={memory.storage_layer.value}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Failed to add memory to Chroma: id={memory.id}, error={e}", exc_info=True)
            raise
    
    async def query_memories(
        self,
        query_text: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
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
            logger.debug(f"Querying memories: user={user_id}, group={group_id}, top_k={top_k}, storage_layer={storage_layer.value if storage_layer else 'all'}")
            
            # 生成查询嵌入
            logger.debug(f"Generating query embedding for: '{query_text[:50]}...' " if len(query_text) > 50 else f"Generating query embedding for: '{query_text}'")
            query_embedding = await self._generate_embedding(query_text)
            if query_embedding is None:
                logger.error("Failed to generate query embedding, returning empty results")
                return []
            logger.debug(f"Query embedding generated: dimension={len(query_embedding)}")

            # 构建多个查询条件，分别查询不同的记忆范围
            all_results = []

            if group_id:
                logger.debug(f"Group chat query mode: group_id={group_id}")
                # 群聊场景：分别查询群组共享记忆、用户个人记忆、全局记忆

                # 1. 群组共享记忆
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
                if results['ids'] and results['ids'][0]:
                    logger.debug(f"Found {len(results['ids'][0])} GROUP_SHARED memories")
                    for i in range(len(results['ids'][0])):
                        all_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                            'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                            'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                        })
                else:
                    logger.debug("No GROUP_SHARED memories found")

                # 2. 用户在群组的个人记忆
                where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
                if storage_layer:
                    where_private["storage_layer"] = storage_layer.value

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=self._build_where_clause(where_private),
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        all_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                            'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                            'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                        })

                # 3. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=self._build_where_clause(where_global),
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        all_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                            'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                            'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                        })
            else:
                # 私聊场景：分别查询用户私有记忆、全局记忆

                # 1. 用户私有记忆
                where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
                if storage_layer:
                    where_user_private["storage_layer"] = storage_layer.value

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=self._build_where_clause(where_user_private),
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        all_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                            'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                            'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                        })

                # 2. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=self._build_where_clause(where_global),
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        all_results.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i] if results.get('documents') and len(results['documents']) > 0 else '',
                            'embedding': results['embeddings'][0][i] if results.get('embeddings') and len(results['embeddings']) > 0 else None,
                            'metadata': results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas']) > 0 else {},
                            'distance': results['distances'][0][i] if results.get('distances') and len(results['distances']) > 0 else None
                        })

            # 去重（如果同一个记忆出现在多个查询结果中）
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            logger.debug(f"Total results: {len(all_results)}, Unique results: {len(unique_results)}")

            # 按 distance 排序并取 top_k
            if unique_results:
                unique_results.sort(key=lambda x: x['distance'] if x['distance'] is not None else float('inf'))
                unique_results = unique_results[:top_k]
                logger.debug(f"Sorted and limited to top {len(unique_results)} results")

            # 转换结果为Memory对象
            memories = []
            for memory_data in unique_results:
                # 移除 distance 字段
                memory_data_without_distance = {k: v for k, v in memory_data.items() if k != 'distance'}
                memory = self._result_to_memory(memory_data_without_distance)
                memories.append(memory)

            logger.info(f"Queried {len(memories)} memories for user={user_id}, group={group_id}, query='{query_text[:30]}...'")
            return memories

        except Exception as e:
            logger.error(f"Failed to query memories: user={user_id}, error={e}", exc_info=True)
            return []
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """构建 ChromaDB 的 where 查询条件
        
        ChromaDB 要求 where 子句只能有一个操作符，多字段需要使用 $and/$or 包装
        
        Args:
            filters: 过滤条件字典
            
        Returns:
            Dict[str, Any]: ChromaDB 兼容的 where 子句
        """
        if len(filters) <= 1:
            return filters
        
        # 多字段条件使用 $and 包装
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _result_to_memory(self, memory_data: Dict[str, Any]) -> Memory:
        """将Chroma查询结果转换为Memory对象
        
        Args:
            memory_data: 从Chroma获取的记录数据
            
        Returns:
            Memory: 记忆对象
        """
        metadata = memory_data.get('metadata', {})
        
        # 创建Memory对象
        memory = Memory(
            id=memory_data['id'],
            content=memory_data['content'],
            user_id=metadata.get('user_id', ''),
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
        
        # 设置嵌入向量
        if 'embedding' in memory_data and memory_data['embedding'] is not None:
            memory.embedding = np.array(memory_data['embedding'])
        
        # 设置时间戳
        if 'created_time' in metadata:
            from datetime import datetime
            try:
                memory.created_time = datetime.fromisoformat(metadata['created_time'])
            except:
                pass
        
        if 'last_access_time' in metadata:
            from datetime import datetime
            try:
                memory.last_access_time = datetime.fromisoformat(metadata['last_access_time'])
            except:
                pass
        
        # 移除系统元数据，保留自定义元数据
        system_keys = {
            'user_id', 'group_id', 'scope', 'type', 'modality', 'quality_level',
            'sensitivity_level', 'storage_layer', 'created_time', 'last_access_time',
            'access_count', 'rif_score', 'importance_score', 'is_user_requested'
        }
        memory.metadata = {k: v for k, v in metadata.items() if k not in system_keys}
        
        return memory
    
    async def update_memory(self, memory: Memory) -> bool:
        """更新记忆
        
        Args:
            memory: 更新后的记忆对象
            
        Returns:
            bool: 是否更新成功
        """
        try:
            # 生成嵌入向量
            if memory.embedding is None:
                embedding = await self._generate_embedding(memory.content)
                memory.embedding = np.array(embedding)
            else:
                embedding = memory.embedding.tolist()
            
            # 构建元数据
            metadata = {
                "user_id": memory.user_id,
                "scope": memory.scope.value,
                "type": memory.type.value,
                "modality": memory.modality.value,
                "quality_level": memory.quality_level.value,
                "sensitivity_level": memory.sensitivity_level.value,
                "storage_layer": memory.storage_layer.value,
                "last_access_time": memory.last_access_time.isoformat(),
                "access_count": memory.access_count,
                "rif_score": memory.rif_score,
                "importance_score": memory.importance_score,
            }

            # 只在有 group_id 时才添加到元数据
            if memory.group_id:
                metadata["group_id"] = memory.group_id
            
            # 添加额外元数据
            metadata.update(memory.metadata)
            
            # 更新Chroma
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
    
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
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
            # 构建过滤条件
            where = {"user_id": user_id}

            # Always include group_id to match metadata structure
            where["group_id"] = group_id if group_id else ""

            # 查询所有符合条件的记忆ID
            results = self.collection.get(where=self._build_where_clause(where))
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} memories for session {user_id}/{group_id}")
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
            all_ids = set()
            
            if in_private_only:
                # 只删除私聊记忆：USER_PRIVATE + GLOBAL(created by user)
                # 1. 用户私有记忆
                where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
                results = self.collection.get(where=self._build_where_clause(where_user_private))
                if results['ids']:
                    all_ids.update(results['ids'])
                
                # 2. 用户创建的全局记忆（私聊场景）
                where_global = {"user_id": user_id, "scope": MemoryScope.GLOBAL.value, "group_id": ""}
                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    all_ids.update(results['ids'])
            else:
                # 删除所有用户相关的记忆（包括私聊和群聊）
                where = {"user_id": user_id}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            
            if all_ids:
                self.collection.delete(ids=list(all_ids))
                logger.info(f"Deleted {len(all_ids)} memories for user {user_id} (private_only={in_private_only})")
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
            all_ids = set()
            
            if scope_filter == "group_shared":
                # 只删除群组共享记忆
                where = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            elif scope_filter == "group_private":
                # 只删除群组个人记忆
                where = {"group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            else:
                # 删除群组所有记忆（共享+所有成员的个人记忆）
                where = {"group_id": group_id}
                results = self.collection.get(where=self._build_where_clause(where))
                if results['ids']:
                    all_ids.update(results['ids'])
            
            if all_ids:
                self.collection.delete(ids=list(all_ids))
                logger.info(f"Deleted {len(all_ids)} memories for group {group_id} (scope_filter={scope_filter})")
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
            # 获取所有记忆ID
            results = self.collection.get()
            
            if results['ids']:
                count = len(results['ids'])
                self.collection.delete(ids=results['ids'])
                logger.warning(f"Deleted ALL {count} memories from database!")
                return True, count
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return False, 0
    
    async def get_all_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """获取用户的所有记忆（支持多层级记忆）

        与 query_memories 和 count_memories 相同的查询策略。

        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            storage_layer: 存储层过滤（可选）

        Returns:
            List[Memory]: 记忆列表
        """
        try:
            all_memories = []

            if group_id:
                # 群聊场景：分别查询群组共享记忆、用户个人记忆、全局记忆

                # 1. 群组共享记忆
                where_shared = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
                if storage_layer:
                    where_shared["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_shared))
                if results['ids']:
                    for i in range(len(results['ids'])):
                        memory_data = self._extract_memory_data(results, i)
                        memory = self._result_to_memory(memory_data)
                        all_memories.append(memory)

                # 2. 用户在群组的个人记忆
                where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
                if storage_layer:
                    where_private["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_private))
                if results['ids']:
                    for i in range(len(results['ids'])):
                        memory_data = self._extract_memory_data(results, i)
                        memory = self._result_to_memory(memory_data)
                        all_memories.append(memory)

                # 3. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    for i in range(len(results['ids'])):
                        memory_data = self._extract_memory_data(results, i)
                        memory = self._result_to_memory(memory_data)
                        all_memories.append(memory)
            else:
                # 私聊场景：分别查询用户私有记忆、全局记忆

                # 1. 用户私有记忆
                where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
                if storage_layer:
                    where_user_private["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_user_private))
                if results['ids']:
                    for i in range(len(results['ids'])):
                        memory_data = self._extract_memory_data(results, i)
                        memory = self._result_to_memory(memory_data)
                        all_memories.append(memory)

                # 2. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    for i in range(len(results['ids'])):
                        memory_data = self._extract_memory_data(results, i)
                        memory = self._result_to_memory(memory_data)
                        all_memories.append(memory)

            # 去重
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

    def _extract_memory_data(self, results: Dict, index: int) -> Dict[str, Any]:
        """从Chroma查询结果中提取记忆数据

        Args:
            results: Chroma查询结果
            index: 索引

        Returns:
            Dict[str, Any]: 记忆数据字典
        """
        # 处理documents和embeddings
        documents = results.get('documents', [])
        embeddings = results.get('embeddings', [])
        metadatas = results.get('metadatas', [])

        # 处理content
        if documents and index < len(documents):
            content = documents[index]
            if isinstance(content, list) and len(content) > 0:
                content = content[0]
        else:
            content = ''

        # 处理embedding
        if embeddings and index < len(embeddings):
            embedding = embeddings[index]
        else:
            embedding = None

        # 处理metadatas
        if metadatas and index < len(metadatas):
            metadata = metadatas[index]
            if isinstance(metadata, list) and len(metadata) > 0:
                metadata = metadata[0]  # 取第一个元素
        else:
            metadata = {}

        return {
            'id': results['ids'][index],
            'content': content,
            'embedding': embedding,
            'metadata': metadata
        }
    
    async def count_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> int:
        """统计记忆数量（支持多层级记忆）

        与 query_memories 相同的查询策略：
        - 群聊：GROUP_SHARED + GROUP_PRIVATE(user_id) + GLOBAL
        - 私聊：USER_PRIVATE + GLOBAL

        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            storage_layer: 存储层过滤（可选）

        Returns:
            int: 记忆数量
        """
        try:
            all_ids = set()

            if group_id:
                # 群聊场景：分别统计群组共享记忆、用户个人记忆、全局记忆

                # 1. 群组共享记忆
                where_shared = {"group_id": group_id, "scope": MemoryScope.GROUP_SHARED.value}
                if storage_layer:
                    where_shared["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_shared))
                if results['ids']:
                    all_ids.update(results['ids'])

                # 2. 用户在群组的个人记忆
                where_private = {"user_id": user_id, "group_id": group_id, "scope": MemoryScope.GROUP_PRIVATE.value}
                if storage_layer:
                    where_private["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_private))
                if results['ids']:
                    all_ids.update(results['ids'])

                # 3. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    all_ids.update(results['ids'])
            else:
                # 私聊场景：分别统计用户私有记忆、全局记忆

                # 1. 用户私有记忆
                where_user_private = {"user_id": user_id, "scope": MemoryScope.USER_PRIVATE.value}
                if storage_layer:
                    where_user_private["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_user_private))
                if results['ids']:
                    all_ids.update(results['ids'])

                # 2. 全局记忆
                where_global = {"scope": MemoryScope.GLOBAL.value}
                if storage_layer:
                    where_global["storage_layer"] = storage_layer.value

                results = self.collection.get(where=self._build_where_clause(where_global))
                if results['ids']:
                    all_ids.update(results['ids'])

            return len(all_ids)

        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0
    
    async def get_memories_by_storage_layer(
        self,
        storage_layer: StorageLayer,
        limit: int = 1000
    ) -> List[Memory]:
        """获取指定存储层的所有记忆
        
        用于记忆升级检查（EPISODIC → SEMANTIC）
        
        Args:
            storage_layer: 存储层类型
            limit: 最大返回数量
            
        Returns:
            List[Memory]: 记忆列表
        """
        try:
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
    
    async def close(self):
        """关闭Chroma客户端"""
        if self.client:
            try:
                self.client.reset()
            except Exception as e:
                logger.debug(f"Error during Chroma client reset: {e}")
            self.client = None
            self.collection = None
            logger.info("Chroma manager closed")
