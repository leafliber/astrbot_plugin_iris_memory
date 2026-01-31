"""
Chroma存储管理器
管理Chroma向量数据库的CRUD操作，支持会话隔离
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np

# Chroma导入
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    Settings = None
    embedding_functions = None

from iris_memory.utils.logger import logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import QualityLevel, SensitivityLevel
from iris_memory.core.types import StorageLayer, MemoryType, ModalityType


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
            config: 插件配置对象（从AstrBotConfig获取）
            data_path: 插件数据目录路径
            plugin_context: AstrBot 插件上下文（用于嵌入API）
        """
        self.config = config
        self.data_path = data_path
        self.client = None
        self.collection = None
        
        # 从配置获取Chroma参数
        self.embedding_model_name = self._get_config("chroma_config.embedding_model", "BAAI/bge-m3")
        self.embedding_dimension = self._get_config("chroma_config.embedding_dimension", 1024)
        self.collection_name = self._get_config("chroma_config.collection_name", "iris_memory")
        self.auto_detect_dimension = self._get_config("chroma_config.auto_detect_dimension", True)
        
        # 嵌入管理器（策略模式）
        from iris_memory.embedding.manager import EmbeddingManager
        self.embedding_manager = EmbeddingManager(config, data_path)
        if plugin_context:
            self.embedding_manager.set_plugin_context(plugin_context)
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔的嵌套键）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = getattr(value, k, value.get(k) if isinstance(value, dict) else default)
            return value if value is not None else default
        except (AttributeError, KeyError):
            return default
    
    async def initialize(self):
        """异步初始化Chroma客户端和集合"""
        try:
            if chromadb is None:
                raise ImportError("chromadb is not installed. Please install it with: pip install chromadb")
            
            # 创建数据目录
            chroma_path = self.data_path / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化Chroma客户端（持久化存储）
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
            except:
                pass  # 集合不存在，需要创建
            
            # 自动检测现有集合的维度（如果启用）
            if self.auto_detect_dimension and existing_collection:
                detected_dimension = await self.embedding_manager.detect_existing_dimension(existing_collection)
                if detected_dimension:
                    logger.info(f"Auto-detected embedding dimension: {detected_dimension}")
                    self.embedding_dimension = detected_dimension
            
            # 初始化嵌入管理器
            await self.embedding_manager.initialize()
            
            # 获取实际使用的维度
            actual_dimension = self.embedding_manager.get_dimension()
            if self.embedding_dimension != actual_dimension:
                logger.warning(f"Configured dimension ({self.embedding_dimension}) differs from provider dimension ({actual_dimension}), using provider dimension")
                self.embedding_dimension = actual_dimension
            
            # 获取或创建集合
            if existing_collection:
                self.collection = existing_collection
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
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
                f"Chroma manager initialized. "
                f"Collection: {self.collection_name}, "
                f"Model: {self.embedding_manager.get_model()}, "
                f"Dimension: {self.embedding_dimension}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma manager: {e}")
            raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量（使用策略模式的嵌入管理器）
        
        Args:
            text: 文本内容
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            # 使用嵌入管理器生成嵌入（自动降级）
            embedding = await self.embedding_manager.embed(text, self.embedding_dimension)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # 作为最后的保底，使用简单的哈希
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            embedding = []
            for i in range(0, min(len(hash_bytes), self.embedding_dimension // 4)):
                byte_val = hash_bytes[i]
                embedding.append(byte_val / 255.0)
            
            while len(embedding) < self.embedding_dimension:
                embedding.append(0.0)
            
            return embedding
    
    async def add_memory(self, memory: Memory) -> str:
        """添加记忆到Chroma
        
        Args:
            memory: 记忆对象
            
        Returns:
            str: 记忆ID
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
                "group_id": memory.group_id if memory.group_id else "",
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
            
            # 添加到Chroma
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            logger.debug(f"Memory added to Chroma: {memory.id}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Failed to add memory to Chroma: {e}")
            raise
    
    async def query_memories(
        self,
        query_text: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """查询相关记忆
        
        Args:
            query_text: 查询文本
            user_id: 用户ID
            group_id: 群组ID（私聊时为None）
            top_k: 返回的最大数量
            storage_layer: 存储层过滤（可选）
            
        Returns:
            List[Memory]: 相关记忆列表
        """
        try:
            # 生成查询嵌入
            query_embedding = await self._generate_embedding(query_text)
            
            # 构建过滤条件（会话隔离）
            where = {
                "user_id": user_id,
                "group_id": group_id if group_id else ""
            }
            
            # 添加存储层过滤
            if storage_layer:
                where["storage_layer"] = storage_layer.value
            
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            
            # 转换结果为Memory对象
            memories = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    memory_data = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'embedding': results['embeddings'][0][i] if 'embeddings' in results else None,
                        'metadata': results['metadatas'][0][i]
                    }
                    memory = self._result_to_memory(memory_data)
                    memories.append(memory)
            
            logger.debug(f"Queried {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to query memories: {e}")
            return []
    
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
        if 'embedding' in memory_data and memory_data['embedding']:
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
            'user_id', 'group_id', 'type', 'modality', 'quality_level',
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
                "group_id": memory.group_id if memory.group_id else "",
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
            where = {
                "user_id": user_id,
                "group_id": group_id if group_id else ""
            }
            
            # 查询所有符合条件的记忆ID
            results = self.collection.get(where=where)
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} memories for session {user_id}/{group_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session memories: {e}")
            return False
    
    async def get_all_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """获取用户的所有记忆
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            storage_layer: 存储层过滤（可选）
            
        Returns:
            List[Memory]: 记忆列表
        """
        try:
            # 构建过滤条件
            where = {
                "user_id": user_id,
                "group_id": group_id if group_id else ""
            }
            
            if storage_layer:
                where["storage_layer"] = storage_layer.value
            
            # 查询所有符合条件的记忆
            results = self.collection.get(where=where)
            
            # 转换为Memory对象
            memories = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    # 处理documents和embeddings
                    documents = results.get('documents', [])
                    embeddings = results.get('embeddings', [])
                    metadatas = results.get('metadatas', [])

                    # 处理content
                    if documents and i < len(documents):
                        content = documents[i]
                        if isinstance(content, list) and len(content) > 0:
                            content = content[0]
                    else:
                        content = ''

                    # 处理embedding
                    if embeddings and i < len(embeddings):
                        embedding = embeddings[i]
                    else:
                        embedding = None

                    # 处理metadatas
                    if metadatas and i < len(metadatas):
                        metadata = metadatas[i]
                        if isinstance(metadata, list) and len(metadata) > 0:
                            metadata = metadata[0]  # 取第一个元素
                    else:
                        metadata = {}

                    memory_data = {
                        'id': results['ids'][i],
                        'content': content,
                        'embedding': embedding,
                        'metadata': metadata
                    }
                    memory = self._result_to_memory(memory_data)
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    async def count_memories(
        self,
        user_id: str,
        group_id: Optional[str] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> int:
        """统计记忆数量
        
        Args:
            user_id: 用户ID
            group_id: 群组ID（可选）
            storage_layer: 存储层过滤（可选）
            
        Returns:
            int: 记忆数量
        """
        try:
            where = {
                "user_id": user_id,
                "group_id": group_id if group_id else ""
            }
            
            if storage_layer:
                where["storage_layer"] = storage_layer.value
            
            results = self.collection.get(where=where)
            return len(results['ids']) if results['ids'] else 0
            
        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0
    
    async def close(self):
        """关闭Chroma客户端"""
        if self.client:
            self.client.reset()
            logger.info("Chroma manager closed")
