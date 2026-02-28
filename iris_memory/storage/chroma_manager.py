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
        
        from iris_memory.core.config_manager import get_config_manager
        from iris_memory.core.defaults import DEFAULTS

        cfg = get_config_manager()

        self.embedding_model_name = cfg.embedding_local_model
        self.embedding_dimension = cfg.embedding_local_dimension
        self.collection_name = DEFAULTS.embedding.collection_name
        self.auto_detect_dimension = DEFAULTS.embedding.auto_detect_dimension
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
            raise RuntimeError(
                "ChromaManager is not initialized. "
                "This may happen during hot-reload. Please wait for initialization to complete."
            )
        if self._queries is None:
            self._queries = ChromaQueries(self)
        if self._operations is None:
            self._operations = ChromaOperations(self)
    
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

            backup_data = self._handle_dimension_conflict(existing_collection, detected_dimension)

            self._create_or_use_collection(existing_collection)

            if backup_data and self.reimport_on_dimension_conflict:
                await self._reimport_memories_with_new_embeddings(backup_data)
            elif backup_data and not self.reimport_on_dimension_conflict:
                logger.warning(
                    f"维度冲突处理：配置禁用了自动重新导入，{len(backup_data.get('ids', []))} 条记忆已备份但未导入。"
                    f"您可以在备份集合中找到这些记忆。"
                )

            self._queries = ChromaQueries(self)
            self._operations = ChromaOperations(self)

            logger.debug(
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
                logger.debug(f"Auto-detected embedding dimension: {detected_dimension}")
                self.embedding_dimension = detected_dimension
        
        logger.debug("Initializing embedding manager...")
        await self.embedding_manager.initialize()
        
        actual_dimension = self.embedding_manager.get_dimension()
        logger.debug(f"Embedding provider dimension: {actual_dimension}, configured: {self.embedding_dimension}")
        if self.embedding_dimension != actual_dimension:
            logger.warning(f"Configured dimension ({self.embedding_dimension}) differs from provider dimension ({actual_dimension}), using provider dimension")
            self.embedding_dimension = actual_dimension
        
        return detected_dimension

    def _get_dimension_from_metadata(self, collection) -> Optional[int]:
        """从集合元数据中获取嵌入维度"""
        try:
            metadata = collection.metadata
            if metadata and "embedding_dimension" in metadata:
                dimension = metadata["embedding_dimension"]
                if isinstance(dimension, int):
                    logger.debug(f"Got embedding dimension from collection metadata: {dimension}")
                    return dimension
        except Exception as e:
            logger.debug(f"Failed to get dimension from collection metadata: {e}")
        return None

    def _handle_dimension_conflict(
        self, existing_collection, detected_dimension: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """处理维度冲突"""
        if not existing_collection:
            return None

        collection_dimension = detected_dimension
        if collection_dimension is None:
            collection_dimension = self._get_dimension_from_metadata(existing_collection)

        if collection_dimension is None:
            logger.debug("Could not determine existing collection dimension, skipping conflict check")
            return None

        actual_dimension = self.embedding_manager.get_dimension()
        if collection_dimension == actual_dimension:
            return None

        old_count = existing_collection.count()
        logger.error(
            f"⚠️ CRITICAL: Embedding dimension conflict detected! "
            f"Collection has {collection_dimension}-dim vectors but provider outputs {actual_dimension}-dim. "
            f"Old memories count: {old_count}. "
            f"This usually happens when the embedding model/provider changes. "
            f"The old collection will be backed up and re-imported with new embeddings."
        )

        backup_data = None
        if old_count > 0:
            backup_data = self._backup_collection_data(existing_collection, old_count)

            if backup_data is None:
                file_backup_ok = self._backup_collection_before_delete(existing_collection, old_count)
                if not file_backup_ok:
                    logger.error(
                        f"All backup methods failed for collection '{self.collection_name}' "
                        f"({old_count} memories). Aborting deletion to prevent data loss. "
                        f"Please manually resolve the dimension conflict."
                    )
                    return None
                logger.warning(
                    f"In-memory backup failed but file-level backup succeeded. "
                    f"Proceeding with collection deletion."
                )

        self.client.delete_collection(name=self.collection_name)
        logger.warning(
            f"Old collection '{self.collection_name}' deleted after dimension conflict. "
            f"{old_count} memories will be re-imported with new embeddings."
        )

        return backup_data

    def _backup_collection_data(self, existing_collection, old_count: int) -> Optional[Dict[str, Any]]:
        """备份集合数据供后续重新导入"""
        try:
            all_data = existing_collection.get(include=["documents", "metadatas"])
            if not all_data or not all_data.get("ids"):
                logger.debug("No data in old collection to backup.")
                return None

            logger.info(f"Backed up {old_count} memories for re-import with new embeddings.")
            return {
                "ids": all_data.get("ids", []),
                "documents": all_data.get("documents", []),
                "metadatas": all_data.get("metadatas", []),
            }
        except Exception as e:
            logger.error(f"Failed to backup collection data: {e}", exc_info=True)
            return None

    def _backup_collection_before_delete(self, existing_collection, old_count: int) -> bool:
        """在删除集合前尝试备份数据到新集合，并额外导出 JSON 文件"""
        backup_name = f"{self.collection_name}_backup_{int(time.time())}"
        chroma_backup_ok = False
        json_backup_ok = False
        try:
            all_data = existing_collection.get(include=["documents", "metadatas", "embeddings"])
            if not all_data or not all_data.get("ids"):
                logger.debug("No data in old collection to backup.")
                return True

            backup_col = self.client.create_collection(name=backup_name)
            batch_size = 500
            ids = all_data["ids"]
            embeddings_data = all_data.get("embeddings")
            has_embeddings = embeddings_data is not None and len(embeddings_data) > 0
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                batch_kwargs = {"ids": ids[i:end]}
                if all_data.get("documents"):
                    batch_kwargs["documents"] = all_data["documents"][i:end]
                if all_data.get("metadatas"):
                    batch_kwargs["metadatas"] = all_data["metadatas"][i:end]
                if has_embeddings:
                    batch_kwargs["embeddings"] = embeddings_data[i:end]
                backup_col.add(**batch_kwargs)

            logger.warning(
                f"Backed up {old_count} memories to collection '{backup_name}' "
                f"before dimension migration. You can manually inspect or delete it later."
            )
            chroma_backup_ok = True
        except Exception as e:
            logger.error(
                f"Failed to backup collection to ChromaDB: {e}.",
                exc_info=True,
            )

        try:
            if not chroma_backup_ok:
                all_data = existing_collection.get(include=["documents", "metadatas"])
            json_backup_path = self.data_path / "backup" / f"{backup_name}.json"
            json_backup_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {
                "ids": all_data.get("ids", []),
                "documents": all_data.get("documents", []),
                "metadatas": all_data.get("metadatas", []),
                "backup_time": time.time(),
                "original_collection": self.collection_name,
                "memory_count": old_count,
            }
            json_backup_path.write_text(json.dumps(export_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.warning(
                f"JSON backup saved to {json_backup_path} ({old_count} memories). "
                f"This can be used for manual recovery."
            )
            json_backup_ok = True
        except Exception as e:
            logger.error(
                f"Failed to save JSON backup: {e}. "
                f"{'ChromaDB backup exists.' if chroma_backup_ok else f'{old_count} memories will be permanently lost.'}",
                exc_info=True,
            )

        return chroma_backup_ok or json_backup_ok

    def _create_or_use_collection(self, existing_collection) -> None:
        """创建或使用现有集合"""
        if existing_collection and self.collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing collection: {self.collection_name}")
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
            logger.debug(f"Created new collection: {self.collection_name}")

    async def _reimport_memories_with_new_embeddings(self, backup_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用新的嵌入模型重新导入备份的记忆数据"""
        ids = backup_data.get("ids", [])
        documents = backup_data.get("documents", [])
        metadatas = backup_data.get("metadatas", [])

        if not ids:
            logger.warning("No memories to re-import.")
            return {"success_count": 0, "failed_count": 0, "failed_ids": [], "failed_details": []}

        total = len(ids)
        logger.info(f"Starting to re-import {total} memories with new embeddings...")

        success_count = 0
        failed_count = 0
        failed_ids: List[str] = []
        failed_details: List[Dict[str, Any]] = []
        batch_size = 50

        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size] if documents else []
            batch_metas = metadatas[i:i + batch_size] if metadatas else []

            batch_embeddings = []
            batch_valid_ids = []
            batch_valid_docs = []
            batch_valid_metas = []

            for j, doc in enumerate(batch_docs):
                mem_id = batch_ids[j]
                if not doc:
                    failed_count += 1
                    failed_ids.append(mem_id)
                    failed_details.append({"id": mem_id, "reason": "empty_document", "stage": "embedding"})
                    continue
                try:
                    embedding = await self._generate_embedding(doc)
                    if embedding:
                        batch_embeddings.append(embedding)
                        batch_valid_ids.append(mem_id)
                        batch_valid_docs.append(doc)
                        if batch_metas and j < len(batch_metas):
                            batch_valid_metas.append(batch_metas[j])
                        else:
                            batch_valid_metas.append({})
                    else:
                        failed_count += 1
                        failed_ids.append(mem_id)
                        failed_details.append({"id": mem_id, "reason": "embedding_generation_failed", "stage": "embedding"})
                        logger.warning(f"Failed to generate embedding for memory {mem_id}")
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(mem_id)
                    failed_details.append({"id": mem_id, "reason": str(e), "stage": "embedding", "error_type": type(e).__name__})
                    logger.error(f"Error generating embedding for memory {mem_id}: {e}")

            if batch_valid_ids:
                try:
                    self.collection.add(
                        ids=batch_valid_ids,
                        embeddings=batch_embeddings,
                        documents=batch_valid_docs,
                        metadatas=batch_valid_metas
                    )
                    success_count += len(batch_valid_ids)
                except Exception as e:
                    failed_count += len(batch_valid_ids)
                    failed_ids.extend(batch_valid_ids)
                    for mid in batch_valid_ids:
                        failed_details.append({"id": mid, "reason": str(e), "stage": "import", "error_type": type(e).__name__})
                    logger.error(f"Failed to import batch: {e}")

        logger.info(
            f"Re-import complete: {success_count} succeeded, {failed_count} failed out of {total} memories"
        )
        if failed_ids:
            logger.warning(f"Failed memory IDs ({len(failed_ids)}): {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")

        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_ids": failed_ids,
            "failed_details": failed_details
        }

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
        )
        
        if 'embedding' in memory_data and memory_data['embedding'] is not None:
            memory.embedding = np.array(memory_data['embedding'])
        
        self._set_memory_timestamps(memory, metadata)
        
        system_keys = {
            'user_id', 'sender_name', 'group_id', 'persona_id', 'scope', 'type', 'modality',
            'quality_level', 'sensitivity_level', 'storage_layer', 'created_time',
            'last_access_time', 'access_count', 'confidence', 'rif_score',
            'importance_score', 'is_user_requested'
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
        except RuntimeError:
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
        except RuntimeError:
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
        except RuntimeError:
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
        except RuntimeError:
            return []

    async def add_memory(self, memory) -> Optional[str]:
        """添加记忆到Chroma（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.add_memory(memory)
        except RuntimeError:
            return None

    async def update_memory(self, memory) -> bool:
        """更新记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.update_memory(memory)
        except RuntimeError:
            return False

    async def batch_update_access_stats(self, memories: list) -> int:
        """批量更新记忆的访问统计（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.batch_update_access_stats(memories)
        except RuntimeError:
            return 0

    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_memory(memory_id)
        except RuntimeError:
            return False

    async def delete_session(self, user_id: str, group_id: Optional[str] = None) -> bool:
        """删除会话的所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_session(user_id, group_id)
        except RuntimeError:
            return False

    async def delete_user_memories(self, user_id: str, in_private_only: bool = False) -> Tuple[bool, int]:
        """删除用户的所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_user_memories(user_id, in_private_only)
        except RuntimeError:
            return False, 0

    async def delete_group_memories(self, group_id: str, scope_filter: Optional[str] = None) -> Tuple[bool, int]:
        """删除群组的记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_group_memories(group_id, scope_filter)
        except RuntimeError:
            return False, 0

    async def delete_all_memories(self) -> Tuple[bool, int]:
        """删除所有记忆（委托到 ChromaOperations）"""
        try:
            self._ensure_ready()
            return await self._operations.delete_all_memories()
        except RuntimeError:
            return False, 0
