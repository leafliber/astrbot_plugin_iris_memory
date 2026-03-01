"""ChromaDB 初始化与维度冲突处理

从 ChromaManager 中提取的初始化流程，包括：
- 集合检查/创建
- 嵌入维度检测
- 维度冲突处理与数据迁移
- 备份与重新导入
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.storage.chroma_manager import ChromaManager

logger = get_logger("chroma_manager.init")

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError as ChromaNotFoundError
except ImportError:
    chromadb = None
    Settings = None  # type: ignore[assignment,misc]
    ChromaNotFoundError = None


class ChromaInitializer:
    """ChromaDB 初始化器

    封装集合创建、维度检测、冲突处理、数据迁移等初始化逻辑。
    通过引用 ChromaManager 实例来读写其属性。
    """

    def __init__(self, mgr: "ChromaManager") -> None:
        self._mgr = mgr

    async def initialize(self) -> None:
        """完整初始化流程"""
        mgr = self._mgr

        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. "
                "Please install it with: pip install chromadb"
            )

        logger.debug("Initializing ChromaManager...")
        chroma_path = mgr.data_path / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Chroma data path: {chroma_path}")

        logger.debug("Creating Chroma persistent client...")
        mgr.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        existing_collection = self._get_or_check_collection()
        detected_dimension = await self._detect_and_init_embedding(existing_collection)
        backup_data = self._handle_dimension_conflict(existing_collection, detected_dimension)
        self._create_or_use_collection(existing_collection)

        if backup_data and mgr.reimport_on_dimension_conflict:
            await self._reimport_memories(backup_data)
        elif backup_data and not mgr.reimport_on_dimension_conflict:
            logger.warning(
                f"维度冲突处理：配置禁用了自动重新导入，"
                f"{len(backup_data.get('ids', []))} 条记忆已备份但未导入。"
                f"您可以在备份集合中找到这些记忆。"
            )

        from iris_memory.storage.chroma_queries import ChromaQueries
        from iris_memory.storage.chroma_operations import ChromaOperations

        mgr._queries = ChromaQueries(mgr)
        mgr._operations = ChromaOperations(mgr)

        logger.debug(
            f"Chroma manager initialized successfully. "
            f"Collection: {mgr.collection_name}, "
            f"Model: {mgr.embedding_manager.get_model()}, "
            f"Dimension: {mgr.embedding_dimension}"
        )

        mgr._is_ready = True

    # ── 集合检查 ──

    def _get_or_check_collection(self) -> Any:
        """获取或检查现有集合"""
        mgr = self._mgr
        existing_collection = None
        try:
            existing_collection = mgr.client.get_collection(name=mgr.collection_name)
            logger.debug(f"Found existing collection: {mgr.collection_name}")
        except Exception as e:
            is_not_found = (
                (ChromaNotFoundError and isinstance(e, ChromaNotFoundError))
                or isinstance(e, ValueError)
                or "not exist" in str(e).lower()
                or "not found" in str(e).lower()
                or "does not exist" in str(e).lower()
            )
            if is_not_found:
                logger.debug(f"Collection does not exist: {mgr.collection_name}")
            else:
                logger.error(f"Error accessing collection {mgr.collection_name}: {e}")
                raise
        return existing_collection

    async def _detect_and_init_embedding(self, existing_collection: Any) -> Optional[int]:
        """检测并初始化嵌入"""
        mgr = self._mgr
        detected_dimension: Optional[int] = None

        if mgr.auto_detect_dimension and existing_collection:
            logger.debug("Auto-detecting embedding dimension from existing collection...")
            detected_dimension = await mgr.embedding_manager.detect_existing_dimension(
                existing_collection
            )
            if detected_dimension:
                logger.debug(f"Auto-detected embedding dimension: {detected_dimension}")
                mgr.embedding_dimension = detected_dimension

        logger.debug("Initializing embedding manager...")
        await mgr.embedding_manager.initialize()

        actual_dimension = mgr.embedding_manager.get_dimension()
        logger.debug(
            f"Embedding provider dimension: {actual_dimension}, "
            f"configured: {mgr.embedding_dimension}"
        )
        if mgr.embedding_dimension != actual_dimension:
            logger.warning(
                f"Configured dimension ({mgr.embedding_dimension}) differs from "
                f"provider dimension ({actual_dimension}), using provider dimension"
            )
            mgr.embedding_dimension = actual_dimension

        return detected_dimension

    # ── 维度冲突 ──

    def _get_dimension_from_metadata(self, collection: Any) -> Optional[int]:
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
        self, existing_collection: Any, detected_dimension: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """处理维度冲突"""
        mgr = self._mgr

        if not existing_collection:
            return None

        collection_dimension = detected_dimension
        if collection_dimension is None:
            collection_dimension = self._get_dimension_from_metadata(existing_collection)

        if collection_dimension is None:
            logger.debug("Could not determine existing collection dimension, skipping conflict check")
            return None

        actual_dimension = mgr.embedding_manager.get_dimension()
        if collection_dimension == actual_dimension:
            return None

        old_count = existing_collection.count()
        logger.error(
            f"⚠️ CRITICAL: Embedding dimension conflict detected! "
            f"Collection has {collection_dimension}-dim vectors but provider outputs "
            f"{actual_dimension}-dim. Old memories count: {old_count}. "
            f"This usually happens when the embedding model/provider changes. "
            f"The old collection will be backed up and re-imported with new embeddings."
        )

        backup_data = None
        if old_count > 0:
            backup_data = self._backup_collection_data(existing_collection, old_count)
            if backup_data is None:
                file_backup_ok = self._backup_collection_before_delete(
                    existing_collection, old_count
                )
                if not file_backup_ok:
                    logger.error(
                        f"All backup methods failed for collection '{mgr.collection_name}' "
                        f"({old_count} memories). Aborting deletion to prevent data loss. "
                        f"Please manually resolve the dimension conflict."
                    )
                    return None
                logger.warning(
                    f"In-memory backup failed but file-level backup succeeded. "
                    f"Proceeding with collection deletion."
                )

        mgr.client.delete_collection(name=mgr.collection_name)
        logger.warning(
            f"Old collection '{mgr.collection_name}' deleted after dimension conflict. "
            f"{old_count} memories will be re-imported with new embeddings."
        )

        return backup_data

    # ── 备份 ──

    def _backup_collection_data(
        self, existing_collection: Any, old_count: int
    ) -> Optional[Dict[str, Any]]:
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

    def _backup_collection_before_delete(
        self, existing_collection: Any, old_count: int
    ) -> bool:
        """在删除集合前备份数据到新集合 + JSON 文件"""
        mgr = self._mgr
        backup_name = f"{mgr.collection_name}_backup_{int(time.time())}"
        chroma_backup_ok = False
        json_backup_ok = False
        all_data: Optional[Dict[str, Any]] = None

        try:
            all_data = existing_collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            if not all_data or not all_data.get("ids"):
                logger.debug("No data in old collection to backup.")
                return True

            backup_col = mgr.client.create_collection(name=backup_name)
            batch_size = 500
            ids = all_data["ids"]
            embeddings_data = all_data.get("embeddings")
            has_embeddings = embeddings_data is not None and len(embeddings_data) > 0

            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                batch_kwargs: Dict[str, Any] = {"ids": ids[i:end]}
                if all_data.get("documents"):
                    batch_kwargs["documents"] = all_data["documents"][i:end]
                if all_data.get("metadatas"):
                    batch_kwargs["metadatas"] = all_data["metadatas"][i:end]
                if has_embeddings:
                    batch_kwargs["embeddings"] = embeddings_data[i:end]
                backup_col.add(**batch_kwargs)

            logger.warning(
                f"Backed up {old_count} memories to collection '{backup_name}' "
                f"before dimension migration."
            )
            chroma_backup_ok = True
        except Exception as e:
            logger.error(
                f"Failed to backup collection to ChromaDB: {e}.", exc_info=True
            )

        try:
            if not chroma_backup_ok:
                all_data = existing_collection.get(include=["documents", "metadatas"])
            json_backup_path = mgr.data_path / "backup" / f"{backup_name}.json"
            json_backup_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {
                "ids": all_data.get("ids", []) if all_data else [],
                "documents": all_data.get("documents", []) if all_data else [],
                "metadatas": all_data.get("metadatas", []) if all_data else [],
                "backup_time": time.time(),
                "original_collection": mgr.collection_name,
                "memory_count": old_count,
            }
            json_backup_path.write_text(
                json.dumps(export_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning(
                f"JSON backup saved to {json_backup_path} ({old_count} memories)."
            )
            json_backup_ok = True
        except Exception as e:
            logger.error(
                f"Failed to save JSON backup: {e}. "
                f"{'ChromaDB backup exists.' if chroma_backup_ok else f'{old_count} memories will be permanently lost.'}",
                exc_info=True,
            )

        return chroma_backup_ok or json_backup_ok

    # ── 集合创建 ──

    def _create_or_use_collection(self, existing_collection: Any) -> None:
        """创建或使用现有集合"""
        mgr = self._mgr
        if (
            existing_collection
            and mgr.collection_name
            in [c.name for c in mgr.client.list_collections()]
        ):
            mgr.collection = mgr.client.get_collection(name=mgr.collection_name)
            logger.debug(f"Using existing collection: {mgr.collection_name}")
        else:
            logger.debug(f"Creating new collection: {mgr.collection_name}")
            mgr.collection = mgr.client.create_collection(
                name=mgr.collection_name,
                metadata={
                    "description": "Iris Memory Plugin - Three-layer memory system",
                    "embedding_model": mgr.embedding_manager.get_model(),
                    "embedding_dimension": mgr.embedding_dimension,
                },
            )
            logger.debug(f"Created new collection: {mgr.collection_name}")

    # ── 重新导入 ──

    async def _reimport_memories(self, backup_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用新的嵌入模型重新导入备份的记忆数据"""
        mgr = self._mgr
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
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size] if documents else []
            batch_metas = metadatas[i : i + batch_size] if metadatas else []

            batch_embeddings: List[Any] = []
            batch_valid_ids: List[str] = []
            batch_valid_docs: List[str] = []
            batch_valid_metas: List[Dict[str, Any]] = []

            for j, doc in enumerate(batch_docs):
                mem_id = batch_ids[j]
                if not doc:
                    failed_count += 1
                    failed_ids.append(mem_id)
                    failed_details.append(
                        {"id": mem_id, "reason": "empty_document", "stage": "embedding"}
                    )
                    continue
                try:
                    embedding = await mgr._generate_embedding(doc)
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
                        failed_details.append(
                            {"id": mem_id, "reason": "embedding_generation_failed", "stage": "embedding"}
                        )
                        logger.warning(f"Failed to generate embedding for memory {mem_id}")
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(mem_id)
                    failed_details.append(
                        {"id": mem_id, "reason": str(e), "stage": "embedding", "error_type": type(e).__name__}
                    )
                    logger.error(f"Error generating embedding for memory {mem_id}: {e}")

            if batch_valid_ids:
                try:
                    mgr.collection.add(
                        ids=batch_valid_ids,
                        embeddings=batch_embeddings,
                        documents=batch_valid_docs,
                        metadatas=batch_valid_metas,
                    )
                    success_count += len(batch_valid_ids)
                except Exception as e:
                    failed_count += len(batch_valid_ids)
                    failed_ids.extend(batch_valid_ids)
                    for mid in batch_valid_ids:
                        failed_details.append(
                            {"id": mid, "reason": str(e), "stage": "import", "error_type": type(e).__name__}
                        )
                    logger.error(f"Failed to import batch: {e}")

        logger.info(
            f"Re-import complete: {success_count} succeeded, {failed_count} failed "
            f"out of {total} memories"
        )
        if failed_ids:
            logger.warning(
                f"Failed memory IDs ({len(failed_ids)}): "
                f"{failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}"
            )

        return {
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_ids": failed_ids,
            "failed_details": failed_details,
        }
