"""
旧版（v2.x）ChromaDB → 新版 L2（FAISS + SQLite）记忆迁移器

映射规则（沿用 v2 io_service.export_to_iris_chat_memory 的字段映射）：
    created_time      → timestamp
    group_id          → group_id
    user_id           → user_id
    last_access_time  → last_access_time
    access_count      → access_count (int)
    confidence        → confidence (float)
    summarized        → source = "summary" / "tool"
    sender_name/type/importance_score/rif_score 原样保留
    storage_layer     → original_storage_layer
    scope             → original_scope
    persona_id        → 条目顶层 persona_id（人格命名空间隔离）
    旧 chroma 条目 id → metadata.legacy_id（新版 add_memory 重新生成 id）
    kg_processed      → True（旧记忆的知识已在旧图谱中，避免梦境管线
                        对全量旧记忆重新做 LLM 提取造成调用量洪峰）

设计说明：
- chromadb 为软依赖：新环境不安装，此处延迟 import，缺包时记日志并跳过
- 旧向量不复用（嵌入模型/维度可能不同）：仅导出 documents+metadatas，
  embedding 由新 L2 管线（MemoryImporter → adapter.add_memory → _embed）重算
- 大批量：分批读取 + 逐批导入 + 进度日志
- 保真导入：skip_duplicates=False，不因相似度静默丢弃旧记忆
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core import get_logger

logger = get_logger("legacy_migration")

#: 每批读取/导入的条目数
BATCH_SIZE = 200

#: 旧版默认 collection 名（v2 config schema: embedding.collection_name）
LEGACY_COLLECTION_NAME = "iris_memory"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def map_legacy_metadata(meta: Dict[str, Any], legacy_id: str) -> Dict[str, Any]:
    """将旧 chroma metadata 映射为新版 entry.metadata"""
    timestamp = meta.get("created_time") or datetime.now().isoformat()
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)

    entry_meta: Dict[str, Any] = {
        # 新版标准字段
        "group_id": meta.get("group_id") or "",
        "user_id": meta.get("user_id") or "",
        "timestamp": timestamp,
        "access_count": _to_int(meta.get("access_count"), 0),
        "confidence": _to_float(meta.get("confidence"), 0.5),
        "source": "summary" if meta.get("summarized") else "tool",
        # 旧记忆的知识已在旧图谱中，标记已处理，避免梦境管线全量重提
        "kg_processed": True,
        # 保留的旧版信息
        "sender_name": meta.get("sender_name") or "",
        "type": meta.get("type"),
        "original_storage_layer": meta.get("storage_layer"),
        "original_scope": meta.get("scope"),
        "importance_score": _to_float(meta.get("importance_score"), 0.5),
        "rif_score": _to_float(meta.get("rif_score"), 0.5),
        "legacy_id": legacy_id,
        "migrated_from": "iris_memory_v2",
    }
    last_access = meta.get("last_access_time")
    if last_access:
        entry_meta["last_access_time"] = last_access

    # 剔除 None，避免写入空值
    return {k: v for k, v in entry_meta.items() if v is not None}


def _get_available_adapter(component_manager: Any) -> Optional[Any]:
    """从组件管理器获取可用的 L2 适配器"""
    try:
        adapter = component_manager.get_component("l2_memory")
    except Exception as e:
        logger.warning(f"获取 L2 组件失败：{e}")
        return None
    if adapter is None or not getattr(adapter, "is_available", False):
        return None
    return adapter


def _get_legacy_collection(client: Any) -> Optional[Any]:
    """获取旧版 collection：先试默认名，再退到第一个可用 collection"""
    try:
        return client.get_collection(LEGACY_COLLECTION_NAME)
    except Exception:
        pass

    try:
        collections = client.list_collections()
    except Exception as e:
        logger.warning(f"列举旧 ChromaDB collection 失败：{e}")
        return None

    for col in collections:
        # chromadb 新版本 list_collections 返回名称字符串，旧版本返回对象
        name = col if isinstance(col, str) else getattr(col, "name", None)
        if not name:
            continue
        try:
            return client.get_collection(name)
        except Exception:
            continue
    return None


def _fetch_all_batches(
    collection: Any, total: int
) -> List[Tuple[List[str], List[Any], List[Any]]]:
    """分批读取 collection 全量数据（ids/documents/metadatas）"""
    batches: List[Tuple[List[str], List[Any], List[Any]]] = []
    offset = 0
    while offset < total:
        res = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = res.get("ids") or []
        if not ids:
            break
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        batches.append((ids, documents, metadatas))
        offset += len(ids)
    return batches


async def migrate_l2(detection: Any, component_manager: Any) -> Dict[str, Any]:
    """迁移旧 ChromaDB 记忆到新版 L2

    Args:
        detection: LegacyDetection 检测结果
        component_manager: 组件管理器

    Returns:
        统计信息
    """
    stats: Dict[str, Any] = {
        "status": "ok",
        "total": 0,
        "imported": 0,
        "skipped": 0,
        "errors": 0,
    }

    if detection.chroma_dir is None:
        stats["status"] = "skipped_no_data"
        return stats

    adapter = _get_available_adapter(component_manager)
    if adapter is None:
        stats["status"] = "skipped_adapter_unavailable"
        logger.warning("L2 记忆库组件不可用，跳过旧 ChromaDB 记忆迁移")
        return stats

    # chromadb 软依赖：新环境不安装，延迟 import
    try:
        import chromadb
    except ImportError:
        stats["status"] = "skipped_missing_chromadb"
        logger.warning(
            "未安装 chromadb，跳过旧向量库迁移。"
            "旧数据已备份到 legacy_backup/chroma/（不会被删除）。"
            "如需迁移：在插件环境安装 chromadb 后，删除 KV 标志 "
            "legacy:migration_done 并重启插件即可重试"
        )
        return stats

    try:
        client = chromadb.PersistentClient(path=str(detection.chroma_dir))
        collection = _get_legacy_collection(client)
        if collection is None:
            stats["status"] = "skipped_no_collection"
            logger.warning("旧 ChromaDB 中未找到 collection，跳过 L2 迁移")
            return stats

        total = int(collection.count())
        stats["total"] = total
        if total == 0:
            logger.info("旧 ChromaDB collection 为空，无需迁移")
            return stats

        logger.info(f"开始迁移旧 ChromaDB 记忆：共 {total} 条，分批大小 {BATCH_SIZE}")

        # MemoryImporter.import_entries 经 adapter.add_memory 写入，
        # 嵌入由新 L2 管线自动重算（旧向量不复用）
        from iris_memory.l2_memory.io import MemoryImporter
        from iris_memory.l2_memory.models import MemoryEntry

        importer = MemoryImporter(adapter)
        processed = 0

        for ids, documents, metadatas in _fetch_all_batches(collection, total):
            entries: List[MemoryEntry] = []
            for i, legacy_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                if not content:
                    stats["skipped"] += 1
                    continue
                meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                entries.append(
                    MemoryEntry(
                        id=str(legacy_id),
                        content=content,
                        metadata=map_legacy_metadata(meta, str(legacy_id)),
                        persona_id=str(meta.get("persona_id") or "default"),
                    )
                )

            if entries:
                # 保真导入：不做相似度去重，避免迁移期静默丢条目
                batch_stats = await importer.import_entries(
                    entries, skip_duplicates=False
                )
                stats["imported"] += batch_stats.imported_count
                stats["skipped"] += batch_stats.skipped_count
                stats["errors"] += batch_stats.error_count

            processed += len(ids)
            logger.info(
                f"L2 迁移进度：{processed}/{total}"
                f"（已导入 {stats['imported']}，跳过 {stats['skipped']}，"
                f"错误 {stats['errors']}）"
            )

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.error(f"旧 ChromaDB 迁移失败：{e}", exc_info=True)
        return stats

    logger.info(
        f"L2 迁移完成：共 {stats['total']} 条，导入 {stats['imported']} 条，"
        f"跳过 {stats['skipped']} 条，错误 {stats['errors']} 条"
    )
    return stats
