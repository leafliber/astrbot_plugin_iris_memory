"""
Iris Chat Memory - L2 记忆库模块

使用 FAISS + SQLite 存储长期记忆向量，支持群聊隔离、人格隔离、降级兜底。
"""

from .models import MemoryEntry, MemorySearchResult
from .adapter import L2MemoryAdapter, SUPPORTED_EMBEDDING_MODELS
from .retriever import MemoryRetriever
from .io import (
    MemoryExporter,
    MemoryImporter,
    ExportStats,
    ImportStats,
    MemoryExport,
    export_memories,
    import_memories,
)

__all__ = [
    # 数据模型
    "MemoryEntry",
    "MemorySearchResult",
    # 核心组件
    "L2MemoryAdapter",
    "SUPPORTED_EMBEDDING_MODELS",
    "MemoryRetriever",
    # 导入导出
    "MemoryExporter",
    "MemoryImporter",
    "ExportStats",
    "ImportStats",
    "MemoryExport",
    "export_memories",
    "import_memories",
]
