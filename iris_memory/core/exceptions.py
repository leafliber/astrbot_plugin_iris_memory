"""Iris Memory 统一异常定义

提供领域特定的异常类型，替代裸 RuntimeError / ValueError，
使上层调用方可以精确捕获和处理不同类别的错误。

异常层次：
    IrisMemoryError
    ├── StorageError          — ChromaDB / SQLite 存储层错误
    │   └── StorageNotReadyError  — 存储未初始化
    ├── EmbeddingError        — 嵌入向量生成/模型加载错误
    ├── ProviderError         — LLM / 嵌入提供者不可用
    └── MigrationError        — 数据迁移/维度冲突错误
"""

from __future__ import annotations


class IrisMemoryError(Exception):
    """Iris Memory 基础异常"""


class StorageError(IrisMemoryError):
    """存储层操作异常（ChromaDB / SQLite）"""


class StorageNotReadyError(StorageError):
    """存储未就绪 — 在调用 initialize() 之前使用了存储"""


class EmbeddingError(IrisMemoryError):
    """嵌入向量生成或模型加载异常"""


class ProviderError(IrisMemoryError):
    """LLM / 嵌入提供者不可用"""


class MigrationError(StorageError):
    """数据迁移失败（维度冲突、schema 升级等）"""
