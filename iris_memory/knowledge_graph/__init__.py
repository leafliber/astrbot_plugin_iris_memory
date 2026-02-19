"""
知识图谱模块 — 基于 SQLite + FTS5 的轻量级知识图谱

提供实体/关系存储、全文检索、多跳推理能力。
架构概述：
- kg_models.py:    KGNode / KGEdge / KGTriple 数据模型
- kg_storage.py:   SQLite + FTS5 持久化层
- kg_extractor.py: 三元组提取器（规则 + LLM）
- kg_reasoning.py: 受限 BFS 多跳推理引擎
- kg_context.py:   知识图谱结果格式化（注入 LLM 上下文）
"""

from iris_memory.knowledge_graph.kg_models import KGNode, KGEdge, KGTriple
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.knowledge_graph.kg_extractor import KGExtractor
from iris_memory.knowledge_graph.kg_reasoning import KGReasoning
from iris_memory.knowledge_graph.kg_context import KGContextFormatter

__all__ = [
    "KGNode",
    "KGEdge",
    "KGTriple",
    "KGStorage",
    "KGExtractor",
    "KGReasoning",
    "KGContextFormatter",
]
