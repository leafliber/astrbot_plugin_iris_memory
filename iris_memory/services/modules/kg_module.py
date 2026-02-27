"""
知识图谱模块 — 封装 KGStorage / KGExtractor / KGReasoning / KGContextFormatter

作为第 7 个 Feature Module 集成到 MemoryService 中。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage
    from iris_memory.knowledge_graph.kg_extractor import KGExtractor
    from iris_memory.knowledge_graph.kg_reasoning import KGReasoning
    from iris_memory.knowledge_graph.kg_context import KGContextFormatter
    from iris_memory.knowledge_graph.kg_models import KGTriple, KGEdge, KGNode
    from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult

logger = get_logger("module.kg")


class KnowledgeGraphModule:
    """知识图谱模块

    职责：
    1. 管理 KGStorage / KGExtractor / KGReasoning 的生命周期
    2. 提供统一接口供 MemoryService 调用
    3. 协同 CaptureModule（提取三元组）和 RetrievalModule（图遍历检索）
    """

    def __init__(self) -> None:
        self._storage: Optional[KGStorage] = None
        self._extractor: Optional[KGExtractor] = None
        self._reasoning: Optional[KGReasoning] = None
        self._formatter: Optional[KGContextFormatter] = None
        self._enabled: bool = True

    # ── 属性 ──

    @property
    def storage(self) -> Optional["KGStorage"]:
        return self._storage

    @property
    def extractor(self) -> Optional["KGExtractor"]:
        return self._extractor

    @property
    def reasoning(self) -> Optional["KGReasoning"]:
        return self._reasoning

    @property
    def formatter(self) -> Optional["KGContextFormatter"]:
        return self._formatter

    @property
    def is_initialized(self) -> bool:
        return self._storage is not None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.is_initialized

    # ── 初始化 ──

    async def initialize(
        self,
        plugin_data_path: Path,
        astrbot_context: Any = None,
        provider_id: Optional[str] = None,
        kg_mode: str = "rule",
        max_depth: int = 3,
        max_nodes_per_hop: int = 10,
        max_facts: int = 8,
        enabled: bool = True,
    ) -> None:
        """初始化知识图谱模块

        Args:
            plugin_data_path: 数据目录
            astrbot_context: AstrBot 上下文（用于 LLM）
            provider_id: LLM provider ID
            kg_mode: 提取模式 ("rule" / "llm" / "hybrid")
            max_depth: BFS 最大跳数
            max_nodes_per_hop: 每跳最大节点数
            max_facts: 注入 LLM 的最大事实数
            enabled: 是否启用
        """
        self._enabled = enabled
        if not enabled:
            logger.debug("KnowledgeGraphModule disabled by config")
            return

        from iris_memory.knowledge_graph.kg_storage import KGStorage
        from iris_memory.knowledge_graph.kg_extractor import KGExtractor
        from iris_memory.knowledge_graph.kg_reasoning import KGReasoning
        from iris_memory.knowledge_graph.kg_context import KGContextFormatter

        # 初始化存储
        db_path = plugin_data_path / "knowledge_graph.db"
        self._storage = KGStorage(db_path)
        await self._storage.initialize(db_path)

        # 初始化提取器
        self._extractor = KGExtractor(
            storage=self._storage,
            mode=kg_mode,
            astrbot_context=astrbot_context,
            provider_id=provider_id,
        )

        # 初始化推理引擎
        self._reasoning = KGReasoning(
            storage=self._storage,
            max_depth=max_depth,
            max_nodes_per_hop=max_nodes_per_hop,
        )

        # 初始化格式化器
        self._formatter = KGContextFormatter(
            max_facts=max_facts,
        )

        logger.debug(
            f"KnowledgeGraphModule initialized: mode={kg_mode}, "
            f"max_depth={max_depth}, db={db_path}"
        )

    async def close(self) -> None:
        """关闭资源"""
        if self._storage:
            await self._storage.close()
            self._storage = None

    # ── 捕获阶段接口 ──

    async def process_memory(
        self,
        memory: Any,
        persona_id: Optional[str] = None,
    ) -> List["KGTriple"]:
        """从记忆中提取三元组并存入图谱

        在 capture_and_store_memory() 之后调用。

        Args:
            memory: Memory 对象
            persona_id: 人格 ID（始终写入节点/边）

        Returns:
            提取到的三元组列表
        """
        if not self.enabled or not self._extractor:
            return []

        try:
            _raw = persona_id or getattr(memory, "persona_id", None)
            _persona = _raw if isinstance(_raw, str) else "default"
            triples = await self._extractor.extract_and_store(
                text=memory.content,
                user_id=memory.user_id,
                group_id=memory.group_id,
                memory_id=memory.id,
                sender_name=memory.sender_name,
                existing_entities=getattr(memory, "detected_entities", None),
                persona_id=_persona,
            )

            # 更新 Memory 的 graph_nodes / graph_edges（可选）
            if triples and hasattr(memory, "graph_nodes"):
                nodes_set = set(memory.graph_nodes or [])
                edges_set = set(memory.graph_edges or [])
                for triple in triples:
                    nodes_set.add(triple.subject)
                    nodes_set.add(triple.object)
                    edges_set.add(f"{triple.subject}->{triple.predicate}->{triple.object}")
                memory.graph_nodes = list(nodes_set)
                memory.graph_edges = list(edges_set)

            return triples

        except Exception as e:
            logger.warning(f"Failed to process memory for KG: {e}")
            return []

    # ── 检索阶段接口 ──

    async def graph_retrieve(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_results: int = 10,
        persona_id: Optional[str] = None,
    ) -> "ReasoningResult":
        """执行图遍历检索 + 多跳推理

        Args:
            query: 查询文本
            user_id: 用户 ID
            group_id: 群组 ID
            max_depth: 最大跳数
            max_results: 最大路径数
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            ReasoningResult
        """
        if not self.enabled or not self._reasoning:
            from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult
            return ReasoningResult()

        return await self._reasoning.reason(
            query=query,
            user_id=user_id,
            group_id=group_id,
            max_depth=max_depth,
            max_results=max_results,
            persona_id=persona_id,
        )

    async def format_graph_context(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> str:
        """执行图检索并格式化为 LLM 上下文

        一站式接口：推理 + 格式化。

        Args:
            query: 查询文本
            user_id: 用户 ID
            group_id: 群组 ID
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            格式化后的知识关联文本（可能为空字符串）
        """
        if not self.enabled or not self._reasoning or not self._formatter:
            return ""

        try:
            result = await self._reasoning.reason(
                query=query,
                user_id=user_id,
                group_id=group_id,
                persona_id=persona_id,
            )

            return self._formatter.format_reasoning_result(result, group_id)

        except Exception as e:
            logger.warning(f"Failed to format graph context: {e}")
            return ""

    # ── 统计 / 管理 ──

    async def get_stats(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """获取统计"""
        if not self._storage:
            return {"nodes": 0, "edges": 0}
        return await self._storage.get_stats(user_id, group_id)

    async def delete_user_data(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> int:
        """删除用户图谱数据"""
        if not self._storage:
            return 0
        return await self._storage.delete_user_data(user_id, group_id)

    async def delete_all(self) -> int:
        """删除所有图谱数据"""
        if not self._storage:
            return 0
        return await self._storage.delete_all()
