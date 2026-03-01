"""知识图谱检索策略"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams
from iris_memory.utils.logger import get_logger
from iris_memory.retrieval.retrieval_logger import retrieval_log

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory

logger = get_logger("retrieval.graph_strategy")


class GraphStrategy:
    """GRAPH_ONLY — 知识图谱 + 向量混合检索

    流程：
    1. 图推理获取相关 memory_id
    2. 向量检索获取候选记忆
    3. 图结果提升相关记忆的排序权重
    4. 合并 + 重排序
    """

    def __init__(
        self,
        chroma_manager: object,
        update_access_fn: Callable,
        emotion_filter_fn: Callable,
        rerank_fn: Callable,
        get_working_memories_fn: Callable,
        merge_memories_fn: Callable,
        kg_module: Optional[object] = None,
        enable_emotion_aware: bool = True,
        enable_working_memory_merge: bool = True,
        session_manager: Optional[object] = None,
    ) -> None:
        self._chroma = chroma_manager
        self._update_access = update_access_fn
        self._apply_emotion_filter = emotion_filter_fn
        self._rerank_memories = rerank_fn
        self._get_working_memories = get_working_memories_fn
        self._merge_memories = merge_memories_fn
        self._kg_module = kg_module
        self._enable_emotion_aware = enable_emotion_aware
        self._enable_working_memory_merge = enable_working_memory_merge
        self._session_manager = session_manager

    def set_kg_module(self, kg_module: object) -> None:
        self._kg_module = kg_module

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        # 向量检索部分
        vector_memories = await self._chroma.query_memories(
            query_text=params.query,
            user_id=params.user_id,
            group_id=params.group_id,
            top_k=params.top_k * 2,
            storage_layer=params.storage_layer,
            persona_id=params.persona_id,
        )

        # 图推理部分
        kg_memory_ids: set = set()
        if self._kg_module and self._kg_module.reasoning:
            try:
                reasoning_result = await self._kg_module.graph_retrieve(
                    query=params.query,
                    user_id=params.user_id,
                    group_id=params.group_id,
                    persona_id=params.persona_id,
                )
                for edge in reasoning_result.get_all_edges():
                    if edge.memory_id:
                        kg_memory_ids.add(edge.memory_id)
            except Exception as e:
                retrieval_log.graph_fallback(
                    params.user_id, f"KG reason error: {e}"
                )

        # 工作记忆合并
        if self._enable_working_memory_merge and self._session_manager:
            working_memories = await self._get_working_memories(
                params.query, params.user_id, params.group_id,
                params.storage_layer,
            )
            if working_memories:
                vector_memories = self._merge_memories(
                    vector_memories, working_memories,
                    max_total=params.top_k * 2,
                )

        if not vector_memories:
            retrieval_log.no_memories_found(params.user_id, params.query)
            return []

        # KG boost
        if kg_memory_ids:
            for mem in vector_memories:
                if mem.id in kg_memory_ids:
                    mem.importance_score = min(1.0, mem.importance_score + 0.3)
                    mem.rif_score = min(1.0, mem.rif_score + 0.2)

        # 情感过滤
        if self._enable_emotion_aware and params.emotional_state:
            vector_memories = self._apply_emotion_filter(
                vector_memories, params.emotional_state, params.user_id
            )

        # 重排序
        ranked = self._rerank_memories(
            vector_memories, params.query,
            params.emotional_state, params.user_id,
        )

        result = ranked[:params.top_k]
        await self._update_access(result)
        return result
