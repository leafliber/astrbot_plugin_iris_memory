"""混合检索策略"""

from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams
from iris_memory.retrieval.retrieval_logger import retrieval_log

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory
    from iris_memory.core.types import StorageLayer


class HybridStrategy:
    """HYBRID — 混合检索（默认策略）

    流程：向量检索 → 工作记忆合并 → 情感过滤 → Reranker 重排序
    """

    def __init__(
        self,
        chroma_manager: object,
        update_access_fn: Callable,
        emotion_filter_fn: Callable,
        rerank_fn: Callable,
        get_working_memories_fn: Callable,
        merge_memories_fn: Callable,
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
        self._enable_emotion_aware = enable_emotion_aware
        self._enable_working_memory_merge = enable_working_memory_merge
        self._session_manager = session_manager

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        candidate_memories = await self._chroma.query_memories(
            query_text=params.query,
            user_id=params.user_id,
            group_id=params.group_id,
            top_k=params.top_k * 2,
            storage_layer=params.storage_layer,
            persona_id=params.persona_id,
        )
        retrieval_log.vector_query(
            params.user_id,
            len(candidate_memories),
            params.storage_layer.value
            if params.storage_layer and hasattr(params.storage_layer, "value")
            else None,
        )

        if self._enable_working_memory_merge and self._session_manager:
            working_memories = await self._get_working_memories(
                params.query, params.user_id, params.group_id,
                params.storage_layer,
            )
            if working_memories:
                retrieval_log.working_memory_merged(
                    params.user_id,
                    len(working_memories),
                    len(candidate_memories) + len(working_memories),
                )
                candidate_memories = self._merge_memories(
                    candidate_memories, working_memories,
                    max_total=params.top_k * 2,
                )

        if not candidate_memories:
            retrieval_log.no_memories_found(params.user_id, params.query)
            return []

        if self._enable_emotion_aware and params.emotional_state:
            before_count = len(candidate_memories)
            candidate_memories = self._apply_emotion_filter(
                candidate_memories, params.emotional_state, params.user_id,
            )
            retrieval_log.emotion_filter_applied(
                params.user_id, before_count,
                len(candidate_memories), "negative_state",
            )

        ranked_memories = self._rerank_memories(
            candidate_memories, params.query,
            params.emotional_state, params.user_id,
        )

        result = ranked_memories[:params.top_k]
        await self._update_access(result)
        return result
