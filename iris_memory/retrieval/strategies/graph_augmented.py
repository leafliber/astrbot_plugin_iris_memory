"""图增强混合检索策略

在 HybridStrategy 基础上插入图扩展阶段：
  阶段1: 向量检索 TOP-K×2
  阶段2: 图扩展（1跳遍历 TOP-5）
  阶段3: 融合去重
  阶段4: 情感过滤
  阶段5: 增强重排序（含图中心性维度）+ Token 管理
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams
from iris_memory.retrieval.retrieval_logger import retrieval_log
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory

logger = get_logger("graph_augmented")


class GraphAugmentedHybridStrategy:
    """HYBRID + 图增强 — 默认策略

    在向量检索和重排序之间插入图扩展阶段，
    实现"向量召回 + 图关系补全 + 融合排序"。
    """

    GRAPH_EXPAND_TOP_N = 5
    GRAPH_EXPAND_DEPTH = 1
    GRAPH_EXPAND_MAX_NEIGHBORS = 3

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
        kg_storage: Optional[object] = None,
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
        self._kg = kg_storage

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        # 阶段 1: 向量检索
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

        # 阶段 2: 图扩展（KG 可用时启用）
        if self._kg and candidate_memories:
            graph_expanded = await self._graph_expand(
                candidate_memories[: self.GRAPH_EXPAND_TOP_N],
                params.user_id,
                params.group_id,
            )
            if graph_expanded:
                seen_ids: Set[str] = {m.id for m in candidate_memories}
                for mem in graph_expanded:
                    if mem.id not in seen_ids:
                        candidate_memories.append(mem)
                        seen_ids.add(mem.id)
                logger.debug(
                    f"Graph expansion added {len(graph_expanded)} candidates "
                    f"for user {params.user_id}"
                )

        # 阶段 3: 合并工作记忆
        if self._enable_working_memory_merge and self._session_manager:
            working_memories = await self._get_working_memories(
                params.query,
                params.user_id,
                params.group_id,
                params.storage_layer,
            )
            if working_memories:
                retrieval_log.working_memory_merged(
                    params.user_id,
                    len(working_memories),
                    len(candidate_memories) + len(working_memories),
                )
                candidate_memories = self._merge_memories(
                    candidate_memories,
                    working_memories,
                    max_total=params.top_k * 2,
                )

        if not candidate_memories:
            retrieval_log.no_memories_found(params.user_id, params.query)
            return []

        # 过滤待审核记忆
        candidate_memories = [
            m
            for m in candidate_memories
            if getattr(m, "review_status", None) not in ("pending_review", "rejected")
        ]

        # 阶段 4: 情感过滤
        if self._enable_emotion_aware and params.emotional_state:
            before_count = len(candidate_memories)
            candidate_memories = self._apply_emotion_filter(
                candidate_memories,
                params.emotional_state,
                params.user_id,
            )
            retrieval_log.emotion_filter_applied(
                params.user_id,
                before_count,
                len(candidate_memories),
                "negative_state",
            )

        # 阶段 5: 增强重排序
        ranked_memories = self._rerank_memories(
            candidate_memories,
            params.query,
            params.emotional_state,
            params.user_id,
        )

        result = ranked_memories[: params.top_k]
        await self._update_access(result)
        return result

    # ── 图扩展 ──

    async def _graph_expand(
        self,
        seed_memories: List["Memory"],
        user_id: str,
        group_id: Optional[str],
    ) -> List["Memory"]:
        """对种子记忆进行 1 跳图遍历"""
        expanded_memory_ids: Set[str] = set()

        for memory in seed_memories:
            if not getattr(memory, "graph_nodes", None):
                continue
            for node_id in memory.graph_nodes[:2]:
                try:
                    neighbors = await self._kg.get_neighbors(
                        node_id=node_id,
                        depth=self.GRAPH_EXPAND_DEPTH,
                        max_neighbors=self.GRAPH_EXPAND_MAX_NEIGHBORS,
                        user_id=user_id,
                        group_id=group_id,
                    )
                    for neighbor in neighbors:
                        mid = getattr(neighbor, "memory_id", None)
                        if mid:
                            expanded_memory_ids.add(mid)
                except Exception:
                    continue

        expanded: List["Memory"] = []
        for mid in expanded_memory_ids:
            try:
                mem = await self._chroma.get_memory(mid)
                if mem:
                    mem.metadata["_source"] = "graph_expansion"
                    expanded.append(mem)
            except Exception:
                continue

        return expanded
