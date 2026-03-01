"""时间感知检索策略"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory


class TimeAwareStrategy:
    """TIME_AWARE — 时间衰减加权检索

    按创建时间的衰减函数加权重新排序。
    """

    def __init__(self, chroma_manager: object, update_access_fn: object) -> None:
        self._chroma = chroma_manager
        self._update_access = update_access_fn

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        memories = await self._chroma.query_memories(
            query_text=params.query,
            user_id=params.user_id,
            group_id=params.group_id,
            top_k=params.top_k * 2,
            storage_layer=params.storage_layer,
            persona_id=params.persona_id,
        )

        scored = [
            (m, m.calculate_time_score(use_created_time=True))
            for m in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [m for m, _ in scored[:params.top_k]]
        await self._update_access(result)
        return result
