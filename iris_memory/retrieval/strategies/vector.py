"""纯向量检索策略"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory


class VectorOnlyStrategy:
    """VECTOR_ONLY — 纯语义向量检索

    适用于简单关键词查询、短文本查询。
    """

    def __init__(self, chroma_manager: object, update_access_fn: object) -> None:
        self._chroma = chroma_manager
        self._update_access = update_access_fn

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        memories = await self._chroma.query_memories(
            query_text=params.query,
            user_id=params.user_id,
            group_id=params.group_id,
            top_k=params.top_k,
            storage_layer=params.storage_layer,
            persona_id=params.persona_id,
        )
        await self._update_access(memories)
        return memories
