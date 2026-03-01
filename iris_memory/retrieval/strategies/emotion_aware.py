"""情感感知检索策略"""

from __future__ import annotations

from typing import Callable, List, TYPE_CHECKING

from iris_memory.retrieval.strategies.base import StrategyParams

if TYPE_CHECKING:
    from iris_memory.models.memory import Memory
    from iris_memory.models.emotion_state import EmotionalState


class EmotionAwareStrategy:
    """EMOTION_AWARE — 情感感知检索

    检索后对情感不匹配的记忆进行降权标记。
    """

    def __init__(
        self,
        chroma_manager: object,
        update_access_fn: object,
        emotion_filter_fn: Callable,
    ) -> None:
        self._chroma = chroma_manager
        self._update_access = update_access_fn
        self._apply_emotion_filter = emotion_filter_fn

    async def execute(self, params: StrategyParams) -> List["Memory"]:
        memories = await self._chroma.query_memories(
            query_text=params.query,
            user_id=params.user_id,
            group_id=params.group_id,
            top_k=params.top_k * 2,
            storage_layer=params.storage_layer,
            persona_id=params.persona_id,
        )

        if params.emotional_state:
            memories = self._apply_emotion_filter(
                memories, params.emotional_state, params.user_id
            )

        result = memories[:params.top_k]
        await self._update_access(result)
        return result
