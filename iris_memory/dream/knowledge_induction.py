"""增量知识归纳：统一模式发现与 L2→L3 实体关系提取。"""

from typing import Optional

from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter
from iris_memory.llm.manager import LLMManager


class KnowledgeInductionPhase:
    """统一的增量知识归纳阶段。

    模式发现写入的派生 L2 条目会直接标记为 ``kg_processed``，因此实体提取
    不会再次消费同一抽象。
    """

    async def execute(
        self,
        l2: L2MemoryAdapter,
        l3: Optional[L3KGAdapter],
        llm: Optional[LLMManager],
        entries: Optional[list] = None,
        persona_id: str = "default",
        component_manager=None,
    ) -> dict:
        details: dict = {}

        from .pattern_discovery import PatternDiscoveryPhase

        details["pattern_discovery"] = await PatternDiscoveryPhase().execute(
            l2, l3, llm, entries=entries, persona_id=persona_id
        )

        from .knowledge_extract import KnowledgeExtractPhase

        details["knowledge_extract"] = await KnowledgeExtractPhase().execute(
            l2,
            l3,
            llm,
            persona_id=persona_id,
            component_manager=component_manager,
        )

        return details
