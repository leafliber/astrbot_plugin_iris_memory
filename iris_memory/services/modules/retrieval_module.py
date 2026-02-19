"""
检索模块 — 封装 MemoryRetrievalEngine 的创建和配置

RetrievalEngine 内部已经聚合了 Reranker 和 RetrievalRouter。
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

logger = get_logger("module.retrieval")


class RetrievalModule:
    """检索模块"""

    def __init__(self) -> None:
        self._retrieval_engine: Optional[MemoryRetrievalEngine] = None

    @property
    def retrieval_engine(self) -> Optional[MemoryRetrievalEngine]:
        return self._retrieval_engine

    def initialize(
        self,
        chroma_manager: Any,
        rif_scorer: Any,
        emotion_analyzer: Any,
        session_manager: Any,
        llm_retrieval_router: Any = None,
    ) -> None:
        """初始化检索引擎"""
        from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine

        self._retrieval_engine = MemoryRetrievalEngine(
            chroma_manager=chroma_manager,
            rif_scorer=rif_scorer,
            emotion_analyzer=emotion_analyzer,
            session_manager=session_manager,
            llm_retrieval_router=llm_retrieval_router,
        )
        logger.info("RetrievalModule initialized")

    def set_kg_module(self, kg_module: Any) -> None:
        """注入知识图谱模块到检索引擎"""
        if self._retrieval_engine:
            self._retrieval_engine.set_kg_module(kg_module)
            logger.info("KG module injected into RetrievalEngine")

    def apply_config(self, cfg: Any) -> None:
        """应用配置到检索引擎"""
        from iris_memory.core.defaults import DEFAULTS

        if self._retrieval_engine:
            self._retrieval_engine.set_config(
                {
                    "max_context_memories": cfg.max_context_memories,
                    "enable_time_aware": DEFAULTS.llm_integration.enable_time_aware,
                    "enable_emotion_aware": DEFAULTS.llm_integration.enable_emotion_aware,
                    "enable_token_budget": cfg.enable_inject,
                    "token_budget": cfg.token_budget,
                    "coordination_strategy": DEFAULTS.llm_integration.coordination_strategy,
                }
            )
