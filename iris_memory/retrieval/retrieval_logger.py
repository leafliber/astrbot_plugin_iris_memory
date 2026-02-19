"""
记忆检索结构化日志器

为记忆检索全生命周期提供统一的 DEBUG 日志。
所有日志事件均以 ``RETRIEVAL.`` 前缀标识，可通过 ``log_level=DEBUG`` 激活。

用法示例::

    from iris_memory.retrieval.retrieval_logger import retrieval_log
    retrieval_log.retrieve_start(user_id, query)
    retrieval_log.strategy_selected(user_id, strategy)
    retrieval_log.retrieve_ok(user_id, count, strategy)
"""

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

_logger = get_logger("retrieval_logger")


class RetrievalLogger:
    """记忆检索结构化日志器 — 统一 DEBUG 输出"""

    def retrieve_start(
        self,
        user_id: str,
        query: str,
        group_id: Optional[str] = None,
        top_k: int = 10
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.START user={user_id} "
            f"group={group_id or 'private'} "
            f"top_k={top_k} query_len={len(query)}"
        )

    def emotional_state(
        self,
        user_id: str,
        primary: str,
        intensity: float
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.EMOTION user={user_id} "
            f"primary={primary} intensity={intensity:.2f}"
        )

    def strategy_selected(
        self,
        user_id: str,
        strategy: str,
        source: str = "router"
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.STRATEGY user={user_id} "
            f"strategy={strategy} source={source}"
        )

    def routing_failed(self, user_id: str, error: str) -> None:
        _logger.debug(
            f"RETRIEVAL.ROUTING.FALLBACK user={user_id} "
            f"error={_trunc(error, 30)}"
        )

    def vector_query(
        self,
        user_id: str,
        candidate_count: int,
        storage_layer: Optional[str] = None
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.VECTOR user={user_id} "
            f"candidates={candidate_count} "
            f"layer={storage_layer or 'all'}"
        )

    def working_memory_merged(
        self,
        user_id: str,
        working_count: int,
        total_count: int
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.WORKING_MERGE user={user_id} "
            f"working={working_count} total={total_count}"
        )

    def emotion_filter_applied(
        self,
        user_id: str,
        before_count: int,
        after_count: int,
        reason: str
    ) -> None:
        filtered = before_count - after_count
        _logger.debug(
            f"RETRIEVAL.EMOTION_FILTER user={user_id} "
            f"filtered={filtered} "
            f"before={before_count} after={after_count} "
            f"reason={reason}"
        )

    def memory_filtered(
        self,
        user_id: str,
        memory_id: str,
        reason: str
    ) -> None:
        _logger.debug(
            f"RETRIEVAL.FILTER user={user_id} "
            f"id={memory_id[:8]}... reason={reason}"
        )

    def no_memories_found(self, user_id: str, query: str) -> None:
        _logger.debug(
            f"RETRIEVAL.EMPTY user={user_id} query_len={len(query)}"
        )

    def retrieve_ok(
        self,
        user_id: str,
        count: int,
        strategy: str
    ) -> None:
        _logger.info(
            f"RETRIEVAL.OK user={user_id} "
            f"count={count} strategy={strategy}"
        )

    def retrieve_error(self, user_id: str, error: Exception) -> None:
        _logger.error(
            f"RETRIEVAL.ERROR user={user_id} error={error}"
        )

    def graph_fallback(self, user_id: str, query: str) -> None:
        _logger.warning(
            f"RETRIEVAL.GRAPH_FALLBACK user={user_id} "
            f"query_len={len(query)} "
            f"reason=not_implemented"
        )


def _trunc(value: Any, max_len: int = 60) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "..."


retrieval_log = RetrievalLogger()
