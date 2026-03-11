"""LLM 统计 Web 服务"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("web.llm_svc")


class LlmWebService:
    """LLM 调用统计数据查询"""

    def __init__(self) -> None:
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            try:
                from iris_memory.stats.registry import get_stats_registry

                self._registry = get_stats_registry()
            except Exception:
                pass
        return self._registry

    def get_summary(self) -> Dict[str, Any]:
        reg = self._get_registry()
        if not reg:
            return {"available": False}

        try:
            summary = reg.get_summary()
            return {
                "available": True,
                "total_calls": summary.total_calls,
                "total_tokens": summary.total_tokens,
                "success_rate": summary.success_rate,
                "avg_duration_ms": summary.avg_duration_ms,
                "top_providers": summary.top_providers,
                "top_sources": summary.top_sources,
                "recent_errors": summary.recent_errors,
            }
        except Exception as e:
            logger.error(f"LLM get_summary error: {e}")
            return {"available": True, "error": str(e)}

    def get_aggregated(self) -> Dict[str, Any]:
        reg = self._get_registry()
        if not reg:
            return {"available": False}

        try:
            stats = reg.get_aggregated()
            return {
                "available": True,
                "total_calls": stats.total_calls,
                "success_calls": stats.successful_calls,
                "failed_calls": stats.failed_calls,
                "total_tokens": stats.total_tokens,
                "avg_tokens_per_call": stats.avg_tokens_per_call,
                "avg_duration_ms": stats.avg_duration_ms,
                "by_provider": stats.calls_by_provider,
                "by_source": stats.calls_by_source,
            }
        except Exception as e:
            logger.error(f"LLM get_aggregated error: {e}")
            return {"available": True, "error": str(e)}

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        reg = self._get_registry()
        if not reg:
            return []

        limit = min(max(1, limit), 500)
        try:
            records = reg.get_recent(limit=limit)
            return [
                {
                    "id": r.record_id,
                    "provider_id": r.provider_id,
                    "source": r.source_module,
                    "success": r.success,
                    "tokens_used": r.tokens_used,
                    "duration_ms": r.duration_ms,
                    "timestamp": r.timestamp,
                    "error": r.error_message,
                    "is_multimodal": r.is_multimodal,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"LLM get_recent error: {e}")
            return []

    def get_by_provider(self, provider_id: str) -> Dict[str, Any]:
        reg = self._get_registry()
        if not reg:
            return {"available": False}
        try:
            return reg.get_by_provider(provider_id)
        except Exception as e:
            return {"error": str(e)}

    def get_by_source(self, source: str) -> Dict[str, Any]:
        reg = self._get_registry()
        if not reg:
            return {"available": False}
        try:
            return reg.get_by_source(source)
        except Exception as e:
            return {"error": str(e)}
