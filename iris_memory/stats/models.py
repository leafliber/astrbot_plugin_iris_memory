"""
统计模块数据模型

定义 LLM 调用记录、聚合统计、查询条件等数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class LLMCallRecord:
    """单次 LLM 调用记录"""
    
    record_id: str
    timestamp: float
    provider_id: str
    source_module: str
    source_class: str
    success: bool
    tokens_used: int
    duration_ms: float
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    group_id: Optional[str] = None
    prompt_preview: str = ""
    response_preview: str = ""
    is_multimodal: bool = False
    image_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMCallRecord:
        return cls(
            record_id=data.get("record_id", ""),
            timestamp=data.get("timestamp", 0.0),
            provider_id=data.get("provider_id", "unknown"),
            source_module=data.get("source_module", "unknown"),
            source_class=data.get("source_class", "unknown"),
            success=data.get("success", False),
            tokens_used=data.get("tokens_used", 0),
            duration_ms=data.get("duration_ms", 0.0),
            error_message=data.get("error_message"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            group_id=data.get("group_id"),
            prompt_preview=data.get("prompt_preview", ""),
            response_preview=data.get("response_preview", ""),
            is_multimodal=data.get("is_multimodal", False),
            image_count=data.get("image_count", 0),
        )


@dataclass
class LLMAggregatedStats:
    """LLM 聚合统计"""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    
    calls_by_provider: Dict[str, int] = field(default_factory=dict)
    tokens_by_provider: Dict[str, int] = field(default_factory=dict)
    calls_by_source: Dict[str, int] = field(default_factory=dict)
    calls_by_date: Dict[str, int] = field(default_factory=dict)
    
    rate_limit_rejections: int = 0
    circuit_breaker_rejections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "calls_by_provider": dict(self.calls_by_provider),
            "tokens_by_provider": dict(self.tokens_by_provider),
            "calls_by_source": dict(self.calls_by_source),
            "calls_by_date": dict(self.calls_by_date),
            "rate_limit_rejections": self.rate_limit_rejections,
            "circuit_breaker_rejections": self.circuit_breaker_rejections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LLMAggregatedStats:
        return cls(
            total_calls=data.get("total_calls", 0),
            successful_calls=data.get("successful_calls", 0),
            failed_calls=data.get("failed_calls", 0),
            total_tokens=data.get("total_tokens", 0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            calls_by_provider=data.get("calls_by_provider", {}),
            tokens_by_provider=data.get("tokens_by_provider", {}),
            calls_by_source=data.get("calls_by_source", {}),
            calls_by_date=data.get("calls_by_date", {}),
            rate_limit_rejections=data.get("rate_limit_rejections", 0),
            circuit_breaker_rejections=data.get("circuit_breaker_rejections", 0),
        )

    def reset(self) -> None:
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_tokens = 0
        self.total_duration_ms = 0.0
        self.calls_by_provider.clear()
        self.tokens_by_provider.clear()
        self.calls_by_source.clear()
        self.calls_by_date.clear()
        self.rate_limit_rejections = 0
        self.circuit_breaker_rejections = 0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def avg_duration_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    @property
    def avg_tokens_per_call(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_tokens / self.total_calls


@dataclass
class StatsQuery:
    """统计查询条件"""
    
    provider_id: Optional[str] = None
    source_module: Optional[str] = None
    source_class: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    group_id: Optional[str] = None
    success: Optional[bool] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    limit: int = 100
    offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StatsQuery:
        return cls(
            provider_id=data.get("provider_id"),
            source_module=data.get("source_module"),
            source_class=data.get("source_class"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            group_id=data.get("group_id"),
            success=data.get("success"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            limit=data.get("limit", 100),
            offset=data.get("offset", 0),
        )


@dataclass
class StatsSummary:
    """统计摘要（用于 Dashboard）"""
    
    total_calls: int = 0
    success_rate: float = 0.0
    total_tokens: int = 0
    avg_duration_ms: float = 0.0
    top_providers: List[Dict[str, Any]] = field(default_factory=list)
    top_sources: List[Dict[str, Any]] = field(default_factory=list)
    recent_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens,
            "avg_duration_ms": self.avg_duration_ms,
            "top_providers": self.top_providers,
            "top_sources": self.top_sources,
            "recent_errors": self.recent_errors,
        }
