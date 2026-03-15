"""
LLM 统计注册表

提供全局单例，自动记录 LLM 调用，支持多维度查询。
"""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Any, Callable, Awaitable, Dict, List, Optional

from iris_memory.stats.models import (
    LLMCallRecord,
    LLMAggregatedStats,
    StatsQuery,
    StatsSummary,
)
from iris_memory.stats.store import StatsKVStore
from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import SOURCE_ALIASES

logger = get_logger("stats.registry")


class LLMStatsRegistry:
    """LLM 统计注册表
    
    全局单例，聚合所有 LLM 调用统计。
    
    Features:
    - 自动推断调用来源
    - 按 provider_id 和 source 分类统计
    - 支持多维度查询
    - 持久化到 AstrBot KV 存储
    
    Usage:
        # 获取单例
        registry = get_stats_registry()
        
        # 设置 KV 接口（插件初始化时）
        registry.set_kv_interface(get_kv_data, put_kv_data)
        await registry.initialize()
        
        # 记录调用（llm_helper 中自动调用）
        await registry.record_call(...)
        
        # 查询
        records = registry.query(StatsQuery(provider_id="openai-gpt-4"))
        aggregated = registry.get_aggregated()
    """
    
    _instance: Optional[LLMStatsRegistry] = None
    
    def __new__(cls) -> LLMStatsRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        self._store = StatsKVStore()
        self._records: List[LLMCallRecord] = []
        self._aggregated = LLMAggregatedStats()
        self._initialized = False
        self._save_pending = False
        self._last_save_time: float = 0
        self._save_interval: float = 5.0
        self._call_context: Dict[str, Any] = {}

    def set_kv_interface(
        self,
        get_kv_data: Callable[[str, Any], Awaitable[Any]],
        put_kv_data: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """设置 KV 接口
        
        Args:
            get_kv_data: AstrBot 的 get_kv_data 方法
            put_kv_data: AstrBot 的 put_kv_data 方法
        """
        self._store.set_kv_interface(get_kv_data, put_kv_data)

    def set_call_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """设置当前调用上下文
        
        用于在记录调用时关联会话信息。
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            group_id: 群组 ID
        """
        self._call_context = {
            "user_id": user_id,
            "session_id": session_id,
            "group_id": group_id,
        }

    def clear_call_context(self) -> None:
        """清除当前调用上下文"""
        self._call_context = {}

    async def initialize(self) -> None:
        """从 KV 加载已有数据"""
        if self._initialized:
            return
        
        if not self._store.is_ready():
            logger.debug("LLMStatsRegistry: KV store not ready, skipping load")
            self._initialized = True
            return
        
        try:
            aggregated_data = await self._store.load_aggregated()
            if aggregated_data:
                self._aggregated = LLMAggregatedStats.from_dict(aggregated_data)
            
            records_data = await self._store.load_records()
            self._records = [LLMCallRecord.from_dict(r) for r in records_data]
            
            self._initialized = True
            logger.info(
                f"LLMStatsRegistry initialized: "
                f"{self._aggregated.total_calls} total calls, "
                f"{len(self._records)} records loaded"
            )
        except Exception as e:
            logger.warning(f"LLMStatsRegistry: Failed to load from KV: {e}")
            self._initialized = True

    def _infer_source(self) -> tuple[str, str]:
        """从调用栈推断来源
        
        Returns:
            (source_alias, source_class)
        """
        stack = inspect.stack()
        
        for frame_info in stack[3:]:
            frame_locals = frame_info.frame.f_locals
            
            if 'self' in frame_locals:
                cls = frame_locals['self'].__class__
                full_name = f"{cls.__module__}.{cls.__name__}"
                alias = SOURCE_ALIASES.get(full_name, full_name.split('.')[-1])
                return alias, cls.__name__
            
            if 'cls' in frame_locals and isinstance(frame_locals['cls'], type):
                cls = frame_locals['cls']
                full_name = f"{cls.__module__}.{cls.__name__}"
                alias = SOURCE_ALIASES.get(full_name, full_name.split('.')[-1])
                return alias, cls.__name__
            
            frame = frame_info.frame
            func_name = frame.f_code.co_name
            module_name = frame.f_globals.get('__name__', '')
            
            if module_name.startswith('iris_memory.') and not module_name.startswith('iris_memory.stats'):
                if module_name == 'iris_memory.utils.llm_helper':
                    continue
                
                module_short = module_name.split('.')[-1]
                alias = SOURCE_ALIASES.get(f"{module_name}.{func_name}", module_short)
                return alias, func_name
        
        return "unknown", "unknown"

    async def record_call(
        self,
        provider_id: Optional[str],
        success: bool,
        tokens_used: int,
        duration_ms: float,
        prompt: str,
        response: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        group_id: Optional[str] = None,
        error: Optional[str] = None,
        is_multimodal: bool = False,
        image_count: int = 0,
        source_module: Optional[str] = None,
        source_class: Optional[str] = None,
    ) -> str:
        """记录一次 LLM 调用
        
        Args:
            provider_id: LLM 提供者 ID
            success: 是否成功
            tokens_used: Token 消耗
            duration_ms: 调用耗时（毫秒）
            prompt: 提示词
            response: 响应内容
            user_id: 用户 ID（可选，从上下文获取）
            session_id: 会话 ID（可选，从上下文获取）
            group_id: 群组 ID（可选，从上下文获取）
            error: 错误信息
            is_multimodal: 是否多模态调用
            image_count: 图片数量
            source_module: 来源模块（可选，由 llm_helper 预先捕获）
            source_class: 来源类名（可选，由 llm_helper 预先捕获）
            
        Returns:
            record_id: 记录唯一 ID
        """
        if source_module is None or source_class is None:
            source_module, source_class = self._infer_source()
        
        if user_id is None:
            user_id = self._call_context.get("user_id")
        if session_id is None:
            session_id = self._call_context.get("session_id")
        if group_id is None:
            group_id = self._call_context.get("group_id")
        
        record = LLMCallRecord(
            record_id=str(uuid.uuid4()),
            timestamp=time.time(),
            provider_id=provider_id or "unknown",
            source_module=source_module,
            source_class=source_class,
            success=success,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            error_message=error,
            user_id=user_id,
            session_id=session_id,
            group_id=group_id,
            prompt_preview=prompt[:100] if prompt else "",
            response_preview=response[:100] if response else "",
            is_multimodal=is_multimodal,
            image_count=image_count,
        )
        
        self._records.append(record)
        
        self._aggregated.total_calls += 1
        self._aggregated.total_tokens += tokens_used
        self._aggregated.total_duration_ms += duration_ms
        
        if success:
            self._aggregated.successful_calls += 1
        else:
            self._aggregated.failed_calls += 1
        
        pid = provider_id or "unknown"
        self._aggregated.calls_by_provider[pid] = self._aggregated.calls_by_provider.get(pid, 0) + 1
        self._aggregated.tokens_by_provider[pid] = self._aggregated.tokens_by_provider.get(pid, 0) + tokens_used
        self._aggregated.calls_by_source[source_module] = self._aggregated.calls_by_source.get(source_module, 0) + 1
        
        date_key = time.strftime("%Y-%m-%d")
        self._aggregated.calls_by_date[date_key] = self._aggregated.calls_by_date.get(date_key, 0) + 1
        
        if len(self._records) > self._store.MAX_RECORDS:
            self._records = self._records[-self._store.MAX_RECORDS:]
        
        await self._save()
        
        logger.debug(
            f"LLMStatsRegistry: Recorded call from {source_module}, "
            f"provider={pid}, tokens={tokens_used}, success={success}"
        )
        
        return record.record_id

    async def _save(self) -> None:
        """持久化到 KV（带节流）"""
        if not self._store.is_ready():
            return
        
        current_time = time.time()
        if current_time - self._last_save_time < self._save_interval:
            self._save_pending = True
            return
        
        await self._store.save_all(
            self._aggregated.to_dict(),
            [r.to_dict() for r in self._records],
        )
        self._last_save_time = current_time
        self._save_pending = False

    async def flush(self) -> None:
        """强制保存所有待保存的数据"""
        if self._store.is_ready():
            await self._store.save_all(
                self._aggregated.to_dict(),
                [r.to_dict() for r in self._records],
            )
            self._last_save_time = time.time()
            self._save_pending = False
            logger.debug("LLMStatsRegistry: Flushed to KV")

    def query(self, query: StatsQuery) -> List[LLMCallRecord]:
        """多维度查询记录
        
        Args:
            query: 查询条件
            
        Returns:
            匹配的记录列表
        """
        results: List[LLMCallRecord] = []
        
        for record in reversed(self._records):
            if query.provider_id and record.provider_id != query.provider_id:
                continue
            if query.source_module and record.source_module != query.source_module:
                continue
            if query.source_class and record.source_class != query.source_class:
                continue
            if query.user_id and record.user_id != query.user_id:
                continue
            if query.session_id and record.session_id != query.session_id:
                continue
            if query.group_id and record.group_id != query.group_id:
                continue
            if query.success is not None and record.success != query.success:
                continue
            if query.start_time and record.timestamp < query.start_time:
                continue
            if query.end_time and record.timestamp > query.end_time:
                continue
            
            results.append(record)
            
            if len(results) >= query.limit:
                break
        
        if query.offset > 0:
            results = results[query.offset:]
        
        return results

    def get_aggregated(self) -> LLMAggregatedStats:
        """获取聚合统计"""
        return self._aggregated

    def get_recent(self, limit: int = 100) -> List[LLMCallRecord]:
        """获取最近 N 条记录"""
        return self._records[-limit:]

    def get_by_provider(self, provider_id: str) -> Dict[str, Any]:
        """按 provider_id 获取统计
        
        Args:
            provider_id: 提供者 ID
            
        Returns:
            该 provider 的统计信息
        """
        calls = self._aggregated.calls_by_provider.get(provider_id, 0)
        tokens = self._aggregated.tokens_by_provider.get(provider_id, 0)
        
        records = [
            r for r in self._records
            if r.provider_id == provider_id
        ]
        
        return {
            "provider_id": provider_id,
            "total_calls": calls,
            "total_tokens": tokens,
            "recent_records": [r.to_dict() for r in records[-20:]],
        }

    def get_by_source(self, source: str) -> Dict[str, Any]:
        """按来源获取统计
        
        Args:
            source: 来源模块名
            
        Returns:
            该来源的统计信息
        """
        calls = self._aggregated.calls_by_source.get(source, 0)
        
        records = [
            r for r in self._records
            if r.source_module == source
        ]
        
        return {
            "source": source,
            "total_calls": calls,
            "recent_records": [r.to_dict() for r in records[-20:]],
        }

    def get_summary(self) -> StatsSummary:
        """获取统计摘要（用于 Dashboard）"""
        sorted_providers = sorted(
            self._aggregated.calls_by_provider.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        sorted_sources = sorted(
            self._aggregated.calls_by_source.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        
        recent_errors = sum(
            1 for r in self._records[-100:]
            if not r.success
        )
        
        return StatsSummary(
            total_calls=self._aggregated.total_calls,
            success_rate=self._aggregated.success_rate,
            total_tokens=self._aggregated.total_tokens,
            avg_duration_ms=self._aggregated.avg_duration_ms,
            top_providers=[
                {"provider_id": p, "calls": c}
                for p, c in sorted_providers
            ],
            top_sources=[
                {"source": s, "calls": c}
                for s, c in sorted_sources
            ],
            recent_errors=recent_errors,
        )

    async def reset(self) -> None:
        """重置所有统计"""
        self._aggregated.reset()
        self._records.clear()
        await self._store.clear_all()
        logger.info("LLMStatsRegistry: All stats reset")


def get_stats_registry() -> LLMStatsRegistry:
    """获取全局统计注册表"""
    return LLMStatsRegistry()
