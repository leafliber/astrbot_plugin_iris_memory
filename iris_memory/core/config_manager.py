"""
配置管理器 — 向后兼容适配层

将所有配置读取委托给 ``iris_memory.config.ConfigStore``，
保留原有公共 API。

新代码推荐直接使用::

    from iris_memory.config import get_store
    store = get_store()
    val = store.get("basic.enable_memory")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from iris_memory.config.schema import ALIAS_MAP, SCHEMA
from iris_memory.config.store import ConfigStore
from iris_memory.config import get_store, init_store, reset_store
from iris_memory.core.activity_config import (
    ActivityAwareConfigProvider, GroupActivityTracker
)
from iris_memory.core.provider_utils import normalize_provider_id


class ConfigManager:
    """配置管理器（向后兼容适配层）

    内部委托给 ``ConfigStore``，保留旧版属性访问和群级自适应方法。
    """

    DEFAULT_CACHE_TTL: float = 10.0

    def __init__(
        self,
        user_config: Any = None,
        *,
        cache_ttl: Optional[float] = None,
        plugin_data_path: Optional[Path] = None,
    ):
        self._store: ConfigStore = init_store(
            user_config=user_config,
            plugin_data_path=plugin_data_path,
            cache_ttl=cache_ttl,
        )
        # 场景自适应组件（延迟初始化）
        self._activity_provider: Optional[ActivityAwareConfigProvider] = None

    # ── 底层 Store 访问 ──

    @property
    def store(self) -> ConfigStore:
        """获取底层 ConfigStore（供新代码直接使用）"""
        return self._store

    def set_user_config(self, config: Any) -> None:
        """替换用户配置并重新加载"""
        self._store.set_user_config(config)

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """清除缓存"""
        self._store.invalidate_cache(key)

    # ========== 核心读取 ==========

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（线程安全，带 TTL 缓存）"""
        return self._store.get(key, default)

    # ========== 属性访问（向后兼容）==========

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        # 委托给 ConfigStore 的别名查找
        key = ALIAS_MAP.get(name)
        if key is not None:
            value = self._store.get(key)
            field = SCHEMA.get(key)
            if field and field.normalize_provider:
                return normalize_provider_id(value)
            return value
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    # ========== 场景自适应配置 ==========

    def init_activity_provider(
        self,
        tracker: GroupActivityTracker,
        enabled: Optional[bool] = None,
    ) -> ActivityAwareConfigProvider:
        if enabled is None:
            enabled = self.get("activity_adaptive.enable", True)
        self._activity_provider = ActivityAwareConfigProvider(
            tracker=tracker,
            enabled=enabled,
        )
        return self._activity_provider

    @property
    def activity_provider(self) -> Optional[ActivityAwareConfigProvider]:
        return self._activity_provider

    def get_group_config(self, group_id: Optional[str], key: str) -> Any:
        """获取群级自适应配置值"""
        if self._activity_provider and self._activity_provider.enabled and group_id:
            return self._activity_provider.get_config(group_id, key)
        advanced_key = f"advanced.{key}"
        val = self._store.get(advanced_key)
        if val is not None:
            return val
        if self._activity_provider:
            return self._activity_provider._get_default(key)
        return None

    @property
    def enable_activity_adaptive(self) -> bool:
        return self.get("activity_adaptive.enable", True)

    # ========== 保留自定义逻辑的属性 ==========

    @property
    def proactive_mode(self) -> str:
        return self.get("proactive_reply.proactive_mode", "rule")

    @property
    def llm_enhanced_enabled(self) -> bool:
        modes = [
            self.sensitivity_mode,
            self.trigger_mode,
            self.emotion_mode,
            self.conflict_mode,
            self.retrieval_mode,
        ]
        return any(mode in ("llm", "hybrid") for mode in modes)

    @property
    def default_persona_id(self) -> str:
        return self.get("persona_isolation.default_persona_id", "default")

    @property
    def persona_id_max_length(self) -> int:
        return self.get("persona_isolation.persona_id_max_length", 64)

    @property
    def persona_llm_provider(self) -> str:
        provider_id = normalize_provider_id(
            self.get("llm_providers.persona_provider_id", "")
        )
        return provider_id or "default"

    # ========== 群级自适应快捷方法 ==========

    def _with_group_override(
        self, group_id: Optional[str], key: str, fallback_key: str, default: Any
    ) -> Any:
        val = self.get_group_config(group_id, key)
        return val if val is not None else self.get(fallback_key, default)

    def get_batch_threshold_count(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "batch_threshold_count",
            "message_processing.batch_threshold_count", 20,
        )

    def get_batch_threshold_interval(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "batch_threshold_interval",
            "message_processing.batch_threshold_interval", 300,
        )

    def get_chat_context_count(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "chat_context_count",
            "advanced.chat_context_count", 15,
        )

    def get_cooldown_seconds(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "cooldown_seconds",
            "proactive_reply.cooldown_seconds", 60,
        )

    def get_max_daily_replies(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "max_daily_replies",
            "proactive_reply.max_daily_replies", 20,
        )

    def get_daily_analysis_budget(self, group_id: Optional[str] = None) -> int:
        return self._with_group_override(
            group_id, "daily_analysis_budget",
            "image_analysis.daily_analysis_budget", 100,
        )

    def get_reply_temperature(self, group_id: Optional[str] = None) -> float:
        return self._with_group_override(
            group_id, "reply_temperature",
            "proactive_reply.reply_temperature", 0.7,
        )

    # ========== 人格 ID ==========

    def get_persona_id_for_storage(self, event_persona_id: Optional[str]) -> str:
        if event_persona_id and event_persona_id.strip():
            normalized = event_persona_id.strip()
            if len(normalized) > self.persona_id_max_length:
                normalized = normalized[:self.persona_id_max_length]
            return normalized
        return self.default_persona_id

    def get_persona_id_for_query(
        self, event_persona_id: Optional[str], module: str = "memory"
    ) -> Optional[str]:
        if module == "memory" and not self.memory_query_by_persona:
            return None
        if module == "knowledge_graph" and not self.kg_query_by_persona:
            return None
        if event_persona_id and event_persona_id.strip():
            normalized = event_persona_id.strip()
            if len(normalized) > self.persona_id_max_length:
                return normalized[:self.persona_id_max_length]
            return normalized
        return self.default_persona_id



