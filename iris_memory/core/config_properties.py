"""配置属性定义表 — 数据驱动的属性映射

``ConfigManager.__getattr__`` 在此表中查找属性名，
自动调用 ``self.get(config_key, default)`` 返回值。

带 ``normalize_provider=True`` 的条目额外调用 ``normalize_provider_id()``。
"""

from __future__ import annotations

from typing import Any, NamedTuple

from iris_memory.core.defaults import DEFAULTS


class _ConfigProp(NamedTuple):
    """单条配置属性定义"""
    config_key: str
    default: Any
    normalize_provider: bool = False


# attribute_name → _ConfigProp
CONFIG_PROPERTIES: dict[str, _ConfigProp] = {
    # ── 基础功能 ──
    "enable_memory": _ConfigProp("basic.enable_memory", True),
    "enable_inject": _ConfigProp("basic.enable_inject", True),
    "log_level": _ConfigProp("logging.log_level", "INFO"),
    "enable_activity_adaptive": _ConfigProp("activity_adaptive.enable", True),

    # ── 记忆设置 ──
    "max_context_memories": _ConfigProp("memory.max_context_memories", 3),
    "token_budget": _ConfigProp(
        "llm_integration.token_budget", DEFAULTS.llm_integration.token_budget
    ),
    "max_working_memory": _ConfigProp(
        "memory.max_working_memory", DEFAULTS.memory.max_working_memory
    ),
    "chat_context_count": _ConfigProp(
        "advanced.chat_context_count", DEFAULTS.llm_integration.chat_context_count
    ),
    "rif_threshold": _ConfigProp(
        "memory.rif_threshold", DEFAULTS.memory.rif_threshold
    ),
    "upgrade_mode": _ConfigProp("memory.upgrade_mode", "rule"),
    "use_llm": _ConfigProp("memory.use_llm", False),

    # ── 主动回复 ──
    "proactive_reply_enabled": _ConfigProp("proactive_reply.enable", False),
    "proactive_reply_max_daily": _ConfigProp(
        "proactive_reply.max_daily_replies",
        DEFAULTS.proactive_reply.max_daily_replies,
    ),
    "proactive_reply_group_whitelist_mode": _ConfigProp(
        "proactive_reply.group_whitelist_mode", False
    ),
    "smart_boost_window_seconds": _ConfigProp(
        "proactive_reply.smart_boost_window_seconds",
        DEFAULTS.proactive_reply.smart_boost_window_seconds,
    ),
    "smart_boost_score_multiplier": _ConfigProp(
        "proactive_reply.smart_boost_score_multiplier",
        DEFAULTS.proactive_reply.smart_boost_score_multiplier,
    ),
    "smart_boost_reply_threshold": _ConfigProp(
        "proactive_reply.smart_boost_reply_threshold",
        DEFAULTS.proactive_reply.smart_boost_reply_threshold,
    ),

    # ── 图片分析 ──
    "image_analysis_enabled": _ConfigProp("image_analysis.enable", True),
    "image_analysis_mode": _ConfigProp("image_analysis.mode", "auto"),
    "image_analysis_max_images": _ConfigProp(
        "image_analysis.max_images_per_message",
        DEFAULTS.image_analysis.max_images_per_message,
    ),
    "image_analysis_daily_budget": _ConfigProp(
        "image_analysis.daily_budget",
        DEFAULTS.image_analysis.daily_analysis_budget,
    ),
    "image_analysis_session_budget": _ConfigProp(
        "image_analysis.session_analysis_budget",
        DEFAULTS.image_analysis.session_analysis_budget,
    ),
    "image_analysis_require_context": _ConfigProp(
        "image_analysis.require_context_relevance",
        DEFAULTS.image_analysis.require_context_relevance,
    ),

    # ── 批量处理 ──
    "batch_threshold_count": _ConfigProp(
        "message_processing.batch_threshold_count",
        DEFAULTS.message_processing.batch_threshold_count,
    ),
    "short_message_threshold": _ConfigProp(
        "message_processing.short_message_threshold",
        DEFAULTS.message_processing.short_message_threshold,
    ),
    "merge_time_window": _ConfigProp(
        "message_processing.merge_time_window",
        DEFAULTS.message_processing.merge_time_window,
    ),
    "max_merge_count": _ConfigProp(
        "message_processing.max_merge_count",
        DEFAULTS.message_processing.max_merge_count,
    ),

    # ── 会话 ──
    "session_timeout": _ConfigProp(
        "session.session_timeout", DEFAULTS.session.session_timeout
    ),

    # ── 嵌入 ──
    "embedding_source": _ConfigProp(
        "embedding.source", DEFAULTS.embedding.source
    ),
    "embedding_astrbot_provider_id": _ConfigProp(
        "embedding.astrbot_provider_id",
        DEFAULTS.embedding.astrbot_provider_id,
        normalize_provider=True,
    ),
    "embedding_local_model": _ConfigProp(
        "embedding.local_model", DEFAULTS.embedding.local_model
    ),
    "embedding_local_dimension": _ConfigProp(
        "embedding.local_dimension", DEFAULTS.embedding.local_dimension
    ),

    # ── 画像 ──
    "persona_auto_update": _ConfigProp(
        "persona.enable_auto_update", DEFAULTS.persona.enable_auto_update
    ),
    "persona_injection_enabled": _ConfigProp(
        "persona.enable_persona_injection",
        DEFAULTS.persona.enable_persona_injection,
    ),
    "persona_max_change_log": _ConfigProp(
        "persona.max_change_log", DEFAULTS.persona.max_change_log
    ),
    "persona_snapshot_interval": _ConfigProp(
        "persona.snapshot_interval", DEFAULTS.persona.snapshot_interval
    ),
    "persona_extraction_mode": _ConfigProp("persona.extraction_mode", "rule"),
    "persona_enable_interest": _ConfigProp(
        "persona.enable_interest_extraction", True
    ),
    "persona_enable_style": _ConfigProp(
        "persona.enable_style_extraction", True
    ),
    "persona_enable_preference": _ConfigProp(
        "persona.enable_preference_extraction", True
    ),
    "persona_llm_max_tokens": _ConfigProp("persona.llm_max_tokens", 300),
    "persona_llm_daily_limit": _ConfigProp("persona.llm_daily_limit", 50),
    "persona_fallback_to_rule": _ConfigProp("persona.fallback_to_rule", True),
    "persona_batch_enabled": _ConfigProp(
        "persona.batch_enabled", DEFAULTS.persona.batch_enabled
    ),
    "persona_batch_threshold": _ConfigProp(
        "persona.batch_threshold", DEFAULTS.persona.batch_threshold
    ),
    "persona_batch_flush_interval": _ConfigProp(
        "persona.batch_flush_interval", DEFAULTS.persona.batch_flush_interval
    ),
    "persona_batch_max_size": _ConfigProp(
        "persona.batch_max_size", DEFAULTS.persona.batch_max_size
    ),

    # ── LLM 增强模式 ──
    "sensitivity_mode": _ConfigProp("llm_enhanced.sensitivity_mode", "rule"),
    "trigger_mode": _ConfigProp("llm_enhanced.trigger_mode", "rule"),
    "emotion_mode": _ConfigProp("llm_enhanced.emotion_mode", "rule"),
    "proactive_mode": _ConfigProp("llm_enhanced.proactive_mode", "rule"),
    "conflict_mode": _ConfigProp("llm_enhanced.conflict_mode", "rule"),
    "retrieval_mode": _ConfigProp("llm_enhanced.retrieval_mode", "rule"),

    # ── LLM 提供者 (normalize_provider) ──
    "llm_provider_id": _ConfigProp(
        "llm_providers.memory_provider_id", "", normalize_provider=True
    ),
    "llm_enhanced_provider_id": _ConfigProp(
        "llm_providers.enhanced_provider_id", "", normalize_provider=True
    ),
    "knowledge_graph_provider_id": _ConfigProp(
        "llm_providers.knowledge_graph_provider_id", "", normalize_provider=True
    ),
    "image_analysis_provider_id": _ConfigProp(
        "llm_providers.image_analysis_provider_id", "", normalize_provider=True
    ),

    # ── 人格隔离 ──
    "memory_query_by_persona": _ConfigProp(
        "persona_isolation.memory_query_by_persona", False
    ),
    "kg_query_by_persona": _ConfigProp(
        "persona_isolation.kg_query_by_persona", False
    ),

    # ── Web UI ──
    "web_ui_enabled": _ConfigProp(
        "web_ui.enable", DEFAULTS.web_ui.enable
    ),
    "web_ui_port": _ConfigProp("web_ui.port", DEFAULTS.web_ui.port),
    "web_ui_access_key": _ConfigProp(
        "web_ui.access_key", DEFAULTS.web_ui.access_key
    ),
    "web_ui_host": _ConfigProp("web_ui.host", DEFAULTS.web_ui.host),
}
