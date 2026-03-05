"""
配置 Schema 定义 — 单一数据源 (Single Source of Truth)

所有配置项的类型、默认值、校验规则、描述和访问级别在此统一定义。
消除 defaults.py 与 Schema 之间的同步问题。

设计参考 Zod 的声明式 Schema 风格：
- ConfigField：单个配置项的完整描述
- SCHEMA：扁平化 key → ConfigField 映射
- 由 Schema 推导类型提示，保证全局类型安全

使用方式::

    from iris_memory.config.schema import SCHEMA, AccessLevel
    field = SCHEMA["basic.enable_memory"]
    print(field.default)       # True
    print(field.access)        # AccessLevel.READONLY
    print(field.value_type)    # <class 'bool'>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Type, Union


# ─── 访问级别 ────────────────────────────────────────────

class AccessLevel(Enum):
    """配置项访问级别

    READONLY  — Level 1：来自 _conf_schema.json / 用户 UI，插件侧只读
    WRITABLE  — Level 2：插件 data 持久化，WebUI 可读写
    INTERNAL  — 内部参数，不暴露给 WebUI
    """
    READONLY = "readonly"
    WRITABLE = "writable"
    INTERNAL = "internal"


# ─── ConfigField ─────────────────────────────────────────

@dataclass(frozen=True)
class ConfigField:
    """单个配置项的完整定义

    Attributes:
        key:         扁平化配置键，如 ``"basic.enable_memory"``
        value_type:  Python 类型 (bool / int / float / str / list / dict)
        default:     内置默认值
        description: 配置项说明
        access:      访问级别
        section:     所属业务域 / 模块名
        choices:     可选值枚举（仅 str 类型适用）
        min_val:     最小值（int / float）
        max_val:     最大值（int / float）
        validator:   自定义校验函数 ``(value) -> value``，非法时抛 ValueError
        attr_alias:  在 ConfigManager 上的属性别名（向后兼容）
        normalize_provider: 是否对值调用 ``normalize_provider_id``
    """
    key: str
    value_type: Type = object
    default: Any = None
    description: str = ""
    access: AccessLevel = AccessLevel.WRITABLE
    section: str = ""
    choices: Optional[Tuple] = None
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    validator: Optional[Callable] = field(default=None, repr=False, compare=False)
    attr_alias: Optional[str] = None
    normalize_provider: bool = False


# ─── 工具函数 ────────────────────────────────────────────

def _f(
    key: str,
    tp: Type,
    default: Any,
    desc: str = "",
    *,
    access: AccessLevel = AccessLevel.WRITABLE,
    section: str = "",
    choices: Optional[Tuple] = None,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    validator: Optional[Callable] = None,
    alias: Optional[str] = None,
    normalize_provider: bool = False,
) -> ConfigField:
    """快捷构造 ConfigField"""
    if not section:
        section = key.split(".")[0] if "." in key else ""
    return ConfigField(
        key=key,
        value_type=tp,
        default=default,
        description=desc,
        access=access,
        section=section,
        choices=choices,
        min_val=min_val,
        max_val=max_val,
        validator=validator,
        attr_alias=alias,
        normalize_provider=normalize_provider,
    )


RO = AccessLevel.READONLY   # 快捷别名
RW = AccessLevel.WRITABLE
INT = AccessLevel.INTERNAL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  全量 Schema 定义
#  按业务域分段，key 格式：section.attr
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FIELDS: List[ConfigField] = [
    # ── 基础功能 ───────────────────────────────────────
    _f("basic.enable_memory", bool, True,
       "启用记忆功能", access=RO, alias="enable_memory"),
    _f("basic.enable_inject", bool, True,
       "自动注入记忆到对话", access=RO, alias="enable_inject"),

    # ── 日志 ──────────────────────────────────────────
    _f("logging.log_level", str, "INFO",
       "日志级别", access=RO,
       choices=("DEBUG", "INFO", "WARNING", "ERROR"),
       alias="log_level"),

    # ── 嵌入向量 ──────────────────────────────────────
    _f("embedding.source", str, "auto",
       "嵌入源选择 (auto/astrbot/local)", access=RO,
       choices=("auto", "astrbot", "local"),
       alias="embedding_source"),
    _f("embedding.astrbot_provider_id", str, "",
       "AstrBot embedding provider ID", access=RO,
       alias="embedding_astrbot_provider_id", normalize_provider=True),
    _f("embedding.local_model", str, "BAAI/bge-small-zh-v1.5",
       "本地嵌入模型名称", access=RO,
       alias="embedding_local_model"),
    _f("embedding.local_dimension", int, 512,
       "本地嵌入模型维度", access=RO,
       alias="embedding_local_dimension"),
    _f("embedding.reimport_on_dimension_conflict", bool, True,
       "维度冲突时重新导入原记忆", access=RO),
    # 内部嵌入参数
    _f("embedding.collection_name", str, "iris_memory",
       "ChromaDB 集合名称", access=INT),
    _f("embedding.auto_detect_dimension", bool, True,
       "自动检测维度", access=INT),

    # ── LLM 提供者 ───────────────────────────────────
    _f("llm_providers.default_provider_id", str, "",
       "默认 LLM 提供者 ID", access=RO,
       alias="default_provider_id", normalize_provider=True),
    _f("llm_providers.memory_provider_id", str, "",
       "记忆功能 LLM 提供者", access=RO,
       alias="llm_provider_id", normalize_provider=True),
    _f("llm_providers.persona_provider_id", str, "",
       "用户画像 LLM 提供者", access=RO,
       alias="persona_llm_provider", normalize_provider=True),
    _f("llm_providers.knowledge_graph_provider_id", str, "",
       "知识图谱 LLM 提供者", access=RO,
       alias="knowledge_graph_provider_id", normalize_provider=True),
    _f("llm_providers.image_analysis_provider_id", str, "",
       "图片分析 LLM 提供者", access=RO,
       alias="image_analysis_provider_id", normalize_provider=True),
    _f("llm_providers.enhanced_provider_id", str, "",
       "智能增强 LLM 提供者", access=RO,
       alias="llm_enhanced_provider_id", normalize_provider=True),

    # ── 记忆核心设置 ─────────────────────────────────
    _f("memory.max_context_memories", int, 10,
       "最大上下文记忆数", access=RO, min_val=1, max_val=100,
       alias="max_context_memories"),
    _f("memory.max_working_memory", int, 10,
       "最大工作记忆数", access=RO, min_val=1, max_val=100,
       alias="max_working_memory"),
    _f("memory.upgrade_mode", str, "rule",
       "记忆升级模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="upgrade_mode"),
    _f("memory.use_llm", bool, False,
       "是否使用 LLM 处理消息", access=RO,
       alias="use_llm"),
    # 内部记忆参数
    _f("memory.rif_threshold", float, 0.4,
       "RIF 评分阈值", access=INT, min_val=0.0, max_val=1.0,
       alias="rif_threshold"),
    _f("memory.llm_upgrade_batch_size", int, 5,
       "LLM 升级批量大小", access=INT),
    _f("memory.llm_upgrade_threshold", float, 0.7,
       "LLM 升级阈值", access=INT),
    _f("memory.min_confidence", float, 0.3,
       "记忆捕获最低置信度", access=INT),
    _f("memory.enable_duplicate_check", bool, True,
       "启用重复检测", access=INT),
    _f("memory.enable_conflict_check", bool, True,
       "启用冲突检测", access=INT),
    _f("memory.enable_entity_extraction", bool, True,
       "启用实体提取", access=INT),

    # ── 用户画像 ──────────────────────────────────────
    _f("persona.enabled", bool, True,
       "启用用户画像功能", access=RO),
    _f("persona.extraction_mode", str, "rule",
       "画像提取模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="persona_extraction_mode"),
    _f("persona.enable_interest_extraction", bool, True,
       "启用兴趣提取", access=RW,
       alias="persona_enable_interest"),
    _f("persona.enable_style_extraction", bool, True,
       "启用风格提取", access=RW,
       alias="persona_enable_style"),
    _f("persona.enable_preference_extraction", bool, True,
       "启用偏好提取", access=RW,
       alias="persona_enable_preference"),
    _f("persona.llm_max_tokens", int, 300,
       "画像提取 LLM 最大 token", access=RW,
       alias="persona_llm_max_tokens"),
    _f("persona.llm_daily_limit", int, 50,
       "画像提取 LLM 每日限制", access=RW,
       alias="persona_llm_daily_limit"),
    _f("persona.fallback_to_rule", bool, True,
       "LLM 失败时回退到规则", access=RW,
       alias="persona_fallback_to_rule"),
    _f("persona.enable_auto_update", bool, True,
       "自动更新画像", access=INT,
       alias="persona_auto_update"),
    _f("persona.max_change_log", int, 200,
       "最大变更日志条数", access=INT,
       alias="persona_max_change_log"),
    _f("persona.snapshot_interval", int, 10,
       "快照间隔", access=INT,
       alias="persona_snapshot_interval"),
    _f("persona.enable_persona_injection", bool, True,
       "启用画像注入", access=INT,
       alias="persona_injection_enabled"),
    _f("persona.batch_enabled", bool, True,
       "批量画像提取", access=INT,
       alias="persona_batch_enabled"),
    _f("persona.batch_threshold", int, 20,
       "批量处理阈值", access=INT,
       alias="persona_batch_threshold"),
    _f("persona.batch_flush_interval", int, 21600,
       "定时刷新间隔（秒）", access=INT,
       alias="persona_batch_flush_interval"),
    _f("persona.batch_max_size", int, 20,
       "单次批量最大消息数", access=INT,
       alias="persona_batch_max_size"),

    # ── 知识图谱 ──────────────────────────────────────
    _f("knowledge_graph.enabled", bool, True,
       "启用知识图谱", access=RO),
    _f("knowledge_graph.extraction_mode", str, "rule",
       "三元组提取模式", access=RO,
       choices=("rule", "llm", "hybrid")),
    _f("knowledge_graph.max_depth", int, 3,
       "多跳推理最大跳数", access=RO, min_val=1, max_val=10),
    _f("knowledge_graph.max_nodes_per_hop", int, 10,
       "每跳最大节点数", access=RO, min_val=1, max_val=50),
    _f("knowledge_graph.max_facts", int, 8,
       "注入 LLM 上下文的最大事实数", access=RO, min_val=1, max_val=30),
    _f("knowledge_graph.min_confidence", float, 0.2,
       "最低置信度阈值", access=RO, min_val=0.0, max_val=1.0),
    # 内部知识图谱参数
    _f("knowledge_graph.maintenance_interval", int, 86400,
       "维护任务间隔（秒）", access=INT),
    _f("knowledge_graph.auto_maintenance", bool, True,
       "自动维护", access=INT),
    _f("knowledge_graph.auto_cleanup_orphans", bool, True,
       "自动清理孤立节点", access=INT),
    _f("knowledge_graph.auto_cleanup_low_confidence", bool, True,
       "自动清理低置信度边", access=INT),
    _f("knowledge_graph.low_confidence_threshold", float, 0.2,
       "低置信度阈值", access=INT),
    _f("knowledge_graph.staleness_days", int, 30,
       "过期天数", access=INT),

    # ── LLM 智能增强 ─────────────────────────────────
    _f("llm_enhanced.sensitivity_mode", str, "rule",
       "敏感度检测模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="sensitivity_mode"),
    _f("llm_enhanced.trigger_mode", str, "rule",
       "触发检测模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="trigger_mode"),
    _f("llm_enhanced.emotion_mode", str, "rule",
       "情感分析模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="emotion_mode"),
    _f("llm_enhanced.proactive_mode", str, "rule",
       "主动回复 L3 模式", access=RO,
       choices=("rule", "hybrid"),
       alias="proactive_mode"),
    _f("llm_enhanced.conflict_mode", str, "rule",
       "冲突检测模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="conflict_mode"),
    _f("llm_enhanced.retrieval_mode", str, "rule",
       "检索路由模式", access=RO,
       choices=("rule", "llm", "hybrid"),
       alias="retrieval_mode"),
    # 内部 LLM 增强参数
    _f("llm_enhanced.sensitivity_confidence_threshold", float, 0.7,
       "敏感度置信阈值", access=INT),
    _f("llm_enhanced.trigger_daily_limit", int, 200,
       "触发每日限制", access=INT),
    _f("llm_enhanced.emotion_llm_weight", float, 0.4,
       "情感 LLM 权重", access=INT),
    _f("llm_enhanced.emotion_enable_context_aware", bool, True,
       "情感上下文感知", access=INT),
    _f("llm_enhanced.proactive_daily_limit", int, 100,
       "主动回复每日限制", access=INT),

    # ── 图片分析 ──────────────────────────────────────
    _f("image_analysis.enable", bool, True,
       "启用图片分析", access=RO,
       alias="image_analysis_enabled"),
    _f("image_analysis.mode", str, "auto",
       "图片分析模式", access=RO,
       choices=("auto", "brief", "detailed", "skip"),
       alias="image_analysis_mode"),
    _f("image_analysis.daily_budget", int, 100,
       "每日图片分析预算", access=RO, min_val=0,
       alias="image_analysis_daily_budget"),
    # 内部图片分析参数
    _f("image_analysis.max_images_per_message", int, 2,
       "每条消息最大图片数", access=INT,
       alias="image_analysis_max_images"),
    _f("image_analysis.skip_sticker", bool, True,
       "跳过表情包", access=INT),
    _f("image_analysis.analysis_cooldown", float, 3.0,
       "分析冷却（秒）", access=INT),
    _f("image_analysis.cache_ttl", int, 3600,
       "缓存 TTL（秒）", access=INT),
    _f("image_analysis.max_cache_size", int, 200,
       "最大缓存条数", access=INT),
    _f("image_analysis.brief_token_cost", int, 100,
       "简要分析 token 消耗", access=INT),
    _f("image_analysis.detailed_token_cost", int, 300,
       "详细分析 token 消耗", access=INT),
    _f("image_analysis.daily_analysis_budget", int, 100,
       "每日分析预算（内部）", access=INT),
    _f("image_analysis.session_analysis_budget", int, 20,
       "每会话分析预算", access=INT,
       alias="image_analysis_session_budget"),
    _f("image_analysis.similar_image_window", int, 60,
       "相似图片检测窗口（秒）", access=INT),
    _f("image_analysis.recent_image_limit", int, 20,
       "最近图片哈希数量", access=INT),
    _f("image_analysis.require_context_relevance", bool, True,
       "要求上下文相关性", access=INT,
       alias="image_analysis_require_context"),

    # ── 主动回复 ──────────────────────────────────────
    _f("proactive_reply.enable", bool, False,
       "启用主动回复", access=RO,
       alias="proactive_reply_enabled"),
    _f("proactive_reply.followup_after_all_replies", bool, False,
       "Bot 回复后继续跟进", access=RO,
       alias="proactive_followup_after_all_replies"),
    _f("proactive_reply.group_whitelist_mode", bool, False,
       "群聊白名单模式", access=RO,
       alias="proactive_reply_group_whitelist_mode"),
    _f("proactive_reply.followup_window_seconds", int, 150,
       "跟进窗口时长（秒）", access=RO,
       alias="proactive_followup_window_seconds"),
    _f("proactive_reply.max_followup_count", int, 3,
       "最大跟进次数", access=RO,
       alias="proactive_max_followup_count"),
    # Level 2（WebUI 可调）
    _f("proactive_reply.cooldown_seconds", int, 60,
       "主动回复冷却秒数", access=RW),
    _f("proactive_reply.max_daily_replies", int, 20,
       "每日最大主动回复次数", access=RW,
       alias="proactive_reply_max_daily"),
    _f("proactive_reply.max_reply_tokens", int, 150,
       "最大回复 token", access=RW),
    _f("proactive_reply.reply_temperature", float, 0.7,
       "回复温度", access=RW),
    _f("proactive_reply.group_whitelist", list, [],
       "群聊白名单", access=RW),
    _f("proactive_reply.quiet_hours", list, [23, 7],
       "静音时段 [start, end]", access=RW,
       alias="proactive_reply_quiet_hours"),
    _f("proactive_reply.max_daily_per_user", int, 5,
       "每用户每日最大主动回复", access=RW,
       alias="proactive_reply_max_daily_per_user"),
    _f("proactive_reply.web_dashboard", bool, False,
       "Web 仪表盘", access=RW,
       alias="proactive_reply_web_dashboard"),
    # 内部主动回复参数
    _f("proactive_reply.signal_check_interval_seconds", int, 30,
       "群定时器检查间隔", access=INT),
    _f("proactive_reply.signal_silence_timeout_seconds", int, 600,
       "沉默超时", access=INT),
    _f("proactive_reply.signal_min_silence_seconds", int, 60,
       "最小沉默时间", access=INT),
    _f("proactive_reply.signal_ttl_emotion_high", int, 180,
       "emotion_high 信号 TTL", access=INT),
    _f("proactive_reply.signal_ttl_rule_match", int, 300,
       "rule_match 信号 TTL", access=INT),
    _f("proactive_reply.signal_weight_direct_reply", float, 0.8,
       "直接回复权重阈值", access=INT),
    _f("proactive_reply.signal_weight_llm_confirm", float, 0.5,
       "LLM 确认权重阈值", access=INT),
    _f("proactive_reply.followup_short_window_seconds", int, 10,
       "短期窗口（秒）", access=INT),
    _f("proactive_reply.followup_llm_max_tokens", int, 1000,
       "跟进 LLM 最大 token", access=INT),
    _f("proactive_reply.followup_llm_temperature", float, 0.3,
       "跟进 LLM 温度", access=INT),
    _f("proactive_reply.followup_fallback_to_rule", bool, True,
       "跟进 LLM 失败降级到规则", access=INT),

    # ── 场景自适应 ────────────────────────────────────
    _f("activity_adaptive.enable", bool, True,
       "启用场景自适应", access=RO,
       alias="enable_activity_adaptive"),
    _f("activity_adaptive.activity_calc_interval", int, 3600,
       "活跃度计算间隔（秒）", access=INT),
    _f("activity_adaptive.config_cache_ttl", int, 300,
       "配置缓存 TTL（秒）", access=INT),

    # ── 高级参数（群聊自适应覆盖）─────────────────────
    _f("advanced.chat_context_count", int, 15,
       "聊天上下文条数", access=RW,
       alias="chat_context_count"),
    _f("advanced.cooldown_seconds", int, 60,
       "主动回复冷却秒数", access=RW),
    _f("advanced.max_daily_replies", int, 20,
       "每日最大主动回复次数", access=RW),
    _f("advanced.reply_temperature", float, 0.7,
       "主动回复温度", access=RW),
    _f("advanced.batch_threshold_count", int, 20,
       "批量处理消息阈值", access=RW),
    _f("advanced.batch_threshold_interval", int, 300,
       "批量处理间隔（秒）", access=RW),
    _f("advanced.daily_analysis_budget", int, 100,
       "每日图片分析预算", access=RW),

    # ── 人格隔离 ──────────────────────────────────────
    _f("persona_isolation.memory_query_by_persona", bool, False,
       "记忆按人格隔离查询", access=RO,
       alias="memory_query_by_persona"),
    _f("persona_isolation.kg_query_by_persona", bool, False,
       "知识图谱按人格隔离查询", access=RO,
       alias="kg_query_by_persona"),
    _f("persona_isolation.default_persona_id", str, "default",
       "默认人格 ID", access=INT),
    _f("persona_isolation.persona_id_max_length", int, 64,
       "persona_id 最大长度", access=INT),

    # ── 错误友好化 ────────────────────────────────────
    _f("error_friendly.enable", bool, True,
       "启用错误消息友好化", access=RO,
       alias="error_friendly_enabled"),

    # ── Markdown 去除器 ───────────────────────────────
    _f("markdown_stripper.enable", bool, True,
       "启用 Markdown 去除功能", access=RO,
       alias="markdown_stripper_enabled"),
    _f("markdown_stripper.preserve_code_blocks", bool, False,
       "保留代码块格式", access=INT),
    _f("markdown_stripper.preserve_links", bool, False,
       "保留链接格式", access=INT),
    _f("markdown_stripper.threshold_offset", int, 0,
       "阈值偏移量", access=INT),
    _f("markdown_stripper.strip_headers", bool, True,
       "去除标题标记", access=INT),
    _f("markdown_stripper.strip_lists", bool, True,
       "去除列表标记", access=INT),

    # ── Web UI ────────────────────────────────────────
    _f("web_ui.enable", bool, False,
       "启用 Web 管理界面", access=RO,
       alias="web_ui_enabled"),
    _f("web_ui.port", int, 8089,
       "Web 服务端口", access=RO, min_val=1024, max_val=65535,
       alias="web_ui_port"),
    _f("web_ui.access_key", str, "",
       "访问密钥", access=RO,
       alias="web_ui_access_key"),
    _f("web_ui.host", str, "127.0.0.1",
       "监听地址", access=RO,
       alias="web_ui_host"),

    # ── 会话管理 ──────────────────────────────────────
    _f("session.session_timeout", int, 86400,
       "会话超时（秒）", access=INT,
       alias="session_timeout"),
    _f("session.session_inactive_timeout", int, 1800,
       "会话不活跃超时（秒）", access=INT),
    _f("session.session_cleanup_interval", int, 3600,
       "会话清理间隔（秒）", access=INT),
    _f("session.max_sessions", int, 100,
       "最大会话数", access=INT),
    _f("session.promotion_interval", int, 3600,
       "记忆升级检查间隔（秒）", access=INT),

    # ── 缓存 ─────────────────────────────────────────
    _f("cache.embedding_cache_size", int, 1000,
       "嵌入缓存大小", access=INT),
    _f("cache.embedding_cache_strategy", str, "lru",
       "嵌入缓存策略", access=INT),
    _f("cache.working_cache_ttl", int, 86400,
       "工作记忆缓存 TTL（秒）", access=INT),
    _f("cache.compression_max_length", int, 200,
       "压缩最大长度", access=INT),

    # ── LLM 集成 ─────────────────────────────────────
    _f("llm_integration.token_budget", int, 512,
       "Token 预算", access=INT,
       alias="token_budget"),
    _f("llm_integration.chat_context_count", int, 15,
       "聊天上下文条数", access=INT),
    _f("llm_integration.injection_mode", str, "suffix",
       "注入模式", access=INT),
    _f("llm_integration.coordination_strategy", str, "hybrid",
       "协调策略", access=INT),
    _f("llm_integration.enable_time_aware", bool, True,
       "时间感知", access=INT),
    _f("llm_integration.enable_emotion_aware", bool, True,
       "情感感知", access=INT),
    _f("llm_integration.enable_token_budget", bool, True,
       "Token 预算控制", access=INT),
    _f("llm_integration.enable_routing", bool, True,
       "路由", access=INT),
    _f("llm_integration.enable_working_memory_merge", bool, True,
       "工作记忆合并", access=INT),

    # ── 消息处理 ──────────────────────────────────────
    _f("message_processing.batch_threshold_count", int, 20,
       "批量处理消息阈值", access=INT,
       alias="batch_threshold_count"),
    _f("message_processing.batch_threshold_interval", int, 300,
       "批量处理间隔（秒）", access=INT),
    _f("message_processing.batch_processing_mode", str, "hybrid",
       "批量处理模式", access=INT),
    _f("message_processing.llm_max_tokens_for_summary", int, 200,
       "LLM 摘要最大 token", access=INT),
    _f("message_processing.short_message_threshold", int, 15,
       "短消息长度阈值（字符）", access=INT,
       alias="short_message_threshold"),
    _f("message_processing.merge_time_window", int, 60,
       "合并时间窗口（秒）", access=INT,
       alias="merge_time_window"),
    _f("message_processing.max_merge_count", int, 5,
       "最大合并消息数", access=INT,
       alias="max_merge_count"),
    _f("message_processing.immediate_trigger_confidence", float, 0.7,
       "即时触发置信度", access=INT),
    _f("message_processing.immediate_emotion_intensity", float, 0.6,
       "即时情感强度", access=INT),
    _f("message_processing.llm_processing_mode", str, "hybrid",
       "LLM 处理模式", access=INT),

    # ── 日志（内部）────────────────────────────────────
    _f("log.console_output", bool, True, "控制台输出", access=INT),
    _f("log.file_output", bool, True, "文件输出", access=INT),
    _f("log.max_file_size", int, 10, "最大文件大小（MB）", access=INT),
    _f("log.backup_count", int, 5, "备份数量", access=INT),

    # ── 语义提取（通道 B）────────────────────────────
    _f("semantic_extraction.enabled", bool, True,
       "启用语义提取", access=INT),
    _f("semantic_extraction.extraction_interval", int, 86400,
       "执行间隔（秒）", access=INT),
    _f("semantic_extraction.min_confidence", float, 0.4,
       "最低置信度", access=INT),
    _f("semantic_extraction.min_age_days", int, 30,
       "最小年龄（天）", access=INT),
    _f("semantic_extraction.min_cluster_size", int, 3,
       "最少出现次数", access=INT),
    _f("semantic_extraction.cluster_time_window_days", int, 90,
       "聚类时间窗口（天）", access=INT),
    _f("semantic_extraction.similarity_threshold", float, 0.75,
       "向量相似度阈值", access=INT),
    _f("semantic_extraction.max_clusters_per_run", int, 20,
       "单次最大聚类数", access=INT),
    _f("semantic_extraction.max_memories_per_cluster", int, 15,
       "单聚类最大记忆数", access=INT),
    _f("semantic_extraction.source_expiry_days", int, 0,
       "源记忆过期天数", access=INT),
    _f("semantic_extraction.llm_max_tokens", int, 500,
       "LLM 最大 token", access=INT),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  导出：SCHEMA (key → ConfigField), ALIAS_MAP (alias → key)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMA: Dict[str, ConfigField] = {f.key: f for f in _FIELDS}
"""全局配置 Schema：key → ConfigField"""

ALIAS_MAP: Dict[str, str] = {
    f.attr_alias: f.key for f in _FIELDS if f.attr_alias
}
"""属性别名 → 配置 key 的映射（向后兼容 ConfigManager.__getattr__）"""

SECTIONS: FrozenSet[str] = frozenset(f.section for f in _FIELDS if f.section)
"""所有配置 section 名称"""


def get_defaults_dict() -> Dict[str, Any]:
    """返回所有默认值的扁平字典 ``{key: default}``"""
    return {f.key: f.default for f in _FIELDS}


def get_section_defaults(section: str) -> Dict[str, Any]:
    """返回指定 section 的默认值"""
    return {
        f.key.split(".", 1)[1]: f.default
        for f in _FIELDS
        if f.section == section
    }
