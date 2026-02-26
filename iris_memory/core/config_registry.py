"""
统一配置注册表

将 config_manager.CONFIG_KEY_MAPPING 和 defaults.py 中分散的默认值
合并为单一数据源 (Single Source of Truth)。

使用方式::

    from iris_memory.core.config_registry import CONFIG_REGISTRY
    defn = CONFIG_REGISTRY["basic.enable_memory"]
    print(defn.default)   # True
    print(defn.section)   # "memory"
    print(defn.attr)      # "auto_capture"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type


@dataclass(frozen=True)
class ConfigDefinition:
    """单个配置项的完整定义

    Attributes:
        key: 用户可见的简化配置键，如 ``"basic.enable_memory"``
        section: 对应 ``defaults.py`` 中 ``AllDefaults`` 的属性名，如 ``"memory"``
        attr: 对应 section dataclass 中的属性名，如 ``"auto_capture"``
        default: 内置默认值
        description: 配置项说明（中文）
        value_type: 值类型，仅用于文档和前端校验提示
    """

    key: str
    section: str
    attr: str
    default: Any
    description: str = ""
    value_type: Type = field(default=object, repr=False)


def _build_registry() -> Dict[str, ConfigDefinition]:
    """构建配置注册表

    集中定义所有配置项，消除 ``CONFIG_KEY_MAPPING`` 和
    ``defaults.py`` 之间需要手动同步的问题。
    """
    entries = [
        # ── 基础功能 ──
        ConfigDefinition(
            key="basic.enable_memory",
            section="memory",
            attr="auto_capture",
            default=True,
            description="是否启用记忆功能",
            value_type=bool,
        ),
        ConfigDefinition(
            key="basic.enable_inject",
            section="llm_integration",
            attr="enable_inject",
            default=True,
            description="是否启用记忆注入",
            value_type=bool,
        ),
        ConfigDefinition(
            key="logging.log_level",
            section="log",
            attr="level",
            default="INFO",
            description="日志等级",
            value_type=str,
        ),

        # ── 记忆设置 ──
        ConfigDefinition(
            key="memory.max_context_memories",
            section="llm_integration",
            attr="max_context_memories",
            default=3,
            description="最大上下文记忆数",
            value_type=int,
        ),
        ConfigDefinition(
            key="memory.max_working_memory",
            section="memory",
            attr="max_working_memory",
            default=10,
            description="最大工作记忆数",
            value_type=int,
        ),
        ConfigDefinition(
            key="memory.upgrade_mode",
            section="memory",
            attr="upgrade_mode",
            default="rule",
            description="记忆升级模式",
            value_type=str,
        ),

        # ── LLM增强处理 ──
        ConfigDefinition(
            key="memory.use_llm",
            section="message_processing",
            attr="use_llm_for_processing",
            default=False,
            description="是否使用LLM处理消息",
            value_type=bool,
        ),
        ConfigDefinition(
            key="memory.provider_id",
            section="llm_providers",
            attr="memory_provider_id",
            default="",
            description="LLM提供者ID",
            value_type=str,
        ),

        # ── 主动回复 ──
        ConfigDefinition(
            key="proactive_reply.enable",
            section="proactive_reply",
            attr="enable",
            default=False,
            description="是否启用主动回复",
            value_type=bool,
        ),
        ConfigDefinition(
            key="proactive_reply.group_whitelist_mode",
            section="proactive_reply",
            attr="group_whitelist_mode",
            default=False,
            description="是否启用群聊白名单模式",
            value_type=bool,
        ),

        # ── 图片分析 ──
        ConfigDefinition(
            key="image_analysis.enable",
            section="image_analysis",
            attr="enable_image_analysis",
            default=True,
            description="是否启用图片分析",
            value_type=bool,
        ),
        ConfigDefinition(
            key="image_analysis.mode",
            section="image_analysis",
            attr="analysis_mode",
            default="auto",
            description="图片分析模式 (auto/brief/detailed/skip)",
            value_type=str,
        ),
        ConfigDefinition(
            key="image_analysis.daily_budget",
            section="image_analysis",
            attr="daily_analysis_budget",
            default=100,
            description="每日图片分析预算",
            value_type=int,
        ),
        ConfigDefinition(
            key="image_analysis.provider_id",
            section="llm_providers",
            attr="image_analysis_provider_id",
            default="",
            description="图片分析LLM提供者ID",
            value_type=str,
        ),

        # ── 场景自适应 ──
        ConfigDefinition(
            key="activity_adaptive.enable",
            section="activity_adaptive",
            attr="enable_activity_adaptive",
            default=True,
            description="是否启用活跃度自适应",
            value_type=bool,
        ),

        # ── 嵌入向量 ──
        ConfigDefinition(
            key="embedding.source",
            section="embedding",
            attr="source",
            default="auto",
            description="嵌入源选择 (auto/astrbot/local)",
            value_type=str,
        ),
        ConfigDefinition(
            key="embedding.astrbot_provider_id",
            section="embedding",
            attr="astrbot_provider_id",
            default="",
            description="AstrBot embedding provider ID（空字符串使用第一个可用的）",
            value_type=str,
        ),
        ConfigDefinition(
            key="embedding.local_model",
            section="embedding",
            attr="local_model",
            default="BAAI/bge-small-zh-v1.5",
            description="本地嵌入模型名称",
            value_type=str,
        ),
        ConfigDefinition(
            key="embedding.local_dimension",
            section="embedding",
            attr="local_dimension",
            default=512,
            description="本地嵌入模型维度",
            value_type=int,
        ),
        ConfigDefinition(
            key="embedding.reimport_on_dimension_conflict",
            section="embedding",
            attr="reimport_on_dimension_conflict",
            default=True,
            description="维度冲突时重新导入原记忆（会增加embedding使用量）",
            value_type=bool,
        ),

        # ── 画像提取 ──
        ConfigDefinition(
            key="persona.enabled",
            section="persona",
            attr="enabled",
            default=True,
            description="是否启用用户画像功能",
            value_type=bool,
        ),
        ConfigDefinition(
            key="persona.extraction_mode",
            section="persona",
            attr="extraction_mode",
            default="rule",
            description="画像提取模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="persona.llm_provider",
            section="llm_providers",
            attr="persona_provider_id",
            default="",
            description="画像提取LLM提供者（空字符串表示使用默认）",
            value_type=str,
        ),
        ConfigDefinition(
            key="persona.enable_interest_extraction",
            section="persona",
            attr="enable_interest_extraction",
            default=True,
            description="是否启用兴趣提取",
            value_type=bool,
        ),
        ConfigDefinition(
            key="persona.enable_style_extraction",
            section="persona",
            attr="enable_style_extraction",
            default=True,
            description="是否启用风格提取",
            value_type=bool,
        ),
        ConfigDefinition(
            key="persona.enable_preference_extraction",
            section="persona",
            attr="enable_preference_extraction",
            default=True,
            description="是否启用偏好提取",
            value_type=bool,
        ),
        ConfigDefinition(
            key="persona.llm_max_tokens",
            section="persona",
            attr="llm_max_tokens",
            default=300,
            description="画像提取LLM最大token数",
            value_type=int,
        ),
        ConfigDefinition(
            key="persona.llm_daily_limit",
            section="persona",
            attr="llm_daily_limit",
            default=50,
            description="画像提取LLM每日限制",
            value_type=int,
        ),
        ConfigDefinition(
            key="persona.fallback_to_rule",
            section="persona",
            attr="fallback_to_rule",
            default=True,
            description="LLM失败时是否回退到规则",
            value_type=bool,
        ),

        # ── LLM提供者 ──
        ConfigDefinition(
            key="llm_providers.default_provider_id",
            section="llm_providers",
            attr="default_provider_id",
            default="",
            description="默认LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_providers.memory_provider_id",
            section="llm_providers",
            attr="memory_provider_id",
            default="",
            description="记忆功能LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_providers.persona_provider_id",
            section="llm_providers",
            attr="persona_provider_id",
            default="",
            description="用户画像LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_providers.knowledge_graph_provider_id",
            section="llm_providers",
            attr="knowledge_graph_provider_id",
            default="",
            description="知识图谱LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_providers.image_analysis_provider_id",
            section="llm_providers",
            attr="image_analysis_provider_id",
            default="",
            description="图片分析LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_providers.enhanced_provider_id",
            section="llm_providers",
            attr="enhanced_provider_id",
            default="",
            description="智能增强LLM提供者ID",
            value_type=str,
        ),

        # ── 高级参数（群聊自适应覆盖） ──
        ConfigDefinition(
            key="advanced.chat_context_count",
            section="llm_integration",
            attr="chat_context_count",
            default=15,
            description="聊天上下文条数",
            value_type=int,
        ),
        ConfigDefinition(
            key="advanced.cooldown_seconds",
            section="proactive_reply",
            attr="cooldown_seconds",
            default=60,
            description="主动回复冷却秒数",
            value_type=int,
        ),
        ConfigDefinition(
            key="advanced.max_daily_replies",
            section="proactive_reply",
            attr="max_daily_replies",
            default=20,
            description="每日最大主动回复次数",
            value_type=int,
        ),
        ConfigDefinition(
            key="advanced.reply_temperature",
            section="proactive_reply",
            attr="reply_temperature",
            default=0.7,
            description="主动回复温度",
            value_type=float,
        ),
        ConfigDefinition(
            key="advanced.batch_threshold_count",
            section="message_processing",
            attr="batch_threshold_count",
            default=20,
            description="批量处理消息阈值",
            value_type=int,
        ),
        ConfigDefinition(
            key="advanced.batch_threshold_interval",
            section="message_processing",
            attr="batch_threshold_interval",
            default=300,
            description="批量处理间隔（秒）",
            value_type=int,
        ),
        ConfigDefinition(
            key="advanced.daily_analysis_budget",
            section="image_analysis",
            attr="daily_analysis_budget",
            default=100,
            description="每日图片分析预算",
            value_type=int,
        ),

        # ── LLM智能增强 ──
        ConfigDefinition(
            key="llm_enhanced.provider_id",
            section="llm_providers",
            attr="enhanced_provider_id",
            default="",
            description="LLM增强提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.sensitivity_mode",
            section="llm_enhanced",
            attr="sensitivity_mode",
            default="rule",
            description="敏感度检测模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.trigger_mode",
            section="llm_enhanced",
            attr="trigger_mode",
            default="rule",
            description="触发检测模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.emotion_mode",
            section="llm_enhanced",
            attr="emotion_mode",
            default="rule",
            description="情感分析模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.proactive_mode",
            section="llm_enhanced",
            attr="proactive_mode",
            default="rule",
            description="主动回复检测模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.conflict_mode",
            section="llm_enhanced",
            attr="conflict_mode",
            default="rule",
            description="冲突检测模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="llm_enhanced.retrieval_mode",
            section="llm_enhanced",
            attr="retrieval_mode",
            default="rule",
            description="检索路由模式 (rule/llm/hybrid)",
            value_type=str,
        ),

        # ── 知识图谱 ──
        ConfigDefinition(
            key="knowledge_graph.enabled",
            section="knowledge_graph",
            attr="enabled",
            default=True,
            description="是否启用知识图谱",
            value_type=bool,
        ),
        ConfigDefinition(
            key="knowledge_graph.extraction_mode",
            section="knowledge_graph",
            attr="extraction_mode",
            default="rule",
            description="三元组提取模式 (rule/llm/hybrid)",
            value_type=str,
        ),
        ConfigDefinition(
            key="knowledge_graph.provider_id",
            section="llm_providers",
            attr="knowledge_graph_provider_id",
            default="",
            description="知识图谱LLM提供者ID",
            value_type=str,
        ),
        ConfigDefinition(
            key="knowledge_graph.max_depth",
            section="knowledge_graph",
            attr="max_depth",
            default=3,
            description="多跳推理最大跳数",
            value_type=int,
        ),
        ConfigDefinition(
            key="knowledge_graph.max_nodes_per_hop",
            section="knowledge_graph",
            attr="max_nodes_per_hop",
            default=10,
            description="每跳最大探索节点数",
            value_type=int,
        ),
        ConfigDefinition(
            key="knowledge_graph.max_facts",
            section="knowledge_graph",
            attr="max_facts",
            default=8,
            description="注入LLM上下文的最大事实数",
            value_type=int,
        ),
        ConfigDefinition(
            key="knowledge_graph.min_confidence",
            section="knowledge_graph",
            attr="min_confidence",
            default=0.2,
            description="最低置信度阈值",
            value_type=float,
        ),
    ]

    return {e.key: e for e in entries}


# 全局配置注册表（单一数据源）
CONFIG_REGISTRY: Dict[str, ConfigDefinition] = _build_registry()


def get_registry_default(key: str) -> Any:
    """从注册表获取默认值

    Args:
        key: 配置键，如 ``"basic.enable_memory"``

    Returns:
        默认值；键不存在时返回 ``None``
    """
    defn = CONFIG_REGISTRY.get(key)
    return defn.default if defn else None


def get_registry_mapping(key: str) -> Optional[tuple]:
    """获取配置键到 (section, attr, default) 的映射

    延续 ``CONFIG_KEY_MAPPING`` 的访问方式。

    Args:
        key: 配置键

    Returns:
        ``(section, attr, default)`` 三元组，或 ``None``
    """
    defn = CONFIG_REGISTRY.get(key)
    if defn is None:
        return None
    return (defn.section, defn.attr, defn.default)
