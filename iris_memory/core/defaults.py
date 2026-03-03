"""
默认配置文件 - 存放高级参数和技术性配置

这些配置通常用户不需要修改，只有在特殊需求时才需要调整。
用户配置请在AstrBot管理界面中修改，将覆盖这里的默认值。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


# ========== 群活跃度枚举 ==========

class GroupActivityLevel(str, Enum):
    """群活跃度等级"""
    QUIET = "quiet"          # < 5条/小时
    MODERATE = "moderate"    # 5-20条/小时
    ACTIVE = "active"        # 20-50条/小时
    INTENSIVE = "intensive"  # > 50条/小时


# ========== 活跃度分级配置预设 ==========

@dataclass
class ActivityBasedPresets:
    """活跃度分级配置预设

    每个字段为 {GroupActivityLevel.value: int/float} 的映射。
    "温和型"陪伴风格：安静群更克制，活跃群更参与但不过度。
    """
    cooldown_seconds: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 120,
        "moderate": 60,
        "active": 45,
        "intensive": 30,
    })
    max_daily_replies: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 8,
        "moderate": 15,
        "active": 22,
        "intensive": 25,
    })
    batch_threshold_count: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 10,
        "moderate": 15,
        "active": 30,
        "intensive": 50,
    })
    batch_threshold_interval: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 600,
        "moderate": 300,
        "active": 180,
        "intensive": 120,
    })
    daily_analysis_budget: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 30,
        "moderate": 60,
        "active": 100,
        "intensive": 150,
    })
    chat_context_count: Dict[str, int] = field(default_factory=lambda: {
        "quiet": 10,
        "moderate": 15,
        "active": 20,
        "intensive": 25,
    })
    reply_temperature: Dict[str, float] = field(default_factory=lambda: {
        "quiet": 0.6,
        "moderate": 0.7,
        "active": 0.75,
        "intensive": 0.8,
    })

    def get(self, key: str, level: GroupActivityLevel) -> Any:
        """按 key 和等级获取预设值"""
        mapping = getattr(self, key, None)
        if mapping is None:
            return None
        return mapping.get(level.value)


# 全局活跃度预设实例
ACTIVITY_PRESETS = ActivityBasedPresets()

# 活跃度阈值定义（条/小时）
ACTIVITY_THRESHOLDS: Dict[str, int] = {
    "quiet_upper": 5,
    "moderate_upper": 20,
    "active_upper": 50,
}

# 滑动窗口和防抖配置
ACTIVITY_WINDOW_HOURS: int = 3          # 滑动窗口小时数
ACTIVITY_HYSTERESIS_RATIO: float = 0.2  # 升降级需比阈值多/少 20%


@dataclass
class MemoryDefaults:
    """记忆系统默认配置"""
    # 工作记忆
    max_working_memory: int = 10
    
    # RIF评分
    rif_threshold: float = 0.4
    
    # 记忆升级（高级）
    upgrade_mode: str = "rule"
    llm_upgrade_batch_size: int = 5
    llm_upgrade_threshold: float = 0.7
    
    # 记忆捕获（高级）
    min_confidence: float = 0.3
    enable_duplicate_check: bool = True
    enable_conflict_check: bool = True
    enable_entity_extraction: bool = True


@dataclass
class SessionDefaults:
    """会话管理默认配置"""
    # 基础配置
    session_timeout: int = 86400  # 24小时
    session_inactive_timeout: int = 1800  # 30分钟
    session_cleanup_interval: int = 3600  # 1小时
    
    # 高级配置
    max_sessions: int = 100
    promotion_interval: int = 3600  # 记忆升级检查间隔


@dataclass
class CacheDefaults:
    """缓存系统默认配置"""
    # 嵌入缓存
    embedding_cache_size: int = 1000
    embedding_cache_strategy: str = "lru"
    
    # 工作记忆缓存
    working_cache_ttl: int = 86400  # 24小时
    
    # 压缩配置
    compression_max_length: int = 200


@dataclass
class EmbeddingDefaults:
    """嵌入向量默认配置"""
    # 源选择：auto / astrbot / local
    source: str = "auto"

    # AstrBot provider 配置
    astrbot_provider_id: str = ""  # 空字符串表示使用第一个可用的

    # 本地模型配置
    local_model: str = "BAAI/bge-small-zh-v1.5"
    local_dimension: int = 512  # 留空则自动检测

    # 集合配置
    collection_name: str = "iris_memory"
    auto_detect_dimension: bool = True

    # 维度冲突处理
    reimport_on_dimension_conflict: bool = True  # 维度冲突时重新导入原记忆


@dataclass
class LLMIntegrationDefaults:
    """LLM集成默认配置"""
    # 基础配置（用户可调）
    max_context_memories: int = 3
    token_budget: int = 512
    chat_context_count: int = 15  # 注入最近N条聊天记录到LLM上下文
    
    # 高级配置
    injection_mode: str = "suffix"
    coordination_strategy: str = "hybrid"
    enable_time_aware: bool = True
    enable_emotion_aware: bool = True
    enable_token_budget: bool = True
    enable_routing: bool = True
    enable_working_memory_merge: bool = True


@dataclass
class MessageProcessingDefaults:
    """消息处理默认配置"""
    # 批量处理 - 优化为20条消息合并处理
    batch_threshold_count: int = 20  # 20条消息触发批量处理
    batch_threshold_interval: int = 300  # 5分钟无新消息就处理
    batch_processing_mode: str = "hybrid"
    llm_max_tokens_for_summary: int = 200

    # 消息合并配置
    short_message_threshold: int = 15  # 短消息长度阈值（字符）
    merge_time_window: int = 60  # 合并时间窗口（秒）
    max_merge_count: int = 5  # 最大合并消息数

    # 触发阈值（高级）
    immediate_trigger_confidence: float = 0.7
    immediate_emotion_intensity: float = 0.6

    # 处理模式
    llm_processing_mode: str = "hybrid"


@dataclass
class ProactiveReplyDefaults:
    """主动回复默认配置

    v3 重构：SignalQueue + GroupScheduler + FollowUpPlanner。
    用户可见的配置通过 AstrBot 管理界面修改，高级参数在此设置默认值。
    """
    # ===== 用户可见配置 =====
    followup_after_all_replies: bool = False  # Bot 所有回复后都创建跟进期待（而非仅主动回复）
    cooldown_seconds: int = 60
    max_daily_replies: int = 20
    max_reply_tokens: int = 150
    reply_temperature: float = 0.7

    # 群聊白名单（空列表表示允许所有群聊）
    group_whitelist: list = field(default_factory=list)

    # 群聊白名单模式（开启后需管理员用指令控制各群聊的主动回复开关）
    group_whitelist_mode: bool = False

    # 静音时段 [start, end]，支持跨午夜，如 [23, 7] 表示 23:00-07:00
    quiet_hours: list = field(default_factory=lambda: [23, 7])

    # 每用户每日最大主动回复
    max_daily_per_user: int = 5

    # Web 界面开关
    web_dashboard: bool = False

    # ===== v3 SignalQueue 高级配置 =====
    signal_check_interval_seconds: int = 30     # 群定时器检查间隔
    signal_silence_timeout_seconds: int = 600   # 沉默超时（定时器销毁）
    signal_min_silence_seconds: int = 60        # 最小沉默时间才触发判断
    signal_ttl_emotion_high: int = 180          # emotion_high 信号 TTL（秒）
    signal_ttl_rule_match: int = 300            # rule_match 信号 TTL（秒）
    signal_weight_direct_reply: float = 0.8     # 直接回复阈值
    signal_weight_llm_confirm: float = 0.5      # LLM 确认阈值

    # ===== v3 FollowUp 高级配置 =====
    followup_window_seconds: int = 120          # FollowUp 窗口时长
    followup_max_count: int = 2                 # 最大跟进次数
    followup_short_window_seconds: int = 10     # 短期窗口
    followup_llm_max_tokens: int = 500          # LLM 判断最大 token
    followup_llm_temperature: float = 0.3       # LLM 判断温度
    followup_fallback_to_rule: bool = True      # LLM 失败时降级到规则判断


@dataclass
class ActivityAdaptiveDefaults:
    """场景自适应配置默认值"""
    # 是否启用活跃度自适应配置
    enable_activity_adaptive: bool = True
    # 活跃度计算间隔（秒），每隔该时间重新计算一次
    activity_calc_interval: int = 3600
    # 配置缓存 TTL（秒），减少重复计算
    config_cache_ttl: int = 300


@dataclass
class ImageAnalysisDefaults:
    """图片分析默认配置"""
    # 基础配置（用户可调）
    enable_image_analysis: bool = True
    analysis_mode: str = "auto"  # auto/brief/detailed/skip
    max_images_per_message: int = 2
    
    # 高级配置
    skip_sticker: bool = True
    analysis_cooldown: float = 3.0  # 秒
    cache_ttl: int = 3600  # 1小时
    max_cache_size: int = 200
    
    # Token预算
    brief_token_cost: int = 100
    detailed_token_cost: int = 300
    
    # 新增：分析预算控制
    daily_analysis_budget: int = 100  # 每日最大分析次数
    session_analysis_budget: int = 20  # 每会话最大分析次数
    
    # 新增：相似图片去重
    similar_image_window: int = 60  # 秒，相似图片检测时间窗口
    recent_image_limit: int = 20  # 保留的最近图片哈希数量
    
    # 新增：上下文相关性过滤
    require_context_relevance: bool = True  # 是否要求上下文相关才分析


@dataclass
class LogDefaults:
    """日志默认配置"""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    max_file_size: int = 10  # MB
    backup_count: int = 5


@dataclass
class PersonaDefaults:
    """用户画像默认配置"""
    enabled: bool = True
    enable_auto_update: bool = True
    max_change_log: int = 200
    snapshot_interval: int = 10
    enable_persona_injection: bool = True
    
    extraction_mode: str = "rule"
    enable_interest_extraction: bool = True
    enable_style_extraction: bool = True
    enable_preference_extraction: bool = True
    llm_max_tokens: int = 300
    llm_daily_limit: int = 50
    fallback_to_rule: bool = True

    # 批量处理配置
    batch_enabled: bool = True          # 是否启用画像批量提取
    batch_threshold: int = 20             # 触发批量处理的消息数量阈值
    batch_flush_interval: int = 21600      # 定时刷新间隔（秒）默认6小时
    batch_max_size: int = 20             # 单次批量处理的最大消息数


@dataclass
class LLMProvidersDefaults:
    """LLM提供者默认配置
    
    统一管理所有LLM提供者的配置，避免分散在各功能模块中。
    """
    default_provider_id: str = ""  # 默认LLM提供者ID，空字符串表示使用AstrBot默认提供者
    memory_provider_id: str = ""  # 记忆功能LLM提供者ID
    persona_provider_id: str = ""  # 用户画像LLM提供者ID
    knowledge_graph_provider_id: str = ""  # 知识图谱LLM提供者ID
    image_analysis_provider_id: str = ""  # 图片分析LLM提供者ID
    enhanced_provider_id: str = ""  # 智能增强LLM提供者ID


@dataclass
class LLMEnhancedDefaults:
    """LLM智能增强默认配置
    
    各模块模式说明：
    - rule: 仅使用规则（快速，零成本）
    - llm: 仅使用LLM（准确，消耗token）
    - hybrid: 混合模式（推荐，规则预筛+LLM确认）
    """
    sensitivity_mode: str = "rule"
    sensitivity_confidence_threshold: float = 0.7
    
    trigger_mode: str = "rule"
    trigger_daily_limit: int = 200
    
    emotion_mode: str = "rule"
    emotion_llm_weight: float = 0.4
    emotion_enable_context_aware: bool = True

    proactive_mode: str = "rule"  # rule=跳过L3, hybrid=正常进行L3
    proactive_daily_limit: int = 100
    
    conflict_mode: str = "rule"
    
    retrieval_mode: str = "rule"


@dataclass
class PersonaIsolationDefaults:
    """人格隔离默认配置

    控制记忆和知识图谱按 AstrBot 人格隔离查询的行为。
    存储时始终记录 persona_id，查询时根据开关决定是否过滤。
    """
    # 用户可见开关
    memory_query_by_persona: bool = False   # 记忆是否按人格隔离查询
    kg_query_by_persona: bool = False       # 知识图谱是否按人格隔离查询

    # 内部默认值（用户无需配置）
    default_persona_id: str = "default"     # 无法获取人格时使用的默认ID
    persona_id_max_length: int = 64         # persona_id 最大长度限制


@dataclass
class KnowledgeGraphDefaults:
    """知识图谱默认配置"""
    enabled: bool = True                   # 是否启用知识图谱
    extraction_mode: str = "rule"           # 提取模式: rule / llm / hybrid
    max_depth: int = 3                     # BFS 最大跳数
    max_nodes_per_hop: int = 10            # 每跳最大节点数
    max_facts: int = 8                     # 注入LLM的最大事实数
    min_confidence: float = 0.2            # 最低置信度阈值

    # 定时维护配置
    maintenance_interval: int = 86400      # 维护任务执行间隔（秒），默认每天
    auto_maintenance: bool = True          # 是否启用自动维护
    auto_cleanup_orphans: bool = True      # 自动清理孤立节点
    auto_cleanup_low_confidence: bool = True  # 自动清理低置信度边
    low_confidence_threshold: float = 0.2  # 低置信度阈值
    staleness_days: int = 30               # 过期天数


@dataclass
class MarkdownStripperDefaults:
    """Markdown 去除器默认配置"""
    enable: bool = True                    # 功能总开关
    preserve_code_blocks: bool = False     # 保留代码块格式
    preserve_links: bool = False           # 保留链接格式
    threshold_offset: int = 0              # 阈值偏移量
    strip_headers: bool = True             # 去除标题标记
    strip_lists: bool = True               # 去除列表标记


@dataclass
class WebUIDefaults:
    """Web管理界面默认配置"""
    enable: bool = False                   # 是否启用Web管理界面
    port: int = 8089                       # Web服务端口
    access_key: str = ""                   # 访问密钥（空表示无需认证）
    host: str = "127.0.0.1"                # 监听地址


@dataclass
class SemanticExtractionDefaults:
    """语义提取隐藏配置 (通道 B)
    
    用于从 EPISODIC 记忆中聚类 + LLM 提取抽象语义记忆。
    这些参数不在用户配置界面中暴露, 仅作为内部调优参数。
    """
    # 是否启用语义提取 (通道 B)
    enabled: bool = True
    
    # 执行间隔（秒），默认每天执行1次
    extraction_interval: int = 86400
    
    # 预筛选：最低置信度
    min_confidence: float = 0.4
    
    # 预筛选：记忆最小年龄（天），排除过新的记忆
    min_age_days: int = 30
    
    # 聚类：同一实体/主题在时间窗口内的最少出现次数
    min_cluster_size: int = 3
    
    # 聚类：时间窗口（天）
    cluster_time_window_days: int = 90
    
    # 向量相似度聚类阈值
    similarity_threshold: float = 0.75
    
    # 单次提取最大聚类数
    max_clusters_per_run: int = 20
    
    # 单个聚类最大记忆数
    max_memories_per_cluster: int = 15
    
    # 提取后源记忆过期天数（0 表示不设置过期）
    source_expiry_days: int = 0
    
    # LLM 最大 token 数
    llm_max_tokens: int = 500


@dataclass  
class AllDefaults:
    """所有默认配置的聚合"""
    memory: MemoryDefaults = field(default_factory=MemoryDefaults)
    session: SessionDefaults = field(default_factory=SessionDefaults)
    cache: CacheDefaults = field(default_factory=CacheDefaults)
    embedding: EmbeddingDefaults = field(default_factory=EmbeddingDefaults)
    llm_integration: LLMIntegrationDefaults = field(default_factory=LLMIntegrationDefaults)
    message_processing: MessageProcessingDefaults = field(default_factory=MessageProcessingDefaults)
    proactive_reply: ProactiveReplyDefaults = field(default_factory=ProactiveReplyDefaults)
    image_analysis: ImageAnalysisDefaults = field(default_factory=ImageAnalysisDefaults)
    log: LogDefaults = field(default_factory=LogDefaults)
    persona: PersonaDefaults = field(default_factory=PersonaDefaults)
    activity_adaptive: ActivityAdaptiveDefaults = field(default_factory=ActivityAdaptiveDefaults)
    llm_providers: LLMProvidersDefaults = field(default_factory=LLMProvidersDefaults)
    llm_enhanced: LLMEnhancedDefaults = field(default_factory=LLMEnhancedDefaults)
    knowledge_graph: KnowledgeGraphDefaults = field(default_factory=KnowledgeGraphDefaults)
    persona_isolation: PersonaIsolationDefaults = field(default_factory=PersonaIsolationDefaults)
    markdown_stripper: MarkdownStripperDefaults = field(default_factory=MarkdownStripperDefaults)
    web_ui: WebUIDefaults = field(default_factory=WebUIDefaults)
    semantic_extraction: SemanticExtractionDefaults = field(default_factory=SemanticExtractionDefaults)


# 全局默认配置实例
DEFAULTS = AllDefaults()


def get_default(section: str, key: str, fallback: Any = None) -> Any:
    """获取默认配置值
    
    Args:
        section: 配置区块（如 "memory", "session"）
        key: 配置键
        fallback: 如果找不到时的回退值
        
    Returns:
        配置值
    """
    section_obj = getattr(DEFAULTS, section, None)
    if section_obj is None:
        return fallback
    return getattr(section_obj, key, fallback)


def get_defaults_dict() -> Dict[str, Dict[str, Any]]:
    """获取所有默认配置为字典格式
    
    Returns:
        嵌套字典形式的所有默认配置
    """
    from dataclasses import asdict
    result = {}
    for section_name in ['memory', 'session', 'cache', 'embedding', 
                         'llm_integration', 'message_processing', 
                         'proactive_reply', 'image_analysis', 'log',
                         'persona', 'activity_adaptive', 'llm_providers',
                         'llm_enhanced', 'knowledge_graph',
                         'persona_isolation', 'markdown_stripper',
                         'web_ui', 'semantic_extraction']:
        section_obj = getattr(DEFAULTS, section_name, None)
        if section_obj:
            result[section_name] = asdict(section_obj)
    return result
