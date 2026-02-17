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
    embedding_strategy: str = "auto"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_models: list = field(default_factory=lambda: ["BAAI/bge-small-zh-v1.5"])
    embedding_dimension: int = 512
    collection_name: str = "iris_memory"
    auto_detect_dimension: bool = True


@dataclass
class LLMIntegrationDefaults:
    """LLM集成默认配置"""
    # 基础配置（用户可调）
    max_context_memories: int = 3
    token_budget: int = 512
    chat_context_count: int = 15  # 注入最近N条聊天记录到LLM上下文
    provider_id: str = ""  # LLM 提供者 ID，空字符串表示使用默认
    
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
    
    注意：触发关键词和检测规则直接在 ProactiveReplyDetector 中定义
    """
    cooldown_seconds: int = 60
    max_daily_replies: int = 20
    max_reply_tokens: int = 150
    reply_temperature: float = 0.7
    
    # 群聊白名单（空列表表示允许所有群聊）
    group_whitelist: list = field(default_factory=list)
    
    # 群聊白名单模式（开启后需管理员用指令控制各群聊的主动回复开关）
    group_whitelist_mode: bool = False
    
    # 检测阈值（高级）
    high_emotion_threshold: float = 0.7
    question_threshold: float = 0.8
    mention_threshold: float = 0.9


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
    provider_id: str = ""  # 图片分析 LLM 提供者 ID，空字符串表示使用默认
    
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
    # 是否启用画像自动更新
    enable_auto_update: bool = True
    # 变更审计日志最大条数
    max_change_log: int = 200
    # 画像快照间隔（每N次更新输出一次DEBUG快照）
    snapshot_interval: int = 10
    # 是否启用画像注入到LLM上下文
    enable_persona_injection: bool = True
    
    # LLM 画像提取
    extraction_mode: str = "rule"       # "rule" | "llm" | "hybrid"
    llm_provider: str = "default"       # "default" 或具体 provider_id
    enable_interest_extraction: bool = True
    enable_style_extraction: bool = True
    enable_preference_extraction: bool = True
    llm_max_tokens: int = 300
    llm_daily_limit: int = 50           # 每日 LLM 提取次数限制
    fallback_to_rule: bool = True       # LLM 失败时回退到规则


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
                         'persona', 'activity_adaptive']:
        section_obj = getattr(DEFAULTS, section_name, None)
        if section_obj:
            result[section_name] = asdict(section_obj)
    return result
