"""
默认配置文件 - 存放高级参数和技术性配置

这些配置通常用户不需要修改，只有在特殊需求时才需要调整。
用户配置请在AstrBot管理界面中修改，将覆盖这里的默认值。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class EmbeddingStrategy(str, Enum):
    """嵌入策略"""
    AUTO = "auto"
    ASTRBOT = "astrbot"
    LOCAL = "local"
    FALLBACK = "fallback"


class InjectionMode(str, Enum):
    """注入模式"""
    PREFIX = "prefix"
    SUFFIX = "suffix"
    EMBEDDED = "embedded"
    HYBRID = "hybrid"


class CacheStrategy(str, Enum):
    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


class UpgradeMode(str, Enum):
    """记忆升级模式"""
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


class ProcessingMode(str, Enum):
    """消息处理模式"""
    LOCAL = "local"
    LLM = "llm"
    HYBRID = "hybrid"


class CoordinationStrategy(str, Enum):
    """人格协调策略"""
    BOT_PRIORITY = "bot_priority"
    USER_PRIORITY = "user_priority"
    HYBRID = "hybrid"
    DYNAMIC = "dynamic"


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
    embedding_dimension: int = 512
    collection_name: str = "iris_memory"
    auto_detect_dimension: bool = True


@dataclass
class LLMIntegrationDefaults:
    """LLM集成默认配置"""
    # 基础配置（用户可调）
    max_context_memories: int = 3
    token_budget: int = 512
    
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
    # 批量处理
    batch_threshold_count: int = 50
    batch_threshold_interval: int = 300  # 5分钟
    batch_processing_mode: str = "hybrid"
    llm_max_tokens_for_summary: int = 200
    
    # 触发阈值（高级）
    immediate_trigger_confidence: float = 0.8
    immediate_emotion_intensity: float = 0.7
    
    # 处理模式
    llm_processing_mode: str = "hybrid"


@dataclass
class ProactiveReplyDefaults:
    """主动回复默认配置"""
    cooldown_seconds: int = 60
    max_daily_replies: int = 20
    max_reply_tokens: int = 150
    reply_temperature: float = 0.7
    
    # 检测阈值（高级）
    high_emotion_threshold: float = 0.7
    question_threshold: float = 0.8
    mention_threshold: float = 0.9


@dataclass
class LogDefaults:
    """日志默认配置"""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    max_file_size: int = 10  # MB
    backup_count: int = 5


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
    log: LogDefaults = field(default_factory=LogDefaults)


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
                         'proactive_reply', 'log']:
        section_obj = getattr(DEFAULTS, section_name, None)
        if section_obj:
            result[section_name] = asdict(section_obj)
    return result
