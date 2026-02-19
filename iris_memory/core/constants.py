"""
常量定义模块 - 集中管理所有硬编码常量和Prompt模板
"""
from enum import Enum
from typing import Final, FrozenSet, Set

from iris_memory.core.types import EmotionType


# ── 共享情感常量 ──

NEGATIVE_EMOTIONS_ALL: Final[FrozenSet[EmotionType]] = frozenset([
    EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR,
    EmotionType.DISGUST, EmotionType.ANXIETY,
])
"""完整负面情绪集（5 种）"""

NEGATIVE_EMOTIONS_CORE: Final[FrozenSet[EmotionType]] = frozenset([
    EmotionType.SADNESS, EmotionType.ANGER, EmotionType.ANXIETY,
])
"""核心负面情绪子集（3 种，常用于检测触发）"""

NEGATIVE_EMOTION_STRINGS: Final[FrozenSet[str]] = frozenset([
    "sadness", "anger", "fear", "disgust", "anxiety",
])
"""负面情绪字符串集（用于 dict-keyed 上下文）"""

DEFAULT_EMOTION: Final[str] = "neutral"
"""默认情绪状态（== EmotionType.NEUTRAL.value）"""


# ── 主动回复 extras key ──

PROACTIVE_EXTRA_KEY: Final[str] = "iris_proactive"
PROACTIVE_CONTEXT_KEY: Final[str] = "iris_proactive_context"


# ── 无限预算哨兵值 ──

UNLIMITED_BUDGET: Final[int] = 999999


class CommandPrefix:
    """指令前缀常量"""
    SLASH: Final[str] = "/"
    MEMORY_SAVE: Final[Set[str]] = frozenset(["/memory_save", "memory_save"])
    MEMORY_SEARCH: Final[Set[str]] = frozenset(["/memory_search", "memory_search"])
    MEMORY_CLEAR: Final[Set[str]] = frozenset(["/memory_clear", "memory_clear"])
    MEMORY_STATS: Final[Set[str]] = frozenset(["/memory_stats", "memory_stats"])
    MEMORY_DELETE: Final[Set[str]] = frozenset(["/memory_delete", "memory_delete"])
    PROACTIVE_REPLY: Final[Set[str]] = frozenset(["/proactive_reply", "proactive_reply"])


class DeleteMainScope(Enum):
    """删除主范围"""
    CURRENT = "current"
    PRIVATE = "private"
    GROUP = "group"
    ALL = "all"


class ConfigKeys:
    """配置键常量"""
    ENABLE_MEMORY: Final[str] = "basic.enable_memory"
    ENABLE_INJECT: Final[str] = "basic.enable_inject"
    LOG_LEVEL: Final[str] = "basic.log_level"
    
    MAX_CONTEXT_MEMORIES: Final[str] = "memory.max_context_memories"
    MAX_WORKING_MEMORY: Final[str] = "memory.max_working_memory"
    UPGRADE_MODE: Final[str] = "memory.upgrade_mode"
    CHAT_CONTEXT_COUNT: Final[str] = "memory.chat_context_count"
    USE_LLM: Final[str] = "memory.use_llm"
    
    PROACTIVE_REPLY_ENABLE: Final[str] = "proactive_reply.enable"
    PROACTIVE_REPLY_MAX_DAILY: Final[str] = "proactive_reply.max_daily"
    PROACTIVE_REPLY_GROUP_WHITELIST_MODE: Final[str] = "proactive_reply.group_whitelist_mode"
    
    IMAGE_ANALYSIS_ENABLE: Final[str] = "image_analysis.enable"
    IMAGE_ANALYSIS_MODE: Final[str] = "image_analysis.mode"
    IMAGE_ANALYSIS_MAX_IMAGES: Final[str] = "image_analysis.max_images"
    IMAGE_ANALYSIS_DAILY_BUDGET: Final[str] = "image_analysis.daily_budget"
    IMAGE_ANALYSIS_SESSION_BUDGET: Final[str] = "image_analysis.session_budget"
    IMAGE_ANALYSIS_REQUIRE_CONTEXT: Final[str] = "image_analysis.require_context"
    
    ERROR_FRIENDLY_ENABLE: Final[str] = "error_friendly.enable"


class SessionScope:
    """会话作用域"""
    PRIVATE: Final[str] = "private"
    GROUP_SHARED: Final[str] = "group_shared"
    GROUP_PRIVATE: Final[str] = "group_private"


class KVStoreKeys:
    """KV存储键"""
    SESSIONS: Final[str] = "sessions"
    LIFECYCLE_STATE: Final[str] = "lifecycle_state"
    BATCH_QUEUES: Final[str] = "batch_queues"
    CHAT_HISTORY: Final[str] = "chat_history"
    LAST_SAVE_PREFIX: Final[str] = "last_save_{user_id}_{group_id}"
    PROACTIVE_REPLY_WHITELIST: Final[str] = "proactive_reply_whitelist"
    MEMBER_IDENTITY: Final[str] = "member_identity"
    USER_PERSONAS: Final[str] = "user_personas"
    GROUP_ACTIVITY: Final[str] = "group_activity"


class ErrorMessages:
    """错误消息模板"""
    EMPTY_CONTENT: Final[str] = "请输入要保存的内容"
    EMPTY_QUERY: Final[str] = "请输入搜索内容"
    PRIVATE_ONLY: Final[str] = "此命令仅限私聊使用"
    GROUP_ONLY: Final[str] = "此命令仅限群聊使用"
    ADMIN_REQUIRED: Final[str] = "权限不足，仅管理员可以执行此操作"
    GROUP_ADMIN_REQUIRED: Final[str] = "权限不足，仅管理员可以删除群聊记忆"
    DELETE_CONFIRM_REQUIRED: Final[str] = "警告：此操作将删除所有记忆！\n请使用 '/memory_delete all confirm' 确认操作"
    INVALID_SCOPE_PARAM: Final[str] = "参数错误，请使用: shared, private 或 all"
    INVALID_DELETE_SCOPE: Final[str] = "参数错误，可用范围: current, private, group [shared|private|all], all confirm"
    CAPTURE_FAILED: Final[str] = "未能保存记忆，可能不满足捕获条件"
    NO_MEMORIES_FOUND: Final[str] = "未找到相关记忆"
    DELETE_FAILED: Final[str] = "未找到记忆或删除失败"


class SuccessMessages:
    """成功消息模板"""
    MEMORY_SAVED: Final[str] = "记忆已保存（类型：{memory_type}，置信度：{confidence:.2f}）"
    MEMORY_CLEARED: Final[str] = "记忆已清除"
    PRIVATE_DELETED: Final[str] = "已删除 {count} 条个人私聊记忆"
    GROUP_DELETED: Final[str] = "已删除当前群聊的 {count} 条{scope_desc}记忆"
    ALL_DELETED: Final[str] = "已删除所有 {count} 条记忆！"
    STATS_TEMPLATE: Final[str] = """记忆统计：
- 工作记忆：{working_count} 条
- 情景记忆：{episodic_count} 条{image_stats}"""


class PersonaStyle:
    """人格风格"""
    NATURAL: Final[str] = "natural"


class BatchProcessingMode:
    """批量处理模式"""
    SUMMARY: Final[str] = "summary"
    FILTER: Final[str] = "filter"
    HYBRID: Final[str] = "hybrid"


class SourceType:
    """来源类型"""
    LOCAL: Final[str] = "local"
    LLM: Final[str] = "llm"


class LogTemplates:
    """日志模板"""
    PLUGIN_INIT_START: Final[str] = "IrisMemory plugin initializing..."
    PLUGIN_INIT_SUCCESS: Final[str] = "IrisMemory plugin initialized successfully"
    PLUGIN_INIT_FAILED: Final[str] = "IrisMemory plugin initialization failed: {error}"
    PLUGIN_TERMINATED: Final[str] = "IrisMemory plugin terminated"
    PLUGIN_TERMINATE_ERROR: Final[str] = "IrisMemory plugin termination error: {error}"
    
    COMPONENT_INIT: Final[str] = "Initializing {component}..."
    COMPONENT_INIT_DISABLED: Final[str] = "{component} is disabled"
    
    SESSION_LOADED: Final[str] = "Loaded {count} sessions"
    SESSION_SAVED: Final[str] = "Saved {count} sessions"
    
    MEMORY_INJECTED: Final[str] = "Injected {count} memories into LLM context"
    MEMORY_CAPTURED: Final[str] = "Auto-captured memory: {memory_id}"
    IMMEDIATE_MEMORY_CAPTURED: Final[str] = "Immediate memory captured: {memory_id}"
    
    FINAL_STATS_HEADER: Final[str] = "=== Final Statistics ==="


# 数值常量
class NumericDefaults:
    """数值默认值"""
    TOP_K_SEARCH: Final[int] = 5
    CONFIRM_PARAM_INDEX: Final[int] = 1
    CONFIRM_VALUE: Final[str] = "confirm"
    SESSION_TIMEOUT_HOURS: Final[int] = 24
    SECONDS_PER_HOUR: Final[int] = 3600


class ErrorFriendlyMessages:
    """错误消息友好化配置"""
    ERROR_PATTERNS: Final[tuple] = (
        "AstrBot 请求失败",
        "错误类型:",
        "错误信息:",
        "请在平台日志查看",
    )
    
    DEFAULT_FRIENDLY_MSG: Final[str] = "呜...遇到了一点问题，请稍后再试试吧~"
    NETWORK_ERROR_MSG: Final[str] = "网络好像不太稳定呢，稍后再试试？"
    RATE_LIMIT_MSG: Final[str] = "我需要休息一下，请稍后再来找我~"
    BAD_REQUEST_MSG: Final[str] = "请求出了点问题，稍后再试试吧~"


# ── LLM 重试与熔断器常量 ──

class LLMRetryConfig:
    """LLM 调用重试配置"""
    MAX_RETRIES: Final[int] = 3
    INITIAL_BACKOFF: Final[float] = 1.0        # 初始退避时间（秒）
    MAX_BACKOFF: Final[float] = 30.0           # 最大退避时间（秒）
    BACKOFF_MULTIPLIER: Final[float] = 2.0     # 退避乘数
    CALL_TIMEOUT: Final[int] = 60              # 单次调用超时（秒）


class CircuitBreakerConfig:
    """熔断器配置"""
    FAILURE_THRESHOLD: Final[int] = 5          # 连续失败次数阈值
    RECOVERY_TIMEOUT: Final[int] = 30          # 熔断恢复超时（秒）
    HALF_OPEN_MAX: Final[int] = 1              # 半开状态最多允许通过的请求数


class LLMRateLimitConfig:
    """LLM 调用速率限制配置"""
    DAILY_CALL_LIMIT: Final[int] = 500         # 每日最大 LLM 调用次数


# ── 批量处理会话管理常量 ──

class InputValidationConfig:
    """输入验证配置"""
    MAX_MESSAGE_LENGTH: Final[int] = 2000      # 单条消息最大长度
    MAX_QUERY_LENGTH: Final[int] = 500         # 搜索查询最大长度
    MAX_SAVE_CONTENT_LENGTH: Final[int] = 1000 # 手动保存内容最大长度

    # 危险内容检测模式（HTML标签、脚本注入等）
    _DANGEROUS_PATTERNS: Final[tuple] = (
        r'<\s*script[^>]*>',        # <script> 标签
        r'javascript\s*:',          # javascript: 协议
        r'on\w+\s*=\s*["\']',       # 事件处理器 (onclick= 等)
        r'<\s*iframe[^>]*>',        # <iframe> 标签
        r'<\s*object[^>]*>',        # <object> 标签
        r'<\s*embed[^>]*>',         # <embed> 标签
    )

    @staticmethod
    def sanitize_input(text: str) -> str:
        """对用户输入进行基本的安全清理
        
        移除 HTML 标签和脚本代码，保留纯文本内容。
        
        Args:
            text: 原始输入文本
            
        Returns:
            str: 清理后的文本
        """
        import re
        # 移除 HTML 标签
        cleaned = re.sub(r'<[^>]+>', '', text)
        # 移除零宽字符和控制字符（保留换行和制表符）
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        return cleaned.strip()

    @staticmethod
    def has_dangerous_content(text: str) -> bool:
        """检查文本是否包含危险内容
        
        Args:
            text: 输入文本
            
        Returns:
            bool: 是否包含危险内容
        """
        import re
        for pattern in InputValidationConfig._DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class RetrievalDefaults:
    """检索引擎默认值"""
    TOKEN_BUDGET: Final[int] = 512             # Token 预算
    MAX_SUMMARY_LENGTH: Final[int] = 100       # 摘要最大长度
    DEFAULT_TOP_K: Final[int] = 10             # 默认 Top-K 检索数量
    SHORT_QUERY_THRESHOLD: Final[int] = 20     # 短查询长度阈值
    EMOTION_FILTER_THRESHOLD: Final[float] = 0.7  # 情感过滤权重阈值
    LLM_ROUTING_CONFIDENCE: Final[float] = 0.6    # LLM 路由置信度阈值
    KG_IMPORTANCE_BOOST: Final[float] = 0.3    # KG 重要性分数提升
    KG_RIF_BOOST: Final[float] = 0.2           # KG RIF 分数提升


class CacheDefaults:
    """缓存默认值"""
    EMBEDDING_CACHE_MAX_SIZE: Final[int] = 1000  # 嵌入缓存最大条目数
    EMBEDDING_CACHE_TTL: Final[float] = 3600.0   # 嵌入缓存 TTL（秒）
    KG_NODE_CACHE_MAX_SIZE: Final[int] = 1000    # KG 节点缓存最大条目数


class BatchSessionConfig:
    """批量处理器会话管理配置"""
    MAX_TRACKED_SESSIONS: Final[int] = 500     # 最大跟踪会话数
    SESSION_EXPIRY_SECONDS: Final[int] = 14400 # 无活动会话过期时间（4小时）


class CaptureQualityThresholds:
    """记忆捕获质量评估阈值"""
    CONFIRMED: Final[float] = 0.9              # 已确认质量
    HIGH: Final[float] = 0.75                  # 高质量
    MODERATE: Final[float] = 0.5               # 中等质量
    LOW: Final[float] = 0.3                    # 低质量
    EMOTION_TRIGGER_INTENSITY: Final[float] = 0.6   # 情感触发强度阈值
    EMOTION_NOTRIGGER_INTENSITY: Final[float] = 0.7 # 无触发时情感强度阈值
    EPISODIC_EMOTION_WEIGHT: Final[float] = 0.6     # 情景记忆情感权重阈值
    EPISODIC_CONFIDENCE: Final[float] = 0.5          # 情景记忆置信度阈值
    USER_REQUESTED_IMPORTANCE: Final[float] = 0.8    # 用户主动请求的最低重要性


class LLMCallDefaults:
    """LLM 调用默认参数"""
    DEFAULT_MAX_TOKENS: Final[int] = 200       # 默认最大 token 数
    CLASSIFICATION_MAX_TOKENS: Final[int] = 150 # 分类调用最大 token 数
    BATCH_MAX_TOKENS: Final[int] = 100         # 批量处理最大 token 数
    DEFAULT_TEMPERATURE: Final[float] = 0.3    # 默认温度
    RESPONSE_TRUNCATION: Final[int] = 500      # 响应截断长度


class TextTruncation:
    """文本截断长度常量"""
    CONTEXT_MESSAGE: Final[int] = 200          # 上下文消息截断
    DIRECTIVE_MESSAGE: Final[int] = 100        # 指令消息截断
    SUMMARY: Final[int] = 100                  # 摘要截断
    LOCAL_SUMMARY: Final[int] = 50             # 本地摘要截断


class UrgencyCooldownMultiplier:
    """紧急度冷却时间乘数"""
    CRITICAL: Final[float] = 0.25
    HIGH: Final[float] = 0.5
    MEDIUM: Final[float] = 1.0
    LOW: Final[float] = 1.5
