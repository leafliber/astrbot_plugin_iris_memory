"""
常量定义模块 - 集中管理所有硬编码常量和Prompt模板
"""
from enum import Enum
from typing import Final, Set


class CommandPrefix:
    """指令前缀常量"""
    SLASH: Final[str] = "/"
    MEMORY_SAVE: Final[Set[str]] = frozenset(["/memory_save", "memory_save"])
    MEMORY_SEARCH: Final[Set[str]] = frozenset(["/memory_search", "memory_search"])
    MEMORY_CLEAR: Final[Set[str]] = frozenset(["/memory_clear", "memory_clear"])
    MEMORY_STATS: Final[Set[str]] = frozenset(["/memory_stats", "memory_stats"])
    MEMORY_DELETE_PRIVATE: Final[Set[str]] = frozenset(["/memory_delete_private", "memory_delete_private"])
    MEMORY_DELETE_GROUP: Final[Set[str]] = frozenset(["/memory_delete_group", "memory_delete_group"])
    MEMORY_DELETE_ALL: Final[Set[str]] = frozenset(["/memory_delete_all", "memory_delete_all"])
    PROACTIVE_REPLY: Final[Set[str]] = frozenset(["/proactive_reply", "proactive_reply"])


class KnownCommands:
    """已知指令列表"""
    ALL_COMMANDS: Final[Set[str]] = frozenset([
        "memory_save", "memory_search", "memory_clear", "memory_stats",
        "memory_delete_private", "memory_delete_group", "memory_delete_all",
        "proactive_reply"
    ])


class ConfigKeys:
    """配置键常量"""
    # 基础功能
    ENABLE_MEMORY: Final[str] = "basic.enable_memory"
    ENABLE_INJECT: Final[str] = "basic.enable_inject"
    LOG_LEVEL: Final[str] = "basic.log_level"
    
    # 记忆设置
    MAX_CONTEXT_MEMORIES: Final[str] = "memory.max_context_memories"
    MAX_WORKING_MEMORY: Final[str] = "memory.max_working_memory"
    UPGRADE_MODE: Final[str] = "memory.upgrade_mode"
    CHAT_CONTEXT_COUNT: Final[str] = "memory.chat_context_count"
    
    # LLM设置
    USE_LLM: Final[str] = "llm.use_llm"
    
    # 主动回复
    PROACTIVE_REPLY_ENABLE: Final[str] = "proactive_reply.enable"
    PROACTIVE_REPLY_MAX_DAILY: Final[str] = "proactive_reply.max_daily"
    PROACTIVE_REPLY_GROUP_WHITELIST_MODE: Final[str] = "proactive_reply.group_whitelist_mode"
    
    # 图片分析
    IMAGE_ANALYSIS_ENABLE: Final[str] = "image_analysis.enable"
    IMAGE_ANALYSIS_MODE: Final[str] = "image_analysis.mode"
    IMAGE_ANALYSIS_MAX_IMAGES: Final[str] = "image_analysis.max_images"
    IMAGE_ANALYSIS_DAILY_BUDGET: Final[str] = "image_analysis.daily_budget"
    IMAGE_ANALYSIS_SESSION_BUDGET: Final[str] = "image_analysis.session_budget"
    IMAGE_ANALYSIS_REQUIRE_CONTEXT: Final[str] = "image_analysis.require_context"


class SessionScope:
    """会话作用域"""
    PRIVATE: Final[str] = "private"
    GROUP_SHARED: Final[str] = "group_shared"
    GROUP_PRIVATE: Final[str] = "group_private"


class DeleteScope:
    """删除范围"""
    SHARED: Final[str] = "shared"
    PRIVATE: Final[str] = "private"
    ALL: Final[str] = "all"


class MessageTypeIndicators:
    """消息类型指示器"""
    WORKING_LAYER: Final[str] = "working"


class KVStoreKeys:
    """KV存储键"""
    SESSIONS: Final[str] = "sessions"
    LIFECYCLE_STATE: Final[str] = "lifecycle_state"
    BATCH_QUEUES: Final[str] = "batch_queues"
    CHAT_HISTORY: Final[str] = "chat_history"
    LAST_SAVE_PREFIX: Final[str] = "last_save_{user_id}_{group_id}"
    PROACTIVE_REPLY_WHITELIST: Final[str] = "proactive_reply_whitelist"
    MEMBER_IDENTITY: Final[str] = "member_identity"


class ErrorMessages:
    """错误消息模板"""
    EMPTY_CONTENT: Final[str] = "请输入要保存的内容"
    EMPTY_QUERY: Final[str] = "请输入搜索内容"
    PRIVATE_ONLY: Final[str] = "此命令仅限私聊使用"
    GROUP_ONLY: Final[str] = "此命令仅限群聊使用"
    ADMIN_REQUIRED: Final[str] = "权限不足，仅管理员可以执行此操作"
    GROUP_ADMIN_REQUIRED: Final[str] = "权限不足，仅管理员可以删除群聊记忆"
    DELETE_CONFIRM_REQUIRED: Final[str] = "警告：此操作将删除所有记忆！\n请使用 '/memory_delete_all confirm' 确认操作"
    INVALID_SCOPE_PARAM: Final[str] = "参数错误，请使用: shared, private 或 all"
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


# 分隔符
class Separators:
    """分隔符"""
    SESSION_KEY: Final[str] = ":"
    KV_KEY: Final[str] = "_"
