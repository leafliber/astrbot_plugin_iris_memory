"""
配置管理器 - 统一配置访问接口

负责：
1. 合并用户配置和默认配置
2. 提供简化的配置访问API
"""

import threading
from typing import Any, Dict, Optional
from iris_memory.core.defaults import DEFAULTS, get_default
from iris_memory.core.activity_config import (
    ActivityAwareConfigProvider, GroupActivityTracker
)


# 配置键映射：简化键 -> (默认配置区块, 默认配置键, 内置默认值)
CONFIG_KEY_MAPPING = {
    # 基础功能
    "basic.enable_memory": ("memory", "auto_capture", True),
    "basic.enable_inject": ("llm_integration", "enable_inject", True),
    "basic.log_level": ("log", "level", "INFO"),
    
    # 记忆设置
    "memory.max_context_memories": ("llm_integration", "max_context_memories", 3),
    "memory.max_working_memory": ("memory", "max_working_memory", 10),
    "memory.upgrade_mode": ("memory", "upgrade_mode", "rule"),
    
    # LLM设置
    "llm.use_llm": ("message_processing", "use_llm_for_processing", False),
    "llm.provider_id": ("llm_integration", "provider_id", ""),
    
    # 主动回复
    "proactive_reply.enable": ("proactive_reply", "enable", False),
    "proactive_reply.group_whitelist_mode": ("proactive_reply", "group_whitelist_mode", False),
    
    # 图片分析
    "image_analysis.enable": ("image_analysis", "enable_image_analysis", True),
    "image_analysis.mode": ("image_analysis", "analysis_mode", "auto"),
    "image_analysis.daily_budget": ("image_analysis", "daily_analysis_budget", 100),
    "image_analysis.provider_id": ("image_analysis", "provider_id", ""),
    
    # 场景自适应
    "activity_adaptive.enable": ("activity_adaptive", "enable_activity_adaptive", True),
    
    # 画像提取
    "persona.extraction_mode": ("persona", "extraction_mode", "rule"),
    "persona.llm_provider": ("persona", "llm_provider", "default"),
    "persona.enable_interest_extraction": ("persona", "enable_interest_extraction", True),
    "persona.enable_style_extraction": ("persona", "enable_style_extraction", True),
    "persona.enable_preference_extraction": ("persona", "enable_preference_extraction", True),
    "persona.llm_max_tokens": ("persona", "llm_max_tokens", 300),
    "persona.llm_daily_limit": ("persona", "llm_daily_limit", 50),
    "persona.fallback_to_rule": ("persona", "fallback_to_rule", True),
    
    # 高级参数（群聊自适应覆盖）
    # 这些配置在群聊启用自适应时会被自动覆盖
    "advanced.chat_context_count": ("llm_integration", "chat_context_count", 10),
    "advanced.cooldown_seconds": ("proactive_reply", "cooldown_seconds", 60),
    "advanced.max_daily_replies": ("proactive_reply", "max_daily_replies", 20),
    "advanced.reply_temperature": ("proactive_reply", "reply_temperature", 0.7),
    "advanced.batch_threshold_count": ("message_processing", "batch_threshold_count", 20),
    "advanced.batch_threshold_interval": ("message_processing", "batch_threshold_interval", 300),
    "advanced.daily_analysis_budget": ("image_analysis", "daily_analysis_budget", 100),
}


class ConfigManager:
    """配置管理器
    
    统一管理用户配置和默认配置的访问。
    """
    
    def __init__(self, user_config: Any = None):
        """初始化配置管理器
        
        Args:
            user_config: AstrBot用户配置对象
        """
        self._user_config = user_config
        self._cache: Dict[str, Any] = {}
        
        # 场景自适应组件（延迟初始化）
        self._activity_provider: Optional[ActivityAwareConfigProvider] = None
        
    def set_user_config(self, config: Any):
        """设置用户配置"""
        self._user_config = config
        self._cache.clear()
    
    # ========== 场景自适应配置 ==========
    
    def init_activity_provider(
        self,
        tracker: GroupActivityTracker,
        enabled: Optional[bool] = None,
    ) -> ActivityAwareConfigProvider:
        """初始化活跃度感知配置提供者
        
        Args:
            tracker: 群活跃度追踪器实例
            enabled: 是否启用（None 则读取配置）
            
        Returns:
            ActivityAwareConfigProvider 实例
        """
        if enabled is None:
            enabled = self.get("activity_adaptive.enable", True)
        self._activity_provider = ActivityAwareConfigProvider(
            tracker=tracker,
            enabled=enabled,
        )
        return self._activity_provider
    
    @property
    def activity_provider(self) -> Optional[ActivityAwareConfigProvider]:
        """获取活跃度感知配置提供者"""
        return self._activity_provider
    
    def get_group_config(self, group_id: Optional[str], key: str) -> Any:
        """获取群级自适应配置值
        
        如果启用了活跃度自适应，返回根据群活跃度调整的值；
        否则返回用户在「高级参数」中设置的值或全局默认值。
        
        Args:
            group_id: 群 ID（None 则返回默认）
            key: 配置键，如 "cooldown_seconds"
            
        Returns:
            配置值
        """
        # 群聊 + 启用自适应 → 返回活跃度调整值
        if self._activity_provider and self._activity_provider.enabled and group_id:
            return self._activity_provider.get_config(group_id, key)
        
        # 私聊 or 禁用自适应 → 读取用户配置的 advanced.* 或默认值
        advanced_key = f"advanced.{key}"
        user_value = self._get_from_user_config(advanced_key)
        if user_value is not None:
            return user_value
        
        # 回退到内置默认值
        if self._activity_provider:
            return self._activity_provider._get_default(key)
        return None
    
    @property
    def enable_activity_adaptive(self) -> bool:
        return self.get("activity_adaptive.enable", True)
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，格式如 "basic.enable_memory"
            default: 默认值（如果未指定，使用内置默认值）
            
        Returns:
            配置值
        """
        # 检查缓存
        if key in self._cache:
            return self._cache[key]
            
        value = self._get_value(key, default)
        self._cache[key] = value
        return value
    
    def _get_value(self, key: str, default: Any = None) -> Any:
        """内部获取配置值"""
        # 1. 尝试直接从用户配置获取
        user_value = self._get_from_user_config(key)
        if user_value is not None:
            return user_value
            
        # 2. 检查配置键映射
        if key in CONFIG_KEY_MAPPING:
            section, attr, builtin_default = CONFIG_KEY_MAPPING[key]
            return self._get_default_value(section, attr, 
                                          default if default is not None else builtin_default)
        
        # 3. 尝试直接访问默认配置（格式：section.key）
        if '.' in key:
            section, attr = key.split('.', 1)
            default_val = get_default(section, attr, default)
            if default_val is not None:
                return default_val
                    
        # 4. 返回默认值
        return default
    
    def _get_from_user_config(self, key: str) -> Any:
        """从用户配置获取值"""
        if self._user_config is None:
            return None
            
        try:
            keys = key.split('.')
            value = self._user_config
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            return value if value is not None else None
        except (AttributeError, KeyError, TypeError):
            return None
    
    def _get_default_value(self, section: str, attr: str, fallback: Any = None) -> Any:
        """从默认配置获取值"""
        return get_default(section, attr, fallback)
    
    # 便捷访问方法 - 基础功能
    @property
    def enable_memory(self) -> bool:
        return self.get("basic.enable_memory", True)
    
    @property
    def enable_inject(self) -> bool:
        return self.get("basic.enable_inject", True)
    
    @property
    def log_level(self) -> str:
        return self.get("basic.log_level", "INFO")
    
    # 记忆设置
    @property
    def max_context_memories(self) -> int:
        return self.get("memory.max_context_memories", 3)
    
    @property
    def token_budget(self) -> int:
        return DEFAULTS.llm_integration.token_budget
    
    @property
    def max_working_memory(self) -> int:
        return self.get("memory.max_working_memory", 10)
    
    @property
    def chat_context_count(self) -> int:
        return self.get("advanced.chat_context_count", 10)
    
    @property
    def rif_threshold(self) -> float:
        return DEFAULTS.memory.rif_threshold
    
    @property
    def upgrade_mode(self) -> str:
        return self.get("memory.upgrade_mode", "rule")

    # 主动回复
    @property
    def proactive_reply_enabled(self) -> bool:
        return self.get("proactive_reply.enable", False)
    
    @property
    def proactive_reply_max_daily(self) -> int:
        return DEFAULTS.proactive_reply.max_daily_replies
    
    @property
    def proactive_reply_group_whitelist_mode(self) -> bool:
        return self.get("proactive_reply.group_whitelist_mode", False)
    
    # 图片分析
    @property
    def image_analysis_enabled(self) -> bool:
        return self.get("image_analysis.enable", True)
    
    @property
    def image_analysis_mode(self) -> str:
        return self.get("image_analysis.mode", "auto")
    
    @property
    def image_analysis_max_images(self) -> int:
        return DEFAULTS.image_analysis.max_images_per_message
    
    @property
    def image_analysis_daily_budget(self) -> int:
        return self.get("image_analysis.daily_budget", 100)
    
    @property
    def image_analysis_session_budget(self) -> int:
        return DEFAULTS.image_analysis.session_analysis_budget
    
    @property
    def image_analysis_require_context(self) -> bool:
        return DEFAULTS.image_analysis.require_context_relevance
    
    @property
    def image_analysis_provider_id(self) -> str:
        return self.get("image_analysis.provider_id", "")
    
    # LLM增强处理
    @property
    def use_llm(self) -> bool:
        return self.get("llm.use_llm", False)
    
    @property
    def llm_provider_id(self) -> str:
        return self.get("llm.provider_id", "")
    
    # 批量处理配置
    @property
    def batch_threshold_count(self) -> int:
        return DEFAULTS.message_processing.batch_threshold_count
    
    @property
    def short_message_threshold(self) -> int:
        return DEFAULTS.message_processing.short_message_threshold
    
    @property
    def merge_time_window(self) -> int:
        return DEFAULTS.message_processing.merge_time_window
    
    @property
    def max_merge_count(self) -> int:
        return DEFAULTS.message_processing.max_merge_count
    
    # 会话管理
    @property
    def session_timeout(self) -> int:
        return DEFAULTS.session.session_timeout
    
    # 嵌入配置
    @property
    def embedding_strategy(self) -> str:
        return DEFAULTS.embedding.embedding_strategy
    
    @property
    def embedding_model(self) -> str:
        return DEFAULTS.embedding.embedding_model
    
    @property
    def embedding_models(self) -> list:
        return DEFAULTS.embedding.embedding_models
    
    @property
    def embedding_dimension(self) -> int:
        return DEFAULTS.embedding.embedding_dimension
    
    # 画像配置
    @property
    def persona_auto_update(self) -> bool:
        return DEFAULTS.persona.enable_auto_update
    
    @property
    def persona_injection_enabled(self) -> bool:
        return DEFAULTS.persona.enable_persona_injection
    
    @property
    def persona_max_change_log(self) -> int:
        return DEFAULTS.persona.max_change_log
    
    @property
    def persona_snapshot_interval(self) -> int:
        return DEFAULTS.persona.snapshot_interval

    @property
    def persona_extraction_mode(self) -> str:
        return self.get("persona.extraction_mode", "rule")

    @property
    def persona_llm_provider(self) -> str:
        return self.get("persona.llm_provider", "default")

    @property
    def persona_enable_interest(self) -> bool:
        return self.get("persona.enable_interest_extraction", True)

    @property
    def persona_enable_style(self) -> bool:
        return self.get("persona.enable_style_extraction", True)

    @property
    def persona_enable_preference(self) -> bool:
        return self.get("persona.enable_preference_extraction", True)

    @property
    def persona_llm_max_tokens(self) -> int:
        return self.get("persona.llm_max_tokens", 300)

    @property
    def persona_llm_daily_limit(self) -> int:
        return self.get("persona.llm_daily_limit", 50)

    @property
    def persona_fallback_to_rule(self) -> bool:
        return self.get("persona.fallback_to_rule", True)
    
    # ========== 批量处理动态配置 ==========
    
    def get_batch_threshold_count(self, group_id: Optional[str] = None) -> int:
        """获取批量处理阈值（群级自适应）"""
        val = self.get_group_config(group_id, "batch_threshold_count")
        return val if val is not None else self.batch_threshold_count
    
    def get_batch_threshold_interval(self, group_id: Optional[str] = None) -> int:
        """获取批量处理间隔（群级自适应）"""
        val = self.get_group_config(group_id, "batch_threshold_interval")
        return val if val is not None else DEFAULTS.message_processing.batch_threshold_interval
    
    def get_chat_context_count(self, group_id: Optional[str] = None) -> int:
        """获取聊天上下文数量（群级自适应）"""
        val = self.get_group_config(group_id, "chat_context_count")
        return val if val is not None else self.chat_context_count
    
    def get_cooldown_seconds(self, group_id: Optional[str] = None) -> int:
        """获取主动回复冷却时间（群级自适应）"""
        val = self.get_group_config(group_id, "cooldown_seconds")
        return val if val is not None else DEFAULTS.proactive_reply.cooldown_seconds
    
    def get_max_daily_replies(self, group_id: Optional[str] = None) -> int:
        """获取每日最大回复次数（群级自适应）"""
        val = self.get_group_config(group_id, "max_daily_replies")
        return val if val is not None else DEFAULTS.proactive_reply.max_daily_replies
    
    def get_daily_analysis_budget(self, group_id: Optional[str] = None) -> int:
        """获取每日分析预算（群级自适应）"""
        val = self.get_group_config(group_id, "daily_analysis_budget")
        return val if val is not None else DEFAULTS.image_analysis.daily_analysis_budget
    
    def get_reply_temperature(self, group_id: Optional[str] = None) -> float:
        """获取回复温度（群级自适应）"""
        val = self.get_group_config(group_id, "reply_temperature")
        return val if val is not None else DEFAULTS.proactive_reply.reply_temperature


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器（线程安全）"""
    global _config_manager
    if _config_manager is None:
        with _config_manager_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
    return _config_manager


def init_config_manager(user_config: Any) -> ConfigManager:
    """初始化全局配置管理器（线程安全）
    
    Args:
        user_config: AstrBot用户配置对象
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    with _config_manager_lock:
        _config_manager = ConfigManager(user_config)
    return _config_manager


def reset_config_manager():
    """重置配置管理器（主要用于测试）"""
    global _config_manager
    with _config_manager_lock:
        _config_manager = None
