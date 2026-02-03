"""
配置管理器 - 统一配置访问接口

负责：
1. 合并用户配置和默认配置
2. 提供简化的配置访问API
3. 配置键映射（新旧格式兼容）
"""

import threading
from typing import Any, Dict, Optional
from iris_memory.core.defaults import DEFAULTS, get_default


# 配置键映射：简化键 -> (默认配置区块, 默认配置键)
# 用于兼容旧配置格式和简化新配置
CONFIG_KEY_MAPPING = {
    # 基础功能
    "basic.enable_memory": ("memory", "auto_capture", True),
    "basic.enable_inject": ("llm_integration", "enable_inject", True),
    "basic.enable_emotion": ("emotion", "enable_emotion", True),
    
    # LLM处理
    "llm_processing.use_llm": ("message_processing", "use_llm_for_processing", False),
    "llm_processing.max_context_memories": ("llm_integration", "max_context_memories", 3),
    
    # 主动回复
    "proactive_reply.enable": ("proactive_reply", "enable", False),
    "proactive_reply.max_daily": ("proactive_reply", "max_daily_replies", 20),
    
    # 嵌入配置
    "embedding.strategy": ("embedding", "embedding_strategy", "auto"),
    "embedding.model": ("embedding", "embedding_model", "BAAI/bge-small-zh-v1.5"),
    "embedding.dimension": ("embedding", "embedding_dimension", 512),
    
    # 日志
    "log_level": ("log", "level", "INFO"),
    
    # 高级设置
    "advanced.max_working_memory": ("memory", "max_working_memory", 10),
    "advanced.rif_threshold": ("memory", "rif_threshold", 0.4),
    "advanced.token_budget": ("llm_integration", "token_budget", 512),
    "advanced.session_timeout_hours": ("session", "session_timeout", 24),
    "advanced.upgrade_mode": ("memory", "upgrade_mode", "rule"),
}

# 旧配置键到新配置键的映射（向后兼容）
LEGACY_KEY_MAPPING = {
    "memory_config.auto_capture": "basic.enable_memory",
    "memory_config.max_working_memory": "advanced.max_working_memory",
    "memory_config.rif_threshold": "advanced.rif_threshold",
    "memory_config.upgrade_mode": "advanced.upgrade_mode",
    "memory_config.session_timeout": "advanced.session_timeout_hours",
    "memory_config.session_cleanup_interval": None,  # 已移除，使用默认值
    "memory_config.session_inactive_timeout": None,
    "memory_config.llm_upgrade_batch_size": None,
    "memory_config.llm_upgrade_threshold": None,
    
    "cache_config.embedding_cache_size": None,
    "cache_config.embedding_cache_strategy": None,
    "cache_config.max_sessions": None,
    "cache_config.working_cache_ttl": None,
    "cache_config.compression_max_length": None,
    
    "chroma_config.embedding_strategy": "embedding.strategy",
    "chroma_config.embedding_model": "embedding.model",
    "chroma_config.embedding_dimension": "embedding.dimension",
    "chroma_config.collection_name": None,
    "chroma_config.auto_detect_dimension": None,
    
    "emotion_config.enable_emotion": "basic.enable_emotion",
    "emotion_config.emotion_model": None,
    
    "llm_integration.enable_inject": "basic.enable_inject",
    "llm_integration.max_context_memories": "llm_processing.max_context_memories",
    "llm_integration.token_budget": "advanced.token_budget",
    "llm_integration.enable_token_budget": None,
    "llm_integration.injection_mode": None,
    "llm_integration.coordination_strategy": None,
    "llm_integration.enable_time_aware": None,
    "llm_integration.enable_emotion_aware": None,
    
    "log_config.level": "log_level",
    "log_config.console_output": None,
    "log_config.file_output": None,
    "log_config.max_file_size": None,
    "log_config.backup_count": None,
    
    "message_processing.enable_batch_processing": None,
    "message_processing.use_llm_for_processing": "llm_processing.use_llm",
    "message_processing.llm_processing_mode": None,
    "message_processing.batch_threshold_count": None,
    "message_processing.batch_threshold_interval": None,
    "message_processing.batch_processing_mode": None,
    "message_processing.immediate_trigger_confidence": None,
    "message_processing.immediate_emotion_intensity": None,
    
    "proactive_reply.enable": "proactive_reply.enable",
    "proactive_reply.cooldown_seconds": None,
    "proactive_reply.max_daily_replies": "proactive_reply.max_daily",
    "proactive_reply.max_reply_tokens": None,
    "proactive_reply.reply_temperature": None,
}


class ConfigManager:
    """配置管理器
    
    统一管理用户配置和默认配置的访问。
    支持新旧配置格式的兼容。
    """
    
    def __init__(self, user_config: Any = None):
        """初始化配置管理器
        
        Args:
            user_config: AstrBot用户配置对象
        """
        self._user_config = user_config
        self._cache: Dict[str, Any] = {}
        
    def set_user_config(self, config: Any):
        """设置用户配置"""
        self._user_config = config
        self._cache.clear()
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        支持三种格式：
        1. 新格式: "basic.enable_memory"
        2. 旧格式: "memory_config.auto_capture"
        3. 直接访问默认配置: "memory.max_working_memory"
        
        Args:
            key: 配置键
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
        # 1. 尝试直接从用户配置获取（支持新格式）
        user_value = self._get_from_user_config(key)
        if user_value is not None:
            return user_value
            
        # 2. 检查是否是旧格式，如果是则映射到新格式
        if key in LEGACY_KEY_MAPPING:
            new_key = LEGACY_KEY_MAPPING[key]
            if new_key is None:
                # 此配置已移除，使用默认值
                return self._get_default_for_legacy_key(key, default)
            return self.get(new_key, default)
            
        # 3. 检查新格式映射
        if key in CONFIG_KEY_MAPPING:
            section, attr, builtin_default = CONFIG_KEY_MAPPING[key]
            return self._get_default_value(section, attr, 
                                          default if default is not None else builtin_default)
        
        # 4. 尝试直接访问默认配置（格式：section.key）
        if '.' in key:
            parts = key.split('.', 1)
            if len(parts) == 2:
                section, attr = parts
                default_val = get_default(section, attr, default)
                if default_val is not None:
                    return default_val
                    
        # 5. 返回默认值
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
    
    def _get_default_for_legacy_key(self, legacy_key: str, default: Any = None) -> Any:
        """为已移除的旧配置键返回合适的默认值"""
        # 定义一些已移除配置的内置默认值
        legacy_defaults = {
            "memory_config.session_cleanup_interval": 3600,
            "memory_config.session_inactive_timeout": 1800,
            "memory_config.llm_upgrade_batch_size": 5,
            "memory_config.llm_upgrade_threshold": 0.7,
            "cache_config.embedding_cache_size": 1000,
            "cache_config.embedding_cache_strategy": "lru",
            "cache_config.max_sessions": 100,
            "cache_config.working_cache_ttl": 86400,
            "cache_config.compression_max_length": 200,
            "chroma_config.collection_name": "iris_memory",
            "chroma_config.auto_detect_dimension": True,
            "emotion_config.emotion_model": "builtin",
            "llm_integration.enable_token_budget": True,
            "llm_integration.injection_mode": "suffix",
            "llm_integration.coordination_strategy": "hybrid",
            "llm_integration.enable_time_aware": True,
            "llm_integration.enable_emotion_aware": True,
            "log_config.console_output": True,
            "log_config.file_output": True,
            "log_config.max_file_size": 10,
            "log_config.backup_count": 5,
            "message_processing.enable_batch_processing": True,
            "message_processing.llm_processing_mode": "hybrid",
            "message_processing.batch_threshold_count": 50,
            "message_processing.batch_threshold_interval": 300,
            "message_processing.batch_processing_mode": "hybrid",
            "message_processing.immediate_trigger_confidence": 0.8,
            "message_processing.immediate_emotion_intensity": 0.7,
            "message_processing.llm_max_tokens_for_summary": 200,
            "proactive_reply.cooldown_seconds": 60,
            "proactive_reply.max_reply_tokens": 150,
            "proactive_reply.reply_temperature": 0.7,
        }
        return legacy_defaults.get(legacy_key, default)
    
    # 便捷访问方法
    @property
    def enable_memory(self) -> bool:
        return self.get("basic.enable_memory", True)
    
    @property
    def enable_inject(self) -> bool:
        return self.get("basic.enable_inject", True)
    
    @property
    def enable_emotion(self) -> bool:
        return self.get("basic.enable_emotion", True)
    
    @property
    def use_llm(self) -> bool:
        return self.get("llm_processing.use_llm", False)
    
    @property
    def max_context_memories(self) -> int:
        return self.get("llm_processing.max_context_memories", 3)
    
    @property
    def proactive_reply_enabled(self) -> bool:
        return self.get("proactive_reply.enable", False)
    
    @property
    def log_level(self) -> str:
        return self.get("log_level", "INFO")
    
    @property
    def max_working_memory(self) -> int:
        return self.get("advanced.max_working_memory", 10)
    
    @property
    def rif_threshold(self) -> float:
        return self.get("advanced.rif_threshold", 0.4)
    
    @property
    def token_budget(self) -> int:
        return self.get("advanced.token_budget", 512)
    
    @property
    def session_timeout(self) -> int:
        """会话超时（秒）"""
        hours = self.get("advanced.session_timeout_hours", 24)
        return hours * 3600
    
    @property
    def upgrade_mode(self) -> str:
        return self.get("advanced.upgrade_mode", "rule")
    
    @property
    def embedding_strategy(self) -> str:
        return self.get("embedding.strategy", "auto")
    
    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "BAAI/bge-small-zh-v1.5")
    
    @property
    def embedding_dimension(self) -> int:
        return self.get("embedding.dimension", 512)


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器（线程安全）"""
    global _config_manager
    if _config_manager is None:
        with _config_manager_lock:
            # 双重检查锁定
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
