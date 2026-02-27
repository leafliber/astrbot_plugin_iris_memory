"""
配置管理器 - 统一配置访问接口

负责：
1. 合并用户配置和默认配置
2. 提供简化的配置访问API
"""

import threading
import time
from typing import Any, Dict, Optional, Tuple
from iris_memory.core.defaults import DEFAULTS, get_default
from iris_memory.core.config_registry import CONFIG_REGISTRY, get_registry_mapping
from iris_memory.core.activity_config import (
    ActivityAwareConfigProvider, GroupActivityTracker
)
from iris_memory.core.provider_utils import normalize_provider_id


# CONFIG_KEY_MAPPING 由 CONFIG_REGISTRY 自动生成
CONFIG_KEY_MAPPING: Dict[str, tuple] = {
    key: (defn.section, defn.attr, defn.default)
    for key, defn in CONFIG_REGISTRY.items()
}


class ConfigManager:
    """配置管理器
    
    统一管理用户配置和默认配置的访问。
    线程安全：对 _cache 和 _user_config 的读写均通过锁保护。
    """
    
    # 默认配置缓存 TTL（秒），可通过构造参数覆盖
    DEFAULT_CACHE_TTL: float = 10.0

    def __init__(self, user_config: Any = None, *, cache_ttl: Optional[float] = None):
        """初始化配置管理器
        
        Args:
            user_config: AstrBot用户配置对象
            cache_ttl: 配置缓存 TTL（秒），None 使用默认值
        """
        self._lock = threading.Lock()
        self._user_config = user_config
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expire_time)
        self._cache_ttl: float = cache_ttl if cache_ttl is not None else self.DEFAULT_CACHE_TTL
        
        # 场景自适应组件（延迟初始化）
        self._activity_provider: Optional[ActivityAwareConfigProvider] = None
        
    def set_user_config(self, config: Any):
        """设置用户配置（线程安全）"""
        with self._lock:
            self._user_config = config
            self._cache.clear()
    
    def invalidate_cache(self, key: Optional[str] = None):
        """主动失效配置缓存
        
        当外部直接修改了配置对象的属性时调用此方法。
        
        Args:
            key: 特定的配置键，为 None 则清除所有缓存
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)
    
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
        """获取配置值（线程安全，带 TTL 缓存）
        
        缓存条目在 _CACHE_TTL 秒后自动过期，确保外部修改能被感知。
        
        Args:
            key: 配置键，格式如 "basic.enable_memory"
            default: 默认值（如果未指定，使用内置默认值）
            
        Returns:
            配置值
        """
        now = time.monotonic()
        with self._lock:
            # 检查缓存（含 TTL 检查）
            if key in self._cache:
                cached_value, expire_at = self._cache[key]
                if now < expire_at:
                    return cached_value
                # 缓存过期，移除并重新获取
                del self._cache[key]
                
            value = self._get_value(key, default)
            self._cache[key] = (value, now + self._cache_ttl)
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
        return self.get("logging.log_level", "INFO")
    
    # 记忆设置
    @property
    def max_context_memories(self) -> int:
        return self.get("memory.max_context_memories", 3)
    
    @property
    def token_budget(self) -> int:
        return self.get("llm_integration.token_budget", DEFAULTS.llm_integration.token_budget)
    
    @property
    def max_working_memory(self) -> int:
        return self.get("memory.max_working_memory", DEFAULTS.memory.max_working_memory)
    
    @property
    def chat_context_count(self) -> int:
        return self.get("advanced.chat_context_count", DEFAULTS.llm_integration.chat_context_count)
    
    @property
    def rif_threshold(self) -> float:
        return self.get("memory.rif_threshold", DEFAULTS.memory.rif_threshold)
    
    @property
    def upgrade_mode(self) -> str:
        return self.get("memory.upgrade_mode", "rule")

    # 主动回复
    @property
    def proactive_reply_enabled(self) -> bool:
        return self.get("proactive_reply.enable", False)
    
    @property
    def proactive_reply_max_daily(self) -> int:
        return self.get("proactive_reply.max_daily_replies", DEFAULTS.proactive_reply.max_daily_replies)
    
    @property
    def proactive_reply_group_whitelist_mode(self) -> bool:
        return self.get("proactive_reply.group_whitelist_mode", False)
    
    @property
    def smart_boost_enabled(self) -> bool:
        """智能增强是否启用（需同时满足：配置开启 + proactive_mode 为 llm 或 hybrid）"""
        enabled = self.get("proactive_reply.smart_boost", DEFAULTS.proactive_reply.smart_boost_enabled)
        if not enabled:
            return False
        mode = self.proactive_mode
        return mode in ("llm", "hybrid")
    
    @property
    def smart_boost_window_seconds(self) -> int:
        return self.get("proactive_reply.smart_boost_window_seconds",
                        DEFAULTS.proactive_reply.smart_boost_window_seconds)
    
    @property
    def smart_boost_score_multiplier(self) -> float:
        return self.get("proactive_reply.smart_boost_score_multiplier",
                        DEFAULTS.proactive_reply.smart_boost_score_multiplier)
    
    @property
    def smart_boost_reply_threshold(self) -> float:
        return self.get("proactive_reply.smart_boost_reply_threshold",
                        DEFAULTS.proactive_reply.smart_boost_reply_threshold)
    
    # 图片分析
    @property
    def image_analysis_enabled(self) -> bool:
        return self.get("image_analysis.enable", True)
    
    @property
    def image_analysis_mode(self) -> str:
        return self.get("image_analysis.mode", "auto")
    
    @property
    def image_analysis_max_images(self) -> int:
        return self.get("image_analysis.max_images_per_message", DEFAULTS.image_analysis.max_images_per_message)
    
    @property
    def image_analysis_daily_budget(self) -> int:
        return self.get("image_analysis.daily_budget", DEFAULTS.image_analysis.daily_analysis_budget)
    
    @property
    def image_analysis_session_budget(self) -> int:
        return self.get("image_analysis.session_analysis_budget", DEFAULTS.image_analysis.session_analysis_budget)
    
    @property
    def image_analysis_require_context(self) -> bool:
        return self.get("image_analysis.require_context_relevance", DEFAULTS.image_analysis.require_context_relevance)
    
    @property
    def image_analysis_provider_id(self) -> str:
        return normalize_provider_id(self.get("llm_providers.image_analysis_provider_id", ""))
    
    # LLM增强处理
    @property
    def use_llm(self) -> bool:
        return self.get("memory.use_llm", False)
    
    @property
    def llm_provider_id(self) -> str:
        return normalize_provider_id(self.get("llm_providers.memory_provider_id", ""))
    
    # 批量处理配置
    @property
    def batch_threshold_count(self) -> int:
        return self.get("message_processing.batch_threshold_count", DEFAULTS.message_processing.batch_threshold_count)
    
    @property
    def short_message_threshold(self) -> int:
        return self.get("message_processing.short_message_threshold", DEFAULTS.message_processing.short_message_threshold)
    
    @property
    def merge_time_window(self) -> int:
        return self.get("message_processing.merge_time_window", DEFAULTS.message_processing.merge_time_window)
    
    @property
    def max_merge_count(self) -> int:
        return self.get("message_processing.max_merge_count", DEFAULTS.message_processing.max_merge_count)
    
    # 会话管理
    @property
    def session_timeout(self) -> int:
        return self.get("session.session_timeout", DEFAULTS.session.session_timeout)
    
    # 嵌入配置
    @property
    def embedding_source(self) -> str:
        """嵌入源选择：auto / astrbot / local"""
        return self.get("embedding.source", DEFAULTS.embedding.source)
    
    @property
    def embedding_astrbot_provider_id(self) -> str:
        """指定的 AstrBot embedding provider ID"""
        return normalize_provider_id(self.get("embedding.astrbot_provider_id", DEFAULTS.embedding.astrbot_provider_id))
    
    @property
    def embedding_local_model(self) -> str:
        """本地嵌入模型名称"""
        return self.get("embedding.local_model", DEFAULTS.embedding.local_model)

    @property
    def embedding_local_dimension(self) -> int:
        """本地嵌入模型维度"""
        return self.get("embedding.local_dimension", DEFAULTS.embedding.local_dimension)
    
    # 画像配置
    @property
    def persona_auto_update(self) -> bool:
        return self.get("persona.enable_auto_update", DEFAULTS.persona.enable_auto_update)
    
    @property
    def persona_injection_enabled(self) -> bool:
        return self.get("persona.enable_persona_injection", DEFAULTS.persona.enable_persona_injection)
    
    @property
    def persona_max_change_log(self) -> int:
        return self.get("persona.max_change_log", DEFAULTS.persona.max_change_log)
    
    @property
    def persona_snapshot_interval(self) -> int:
        return self.get("persona.snapshot_interval", DEFAULTS.persona.snapshot_interval)

    @property
    def persona_extraction_mode(self) -> str:
        return self.get("persona.extraction_mode", "rule")

    @property
    def persona_llm_provider(self) -> str:
        provider_id = normalize_provider_id(self.get("llm_providers.persona_provider_id", ""))
        return provider_id or "default"

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

    # 画像批量处理配置
    @property
    def persona_batch_enabled(self) -> bool:
        return self.get("persona.batch_enabled", DEFAULTS.persona.batch_enabled)
    
    @property
    def persona_batch_threshold(self) -> int:
        return self.get("persona.batch_threshold", DEFAULTS.persona.batch_threshold)
    
    @property
    def persona_batch_flush_interval(self) -> int:
        return self.get("persona.batch_flush_interval", DEFAULTS.persona.batch_flush_interval)
    
    @property
    def persona_batch_max_size(self) -> int:
        return self.get("persona.batch_max_size", DEFAULTS.persona.batch_max_size)
    
    # LLM智能增强配置
    @property
    def llm_enhanced_provider_id(self) -> str:
        return normalize_provider_id(self.get("llm_providers.enhanced_provider_id", ""))
    
    @property
    def sensitivity_mode(self) -> str:
        return self.get("llm_enhanced.sensitivity_mode", "rule")
    
    @property
    def trigger_mode(self) -> str:
        return self.get("llm_enhanced.trigger_mode", "rule")
    
    @property
    def emotion_mode(self) -> str:
        return self.get("llm_enhanced.emotion_mode", "rule")
    
    @property
    def proactive_mode(self) -> str:
        return self.get("llm_enhanced.proactive_mode", "rule")
    
    @property
    def conflict_mode(self) -> str:
        return self.get("llm_enhanced.conflict_mode", "rule")
    
    @property
    def retrieval_mode(self) -> str:
        return self.get("llm_enhanced.retrieval_mode", "rule")
    
    @property
    def llm_enhanced_enabled(self) -> bool:
        """判断是否有任何模块启用了LLM增强"""
        modes = [
            self.sensitivity_mode,
            self.trigger_mode,
            self.emotion_mode,
            self.proactive_mode,
            self.conflict_mode,
            self.retrieval_mode,
        ]
        return any(mode in ("llm", "hybrid") for mode in modes)

    # 知识图谱配置
    @property
    def knowledge_graph_provider_id(self) -> str:
        return normalize_provider_id(self.get("llm_providers.knowledge_graph_provider_id", ""))
    
    # ========== 批量处理动态配置 ==========
    
    def _with_group_override(self, group_id: Optional[str], key: str, fallback_key: str, default: Any) -> Any:
        """通用群级自适应配置查询。

        优先返回群级配置 ``get_group_config(group_id, key)``，
        不存在时回退到全局 ``self.get(fallback_key, default)``。
        """
        val = self.get_group_config(group_id, key)
        return val if val is not None else self.get(fallback_key, default)
    
    def get_batch_threshold_count(self, group_id: Optional[str] = None) -> int:
        """获取批量处理阈值（群级自适应）"""
        return self._with_group_override(group_id, "batch_threshold_count",
                                         "message_processing.batch_threshold_count",
                                         DEFAULTS.message_processing.batch_threshold_count)
    
    def get_batch_threshold_interval(self, group_id: Optional[str] = None) -> int:
        """获取批量处理间隔（群级自适应）"""
        return self._with_group_override(group_id, "batch_threshold_interval",
                                         "message_processing.batch_threshold_interval",
                                         DEFAULTS.message_processing.batch_threshold_interval)
    
    def get_chat_context_count(self, group_id: Optional[str] = None) -> int:
        """获取聊天上下文数量（群级自适应）"""
        return self._with_group_override(group_id, "chat_context_count",
                                         "advanced.chat_context_count",
                                         DEFAULTS.llm_integration.chat_context_count)
    
    def get_cooldown_seconds(self, group_id: Optional[str] = None) -> int:
        """获取主动回复冷却时间（群级自适应）"""
        return self._with_group_override(group_id, "cooldown_seconds",
                                         "proactive_reply.cooldown_seconds",
                                         DEFAULTS.proactive_reply.cooldown_seconds)
    
    def get_max_daily_replies(self, group_id: Optional[str] = None) -> int:
        """获取每日最大回复次数（群级自适应）"""
        return self._with_group_override(group_id, "max_daily_replies",
                                         "proactive_reply.max_daily_replies",
                                         DEFAULTS.proactive_reply.max_daily_replies)
    
    def get_daily_analysis_budget(self, group_id: Optional[str] = None) -> int:
        """获取每日分析预算（群级自适应）"""
        return self._with_group_override(group_id, "daily_analysis_budget",
                                         "image_analysis.daily_analysis_budget",
                                         DEFAULTS.image_analysis.daily_analysis_budget)
    
    def get_reply_temperature(self, group_id: Optional[str] = None) -> float:
        """获取回复温度（群级自适应）"""
        return self._with_group_override(group_id, "reply_temperature",
                                         "proactive_reply.reply_temperature",
                                         DEFAULTS.proactive_reply.reply_temperature)
    
    # Web UI 配置
    @property
    def web_ui_enabled(self) -> bool:
        """是否启用Web管理界面"""
        return self.get("web_ui.enable", DEFAULTS.web_ui.enable)
    
    @property
    def web_ui_port(self) -> int:
        """Web服务端口"""
        return self.get("web_ui.port", DEFAULTS.web_ui.port)
    
    @property
    def web_ui_access_key(self) -> str:
        """Web访问密钥"""
        return self.get("web_ui.access_key", DEFAULTS.web_ui.access_key)
    
    @property
    def web_ui_host(self) -> str:
        """Web服务监听地址"""
        return self.get("web_ui.host", DEFAULTS.web_ui.host)


# 全局配置管理器 — 通过 ServiceContainer 管理
# 保留原有公共 API (get_config_manager / init_config_manager / reset_config_manager)

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器（线程安全）"""
    from iris_memory.core.service_container import ServiceContainer
    container = ServiceContainer.instance()
    mgr = container.get("config_manager")
    if mgr is None:
        mgr = ConfigManager()
        container.register("config_manager", mgr)
    return mgr


def init_config_manager(user_config: Any) -> ConfigManager:
    """初始化全局配置管理器（线程安全）
    
    Args:
        user_config: AstrBot用户配置对象
        
    Returns:
        配置管理器实例
    """
    from iris_memory.core.service_container import ServiceContainer
    container = ServiceContainer.instance()
    mgr = ConfigManager(user_config)
    container.register("config_manager", mgr)
    return mgr


def reset_config_manager():
    """重置配置管理器（主要用于测试）"""
    from iris_memory.core.service_container import ServiceContainer
    container = ServiceContainer.instance()
    container.unregister("config_manager")
