"""Core module for iris memory"""
from iris_memory.core.defaults import (
    DEFAULTS, get_default, get_defaults_dict,
    GroupActivityLevel, ActivityBasedPresets, ACTIVITY_PRESETS,
)
from iris_memory.core.config_manager import ConfigManager
from iris_memory.core.activity_config import (
    GroupActivityTracker,
    ActivityAwareConfigProvider,
)
from iris_memory.core.service_container import ServiceContainer
from iris_memory.core.upgrade_evaluator import UpgradeEvaluator, UpgradeMode
from iris_memory.core.provider_utils import (
    normalize_provider_id,
    extract_provider_id,
    get_provider_by_id,
    get_default_provider,
)
from iris_memory.core.detection import (
    BaseDetectionResult,
    DetectionMode,
    LLMEnhancedDetector,
)

# 新配置系统（推荐）
from iris_memory.config import (
    ConfigStore,
    get_store,
    init_store,
    reset_store,
    SCHEMA,
    AccessLevel,
    config_events,
)

__all__ = [
    'DEFAULTS',
    'get_default',
    'get_defaults_dict',
    'GroupActivityLevel',
    'ActivityBasedPresets',
    'ACTIVITY_PRESETS',
    'ConfigManager',
    'GroupActivityTracker',
    'ActivityAwareConfigProvider',
    'ServiceContainer',
    'UpgradeEvaluator',
    'UpgradeMode',
    'normalize_provider_id',
    'extract_provider_id',
    'get_provider_by_id',
    'get_default_provider',
    'BaseDetectionResult',
    'DetectionMode',
    'LLMEnhancedDetector',
    # 新配置系统
    'ConfigStore',
    'get_store',
    'init_store',
    'reset_store',
    'SCHEMA',
    'AccessLevel',
    'config_events',
]