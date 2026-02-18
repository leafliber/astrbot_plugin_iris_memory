"""Core module for iris memory"""
from iris_memory.core.defaults import (
    DEFAULTS, get_default, get_defaults_dict,
    GroupActivityLevel, ActivityBasedPresets, ACTIVITY_PRESETS,
)
from iris_memory.core.config_manager import (
    ConfigManager, 
    get_config_manager, 
    init_config_manager
)
from iris_memory.core.activity_config import (
    GroupActivityTracker,
    ActivityAwareConfigProvider,
)
from iris_memory.core.config_registry import (
    ConfigDefinition,
    CONFIG_REGISTRY,
    get_registry_default,
    get_registry_mapping,
)
from iris_memory.core.service_container import ServiceContainer

__all__ = [
    'DEFAULTS',
    'get_default',
    'get_defaults_dict',
    'GroupActivityLevel',
    'ActivityBasedPresets',
    'ACTIVITY_PRESETS',
    'ConfigManager',
    'get_config_manager',
    'init_config_manager',
    'GroupActivityTracker',
    'ActivityAwareConfigProvider',
    'ConfigDefinition',
    'CONFIG_REGISTRY',
    'get_registry_default',
    'get_registry_mapping',
    'ServiceContainer',
]