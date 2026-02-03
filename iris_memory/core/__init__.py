"""Core module for iris memory"""
from iris_memory.core.defaults import DEFAULTS, get_default, get_defaults_dict
from iris_memory.core.config_manager import (
    ConfigManager, 
    get_config_manager, 
    init_config_manager
)

__all__ = [
    'DEFAULTS',
    'get_default',
    'get_defaults_dict',
    'ConfigManager',
    'get_config_manager',
    'init_config_manager',
]