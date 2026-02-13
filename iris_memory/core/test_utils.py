"""
测试工具模块 - 提供测试专用的配置管理

使用示例：
    from iris_memory.core.test_utils import setup_test_config
    
    # 在测试fixture中
    @pytest.fixture(autouse=True)
    def setup_config():
        setup_test_config({
            'basic': {'enable_memory': True},
            'memory': {'max_working_memory': 10}
        })
"""

from typing import Dict, Any
from unittest.mock import Mock


def setup_test_config(config_dict: Dict[str, Any] = None):
    """设置测试配置
    
    Args:
        config_dict: 测试配置字典，格式如下：
            {
                'basic': {'enable_memory': True, 'log_level': 'DEBUG'},
                'memory': {'max_working_memory': 10},
                'proactive_reply': {'enable': False}
            }
    """
    from iris_memory.core.config_manager import init_config_manager
    
    mock_config = Mock()
    
    if config_dict:
        for section, values in config_dict.items():
            if isinstance(values, dict):
                setattr(mock_config, section, values)
            else:
                setattr(mock_config, section, values)
    
    init_config_manager(mock_config)


def setup_simple_test_config(**kwargs):
    """设置简单测试配置
    
    Args:
        **kwargs: 配置键值对，如：
            enable_memory=True,
            embedding_dimension=1024
    """
    from iris_memory.core.config_manager import init_config_manager
    
    mock_config = Mock()
    
    config_mapping = {
        'enable_memory': ('basic', 'enable_memory'),
        'enable_inject': ('basic', 'enable_inject'),
        'log_level': ('basic', 'log_level'),
        'embedding_model': ('embedding', 'embedding_model'),
        'embedding_dimension': ('embedding', 'embedding_dimension'),
        'collection_name': ('embedding', 'collection_name'),
        'auto_detect_dimension': ('embedding', 'auto_detect_dimension'),
        'max_working_memory': ('memory', 'max_working_memory'),
        'upgrade_mode': ('memory', 'upgrade_mode'),
    }
    
    grouped_config = {}
    for key, value in kwargs.items():
        if key in config_mapping:
            section, attr = config_mapping[key]
            if section not in grouped_config:
                grouped_config[section] = {}
            grouped_config[section][attr] = value
        else:
            print(f"Warning: Unknown config key '{key}' in test config")
    
    for section, values in grouped_config.items():
        setattr(mock_config, section, values)
    
    init_config_manager(mock_config)


def reset_config_manager():
    """重置配置管理器（测试清理用）"""
    from iris_memory.core.config_manager import reset_config_manager as _reset
    _reset()


class TestConfigContext:
    """测试配置上下文管理器
    
    用法：
        with TestConfigContext(enable_memory=True):
            # 在此上下文中使用特定配置
            ...
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.original_manager = None
    
    def __enter__(self):
        from iris_memory.core import config_manager
        self.original_manager = getattr(config_manager, '_config_manager', None)
        setup_simple_test_config(**self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from iris_memory.core import config_manager
        config_manager._config_manager = self.original_manager