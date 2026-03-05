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
    from iris_memory.config import init_store

    mock_config = Mock()

    if config_dict:
        for section, values in config_dict.items():
            if isinstance(values, dict):
                setattr(mock_config, section, values)
            else:
                setattr(mock_config, section, values)

    init_store(mock_config)
