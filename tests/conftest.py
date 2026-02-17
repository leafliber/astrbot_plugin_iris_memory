"""
pytest测试配置文件

为所有测试提供基础fixtures，统一配置管理器初始化。
"""

import pytest
from unittest.mock import Mock
from iris_memory.core.test_utils import setup_test_config, reset_config_manager


@pytest.fixture(autouse=True, scope="function")
def setup_test_environment():
    """自动为每个测试设置基础环境"""
    basic_config = {
        'basic': {
            'enable_memory': True,
            'enable_inject': True,
            'log_level': 'INFO',
        },
        'memory': {
            'max_working_memory': 10,
            'upgrade_mode': 'rule',
            'use_llm': False,
        },
        'proactive_reply': {
            'enable': False,
        },
        'embedding': {
            'embedding_model': 'bge-small-zh-v1.5',
            'embedding_dimension': 1536,
            'collection_name': 'test_iris_memory',
            'auto_detect_dimension': True,
        },
    }
    
    setup_test_config(basic_config)
    
    yield
    
    reset_config_manager()


@pytest.fixture
def mock_embedding_config():
    """嵌入向量配置Mock"""
    return {
        'embedding_dimension': 1024,
        'collection_name': 'test_collection',
        'embedding_model': 'test-model',
        'auto_detect_dimension': False,
    }


@pytest.fixture
def mock_memory_config():
    """记忆配置Mock"""
    return {
        'max_working_memory': 10,
        'rif_threshold': 0.4,
        'upgrade_mode': 'rule',
    }


@pytest.fixture
def custom_test_config():
    """自定义测试配置辅助函数"""
    def _setup_config(**kwargs):
        config_dict = {}
        for key, value in kwargs.items():
            if key.startswith('embedding_'):
                if 'embedding' not in config_dict:
                    config_dict['embedding'] = {}
                config_dict['embedding'][key[10:]] = value
            elif key.startswith('memory_'):
                if 'memory' not in config_dict:
                    config_dict['memory'] = {}
                config_dict['memory'][key[7:]] = value
            else:
                if 'basic' not in config_dict:
                    config_dict['basic'] = {}
                config_dict['basic'][key] = value
        
        setup_test_config(config_dict)
    
    return _setup_config