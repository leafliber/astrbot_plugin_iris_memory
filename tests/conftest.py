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
    # 测试开始前：设置基础配置
    basic_config = {
        'basic': {
            'enable_memory': True,
            'memory_max_total_size': 100000000,
            'memory_max_single_size': 5000000,
        },
        'llm_processing': {
            'enable_llm_upgrade': True,
            'llm_upgrade_threshold': 0.7,
            'llm_upgrade_prompt': 'Upgrade this memory: {content}',
        },
        'proactive_reply': {
            'enable_proactive': False,
            'proactive_probability': 0.05,
            'proactive_cooldown': 300,
        },
        'embedding': {
            'provider': 'astrbot',
            'embedding_model': 'bge-small-zh-v1.5',
        },
        'advanced': {
            'log_level': 'INFO',
        },
        # 向后兼容的旧配置格式
        'emotion_config': {
            'enable_emotion': True,
            'emotion_model': 'builtin',
        },
        'chroma_config': {
            'embedding_dimension': 1536,
            'collection_name': 'test_iris_memory',
            'embedding_model': 'text-embedding-ada-002',
            'auto_detect_dimension': True,
        },
        'memory_config': {
            'session_cleanup_interval': 3600,
            'session_inactive_timeout': 1800,
            'rif_scale_factor': 10.0,
        },
        'cache_config': {
            'embedding_cache_size': 100,
            'max_sessions': 50,
        }
    }
    
    setup_test_config(basic_config)
    
    yield  # 测试运行
    
    # 测试结束后：清理
    reset_config_manager()


@pytest.fixture
def mock_emotion_config():
    """情感分析配置Mock"""
    return {
        'enable_emotion': False,
        'emotion_model': 'builtin',
    }


@pytest.fixture
def mock_chroma_config():
    """ChromaDB配置Mock"""
    return {
        'embedding_dimension': 1024,
        'collection_name': 'test_collection',
        'embedding_model': 'test-model',
        'auto_detect_dimension': False,
    }


@pytest.fixture
def mock_memory_config():
    """内存配置Mock"""
    return {
        'session_cleanup_interval': 3600,
        'session_inactive_timeout': 1800,
        'rif_scale_factor': 5.0,
    }


@pytest.fixture
def custom_test_config():
    """自定义测试配置辅助函数"""
    def _setup_config(**kwargs):
        config_dict = {}
        for key, value in kwargs.items():
            if key.startswith('emotion_'):
                if 'emotion_config' not in config_dict:
                    config_dict['emotion_config'] = {}
                config_dict['emotion_config'][key[8:]] = value  # 移除 'emotion_' 前缀
            elif key.startswith('chroma_'):
                if 'chroma_config' not in config_dict:
                    config_dict['chroma_config'] = {}
                config_dict['chroma_config'][key[7:]] = value  # 移除 'chroma_' 前缀
            elif key.startswith('memory_'):
                if 'memory_config' not in config_dict:
                    config_dict['memory_config'] = {}
                config_dict['memory_config'][key[7:]] = value  # 移除 'memory_' 前缀
            else:
                # 直接设置到基础配置
                if 'basic' not in config_dict:
                    config_dict['basic'] = {}
                config_dict['basic'][key] = value
        
        setup_test_config(config_dict)
    
    return _setup_config