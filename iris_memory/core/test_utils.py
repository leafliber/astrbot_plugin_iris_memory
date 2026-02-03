"""
测试工具模块 - 提供测试专用的配置管理

使用示例：
    from iris_memory.core.test_utils import setup_test_config
    
    # 在测试fixture中
    @pytest.fixture(autouse=True)
    def setup_config():
        setup_test_config({
            'emotion_config': {'enable_emotion': False},
            'chroma_config': {'embedding_dimension': 1024}
        })
"""

from typing import Dict, Any
from unittest.mock import Mock


def setup_test_config(config_dict: Dict[str, Any] = None):
    """设置测试配置
    
    Args:
        config_dict: 测试配置字典，格式如下：
            {
                'emotion_config': {'enable_emotion': False},
                'chroma_config': {'embedding_dimension': 1024},
                'basic': {'enable_memory': True}
            }
    """
    from iris_memory.core.config_manager import init_config_manager
    
    # 创建Mock配置对象
    mock_config = Mock()
    
    if config_dict:
        # 设置旧格式配置（用于向后兼容测试）
        for section, values in config_dict.items():
            if isinstance(values, dict):
                # 创建具体的字典而不是Mock对象
                setattr(mock_config, section, values)
            else:
                setattr(mock_config, section, values)
    
    # 初始化配置管理器
    init_config_manager(mock_config)


def setup_simple_test_config(**kwargs):
    """设置简单测试配置
    
    Args:
        **kwargs: 直接的配置键值对，如：
            enable_emotion=False,
            embedding_dimension=1024
    """
    from iris_memory.core.config_manager import init_config_manager
    
    # 创建Mock配置对象
    mock_config = Mock()
    
    # 映射简单配置到旧格式
    config_mapping = {
        'enable_emotion': ('emotion_config', 'enable_emotion'),
        'emotion_enable_emotion': ('emotion_config', 'enable_emotion'),
        'emotion_model': ('emotion_config', 'emotion_model'),
        'embedding_model': ('chroma_config', 'embedding_model'),
        'embedding_dimension': ('chroma_config', 'embedding_dimension'),
        'chroma_embedding_dimension': ('chroma_config', 'embedding_dimension'),
        'chroma_collection_name': ('chroma_config', 'collection_name'),
        'collection_name': ('chroma_config', 'collection_name'),
        'auto_detect_dimension': ('chroma_config', 'auto_detect_dimension'),
        'chroma_auto_detect_dimension': ('chroma_config', 'auto_detect_dimension'),
    }
    
    # 构建分组配置
    grouped_config = {}
    for key, value in kwargs.items():
        if key in config_mapping:
            section, attr = config_mapping[key]
            if section not in grouped_config:
                grouped_config[section] = {}
            grouped_config[section][attr] = value
        else:
            # 未映射的配置跳过或记录警告
            print(f"Warning: Unknown config key '{key}' in test config")
    
    # 设置到mock对象
    for section, values in grouped_config.items():
        setattr(mock_config, section, values)
    
    # 确保基础配置不是Mock对象，将emotion配置同步到基础配置
    if 'emotion_config' in grouped_config and 'enable_emotion' in grouped_config['emotion_config']:
        if not hasattr(mock_config, 'basic'):
            mock_config.basic = {}
        elif isinstance(mock_config.basic, Mock):
            mock_config.basic = {}
        mock_config.basic['enable_emotion'] = grouped_config['emotion_config']['enable_emotion']
    
    # 初始化配置管理器
    init_config_manager(mock_config)


def reset_config_manager():
    """重置配置管理器（测试清理用）"""
    from iris_memory.core.config_manager import _config_manager
    global _config_manager
    _config_manager = None


class TestConfigContext:
    """测试配置上下文管理器
    
    用法：
        with TestConfigContext(enable_emotion=False):
            # 在此上下文中，emotion被禁用
            analyzer = EmotionAnalyzer()
            assert analyzer.enable_emotion is False
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.original_manager = None
    
    def __enter__(self):
        # 保存原配置管理器
        from iris_memory.core import config_manager
        self.original_manager = getattr(config_manager, '_config_manager', None)
        
        # 设置测试配置
        setup_simple_test_config(**self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原配置管理器
        from iris_memory.core import config_manager
        config_manager._config_manager = self.original_manager