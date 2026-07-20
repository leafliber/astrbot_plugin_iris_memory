"""
Iris Chat Memory - 配置管理模块

提供统一的配置管理接口，支持用户配置与隐藏配置的分层管理。

公共 API:
    - Config: 配置管理主类
    - init_config(): 初始化全局配置实例(插件启动时调用)
    - get_config(): 获取全局配置实例(其他模块使用)
"""

from .config import Config, init_config, get_config

__all__ = [
    "Config",
    "init_config",
    "get_config",
]
