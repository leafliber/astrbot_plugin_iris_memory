"""
Logger工具模块
统一处理logger导入，确保在测试环境和生产环境都能正常工作
"""

import logging

# 尝试导入astrbot的logger，如果失败则使用标准logging
try:
    from astrbot.api import logger as _logger
    # 如果成功导入，使用astrbot的logger
    logger = _logger
    USE_ASTRBOT_LOGGER = True
except ImportError:
    # 测试环境或其他环境，使用标准logging
    logger = logging.getLogger(__name__)
    
    # 配置日志格式
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    USE_ASTRBOT_LOGGER = False


__all__ = ['logger', 'USE_ASTRBOT_LOGGER']
