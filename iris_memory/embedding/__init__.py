"""
嵌入模块 - 策略模式实现多种嵌入源
支持 AstrBot API、本地模型、降级策略无缝切换
"""

from .base import EmbeddingProvider, EmbeddingRequest, EmbeddingResponse
from .astrbot_provider import AstrBotProvider
from .local_provider import LocalProvider
from .fallback_provider import FallbackProvider
from .manager import EmbeddingManager, EmbeddingStrategy

__all__ = [
    'EmbeddingProvider',
    'EmbeddingRequest',
    'EmbeddingResponse',
    'AstrBotProvider',
    'LocalProvider',
    'FallbackProvider',
    'EmbeddingManager',
    'EmbeddingStrategy'
]
