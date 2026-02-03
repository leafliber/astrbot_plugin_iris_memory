"""
多模态处理模块
利用 AstrBot 现有模型实现图像、语音处理
"""

from .image_analyzer import (
    ImageAnalyzer,
    ImageAnalysisLevel,
    ImageAnalysisResult,
    ImageInfo
)

__all__ = [
    'ImageAnalyzer',
    'ImageAnalysisLevel',
    'ImageAnalysisResult',
    'ImageInfo'
]
