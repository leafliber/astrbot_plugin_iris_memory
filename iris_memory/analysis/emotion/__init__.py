"""Emotion analysis submodule - 情感分析"""

from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.emotion.llm_emotion_analyzer import (
    LLMEmotionAnalyzer,
    EmotionAnalysisResult,
)

__all__ = [
    'EmotionAnalyzer',
    'LLMEmotionAnalyzer',
    'EmotionAnalysisResult',
]
