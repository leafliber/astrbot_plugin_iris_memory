"""Detection base classes - 检测基类"""

from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMCallResult,
    LLMEnhancedBase,
    LLMEnhancedDetector,
)

__all__ = [
    'BaseDetectionResult',
    'DetectionMode',
    'LLMCallResult',
    'LLMEnhancedBase',
    'LLMEnhancedDetector',
]
