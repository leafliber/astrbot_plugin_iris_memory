"""
处理模块

提供消息处理的核心能力：
- LLM消息分类
- LLM摘要生成
- 混合处理策略
- LLM记忆升级评估
- 通用检测结果基类
- LLM增强检测器泛型基类
"""

from .llm_processor import (
    LLMMessageProcessor,
    LLMClassificationResult,
    LLMSummaryResult
)

from .upgrade_evaluator import (
    UpgradeEvaluator,
    UpgradeMode
)

from .detection_result import BaseDetectionResult

from .llm_enhanced_base import (
    LLMEnhancedDetector,
    DetectionMode,
)

__all__ = [
    'LLMMessageProcessor',
    'LLMClassificationResult',
    'LLMSummaryResult',
    'UpgradeEvaluator',
    'UpgradeMode',
    'BaseDetectionResult',
    'LLMEnhancedDetector',
    'DetectionMode',
]
