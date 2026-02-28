"""
处理模块

提供消息处理的核心能力：
- LLM消息分类
- LLM摘要生成
- 混合处理策略
- LLM记忆升级评估
- 消息处理器（LLM Hook、消息装饰、普通消息处理）

检测基类统一由 iris_memory.core.detection 提供。
"""

from .llm_processor import (
    LLMMessageProcessor,
    LLMClassificationResult,
    LLMSummaryResult
)
from .message_processor import MessageProcessor, ErrorFriendlyProcessor

# Re-exported from core for backward compatibility
from iris_memory.core.upgrade_evaluator import (
    UpgradeEvaluator,
    UpgradeMode
)

from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    LLMEnhancedDetector,
    DetectionMode,
)

__all__ = [
    'LLMMessageProcessor',
    'LLMClassificationResult',
    'LLMSummaryResult',
    'MessageProcessor',
    'ErrorFriendlyProcessor',
    'UpgradeEvaluator',
    'UpgradeMode',
    'BaseDetectionResult',
    'LLMEnhancedDetector',
    'DetectionMode',
]
