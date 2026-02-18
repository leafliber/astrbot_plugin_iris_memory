"""Conflict resolution submodule - 冲突检测与解决"""

from iris_memory.capture.conflict.conflict_resolver import ConflictResolver
from iris_memory.capture.conflict.llm_conflict_resolver import (
    LLMConflictResolver,
    ConflictDetectionResult,
)
from iris_memory.capture.conflict.similarity_calculator import (
    SimilarityCalculator,
    sanitize_for_log,
)

__all__ = [
    'ConflictResolver',
    'LLMConflictResolver',
    'ConflictDetectionResult',
    'SimilarityCalculator',
    'sanitize_for_log',
]
