"""
Iris Chat Memory - 梦境模块

记忆的离线深度加工，包含 6 个阶段：
1. ConsolidationPhase: 合并重复项
2. TemporalAnchorPhase: 时间锚定
3. ContradictionPhase: 矛盾消解
4. PatternDiscoveryPhase: 模式挖掘
5. KnowledgeExtractPhase: 知识提取
6. PruningPhase: 遗忘清洗
"""

from iris_memory.core import get_logger

__all__ = [
    "DreamTask",
    "DreamReport",
    "DreamPhaseReport",
]


def __getattr__(name: str):
    if name == "DreamTask":
        from .dream_task import DreamTask

        return DreamTask
    elif name == "DreamReport":
        from .dream_task import DreamReport

        return DreamReport
    elif name == "DreamPhaseReport":
        from .dream_task import DreamPhaseReport

        return DreamPhaseReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = get_logger("dream")
logger.debug("梦境模块已加载")
