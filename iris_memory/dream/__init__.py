"""
Iris Chat Memory - 梦境模块

记忆的离线深度加工，提供 5 个执行阶段开关：
1. TemporalAnchorPhase: 零 LLM 时间锚定
2. ReconciliationPhase: 共享近邻扫描的重复合并与矛盾消解
3. KnowledgeInductionPhase: 增量模式挖掘与知识提取
4. PruningPhase: persona 级 L2 遗忘清洗
5. Global L3 Maintenance: 每轮一次全局图谱维护
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
