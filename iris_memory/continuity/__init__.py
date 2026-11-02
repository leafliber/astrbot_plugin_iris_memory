"""
Iris Chat Memory - 连续性层（待续事项 / 出站账本 / 主动回合 / 收尾 / 频道观察）

见 tasks/2026-11-02-continuity/DESIGN.md。门面为 ``service.ContinuityService``。
"""

from .service import ContinuityConfig, ContinuityService, is_silent_or_forbidden
from .store import SCHEMA_VERSION, ContinuityStore, ItemNotFound, RevisionConflict

__all__ = [
    "ContinuityStore",
    "ItemNotFound",
    "RevisionConflict",
    "SCHEMA_VERSION",
    "ContinuityConfig",
    "ContinuityService",
    "is_silent_or_forbidden",
]
