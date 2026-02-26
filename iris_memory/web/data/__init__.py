"""数据访问层

包含各领域仓库实现，用于隔离数据访问逻辑。
"""

from iris_memory.web.data.interfaces import (
    EmotionRepository,
    KnowledgeGraphRepository,
    MemoryRepository,
    PersonaRepository,
    SessionRepository,
)

from iris_memory.web.data.kg_repo import KnowledgeGraphRepositoryImpl
from iris_memory.web.data.memory_repo import MemoryRepositoryImpl
from iris_memory.web.data.persona_repo import EmotionRepositoryImpl, PersonaRepositoryImpl
from iris_memory.web.data.session_repo import SessionRepositoryImpl

__all__ = [
    "MemoryRepository",
    "MemoryRepositoryImpl",
    "PersonaRepository",
    "PersonaRepositoryImpl",
    "EmotionRepository",
    "EmotionRepositoryImpl",
    "KnowledgeGraphRepository",
    "KnowledgeGraphRepositoryImpl",
    "SessionRepository",
    "SessionRepositoryImpl",
]
