"""Models module for iris memory

核心数据模型层，定义系统的实体结构。

注意：为避免循环导入，核心模型 Memory, EmotionalState, UserPersona 
需要从各自的模块直接导入，而非从此 __init__.py 导入。
"""

from iris_memory.models.persona_change import PersonaChangeRecord
from iris_memory.models.persona_view import build_injection_view
from iris_memory.models.persona_extraction_applier import (
    apply_extraction_result as apply_persona_extraction,
)
from iris_memory.models.protection import ProtectionFlag, ProtectionMixin, ProtectionRules

__all__ = [
    "PersonaChangeRecord",
    "build_injection_view",
    "apply_persona_extraction",
    "ProtectionFlag",
    "ProtectionMixin",
    "ProtectionRules",
]
