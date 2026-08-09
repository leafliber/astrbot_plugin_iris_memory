"""
Iris Chat Memory - 人格自迭代子模块

根据群聊真人发言的表达风格，通过"风格归纳 → 人格生成 →
确定性校验 → 发布"流水线迭代 AstrBot Persona 的表达方式。
本包拥有独立的语料池、数据模型、状态机与审计历史。
"""

from .analyzer import StyleAnalyzer, extract_json_object, parse_style_profile
from .component import PersonaEvolutionComponent
from .collector import PersonaCollector
from .generator import CandidateGenerator, parse_generation
from .goals import GOAL_PRESET_VERSION, build_goal_snapshot, get_goal_preset
from .models import (
    ApprovalMode,
    EditMode,
    ErrorCode,
    EvolutionJob,
    EvolutionRun,
    JobStatus,
    PersonaRevision,
    RevisionStatus,
    RunStatus,
    StyleSample,
    TriggerType,
)
from .publisher import (
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    MARKERS_ABSENT,
    MARKERS_INVALID,
    MARKERS_OK,
    PersonaPublisher,
    append_managed_block,
    persona_hash,
    split_managed_block,
)
from .reviewer import DEFAULT_THRESHOLDS, PromptReviewer, parse_review, review_passed
from .sampler import stratified_sample
from .service import PersonaEvolutionService
from .storage import PersonaEvolutionStorage
from .validator import ValidationOutcome, validate_candidate

__all__ = [
    "PersonaEvolutionComponent",
    "PersonaEvolutionStorage",
    "PersonaEvolutionService",
    "PersonaCollector",
    "PersonaPublisher",
    "StyleAnalyzer",
    "CandidateGenerator",
    "PromptReviewer",
    "stratified_sample",
    "get_goal_preset",
    "build_goal_snapshot",
    "GOAL_PRESET_VERSION",
    "EvolutionJob",
    "EvolutionRun",
    "PersonaRevision",
    "StyleSample",
    "JobStatus",
    "EditMode",
    "ApprovalMode",
    "RevisionStatus",
    "RunStatus",
    "TriggerType",
    "ErrorCode",
    "MANAGED_BLOCK_BEGIN",
    "MANAGED_BLOCK_END",
    "MARKERS_OK",
    "MARKERS_ABSENT",
    "MARKERS_INVALID",
    "persona_hash",
    "split_managed_block",
    "append_managed_block",
    "extract_json_object",
    "parse_style_profile",
    "parse_generation",
    "parse_review",
    "review_passed",
    "DEFAULT_THRESHOLDS",
    "ValidationOutcome",
    "validate_candidate",
]
