"""
ExtractionResult 的数据驱动应用逻辑

将 UserPersona.apply_extraction_result() 中 ~350 行的重复 apply_change 调用
压缩为声明式字段描述符 + 统一循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence, TYPE_CHECKING

from iris_memory.models.persona_change import PersonaChangeRecord

if TYPE_CHECKING:
    from iris_memory.models.user_persona import UserPersona
    from iris_memory.analysis.persona.keyword_maps import ExtractionResult


# ---------------------------------------------------------------------------
# 字段描述符
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScalarField:
    """result 属性 → persona 标量字段"""
    result_attr: str
    persona_field: str
    mem_type: str = "fact"
    rule_suffix: str = ""
    conf_modifier: float = 1.0


@dataclass(frozen=True)
class _ListAppendField:
    """result 属性 (str) → persona list 字段（追加）"""
    result_attr: str
    persona_field: str
    mem_type: str = "fact"
    rule_suffix: str = ""


@dataclass(frozen=True)
class _DictMergeField:
    """result 属性 (dict) → persona dict 字段（合并）"""
    result_attr: str
    persona_field: str
    mem_type: str = "fact"
    rule_suffix: str = ""
    conf_modifier: float = 1.0


@dataclass(frozen=True)
class _DeltaField:
    """result 属性 (float delta) → persona 数值字段"""
    result_attr: str
    persona_field: str
    mem_type: str = "relationship"
    rule_suffix: str = ""
    max_val: float = 1.0
    positive_only: bool = True  # True: 仅 >0 时应用；False: !=0 时应用


@dataclass(frozen=True)
class _AdjustmentField:
    """result 属性 (float adjustment) → persona 数值字段，rule 直加 / LLM ×0.2"""
    result_attr: str
    persona_field: str
    mem_type: str = "interaction"
    rule_suffix: str = ""


@dataclass(frozen=True)
class _ListIterateField:
    """result 属性 (List[str]) → persona list（逐项追加去重）"""
    result_attr: str
    persona_field: str
    mem_type: str = "interaction"
    rule_suffix: str = ""


# ── 字段注册表 ──

_SCALAR_FIELDS: Sequence[_ScalarField] = (
    _ScalarField("social_style", "social_style", "relationship", "social_style", 0.8),
    _ScalarField("reply_style_preference", "preferred_reply_style", "interaction", "reply_style"),
    _ScalarField("work_style", "work_style", "fact", "work_style"),
    _ScalarField("lifestyle", "lifestyle", "fact", "lifestyle"),
)

_LIST_APPEND_FIELDS: Sequence[_ListAppendField] = (
    _ListAppendField("work_info", "work_goals", "fact", "work"),
    _ListAppendField("life_info", "habits", "fact", "life"),
    _ListAppendField("work_challenge", "work_challenges", "fact", "work_challenge"),
)

_DICT_MERGE_FIELDS: Sequence[_DictMergeField] = (
    _DictMergeField("work_preferences", "work_preferences", "fact", "work_pref"),
    _DictMergeField("life_preferences", "life_preferences", "fact", "life_pref"),
    _DictMergeField("emotional_soothers", "emotional_soothers", "emotion", "emotion_soother"),
    _DictMergeField("social_boundaries", "social_boundaries", "relationship", "social_boundary", 0.9),
)

_DELTA_FIELDS: Sequence[_DeltaField] = (
    _DeltaField("trust_delta", "trust_level", "relationship", "trust"),
    _DeltaField("intimacy_delta", "intimacy_level", "relationship", "intimacy"),
    _DeltaField("proactive_reply_delta", "proactive_reply_preference", "interaction", "proactive_reply",
               positive_only=False),
)

_ADJUSTMENT_FIELDS: Sequence[_AdjustmentField] = (
    _AdjustmentField("formality_adjustment", "communication_formality", "interaction", "formality"),
    _AdjustmentField("directness_adjustment", "communication_directness", "interaction", "directness"),
    _AdjustmentField("humor_adjustment", "communication_humor", "interaction", "humor"),
    _AdjustmentField("empathy_adjustment", "communication_empathy", "interaction", "empathy"),
)

_LIST_ITERATE_FIELDS: Sequence[_ListIterateField] = (
    _ListIterateField("topic_blacklist", "topic_blacklist", "interaction", "topic_blacklist"),
    _ListIterateField("emotional_triggers", "emotional_triggers", "emotion", "emotion_trigger"),
)

# Big Five 特质名称列表
_PERSONALITY_TRAITS = (
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism",
)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def apply_extraction_result(
    persona: "UserPersona",
    result: "ExtractionResult",
    source_memory_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    base_confidence: float = 0.5,
) -> List[PersonaChangeRecord]:
    """将 PersonaExtractor 的提取结果应用到画像。

    这是 LLM / hybrid 提取模式的主要入口，替代旧的硬编码关键词匹配。
    数据驱动实现：通过字段描述符表替代大量重复的 apply_change 调用。

    Args:
        persona: UserPersona 实例
        result: 提取结果
        source_memory_id: 来源记忆 ID
        memory_type: 记忆类型
        base_confidence: 基础置信度

    Returns:
        变更记录列表
    """
    changes: List[PersonaChangeRecord] = []
    conf = base_confidence * result.confidence if result.confidence > 0 else base_confidence
    evidence = "confirmed" if result.source == "llm" else "inferred"
    rule_prefix = f"extraction_{result.source}"

    # ── 兴趣（特殊处理: rule 增量 vs LLM 绝对值）──
    for interest, weight in result.interests.items():
        old_w = persona.interests.get(interest, 0.0)
        if result.source == "rule":
            new_w = min(1.0, old_w + weight)
        else:
            new_w = min(1.0, max(old_w, weight))
        if new_w != old_w:
            persona.interests[interest] = new_w
            changes.append(PersonaChangeRecord(
                timestamp=datetime.now().isoformat(),
                field_name=f"interests.{interest}",
                old_value=round(old_w, 2),
                new_value=round(new_w, 2),
                source_memory_id=source_memory_id,
                memory_type=memory_type,
                rule_id=f"{rule_prefix}_interest",
                confidence=conf,
                evidence_type=evidence,
            ))

    # ── 标量字段 ──
    for fd in _SCALAR_FIELDS:
        val = getattr(result, fd.result_attr, None)
        if val:
            _apply(persona, changes, fd.persona_field, val,
                   source_memory_id, memory_type or fd.mem_type,
                   f"{rule_prefix}_{fd.rule_suffix}", conf * fd.conf_modifier, evidence)

    # ── 列表追加字段 ──
    for fd in _LIST_APPEND_FIELDS:
        val = getattr(result, fd.result_attr, None)
        if val:
            _apply(persona, changes, fd.persona_field, val,
                   source_memory_id, memory_type or fd.mem_type,
                   f"{rule_prefix}_{fd.rule_suffix}", conf, evidence)

    # ── 字典合并字段 ──
    for fd in _DICT_MERGE_FIELDS:
        val = getattr(result, fd.result_attr, None)
        if val:
            _apply(persona, changes, fd.persona_field, val,
                   source_memory_id, memory_type or fd.mem_type,
                   f"{rule_prefix}_{fd.rule_suffix}", conf * fd.conf_modifier, evidence)

    # ── 列表逐项追加 ──
    for fd in _LIST_ITERATE_FIELDS:
        items: List[str] = getattr(result, fd.result_attr, [])
        current_list = getattr(persona, fd.persona_field, [])
        for item in items:
            if item and item not in current_list:
                _apply(persona, changes, fd.persona_field, item,
                       source_memory_id, memory_type or fd.mem_type,
                       f"{rule_prefix}_{fd.rule_suffix}", conf, evidence)

    # ── Delta 字段 ──
    for fd in _DELTA_FIELDS:
        delta: float = getattr(result, fd.result_attr, 0.0)
        should_apply = (delta > 0) if fd.positive_only else (delta != 0.0)
        if should_apply:
            old_val = getattr(persona, fd.persona_field, 0.0)
            new_val = round(max(0.0, min(fd.max_val, old_val + delta)), 3)
            _apply(persona, changes, fd.persona_field, new_val,
                   source_memory_id, memory_type or fd.mem_type,
                   f"{rule_prefix}_{fd.rule_suffix}", conf, evidence)

    # ── Adjustment 字段 (rule 直加, LLM ×0.2) ──
    for fd in _ADJUSTMENT_FIELDS:
        adj: float = getattr(result, fd.result_attr, 0.0)
        if adj != 0.0:
            old_val = getattr(persona, fd.persona_field, 0.5)
            effective = adj if result.source == "rule" else adj * 0.2
            new_val = round(max(0.0, min(1.0, old_val + effective)), 3)
            _apply(persona, changes, fd.persona_field, new_val,
                   source_memory_id, memory_type or fd.mem_type,
                   f"{rule_prefix}_{fd.rule_suffix}", conf, evidence)

    # ── Big Five 人格特质 ──
    for trait in _PERSONALITY_TRAITS:
        delta = getattr(result, f"personality_{trait}_delta", 0.0)
        if delta != 0.0:
            field_name = f"personality_{trait}"
            old_val = getattr(persona, field_name, 0.5)
            new_val = round(max(0.0, min(1.0, old_val + delta)), 3)
            _apply(persona, changes, field_name, new_val,
                   source_memory_id, memory_type or "interaction",
                   f"{rule_prefix}_personality_{trait}", conf * 0.7, evidence)

    return changes


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _apply(
    persona: "UserPersona",
    changes: List[PersonaChangeRecord],
    field_name: str,
    value: Any,
    source_memory_id: Optional[str],
    mem_type: str,
    rule_id: str,
    confidence: float,
    evidence: str,
) -> None:
    """调用 persona.apply_change 并将结果追加到 changes。"""
    rec = persona.apply_change(
        field_name, value,
        source_memory_id=source_memory_id,
        memory_type=mem_type,
        rule_id=rule_id,
        confidence=confidence,
        evidence_type=evidence,
    )
    if rec:
        changes.append(rec)
