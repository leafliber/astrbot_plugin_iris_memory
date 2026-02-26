"""
UserPersona数据模型（v2 - 画像补完）
支持：变更审计日志、注入视图生成、主动回复偏好、结构化DEBUG
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from iris_memory.core.types import DecayRate, MemoryType
from iris_memory.core.constants import DEFAULT_EMOTION, NEGATIVE_EMOTION_STRINGS
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.analysis.persona.keyword_maps import ExtractionResult


logger = get_logger("user_persona")


# ---------------------------------------------------------------------------
# 变更审计记录
# ---------------------------------------------------------------------------
@dataclass
class PersonaChangeRecord:
    """画像单次变更的审计记录"""
    timestamp: str = ""
    field_name: str = ""
    old_value: Any = None
    new_value: Any = None
    source_memory_id: Optional[str] = None
    memory_type: Optional[str] = None
    rule_id: str = ""          # 触发规则标识
    confidence: float = 0.0
    evidence_type: str = "inferred"  # confirmed / inferred / contested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "field": self.field_name,
            "old": self.old_value,
            "new": self.new_value,
            "mem_id": self.source_memory_id,
            "mem_type": self.memory_type,
            "rule": self.rule_id,
            "conf": self.confidence,
            "ev": self.evidence_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaChangeRecord":
        return cls(
            timestamp=d.get("ts", ""),
            field_name=d.get("field", ""),
            old_value=d.get("old"),
            new_value=d.get("new"),
            source_memory_id=d.get("mem_id"),
            memory_type=d.get("mem_type"),
            rule_id=d.get("rule", ""),
            confidence=d.get("conf", 0.0),
            evidence_type=d.get("ev", "inferred"),
        )


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
@dataclass
class UserPersona:
    """用户画像数据模型 v2

    多维度画像，记录用户的特征、偏好、情感状态等。
    所有通过 ``apply_change`` 进行的字段变更都会被记录到 ``change_log`` 审计日志中。
    """

    # ========== 基础信息 ==========
    user_id: str = ""
    display_name: Optional[str] = None  # 用户昵称/姓名
    version: int = 2
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0  # 累计更新次数

    # ========== 工作维度 ==========
    work_style: Optional[str] = None
    work_goals: List[str] = field(default_factory=list)
    work_challenges: List[str] = field(default_factory=list)
    work_preferences: Dict[str, Any] = field(default_factory=dict)

    # ========== 生活维度 ==========
    lifestyle: Optional[str] = None
    interests: Dict[str, float] = field(default_factory=dict)
    habits: List[str] = field(default_factory=list)
    life_preferences: Dict[str, Any] = field(default_factory=dict)

    # ========== 情感维度 ==========
    emotional_baseline: str = DEFAULT_EMOTION
    emotional_volatility: float = 0.5
    emotional_triggers: List[str] = field(default_factory=list)
    emotional_soothers: Dict[str, Any] = field(default_factory=dict)
    emotional_patterns: Dict[str, int] = field(default_factory=dict)
    emotional_trajectory: Optional[str] = None
    negative_ratio: float = 0.3

    # ========== 关系维度 ==========
    social_style: Optional[str] = None
    social_boundaries: Dict[str, Any] = field(default_factory=dict)
    trust_level: float = 0.5
    intimacy_level: float = 0.5

    # ========== 人格维度（Big Five）==========
    personality_openness: float = 0.5
    personality_conscientiousness: float = 0.5
    personality_extraversion: float = 0.5
    personality_agreeableness: float = 0.5
    personality_neuroticism: float = 0.5
    confidence_decay: float = DecayRate.PERSONALITY

    # ========== 沟通维度 ==========
    communication_formality: float = 0.5
    communication_directness: float = 0.5
    communication_humor: float = 0.5
    communication_empathy: float = 0.5

    # ========== 交互偏好 ==========
    proactive_reply_preference: float = 0.5  # 0→不希望被主动回复  1→欢迎
    preferred_reply_style: Optional[str] = None  # brief / detailed / default
    topic_blacklist: List[str] = field(default_factory=list)  # 用户排斥话题

    # ========== 行为模式 ==========
    hourly_distribution: List[float] = field(default_factory=lambda: [0.0] * 24)
    topic_sequences: List[str] = field(default_factory=list)
    memory_cooccurrence: Dict[str, List[str]] = field(default_factory=dict)

    # ========== 证据追踪 ==========
    evidence_confirmed: List[str] = field(default_factory=list)
    evidence_inferred: List[str] = field(default_factory=list)
    evidence_contested: List[str] = field(default_factory=list)

    # ========== 变更审计 ==========
    change_log: List[PersonaChangeRecord] = field(default_factory=list)
    _max_change_log: int = field(default=200, repr=False)

    # ========== 元数据 ==========
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典"""
        data: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif key == "change_log":
                data[key] = [r.to_dict() for r in value]
            else:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPersona":
        """从字典恢复"""
        data = dict(data)  # shallow copy
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        if "change_log" in data:
            data["change_log"] = [
                PersonaChangeRecord.from_dict(r) if isinstance(r, dict) else r
                for r in data["change_log"]
            ]
        # 过滤掉私有字段和无效字段
        valid_fields = {
            f.name for f in cls.__dataclass_fields__.values()
            if not f.name.startswith("_")
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Dict 兼容接口 — 让 UserPersona 可用于 .get() 等字典访问模式
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible get: 委托到 to_injection_view()。

        这使得下游代码无需关心收到的是 UserPersona 对象还是字典，
        ``user_persona.get("preferences", {})`` 总是有效。
        """
        return self.to_injection_view().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_injection_view()

    # ------------------------------------------------------------------
    # 注入视图 — 用于传给 LLM 上下文 / PersonaCoordinator
    # ------------------------------------------------------------------
    def to_injection_view(self) -> Dict[str, Any]:
        """生成精简的画像视图（供 prompt 注入使用，不含审计日志）"""
        view: Dict[str, Any] = {}

        # 情感摘要
        if self.emotional_baseline != DEFAULT_EMOTION or self.emotional_trajectory:
            emotional: Dict[str, Any] = {
                "baseline": self.emotional_baseline,
                "trajectory": self.emotional_trajectory,
                "volatility": round(self.emotional_volatility, 2),
            }
            if self.emotional_triggers:
                emotional["triggers"] = self.emotional_triggers[:5]
            if self.emotional_soothers:
                emotional["soothers"] = dict(list(self.emotional_soothers.items())[:3])
            view["emotional"] = emotional

        # 兴趣 / 习惯
        if self.interests:
            top = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)[:5]
            view["interests"] = {k: round(v, 2) for k, v in top}
        if self.habits:
            view["habits"] = self.habits[:10]

        # 工作维度
        work: Dict[str, Any] = {}
        if self.work_style:
            work["style"] = self.work_style
        if self.work_goals:
            work["goals"] = self.work_goals[:5]
        if self.work_challenges:
            work["challenges"] = self.work_challenges[:5]
        if self.work_preferences:
            work["preferences"] = dict(list(self.work_preferences.items())[:5])
        if work:
            view["work"] = work

        # 生活维度
        life: Dict[str, Any] = {}
        if self.lifestyle:
            life["style"] = self.lifestyle
        if self.life_preferences:
            life["preferences"] = dict(list(self.life_preferences.items())[:5])
        if life:
            view["life"] = life

        # 沟通偏好
        view["communication"] = {
            "formality": round(self.communication_formality, 2),
            "directness": round(self.communication_directness, 2),
            "humor": round(self.communication_humor, 2),
            "empathy": round(self.communication_empathy, 2),
        }

        # 人格特质（Big Five）— 仅展示偏离默认值(0.5)的维度
        personality: Dict[str, float] = {}
        for trait in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"):
            val = getattr(self, f"personality_{trait}", 0.5)
            if abs(val - 0.5) > 0.05:
                personality[trait] = round(val, 2)
        if personality:
            view["personality"] = personality

        # 交互偏好
        view["preferences"] = {}
        if self.preferred_reply_style:
            view["preferences"]["style"] = self.preferred_reply_style
        view["preferences"]["proactive_reply"] = round(self.proactive_reply_preference, 2)
        if self.topic_blacklist:
            view["preferences"]["topic_blacklist"] = self.topic_blacklist[:10]

        # 关系
        relationship: Dict[str, Any] = {
            "trust": round(self.trust_level, 2),
            "intimacy": round(self.intimacy_level, 2),
        }
        if self.social_style:
            relationship["social_style"] = self.social_style
        if self.social_boundaries:
            relationship["boundaries"] = dict(list(self.social_boundaries.items())[:5])
        view["relationship"] = relationship

        return view

    # ------------------------------------------------------------------
    # 变更接口 — 统一入口，自动审计
    # ------------------------------------------------------------------
    def apply_change(
        self,
        field_name: str,
        new_value: Any,
        *,
        source_memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        rule_id: str = "",
        confidence: float = 0.5,
        evidence_type: str = "inferred",
    ) -> Optional[PersonaChangeRecord]:
        """安全地更新一个字段并写入审计日志。

        对于 ``list`` 类型字段，``new_value`` (str) 的语义为 *追加*（去重）。
        对于 ``dict`` 类型字段，``new_value`` (dict) 的语义为 *合并*。
        标量字段直接替换。

        Returns:
            变更记录；如果值未变化则返回 ``None``。
        """
        if not hasattr(self, field_name):
            return None

        old_value = getattr(self, field_name)

        # list → append（去重）
        if isinstance(old_value, list) and isinstance(new_value, str):
            if new_value and new_value not in old_value:
                old_snapshot = list(old_value)
                old_value.append(new_value)
                old_value_for_log = old_snapshot
                new_value_for_log = list(old_value)
            else:
                return None
        # dict → merge
        elif isinstance(old_value, dict) and isinstance(new_value, dict):
            old_snapshot = dict(old_value)
            old_value.update(new_value)
            if old_value != old_snapshot:
                old_value_for_log = old_snapshot
                new_value_for_log = dict(old_value)
            else:
                return None
        # scalar
        else:
            if old_value == new_value:
                return None
            old_value_for_log = old_value
            new_value_for_log = new_value
            setattr(self, field_name, new_value)

        record = PersonaChangeRecord(
            timestamp=datetime.now().isoformat(),
            field_name=field_name,
            old_value=self._safe_log_value(old_value_for_log),
            new_value=self._safe_log_value(new_value_for_log),
            source_memory_id=source_memory_id,
            memory_type=memory_type,
            rule_id=rule_id,
            confidence=confidence,
            evidence_type=evidence_type,
        )
        self.change_log.append(record)
        if len(self.change_log) > self._max_change_log:
            self.change_log = self.change_log[-self._max_change_log:]

        self.last_updated = datetime.now()
        self.update_count += 1
        return record

    @staticmethod
    def _safe_log_value(value: Any) -> Any:
        """截断过大的值，防止日志膨胀"""
        if isinstance(value, str) and len(value) > 200:
            return value[:200] + "..."
        if isinstance(value, list) and len(value) > 20:
            return value[:20]
        if isinstance(value, dict) and len(value) > 20:
            return dict(list(value.items())[:20])
        return value

    # ------------------------------------------------------------------
    # 证据追踪
    # ------------------------------------------------------------------
    def add_memory_evidence(self, memory_id: str, evidence_type: str = "confirmed"):
        if evidence_type == "confirmed" and memory_id not in self.evidence_confirmed:
            self.evidence_confirmed.append(memory_id)
        elif evidence_type == "inferred" and memory_id not in self.evidence_inferred:
            self.evidence_inferred.append(memory_id)
        elif evidence_type == "contested" and memory_id not in self.evidence_contested:
            self.evidence_contested.append(memory_id)

    # ------------------------------------------------------------------
    # 从记忆更新画像（规则引擎）
    # ------------------------------------------------------------------
    def update_from_memory(self, memory) -> List[PersonaChangeRecord]:
        """从一条 Memory 推导并更新画像字段，返回本次变更列表。"""
        changes: List[PersonaChangeRecord] = []
        mem_id = getattr(memory, "id", None)
        mem_type_raw = getattr(memory, "type", None)
        mem_type = mem_type_raw.value if hasattr(mem_type_raw, "value") else str(mem_type_raw)
        confidence = getattr(memory, "confidence", 0.5)

        # 分发到维度处理器
        if mem_type in (MemoryType.EMOTION.value, "emotion"):
            try:
                changes.extend(self._update_emotional(memory, mem_id, confidence))
            except Exception as e:
                logger.warning(f"update_from_memory emotional update failed: {e}")
        if mem_type in (MemoryType.FACT.value, "fact"):
            try:
                changes.extend(self._update_facts(memory, mem_id, confidence))
            except Exception as e:
                logger.warning(f"update_from_memory fact update failed: {e}")
        if mem_type in (MemoryType.RELATIONSHIP.value, "relationship"):
            try:
                changes.extend(self._update_social(memory, mem_id, confidence))
            except Exception as e:
                logger.warning(f"update_from_memory social update failed: {e}")
        if mem_type in (MemoryType.INTERACTION.value, "interaction"):
            try:
                changes.extend(self._update_interaction(memory, mem_id, confidence))
            except Exception as e:
                logger.warning(f"update_from_memory interaction update failed: {e}")

        # 更新活跃时段
        created = getattr(memory, "created_time", None)
        if created and isinstance(created, datetime):
            hour = created.hour
            self.hourly_distribution[hour] += 1.0

        # 更新话题序列（记录最近的记忆类型序列）
        if mem_type:
            self.topic_sequences.append(mem_type)
            # 保持最多50条记录
            if len(self.topic_sequences) > 50:
                self.topic_sequences = self.topic_sequences[-50:]

        # 更新记忆共现关系
        entities = getattr(memory, "detected_entities", None)
        if entities and isinstance(entities, list) and len(entities) >= 2:
            for entity in entities:
                others = [e for e in entities if e != entity]
                if entity in self.memory_cooccurrence:
                    # 追加去重
                    existing = self.memory_cooccurrence[entity]
                    for o in others:
                        if o not in existing:
                            existing.append(o)
                    # 限制每个实体最多20条共现
                    self.memory_cooccurrence[entity] = existing[:20]
                else:
                    self.memory_cooccurrence[entity] = others[:20]

        return changes

    # ------------------------------------------------------------------
    # 从 ExtractionResult 更新画像（支持规则 / LLM / 混合）
    # ------------------------------------------------------------------
    def apply_extraction_result(
        self,
        result: "ExtractionResult",
        source_memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        base_confidence: float = 0.5,
    ) -> List[PersonaChangeRecord]:
        """将 PersonaExtractor 的提取结果应用到画像。

        这是 LLM / hybrid 提取模式的主要入口，替代旧的硬编码关键词匹配。

        Args:
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

        # 兴趣
        for interest, weight in result.interests.items():
            old_w = self.interests.get(interest, 0.0)
            # 规则模式：权重为增量（+0.1）；LLM 模式：权重为绝对值
            if result.source == "rule":
                new_w = min(1.0, old_w + weight)
            else:
                new_w = min(1.0, max(old_w, weight))
            if new_w != old_w:
                self.interests[interest] = new_w
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

        # 社交风格
        if result.social_style:
            rec = self.apply_change(
                "social_style", result.social_style,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "relationship",
                rule_id=f"{rule_prefix}_social_style",
                confidence=conf * 0.8,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 回复风格偏好
        if result.reply_style_preference:
            rec = self.apply_change(
                "preferred_reply_style", result.reply_style_preference,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_reply_style",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 沟通正式度
        if result.formality_adjustment != 0.0:
            if result.source == "rule":
                new_formality = max(0.0, min(1.0, self.communication_formality + result.formality_adjustment))
            else:
                # LLM 返回的是绝对变化量，按比例应用
                new_formality = max(0.0, min(1.0, self.communication_formality + result.formality_adjustment * 0.2))
            rec = self.apply_change(
                "communication_formality", round(new_formality, 3),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_formality",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 话题黑名单
        for topic in result.topic_blacklist:
            if topic and topic not in self.topic_blacklist:
                rec = self.apply_change(
                    "topic_blacklist", topic,
                    source_memory_id=source_memory_id,
                    memory_type=memory_type or "interaction",
                    rule_id=f"{rule_prefix}_topic_blacklist",
                    confidence=conf,
                    evidence_type=evidence,
                )
                if rec:
                    changes.append(rec)

        # 工作信息
        if result.work_info:
            rec = self.apply_change(
                "work_goals", result.work_info,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_work",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 生活信息
        if result.life_info:
            rec = self.apply_change(
                "habits", result.life_info,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_life",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 信任度
        if result.trust_delta > 0:
            rec = self.apply_change(
                "trust_level", min(1.0, self.trust_level + result.trust_delta),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "relationship",
                rule_id=f"{rule_prefix}_trust",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 亲密度
        if result.intimacy_delta > 0:
            rec = self.apply_change(
                "intimacy_level", min(1.0, self.intimacy_level + result.intimacy_delta),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "relationship",
                rule_id=f"{rule_prefix}_intimacy",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # ── v2 新增维度应用 ──

        # 工作风格
        if result.work_style:
            rec = self.apply_change(
                "work_style", result.work_style,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_work_style",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 工作挑战
        if result.work_challenge:
            rec = self.apply_change(
                "work_challenges", result.work_challenge,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_work_challenge",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 工作偏好
        if result.work_preferences:
            rec = self.apply_change(
                "work_preferences", result.work_preferences,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_work_pref",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 生活方式
        if result.lifestyle:
            rec = self.apply_change(
                "lifestyle", result.lifestyle,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_lifestyle",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 生活偏好
        if result.life_preferences:
            rec = self.apply_change(
                "life_preferences", result.life_preferences,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "fact",
                rule_id=f"{rule_prefix}_life_pref",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 情感触发器
        for trigger in result.emotional_triggers:
            if trigger and trigger not in self.emotional_triggers:
                rec = self.apply_change(
                    "emotional_triggers", trigger,
                    source_memory_id=source_memory_id,
                    memory_type=memory_type or "emotion",
                    rule_id=f"{rule_prefix}_emotion_trigger",
                    confidence=conf,
                    evidence_type=evidence,
                )
                if rec:
                    changes.append(rec)

        # 情感安慰物
        if result.emotional_soothers:
            rec = self.apply_change(
                "emotional_soothers", result.emotional_soothers,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "emotion",
                rule_id=f"{rule_prefix}_emotion_soother",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 社交边界
        if result.social_boundaries:
            rec = self.apply_change(
                "social_boundaries", result.social_boundaries,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "relationship",
                rule_id=f"{rule_prefix}_social_boundary",
                confidence=conf * 0.9,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 人格特质（Big Five） — 渐进式微调
        personality_traits = [
            ("openness", result.personality_openness_delta),
            ("conscientiousness", result.personality_conscientiousness_delta),
            ("extraversion", result.personality_extraversion_delta),
            ("agreeableness", result.personality_agreeableness_delta),
            ("neuroticism", result.personality_neuroticism_delta),
        ]
        for trait_name, delta in personality_traits:
            if delta != 0.0:
                field_name = f"personality_{trait_name}"
                old_val = getattr(self, field_name, 0.5)
                # LLM 模式直接应用 delta，rule 模式使用更小步长
                effective_delta = delta if result.source != "rule" else delta
                new_val = round(max(0.0, min(1.0, old_val + effective_delta)), 3)
                rec = self.apply_change(
                    field_name, new_val,
                    source_memory_id=source_memory_id,
                    memory_type=memory_type or "interaction",
                    rule_id=f"{rule_prefix}_personality_{trait_name}",
                    confidence=conf * 0.7,
                    evidence_type=evidence,
                )
                if rec:
                    changes.append(rec)

        # 沟通直接度
        if result.directness_adjustment != 0.0:
            if result.source == "rule":
                new_val = max(0.0, min(1.0, self.communication_directness + result.directness_adjustment))
            else:
                new_val = max(0.0, min(1.0, self.communication_directness + result.directness_adjustment * 0.2))
            rec = self.apply_change(
                "communication_directness", round(new_val, 3),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_directness",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 幽默度
        if result.humor_adjustment != 0.0:
            if result.source == "rule":
                new_val = max(0.0, min(1.0, self.communication_humor + result.humor_adjustment))
            else:
                new_val = max(0.0, min(1.0, self.communication_humor + result.humor_adjustment * 0.2))
            rec = self.apply_change(
                "communication_humor", round(new_val, 3),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_humor",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 共情度
        if result.empathy_adjustment != 0.0:
            if result.source == "rule":
                new_val = max(0.0, min(1.0, self.communication_empathy + result.empathy_adjustment))
            else:
                new_val = max(0.0, min(1.0, self.communication_empathy + result.empathy_adjustment * 0.2))
            rec = self.apply_change(
                "communication_empathy", round(new_val, 3),
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_empathy",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        # 主动回复偏好
        if result.proactive_reply_delta != 0.0:
            new_val = round(max(0.0, min(1.0,
                self.proactive_reply_preference + result.proactive_reply_delta)), 3)
            rec = self.apply_change(
                "proactive_reply_preference", new_val,
                source_memory_id=source_memory_id,
                memory_type=memory_type or "interaction",
                rule_id=f"{rule_prefix}_proactive_reply",
                confidence=conf,
                evidence_type=evidence,
            )
            if rec:
                changes.append(rec)

        return changes

    # --- 情感维度 ---
    def _update_emotional(self, memory, mem_id, confidence) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        subtype = getattr(memory, "subtype", None)
        weight = getattr(memory, "emotional_weight", 0.0)
        content = getattr(memory, "content", "") or ""

        # 更新情感模式统计
        if subtype:
            old_count = self.emotional_patterns.get(subtype, 0)
            self.emotional_patterns[subtype] = old_count + 1
            changes.append(PersonaChangeRecord(
                timestamp=datetime.now().isoformat(),
                field_name="emotional_patterns",
                old_value={subtype: old_count},
                new_value={subtype: old_count + 1},
                source_memory_id=mem_id,
                memory_type="emotion",
                rule_id="emotion_pattern_count",
                confidence=confidence,
                evidence_type="confirmed",
            ))

        # 基线更新（仅高强度）
        if weight > 0.7 and subtype:
            rec = self.apply_change(
                "emotional_baseline", subtype,
                source_memory_id=mem_id, memory_type="emotion",
                rule_id="emotion_baseline_high_weight",
                confidence=confidence, evidence_type="confirmed",
            )
            if rec:
                changes.append(rec)

        # 重新计算负面占比
        total = sum(self.emotional_patterns.values()) or 1
        neg_count = sum(self.emotional_patterns.get(k, 0) for k in NEGATIVE_EMOTION_STRINGS)
        new_ratio = round(neg_count / total, 3)
        rec = self.apply_change(
            "negative_ratio", new_ratio,
            source_memory_id=mem_id, memory_type="emotion",
            rule_id="negative_ratio_recalc", confidence=0.8,
            evidence_type="confirmed",
        )
        if rec:
            changes.append(rec)

        # 计算情感轨迹
        trajectory = self._infer_trajectory()
        if trajectory != self.emotional_trajectory:
            rec = self.apply_change(
                "emotional_trajectory", trajectory,
                source_memory_id=mem_id, memory_type="emotion",
                rule_id="trajectory_inference", confidence=0.6,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 计算情感波动性（基于情感模式多样性和负面占比差异）
        new_volatility = self._compute_volatility()
        if abs(new_volatility - self.emotional_volatility) > 0.02:
            rec = self.apply_change(
                "emotional_volatility", round(new_volatility, 3),
                source_memory_id=mem_id, memory_type="emotion",
                rule_id="volatility_recalc", confidence=0.7,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 情感触发器推断（高强度负面情绪时，尝试提取触发上下文）
        if weight > 0.6 and subtype in NEGATIVE_EMOTION_STRINGS and content:
            trigger_snippet = content[:50].strip()
            if trigger_snippet and trigger_snippet not in self.emotional_triggers:
                rec = self.apply_change(
                    "emotional_triggers", trigger_snippet,
                    source_memory_id=mem_id, memory_type="emotion",
                    rule_id="emotion_trigger_high_weight",
                    confidence=confidence * 0.6,
                    evidence_type="inferred",
                )
                if rec:
                    changes.append(rec)

        return changes

    def _compute_volatility(self) -> float:
        """根据情感模式多样性和变化频率计算波动性

        波动性 = 情感类型多样性 × 负面-正面比例偏差
        范围: 0.0 (完全稳定) ~ 1.0 (高度波动)
        """
        if not self.emotional_patterns:
            return 0.5  # 默认中等

        total = sum(self.emotional_patterns.values())
        if total < 3:
            return 0.5  # 样本不足

        # 因子1：情感类型多样性（Shannon 熵归一化）
        import math
        n_types = len(self.emotional_patterns)
        if n_types <= 1:
            diversity = 0.0
        else:
            entropy = 0.0
            for count in self.emotional_patterns.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n_types) if n_types > 1 else 1.0
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # 因子2：负面占比偏离中性的程度
        neg_count = sum(self.emotional_patterns.get(k, 0) for k in NEGATIVE_EMOTION_STRINGS)
        neg_ratio = neg_count / total
        # 偏离0.3（理想中性点）越远越波动
        deviation = abs(neg_ratio - 0.3) * 2.0

        volatility = min(1.0, diversity * 0.6 + deviation * 0.4)
        return volatility

    def _infer_trajectory(self) -> Optional[str]:
        """根据情感模式推断趋势"""
        if not self.emotional_patterns:
            return None
        total = sum(self.emotional_patterns.values())
        if total < 3:
            return None
        neg_count = sum(self.emotional_patterns.get(k, 0) for k in NEGATIVE_EMOTION_STRINGS)
        ratio = neg_count / total
        if ratio > 0.6:
            return "deteriorating"
        elif ratio > 0.4:
            return "volatile"
        elif ratio < 0.2:
            return "improving"
        return "stable"

    # --- 事实维度 ---
    def _update_facts(
        self, memory, mem_id, confidence,
        keyword_maps=None,
    ) -> List[PersonaChangeRecord]:
        """基于关键词规则更新事实维度。

        关键词来源优先级：
        1. 外部传入的 ``keyword_maps``
        2. 内置默认值（保持既有行为）
        """
        changes: List[PersonaChangeRecord] = []
        content = getattr(memory, "content", "") or ""
        summary = getattr(memory, "summary", None)
        content_lower = content.lower()

        # 获取关键词配置
        if keyword_maps is not None:
            work_keywords = keyword_maps.work_keywords
            life_keywords = keyword_maps.life_keywords
            interest_map = keyword_maps.interests
            work_styles = getattr(keyword_maps, "work_styles", {})
            work_challenge_kws = getattr(keyword_maps, "work_challenge_keywords", [])
            lifestyles = getattr(keyword_maps, "lifestyles", {})
        else:
            work_keywords = ["工作", "公司", "项目", "同事", "老板", "职业", "事业", "上班"]
            life_keywords = ["喜欢", "爱好", "兴趣", "习惯", "运动", "娱乐", "爱吃", "讨厌"]
            interest_map = {
                "编程": ["编程", "代码", "开发", "程序"],
                "阅读": ["阅读", "读书", "看书", "书"],
                "运动": ["运动", "跑步", "健身", "锻炼"],
                "音乐": ["音乐", "歌", "唱"],
                "游戏": ["游戏", "打游戏", "玩游戏"],
                "美食": ["吃", "美食", "餐厅", "做饭"],
                "旅行": ["旅行", "旅游", "出游"],
            }
            work_styles = {
                "远程": ["远程", "在家办公", "remote"],
                "坐班": ["坐班", "朝九晚五", "打卡"],
                "自由": ["自由职业", "弹性", "灵活"],
            }
            work_challenge_kws = ["压力", "加班", "困难", "挑战", "焦虑", "紧急"]
            lifestyles = {
                "夜猫子": ["夜猫子", "熬夜", "晚睡"],
                "早起": ["早起", "早睡早起", "晨跑"],
                "宅": ["宅", "宅家", "不出门"],
            }

        if any(kw in content_lower for kw in work_keywords) and summary:
            rec = self.apply_change(
                "work_goals", summary,
                source_memory_id=mem_id, memory_type="fact",
                rule_id="fact_work_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        if any(kw in content_lower for kw in life_keywords) and summary:
            rec = self.apply_change(
                "habits", summary,
                source_memory_id=mem_id, memory_type="fact",
                rule_id="fact_life_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 兴趣权重更新
        for interest, keywords in interest_map.items():
            if any(kw in content_lower for kw in keywords):
                old_w = self.interests.get(interest, 0.0)
                new_w = min(1.0, old_w + 0.1)
                if new_w != old_w:
                    self.interests[interest] = new_w
                    changes.append(PersonaChangeRecord(
                        timestamp=datetime.now().isoformat(),
                        field_name=f"interests.{interest}",
                        old_value=round(old_w, 2),
                        new_value=round(new_w, 2),
                        source_memory_id=mem_id,
                        memory_type="fact",
                        rule_id="interest_weight_increment",
                        confidence=confidence,
                        evidence_type="inferred",
                    ))

        # 工作风格推断
        for style, keywords in work_styles.items():
            if any(kw in content_lower for kw in keywords):
                rec = self.apply_change(
                    "work_style", style,
                    source_memory_id=mem_id, memory_type="fact",
                    rule_id="fact_work_style_keyword",
                    confidence=confidence * 0.8,
                    evidence_type="inferred",
                )
                if rec:
                    changes.append(rec)
                break

        # 工作挑战
        if any(kw in content_lower for kw in work_challenge_kws) and summary:
            rec = self.apply_change(
                "work_challenges", summary,
                source_memory_id=mem_id, memory_type="fact",
                rule_id="fact_work_challenge_keyword",
                confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 生活方式推断
        for style, keywords in lifestyles.items():
            if any(kw in content_lower for kw in keywords):
                rec = self.apply_change(
                    "lifestyle", style,
                    source_memory_id=mem_id, memory_type="fact",
                    rule_id="fact_lifestyle_keyword",
                    confidence=confidence * 0.8,
                    evidence_type="inferred",
                )
                if rec:
                    changes.append(rec)
                break

        return changes

    # --- 关系维度 ---
    def _update_social(
        self, memory, mem_id, confidence,
        keyword_maps=None,
    ) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        content = getattr(memory, "content", "") or ""
        summary = getattr(memory, "summary", "") or ""
        text = content + summary

        # 关键词配置
        if keyword_maps is not None:
            trust_kws = keyword_maps.trust_keywords
            intimacy_kws = keyword_maps.intimacy_keywords
            style_map = keyword_maps.social_styles
            boundary_kws = getattr(keyword_maps, "social_boundary_keywords", [])
        else:
            trust_kws = ["信任"]
            intimacy_kws = ["亲密"]
            style_map = {
                "外向": ["外向", "活泼", "爱交际"],
                "内向": ["内向", "安静", "独处"],
                "温和": ["温和", "和善", "温柔"],
            }
            boundary_kws = ["别聊", "不想说", "不讨论", "别问", "不说"]

        if any(kw in text for kw in trust_kws):
            rec = self.apply_change(
                "trust_level", min(1.0, self.trust_level + 0.1),
                source_memory_id=mem_id, memory_type="relationship",
                rule_id="trust_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        if any(kw in text for kw in intimacy_kws):
            rec = self.apply_change(
                "intimacy_level", min(1.0, self.intimacy_level + 0.1),
                source_memory_id=mem_id, memory_type="relationship",
                rule_id="intimacy_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 社交风格推断
        for style, keywords in style_map.items():
            if any(kw in text for kw in keywords):
                rec = self.apply_change(
                    "social_style", style,
                    source_memory_id=mem_id, memory_type="relationship",
                    rule_id="social_style_keyword", confidence=confidence * 0.8,
                    evidence_type="inferred",
                )
                if rec:
                    changes.append(rec)
                break

        # 社交边界提取
        for kw in boundary_kws:
            if kw in content:
                idx = content.index(kw)
                boundary_ctx = content[idx:idx + 20].strip()
                if boundary_ctx:
                    rec = self.apply_change(
                        "social_boundaries", {kw: boundary_ctx},
                        source_memory_id=mem_id, memory_type="relationship",
                        rule_id="social_boundary_keyword",
                        confidence=confidence * 0.8,
                        evidence_type="inferred",
                    )
                    if rec:
                        changes.append(rec)

        return changes

    # --- 交互维度 ---
    def _update_dimension_by_keywords(
        self,
        content: str,
        field_name: str,
        positive_keywords: List[str],
        negative_keywords: List[str],
        positive_rule_id: str,
        negative_rule_id: str,
        step: float,
        mem_id: Optional[str],
        confidence: float,
    ) -> Optional[PersonaChangeRecord]:
        """根据正负关键词对连续数值维度做增减更新。"""
        old_val = float(getattr(self, field_name, 0.5))

        if any(str(kw).lower() in content for kw in positive_keywords if kw is not None):
            new_val = round(min(1.0, old_val + step), 3)
            return self.apply_change(
                field_name,
                new_val,
                source_memory_id=mem_id,
                memory_type="interaction",
                rule_id=positive_rule_id,
                confidence=confidence,
                evidence_type="inferred",
            )

        if any(str(kw).lower() in content for kw in negative_keywords if kw is not None):
            new_val = round(max(0.0, old_val - step), 3)
            return self.apply_change(
                field_name,
                new_val,
                source_memory_id=mem_id,
                memory_type="interaction",
                rule_id=negative_rule_id,
                confidence=confidence,
                evidence_type="inferred",
            )

        return None

    def _update_personality_by_keywords(
        self,
        content: str,
        trait: str,
        high_keywords: List[str],
        low_keywords: List[str],
        step: float,
        mem_id: Optional[str],
        confidence: float,
    ) -> Optional[PersonaChangeRecord]:
        """根据关键词更新单个人格特质。"""
        field_name = f"personality_{trait}"
        old_val = float(getattr(self, field_name, 0.5))

        if any(str(kw).lower() in content for kw in high_keywords if kw is not None):
            new_val = round(min(1.0, old_val + step), 3)
            return self.apply_change(
                field_name,
                new_val,
                source_memory_id=mem_id,
                memory_type="interaction",
                rule_id=f"personality_{trait}_increase",
                confidence=confidence * 0.7,
                evidence_type="inferred",
            )

        if any(str(kw).lower() in content for kw in low_keywords if kw is not None):
            new_val = round(max(0.0, old_val - step), 3)
            return self.apply_change(
                field_name,
                new_val,
                source_memory_id=mem_id,
                memory_type="interaction",
                rule_id=f"personality_{trait}_decrease",
                confidence=confidence * 0.7,
                evidence_type="inferred",
            )

        return None

    def _update_interaction(
        self, memory, mem_id, confidence,
        keyword_maps=None,
    ) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        content = (getattr(memory, "content", "") or "").lower()

        # 关键词配置
        if keyword_maps is not None:
            reply_style = keyword_maps.reply_style
            formality = keyword_maps.formality
            directness = getattr(keyword_maps, "directness", {})
            humor = getattr(keyword_maps, "humor", {})
            empathy = getattr(keyword_maps, "empathy", {})
            proactive_pref = getattr(keyword_maps, "proactive_preference", {})
            personality_cfg = getattr(keyword_maps, "personality", {})
        else:
            reply_style = {
                "brief": ["简短", "简洁", "不要太多"],
                "detailed": ["详细", "具体", "展开说"],
            }
            formality = {
                "formal": ["正式", "礼貌", "敬语"],
                "casual": ["随意", "口语", "不用客气"],
            }
            directness = {
                "direct": ["直说", "别绕弯", "说重点", "直接"],
                "indirect": ["委婉", "含蓄", "暗示"],
            }
            humor = {
                "high": ["哈哈", "233", "笑死", "段子", "幽默", "lol", "搞笑"],
                "low": ["严肃", "认真", "正经"],
            }
            empathy = {
                "high": ["理解", "共情", "体谅", "安慰", "换位思考"],
                "low": ["冷漠", "无所谓", "别矫情", "不关心"],
            }
            proactive_pref = {
                "welcome": ["多聊聊", "常来", "找我聊", "欢迎"],
                "unwanted": ["别打扰", "别找我", "不用管我", "少说话"],
            }
            personality_cfg = {
                "openness": {
                    "high": ["新鲜", "创意", "创新", "尝试", "探索", "好奇"],
                    "low": ["传统", "保守", "不变", "墨守成规"],
                },
                "conscientiousness": {
                    "high": ["计划", "规律", "条理", "准时", "认真", "仔细"],
                    "low": ["随性", "拖延", "随便", "懒"],
                },
                "extraversion": {
                    "high": ["外向", "社交", "聚会", "热闹", "活泼"],
                    "low": ["内向", "独处", "安静", "一个人"],
                },
                "agreeableness": {
                    "high": ["温和", "随和", "配合", "善良", "体贴"],
                    "low": ["坚持", "固执", "不妥协", "竞争"],
                },
                "neuroticism": {
                    "high": ["紧张", "焦虑", "担心", "多虑", "敏感"],
                    "low": ["淡定", "冷静", "平和", "稳重"],
                },
            }

        # 回复风格偏好
        for style, keywords in reply_style.items():
            if any(kw in content for kw in keywords):
                rec = self.apply_change(
                    "preferred_reply_style", style,
                    source_memory_id=mem_id, memory_type="interaction",
                    rule_id=f"reply_style_{style}", confidence=confidence,
                    evidence_type="inferred",
                )
                if rec:
                    changes.append(rec)
                break

        # 沟通正式度
        formal_kws = formality.get("formal", [])
        casual_kws = formality.get("casual", [])

        if any(kw in content for kw in formal_kws):
            rec = self.apply_change(
                "communication_formality", min(1.0, self.communication_formality + 0.1),
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="formality_increase", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)
        elif any(kw in content for kw in casual_kws):
            rec = self.apply_change(
                "communication_formality", max(0.0, self.communication_formality - 0.1),
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="formality_decrease", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 沟通直接度
        rec = self._update_dimension_by_keywords(
            content=content,
            field_name="communication_directness",
            positive_keywords=directness.get("direct", []),
            negative_keywords=directness.get("indirect", []),
            positive_rule_id="directness_increase",
            negative_rule_id="directness_decrease",
            step=0.1,
            mem_id=mem_id,
            confidence=confidence,
        )
        if rec:
            changes.append(rec)

        # 幽默度
        rec = self._update_dimension_by_keywords(
            content=content,
            field_name="communication_humor",
            positive_keywords=humor.get("high", []),
            negative_keywords=humor.get("low", []),
            positive_rule_id="humor_increase",
            negative_rule_id="humor_decrease",
            step=0.1,
            mem_id=mem_id,
            confidence=confidence,
        )
        if rec:
            changes.append(rec)

        # 共情度（Rule 模式）
        rec = self._update_dimension_by_keywords(
            content=content,
            field_name="communication_empathy",
            positive_keywords=empathy.get("high", []),
            negative_keywords=empathy.get("low", []),
            positive_rule_id="empathy_increase",
            negative_rule_id="empathy_decrease",
            step=0.1,
            mem_id=mem_id,
            confidence=confidence,
        )
        if rec:
            changes.append(rec)

        # 主动回复偏好
        rec = self._update_dimension_by_keywords(
            content=content,
            field_name="proactive_reply_preference",
            positive_keywords=proactive_pref.get("welcome", []),
            negative_keywords=proactive_pref.get("unwanted", []),
            positive_rule_id="proactive_welcome",
            negative_rule_id="proactive_unwanted",
            step=0.1,
            mem_id=mem_id,
            confidence=confidence,
        )
        if rec:
            changes.append(rec)

        # 人格特质（Big Five）— 微调
        for trait, directions in personality_cfg.items():
            rec = self._update_personality_by_keywords(
                content=content,
                trait=trait,
                high_keywords=directions.get("high", []),
                low_keywords=directions.get("low", []),
                step=0.05,
                mem_id=mem_id,
                confidence=confidence,
            )
            if rec:
                changes.append(rec)

        return changes
