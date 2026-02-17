"""
UserPersona数据模型（v2 - 画像补完重构）
支持：变更审计日志、注入视图生成、主动回复偏好、结构化DEBUG
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from iris_memory.core.types import DecayRate, MemoryType


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
    emotional_baseline: str = "neutral"
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
    # 注入视图 — 用于传给 LLM 上下文 / PersonaCoordinator
    # ------------------------------------------------------------------
    def to_injection_view(self) -> Dict[str, Any]:
        """生成精简的画像视图（供 prompt 注入使用，不含审计日志）"""
        view: Dict[str, Any] = {}

        # 情感摘要
        if self.emotional_baseline != "neutral" or self.emotional_trajectory:
            view["emotional"] = {
                "baseline": self.emotional_baseline,
                "trajectory": self.emotional_trajectory,
                "volatility": round(self.emotional_volatility, 2),
            }

        # 兴趣 / 习惯
        if self.interests:
            top = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)[:5]
            view["interests"] = {k: round(v, 2) for k, v in top}
        if self.habits:
            view["habits"] = self.habits[:10]

        # 工作
        if self.work_style:
            view["work_style"] = self.work_style
        if self.work_goals:
            view["work_goals"] = self.work_goals[:5]

        # 沟通偏好
        view["communication"] = {
            "formality": round(self.communication_formality, 2),
            "directness": round(self.communication_directness, 2),
            "humor": round(self.communication_humor, 2),
        }

        # 交互偏好
        view["preferences"] = {}
        if self.preferred_reply_style:
            view["preferences"]["style"] = self.preferred_reply_style
        view["preferences"]["proactive_reply"] = round(self.proactive_reply_preference, 2)
        if self.topic_blacklist:
            view["preferences"]["topic_blacklist"] = self.topic_blacklist[:10]

        # 关系
        view["relationship"] = {
            "trust": round(self.trust_level, 2),
            "intimacy": round(self.intimacy_level, 2),
        }
        if self.social_style:
            view["relationship"]["social_style"] = self.social_style

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
            changes.extend(self._update_emotional(memory, mem_id, confidence))
        if mem_type in (MemoryType.FACT.value, "fact"):
            changes.extend(self._update_facts(memory, mem_id, confidence))
        if mem_type in (MemoryType.RELATIONSHIP.value, "relationship"):
            changes.extend(self._update_social(memory, mem_id, confidence))
        if mem_type in (MemoryType.INTERACTION.value, "interaction"):
            changes.extend(self._update_interaction(memory, mem_id, confidence))

        # 更新活跃时段
        created = getattr(memory, "created_time", None)
        if created and isinstance(created, datetime):
            hour = created.hour
            self.hourly_distribution[hour] += 1.0

        return changes

    # --- 情感维度 ---
    def _update_emotional(self, memory, mem_id, confidence) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        subtype = getattr(memory, "subtype", None)
        weight = getattr(memory, "emotional_weight", 0.0)

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
        neg_keys = {"sadness", "anger", "fear", "disgust", "anxiety"}
        neg_count = sum(self.emotional_patterns.get(k, 0) for k in neg_keys)
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

        return changes

    def _infer_trajectory(self) -> Optional[str]:
        """根据情感模式推断趋势"""
        if not self.emotional_patterns:
            return None
        total = sum(self.emotional_patterns.values())
        if total < 3:
            return None
        neg_keys = {"sadness", "anger", "fear", "disgust", "anxiety"}
        neg_count = sum(self.emotional_patterns.get(k, 0) for k in neg_keys)
        ratio = neg_count / total
        if ratio > 0.6:
            return "deteriorating"
        elif ratio > 0.4:
            return "volatile"
        elif ratio < 0.2:
            return "improving"
        return "stable"

    # --- 事实维度 ---
    def _update_facts(self, memory, mem_id, confidence) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        content = getattr(memory, "content", "") or ""
        summary = getattr(memory, "summary", None)
        content_lower = content.lower()

        work_keywords = ["工作", "公司", "项目", "同事", "老板", "职业", "事业", "上班"]
        life_keywords = ["喜欢", "爱好", "兴趣", "习惯", "运动", "娱乐", "爱吃", "讨厌"]

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
        interest_map = {
            "编程": ["编程", "代码", "开发", "程序"],
            "阅读": ["阅读", "读书", "看书", "书"],
            "运动": ["运动", "跑步", "健身", "锻炼"],
            "音乐": ["音乐", "歌", "唱"],
            "游戏": ["游戏", "打游戏", "玩游戏"],
            "美食": ["吃", "美食", "餐厅", "做饭"],
            "旅行": ["旅行", "旅游", "出游"],
        }
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

        return changes

    # --- 关系维度 ---
    def _update_social(self, memory, mem_id, confidence) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        content = getattr(memory, "content", "") or ""
        summary = getattr(memory, "summary", "") or ""
        text = content + summary

        if "信任" in text:
            rec = self.apply_change(
                "trust_level", min(1.0, self.trust_level + 0.1),
                source_memory_id=mem_id, memory_type="relationship",
                rule_id="trust_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        if "亲密" in text:
            rec = self.apply_change(
                "intimacy_level", min(1.0, self.intimacy_level + 0.1),
                source_memory_id=mem_id, memory_type="relationship",
                rule_id="intimacy_keyword", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 社交风格推断
        style_map = {
            "外向": ["外向", "活泼", "爱交际"],
            "内向": ["内向", "安静", "独处"],
            "温和": ["温和", "和善", "温柔"],
        }
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

        return changes

    # --- 交互维度 ---
    def _update_interaction(self, memory, mem_id, confidence) -> List[PersonaChangeRecord]:
        changes: List[PersonaChangeRecord] = []
        content = (getattr(memory, "content", "") or "").lower()

        # 回复风格偏好
        if any(kw in content for kw in ["简短", "简洁", "不要太多"]):
            rec = self.apply_change(
                "preferred_reply_style", "brief",
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="reply_style_brief", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)
        elif any(kw in content for kw in ["详细", "具体", "展开说"]):
            rec = self.apply_change(
                "preferred_reply_style", "detailed",
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="reply_style_detailed", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        # 沟通正式度
        if any(kw in content for kw in ["正式", "礼貌", "敬语"]):
            rec = self.apply_change(
                "communication_formality", min(1.0, self.communication_formality + 0.1),
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="formality_increase", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)
        elif any(kw in content for kw in ["随意", "口语", "不用客气"]):
            rec = self.apply_change(
                "communication_formality", max(0.0, self.communication_formality - 0.1),
                source_memory_id=mem_id, memory_type="interaction",
                rule_id="formality_decrease", confidence=confidence,
                evidence_type="inferred",
            )
            if rec:
                changes.append(rec)

        return changes
