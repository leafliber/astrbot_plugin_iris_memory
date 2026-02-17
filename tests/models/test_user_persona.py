"""
UserPersona v2 测试
测试用户画像数据模型核心功能（apply_change审计、to_injection_view、update_from_memory规则引擎）
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from iris_memory.models.user_persona import UserPersona, PersonaChangeRecord
from iris_memory.core.types import MemoryType


# ==============================================================
# Fixtures
# ==============================================================

@pytest.fixture
def empty_persona():
    """空画像"""
    return UserPersona(user_id="u_empty")


@pytest.fixture
def sample_persona():
    """示例用户画像"""
    return UserPersona(
        user_id="user_123",
        work_style="创新",
        work_goals=["完成项目", "提升技能"],
        lifestyle="规律",
        interests={"编程": 0.9, "阅读": 0.7},
        emotional_baseline="joy",
        emotional_volatility=0.3,
        personality_openness=0.8,
        personality_conscientiousness=0.7,
        personality_extraversion=0.6,
        communication_formality=0.4,
    )


# ==============================================================
# PersonaChangeRecord 测试
# ==============================================================

class TestPersonaChangeRecord:
    """变更审计记录测试"""

    def test_to_dict(self):
        r = PersonaChangeRecord(
            timestamp="2024-01-01T00:00:00",
            field_name="trust_level",
            old_value=0.5,
            new_value=0.6,
            source_memory_id="m1",
            memory_type="relationship",
            rule_id="trust_keyword",
            confidence=0.8,
            evidence_type="inferred",
        )
        d = r.to_dict()
        assert d["ts"] == "2024-01-01T00:00:00"
        assert d["field"] == "trust_level"
        assert d["old"] == 0.5
        assert d["new"] == 0.6
        assert d["mem_id"] == "m1"
        assert d["mem_type"] == "relationship"
        assert d["rule"] == "trust_keyword"
        assert d["conf"] == 0.8
        assert d["ev"] == "inferred"

    def test_from_dict_roundtrip(self):
        r = PersonaChangeRecord(
            timestamp="T", field_name="f", old_value=1, new_value=2,
            rule_id="r", confidence=0.9, evidence_type="confirmed",
        )
        d = r.to_dict()
        r2 = PersonaChangeRecord.from_dict(d)
        assert r2.timestamp == r.timestamp
        assert r2.field_name == r.field_name
        assert r2.old_value == r.old_value
        assert r2.new_value == r.new_value


# ==============================================================
# 初始化测试
# ==============================================================

class TestUserPersonaInit:
    """初始化功能测试"""

    def test_defaults(self, empty_persona):
        p = empty_persona
        assert p.user_id == "u_empty"
        assert p.version == 2
        assert isinstance(p.last_updated, datetime)
        assert p.update_count == 0
        assert p.emotional_baseline == "neutral"
        assert p.proactive_reply_preference == 0.5
        assert p.preferred_reply_style is None
        assert p.topic_blacklist == []
        assert p.change_log == []
        assert len(p.hourly_distribution) == 24

    def test_init_with_values(self, sample_persona):
        assert sample_persona.user_id == "user_123"
        assert sample_persona.work_style == "创新"
        assert "完成项目" in sample_persona.work_goals
        assert sample_persona.interests["编程"] == 0.9
        assert sample_persona.emotional_baseline == "joy"
        assert sample_persona.personality_openness == 0.8

    def test_big_five(self):
        p = UserPersona(
            personality_openness=0.9,
            personality_conscientiousness=0.8,
            personality_extraversion=0.7,
            personality_agreeableness=0.6,
            personality_neuroticism=0.2,
        )
        assert p.personality_openness == 0.9
        assert p.personality_neuroticism == 0.2

    def test_communication_dimensions(self):
        p = UserPersona(
            communication_formality=0.8,
            communication_directness=0.7,
            communication_humor=0.6,
            communication_empathy=0.9,
        )
        assert p.communication_formality == 0.8
        assert p.communication_empathy == 0.9


# ==============================================================
# apply_change 审计测试
# ==============================================================

class TestApplyChange:
    """apply_change 统一变更入口测试"""

    def test_scalar_change(self, empty_persona):
        rec = empty_persona.apply_change(
            "emotional_baseline", "joy",
            rule_id="test", confidence=0.8,
        )
        assert rec is not None
        assert rec.field_name == "emotional_baseline"
        assert rec.old_value == "neutral"
        assert rec.new_value == "joy"
        assert empty_persona.emotional_baseline == "joy"
        assert empty_persona.update_count == 1
        assert len(empty_persona.change_log) == 1

    def test_scalar_no_change(self, empty_persona):
        """值相同时不产生变更"""
        rec = empty_persona.apply_change(
            "emotional_baseline", "neutral",
        )
        assert rec is None
        assert empty_persona.update_count == 0

    def test_list_append(self, empty_persona):
        rec = empty_persona.apply_change("work_goals", "目标A")
        assert rec is not None
        assert "目标A" in empty_persona.work_goals
        # 去重
        rec2 = empty_persona.apply_change("work_goals", "目标A")
        assert rec2 is None

    def test_dict_merge(self, empty_persona):
        rec = empty_persona.apply_change(
            "interests", {"编程": 0.9, "阅读": 0.7}
        )
        assert rec is not None
        assert empty_persona.interests["编程"] == 0.9

    def test_dict_no_change(self, empty_persona):
        empty_persona.interests = {"a": 1}
        rec = empty_persona.apply_change("interests", {"a": 1})
        assert rec is None

    def test_invalid_field(self, empty_persona):
        rec = empty_persona.apply_change("nonexistent_field", "val")
        assert rec is None

    def test_change_log_capped(self, empty_persona):
        empty_persona._max_change_log = 5
        for i in range(10):
            empty_persona.apply_change(
                "emotional_baseline", f"state_{i}"
            )
        assert len(empty_persona.change_log) <= 5

    def test_safe_log_value_truncates(self, empty_persona):
        long_str = "x" * 300
        rec = empty_persona.apply_change("work_style", long_str)
        assert rec is not None
        assert len(str(rec.new_value)) <= 210  # 200 + "..."


# ==============================================================
# to_injection_view 测试
# ==============================================================

class TestToInjectionView:
    """to_injection_view 注入视图测试"""

    def test_basic_view(self, sample_persona):
        view = sample_persona.to_injection_view()
        assert "interests" in view
        assert "communication" in view
        assert "relationship" in view
        assert "preferences" in view

    def test_emotional_section(self, sample_persona):
        view = sample_persona.to_injection_view()
        assert "emotional" in view
        assert view["emotional"]["baseline"] == "joy"

    def test_interests_top5(self):
        p = UserPersona(interests={
            f"i{i}": float(i) / 10 for i in range(10)
        })
        view = p.to_injection_view()
        assert len(view.get("interests", {})) <= 5

    def test_proactive_preference_in_view(self, empty_persona):
        empty_persona.proactive_reply_preference = 0.8
        view = empty_persona.to_injection_view()
        assert view["preferences"]["proactive_reply"] == 0.8

    def test_topic_blacklist_in_view(self, empty_persona):
        empty_persona.topic_blacklist = ["政治"]
        view = empty_persona.to_injection_view()
        assert "政治" in view["preferences"]["topic_blacklist"]

    def test_no_audit_log_in_view(self, sample_persona):
        sample_persona.apply_change("work_style", "严谨")
        view = sample_persona.to_injection_view()
        assert "change_log" not in view


# ==============================================================
# 序列化 / 反序列化 测试
# ==============================================================

class TestSerialization:
    """序列化功能测试"""

    def test_to_dict_basic(self, sample_persona):
        d = sample_persona.to_dict()
        assert d["user_id"] == "user_123"
        assert d["version"] == 2
        assert isinstance(d["last_updated"], str)

    def test_from_dict_basic(self):
        d = {"user_id": "u456", "version": 2, "work_style": "严谨"}
        p = UserPersona.from_dict(d)
        assert p.user_id == "u456"
        assert p.work_style == "严谨"

    def test_from_dict_datetime(self):
        d = {"user_id": "u", "last_updated": "2024-01-15T10:30:00"}
        p = UserPersona.from_dict(d)
        assert isinstance(p.last_updated, datetime)
        assert p.last_updated.year == 2024

    def test_roundtrip(self, sample_persona):
        sample_persona.apply_change("work_style", "严谨")
        d = sample_persona.to_dict()
        p2 = UserPersona.from_dict(d)
        assert p2.user_id == sample_persona.user_id
        assert p2.work_style == "严谨"
        assert len(p2.change_log) == len(sample_persona.change_log)

    def test_change_log_roundtrip(self, empty_persona):
        empty_persona.apply_change("trust_level", 0.8, rule_id="test")
        d = empty_persona.to_dict()
        p2 = UserPersona.from_dict(d)
        assert len(p2.change_log) == 1
        assert p2.change_log[0].field_name == "trust_level"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"user_id": "u", "unknown_key": 42}
        p = UserPersona.from_dict(d)
        assert p.user_id == "u"
        assert not hasattr(p, "unknown_key") or "unknown_key" not in p.to_dict()


# ==============================================================
# 证据追踪测试
# ==============================================================

class TestEvidenceTracking:
    """证据追踪功能测试"""

    def test_add_confirmed(self, empty_persona):
        empty_persona.add_memory_evidence("m1", "confirmed")
        assert "m1" in empty_persona.evidence_confirmed

    def test_add_inferred(self, empty_persona):
        empty_persona.add_memory_evidence("m2", "inferred")
        assert "m2" in empty_persona.evidence_inferred

    def test_add_contested(self, empty_persona):
        empty_persona.add_memory_evidence("m3", "contested")
        assert "m3" in empty_persona.evidence_contested

    def test_dedup(self, empty_persona):
        empty_persona.add_memory_evidence("m1", "confirmed")
        empty_persona.add_memory_evidence("m1", "confirmed")
        assert empty_persona.evidence_confirmed.count("m1") == 1


# ==============================================================
# update_from_memory 规则引擎测试
# ==============================================================

class TestUpdateFromMemory:
    """从记忆更新画像功能测试"""

    def _make_memory(self, **kwargs):
        defaults = {
            "type": MemoryType.FACT,
            "content": "",
            "user_id": "u",
            "summary": None,
            "subtype": None,
            "emotional_weight": 0.0,
            "confidence": 0.5,
            "id": "test_mem",
            "created_time": datetime.now(),
        }
        defaults.update(kwargs)
        m = Mock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    # --- 情感维度 ---

    def test_emotion_pattern_count(self, empty_persona):
        m = self._make_memory(type=MemoryType.EMOTION, subtype="joy", emotional_weight=0.3)
        changes = empty_persona.update_from_memory(m)
        assert empty_persona.emotional_patterns.get("joy") == 1
        assert any(c.field_name == "emotional_patterns" for c in changes)

    def test_emotion_baseline_high_weight(self, empty_persona):
        m = self._make_memory(type=MemoryType.EMOTION, subtype="anger", emotional_weight=0.9)
        changes = empty_persona.update_from_memory(m)
        assert empty_persona.emotional_baseline == "anger"
        assert any(c.rule_id == "emotion_baseline_high_weight" for c in changes)

    def test_emotion_baseline_low_weight_no_change(self, empty_persona):
        m = self._make_memory(type=MemoryType.EMOTION, subtype="sadness", emotional_weight=0.5)
        empty_persona.update_from_memory(m)
        assert empty_persona.emotional_baseline == "neutral"

    def test_negative_ratio_recalc(self, empty_persona):
        m = self._make_memory(type=MemoryType.EMOTION, subtype="sadness", emotional_weight=0.3)
        empty_persona.update_from_memory(m)
        assert empty_persona.negative_ratio > 0

    def test_trajectory_inference(self, empty_persona):
        """足够多的负面情感应推断出 deteriorating"""
        for _ in range(5):
            m = self._make_memory(type=MemoryType.EMOTION, subtype="sadness", emotional_weight=0.3)
            empty_persona.update_from_memory(m)
        assert empty_persona.emotional_trajectory in ("deteriorating", "volatile")

    # --- 事实维度 ---

    def test_fact_work_keyword(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.FACT,
            content="我在工作中想提升",
            summary="提升技能",
        )
        changes = empty_persona.update_from_memory(m)
        assert "提升技能" in empty_persona.work_goals

    def test_fact_life_keyword(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.FACT,
            content="我喜欢运动",
            summary="运动",
        )
        changes = empty_persona.update_from_memory(m)
        assert "运动" in empty_persona.habits

    def test_interest_weight_increment(self, empty_persona):
        m = self._make_memory(type=MemoryType.FACT, content="我最近在学编程")
        empty_persona.update_from_memory(m)
        assert empty_persona.interests.get("编程", 0) > 0

    # --- 关系维度 ---

    def test_trust_keyword(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.RELATIONSHIP,
            content="我很信任你", summary="信任",
        )
        old = empty_persona.trust_level
        empty_persona.update_from_memory(m)
        assert empty_persona.trust_level > old

    def test_intimacy_keyword(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.RELATIONSHIP,
            content="我们关系很亲密", summary="亲密",
        )
        old = empty_persona.intimacy_level
        empty_persona.update_from_memory(m)
        assert empty_persona.intimacy_level > old

    def test_social_style_inferred(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.RELATIONSHIP,
            content="我是一个外向的人", summary="",
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.social_style == "外向"

    def test_trust_cap_at_1(self, empty_persona):
        empty_persona.trust_level = 0.95
        m = self._make_memory(
            type=MemoryType.RELATIONSHIP,
            content="信任", summary="信任",
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.trust_level <= 1.0

    # --- 交互维度 ---

    def test_reply_style_brief(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.INTERACTION,
            content="回复简短就好",
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.preferred_reply_style == "brief"

    def test_reply_style_detailed(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.INTERACTION,
            content="请详细展开说",
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.preferred_reply_style == "detailed"

    def test_formality_increase(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.INTERACTION,
            content="请正式一些",
        )
        old = empty_persona.communication_formality
        empty_persona.update_from_memory(m)
        assert empty_persona.communication_formality > old

    def test_formality_decrease(self, empty_persona):
        empty_persona.communication_formality = 0.8
        m = self._make_memory(
            type=MemoryType.INTERACTION,
            content="不用客气，随意就好",
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.communication_formality < 0.8

    # --- 活跃时段 ---

    def test_hourly_distribution_updated(self, empty_persona):
        hour = 14
        m = self._make_memory(
            type=MemoryType.FACT,
            content="test",
            created_time=datetime(2024, 1, 1, hour, 0, 0),
        )
        empty_persona.update_from_memory(m)
        assert empty_persona.hourly_distribution[hour] == 1.0

    # --- 返回值 ---

    def test_returns_change_list(self, empty_persona):
        m = self._make_memory(
            type=MemoryType.EMOTION,
            subtype="joy",
            emotional_weight=0.9,
            content="happy",
        )
        changes = empty_persona.update_from_memory(m)
        assert isinstance(changes, list)
        assert all(isinstance(c, PersonaChangeRecord) for c in changes)


# ==============================================================
# 边界情况测试
# ==============================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_user_id(self):
        p = UserPersona(user_id="")
        assert p.user_id == ""

    def test_unicode_content(self):
        p = UserPersona(user_id="用户_123", work_style="创新", emotional_baseline="😊")
        assert p.emotional_baseline == "😊"

    def test_large_values(self):
        p = UserPersona(emotional_volatility=100.0, trust_level=1000.0)
        assert p.emotional_volatility == 100.0

    def test_hourly_distribution_length(self):
        p = UserPersona()
        assert len(p.hourly_distribution) == 24


# ==============================================================
# 集成测试
# ==============================================================

class TestIntegration:
    """集成场景测试"""

    def test_full_workflow(self):
        """创建画像 → 多次更新 → 序列化 → 反序列化"""
        persona = UserPersona(user_id="u_int")

        # 情感更新
        m1 = Mock(
            type=MemoryType.EMOTION, subtype="joy", emotional_weight=0.8,
            content="开心", id="m1", confidence=0.7, created_time=datetime.now(),
        )
        c1 = persona.update_from_memory(m1)
        assert persona.emotional_baseline == "joy"

        # 事实更新
        m2 = Mock(
            type=MemoryType.FACT, content="我在工作中提升技能",
            summary="提升技能", subtype=None, emotional_weight=0,
            id="m2", confidence=0.5, created_time=datetime.now(),
        )
        persona.update_from_memory(m2)
        assert "提升技能" in persona.work_goals

        # 添加证据
        persona.add_memory_evidence("m1", "confirmed")

        # 序列化往返
        d = persona.to_dict()
        p2 = UserPersona.from_dict(d)
        assert p2.user_id == "u_int"
        assert p2.emotional_baseline == "joy"
        assert "提升技能" in p2.work_goals
        assert len(p2.change_log) > 0

        # 注入视图
        view = p2.to_injection_view()
        assert "emotional" in view
        assert "preferences" in view
