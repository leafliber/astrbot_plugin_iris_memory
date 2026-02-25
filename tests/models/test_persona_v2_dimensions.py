"""
UserPersona v2 维度补完测试

覆盖所有新增维度在 rule、llm、hybrid 三种模式下的更新逻辑:
- 工作维度: work_style, work_challenges, work_preferences
- 生活维度: lifestyle, life_preferences
- 情感维度: emotional_volatility, emotional_triggers, emotional_soothers
- 关系维度: social_boundaries
- 人格维度: Big Five 五个特质
- 沟通维度: communication_directness, communication_humor, communication_empathy
- 交互偏好: proactive_reply_preference
- 行为模式: topic_sequences, memory_cooccurrence
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from iris_memory.models.user_persona import UserPersona
from iris_memory.analysis.persona.keyword_maps import ExtractionResult, KeywordMaps
from iris_memory.analysis.persona.rule_extractor import RuleExtractor
from iris_memory.analysis.persona.llm_extractor import LLMExtractor
from iris_memory.analysis.persona.persona_extractor import PersonaExtractor
from iris_memory.core.types import MemoryType


# ==============================================================
# Fixtures
# ==============================================================

@pytest.fixture
def persona():
    return UserPersona(user_id="test_user")


@pytest.fixture
def keyword_maps():
    kw = KeywordMaps()
    kw.load()
    return kw


@pytest.fixture
def rule_extractor(keyword_maps):
    return RuleExtractor(keyword_maps)


def _make_memory(content, mem_type="fact", summary=None, subtype=None,
                 weight=0.5, confidence=0.7, entities=None):
    """工具函数：创建 mock memory 对象"""
    m = Mock()
    m.id = "mem_test_001"
    m.content = content
    m.summary = summary or content[:30]
    m.type = MemoryType(mem_type) if mem_type in [t.value for t in MemoryType] else mem_type
    m.subtype = subtype
    m.emotional_weight = weight
    m.confidence = confidence
    m.created_time = datetime(2024, 6, 15, 14, 30)
    m.detected_entities = entities or []
    return m


# ==============================================================
# Rule 模式 — update_from_memory 测试
# ==============================================================

class TestRuleMode_WorkDimension:
    """Rule模式: 工作维度更新"""

    def test_work_style_remote(self, persona):
        m = _make_memory("我现在都是远程办公", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.work_style == "远程"

    def test_work_style_office(self, persona):
        m = _make_memory("每天朝九晚五打卡上班", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.work_style == "坐班"

    def test_work_challenges(self, persona):
        m = _make_memory("最近项目压力很大", mem_type="fact",
                         summary="项目压力大")
        persona.update_from_memory(m)
        assert "项目压力大" in persona.work_challenges

    def test_work_style_freelance(self, persona):
        m = _make_memory("我是自由职业者", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.work_style == "自由"


class TestRuleMode_LifeDimension:
    """Rule模式: 生活维度更新"""

    def test_lifestyle_nightowl(self, persona):
        m = _make_memory("我是夜猫子，经常熬夜", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.lifestyle == "夜猫子"

    def test_lifestyle_earlybird(self, persona):
        m = _make_memory("我每天早起跑步", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.lifestyle == "早起"

    def test_lifestyle_homebody(self, persona):
        m = _make_memory("周末喜欢宅在家", mem_type="fact")
        persona.update_from_memory(m)
        assert persona.lifestyle == "宅"


class TestRuleMode_EmotionalDimension:
    """Rule模式: 情感维度更新"""

    def test_emotional_volatility_computed(self, persona):
        """多种情绪模式应提高波动性"""
        persona.emotional_patterns = {
            "joy": 3, "sadness": 2, "anger": 1, "neutral": 2
        }
        m = _make_memory("今天很开心", mem_type="emotion", subtype="joy",
                         weight=0.8)
        persona.update_from_memory(m)
        # 波动性应该被重新计算（不再是默认0.5，因为有多种情绪模式）
        assert isinstance(persona.emotional_volatility, float)

    def test_emotional_trigger_high_weight_negative(self, persona):
        """高强度负面情绪应提取触发器"""
        m = _make_memory("被老板批评了很伤心", mem_type="emotion",
                         subtype="sadness", weight=0.8)
        persona.update_from_memory(m)
        assert len(persona.emotional_triggers) > 0

    def test_emotional_trigger_low_weight_no_extract(self, persona):
        """低强度不应提取触发器"""
        m = _make_memory("有点小烦", mem_type="emotion",
                         subtype="sadness", weight=0.3)
        persona.update_from_memory(m)
        # 触发器不应被添加（weight < 0.6）
        assert len(persona.emotional_triggers) == 0

    def test_volatility_stable_pattern(self, persona):
        """单一情绪模式应该产生较低波动性"""
        persona.emotional_patterns = {"neutral": 10}
        m = _make_memory("一般般", mem_type="emotion",
                         subtype="neutral", weight=0.3)
        persona.update_from_memory(m)
        assert persona.emotional_volatility < 0.5


class TestRuleMode_SocialDimension:
    """Rule模式: 社交维度更新"""

    def test_social_boundary_detected(self, persona):
        m = _make_memory("别聊政治话题", mem_type="relationship")
        persona.update_from_memory(m)
        assert len(persona.social_boundaries) > 0

    def test_social_boundary_multiple(self, persona):
        m1 = _make_memory("不想说家庭的事", mem_type="relationship")
        persona.update_from_memory(m1)
        m2 = _make_memory("别问我收入", mem_type="relationship")
        persona.update_from_memory(m2)
        assert len(persona.social_boundaries) >= 2


class TestRuleMode_InteractionDimension:
    """Rule模式: 交互维度更新"""

    def test_directness_increase(self, persona):
        m = _make_memory("直说吧别绕弯子", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_directness > 0.5

    def test_directness_decrease(self, persona):
        m = _make_memory("说话委婉一点", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_directness < 0.5

    def test_humor_increase(self, persona):
        m = _make_memory("哈哈哈笑死我了", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_humor > 0.5

    def test_humor_decrease(self, persona):
        m = _make_memory("请严肃对待", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_humor < 0.5

    def test_proactive_welcome(self, persona):
        m = _make_memory("欢迎多聊聊", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.proactive_reply_preference > 0.5

    def test_proactive_unwanted(self, persona):
        m = _make_memory("别打扰我", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.proactive_reply_preference < 0.5

    def test_empathy_increase(self, persona):
        m = _make_memory("谢谢你的理解和安慰", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_empathy > 0.5

    def test_empathy_decrease(self, persona):
        m = _make_memory("别矫情，我不关心", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.communication_empathy < 0.5


class TestRuleMode_PersonalityDimension:
    """Rule模式: 人格特质 Big Five 更新"""

    def test_openness_increase(self, persona):
        m = _make_memory("我很好奇想去探索新事物", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_openness > 0.5

    def test_openness_decrease(self, persona):
        m = _make_memory("我比较保守不喜欢变化", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_openness < 0.5

    def test_conscientiousness_increase(self, persona):
        m = _make_memory("我做事很有计划很自律", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_conscientiousness > 0.5

    def test_conscientiousness_decrease(self, persona):
        m = _make_memory("我经常拖延", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_conscientiousness < 0.5

    def test_extraversion_increase(self, persona):
        m = _make_memory("我很外向爱社交", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_extraversion > 0.5

    def test_extraversion_decrease(self, persona):
        m = _make_memory("我喜欢独处安静", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_extraversion < 0.5

    def test_agreeableness_increase(self, persona):
        m = _make_memory("我很温和随和", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_agreeableness > 0.5

    def test_neuroticism_increase(self, persona):
        m = _make_memory("我总是很焦虑担心", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_neuroticism > 0.5

    def test_neuroticism_decrease(self, persona):
        m = _make_memory("我很淡定冷静", mem_type="interaction")
        persona.update_from_memory(m)
        assert persona.personality_neuroticism < 0.5


class TestRuleMode_BehaviorDimension:
    """Rule模式: 行为模式更新"""

    def test_topic_sequences_tracked(self, persona):
        m1 = _make_memory("测试1", mem_type="fact")
        m2 = _make_memory("测试2", mem_type="emotion", subtype="joy")
        persona.update_from_memory(m1)
        persona.update_from_memory(m2)
        assert len(persona.topic_sequences) == 2
        assert persona.topic_sequences[0] == "fact"
        assert persona.topic_sequences[1] == "emotion"

    def test_topic_sequences_capped(self, persona):
        for i in range(60):
            m = _make_memory(f"消息{i}", mem_type="fact")
            persona.update_from_memory(m)
        assert len(persona.topic_sequences) <= 50

    def test_memory_cooccurrence(self, persona):
        m = _make_memory("张三和李四一起吃饭", mem_type="fact",
                         entities=["张三", "李四", "吃饭"])
        persona.update_from_memory(m)
        assert "张三" in persona.memory_cooccurrence
        assert "李四" in persona.memory_cooccurrence["张三"]

    def test_memory_cooccurrence_dedup(self, persona):
        entities = ["张三", "李四"]
        m1 = _make_memory("张三和李四", mem_type="fact", entities=entities)
        m2 = _make_memory("张三和李四", mem_type="fact", entities=entities)
        persona.update_from_memory(m1)
        persona.update_from_memory(m2)
        assert persona.memory_cooccurrence["张三"].count("李四") == 1


# ==============================================================
# RuleExtractor 新维度测试
# ==============================================================

class TestRuleExtractor_NewDimensions:
    """RuleExtractor 新维度提取"""

    def test_work_style_extraction(self, rule_extractor):
        result = rule_extractor.extract("我在家远程办公")
        assert result.work_style == "远程"

    def test_work_challenge_extraction(self, rule_extractor):
        result = rule_extractor.extract("最近压力好大", summary="工作压力大")
        assert result.work_challenge == "工作压力大"

    def test_lifestyle_extraction(self, rule_extractor):
        result = rule_extractor.extract("我是夜猫子")
        assert result.lifestyle == "夜猫子"

    def test_emotional_trigger_extraction(self, rule_extractor):
        result = rule_extractor.extract("我最怕被人误解")
        assert len(result.emotional_triggers) > 0

    def test_emotional_soother_extraction(self, rule_extractor):
        result = rule_extractor.extract("听音乐能让我放松")
        assert len(result.emotional_soothers) > 0

    def test_social_boundary_extraction(self, rule_extractor):
        result = rule_extractor.extract("别聊政治")
        assert len(result.social_boundaries) > 0

    def test_directness_direct(self, rule_extractor):
        result = rule_extractor.extract("直说就行")
        assert result.directness_adjustment > 0

    def test_directness_indirect(self, rule_extractor):
        result = rule_extractor.extract("说话委婉点")
        assert result.directness_adjustment < 0

    def test_humor_high(self, rule_extractor):
        result = rule_extractor.extract("哈哈哈太搞笑了")
        assert result.humor_adjustment > 0

    def test_humor_low(self, rule_extractor):
        result = rule_extractor.extract("请严肃认真")
        assert result.humor_adjustment < 0

    def test_empathy_high(self, rule_extractor):
        result = rule_extractor.extract("我会理解你并且安慰你")
        assert result.empathy_adjustment > 0

    def test_empathy_low(self, rule_extractor):
        result = rule_extractor.extract("别矫情，我不关心")
        assert result.empathy_adjustment < 0

    def test_proactive_welcome(self, rule_extractor):
        result = rule_extractor.extract("欢迎多聊聊")
        assert result.proactive_reply_delta > 0

    def test_proactive_unwanted(self, rule_extractor):
        result = rule_extractor.extract("别打扰我")
        assert result.proactive_reply_delta < 0

    def test_personality_openness_high(self, rule_extractor):
        result = rule_extractor.extract("我很好奇想探索新事物")
        assert result.personality_openness_delta > 0

    def test_personality_conscientiousness_high(self, rule_extractor):
        result = rule_extractor.extract("我做事很有计划")
        assert result.personality_conscientiousness_delta > 0

    def test_personality_extraversion_low(self, rule_extractor):
        result = rule_extractor.extract("我喜欢独处安静")
        assert result.personality_extraversion_delta < 0

    def test_personality_neuroticism_high(self, rule_extractor):
        result = rule_extractor.extract("我总是很焦虑紧张")
        assert result.personality_neuroticism_delta > 0

    def test_confidence_increases_with_more_hits(self, rule_extractor):
        """多维度命中应提高置信度"""
        r1 = rule_extractor.extract("编程")
        r2 = rule_extractor.extract("我远程办公写代码，喜欢直说，哈哈哈")
        assert r2.confidence > r1.confidence


# ==============================================================
# LLMExtractor 解析测试
# ==============================================================

class TestLLMExtractor_NewDimensions:
    """LLMExtractor 新维度解析"""

    def test_parse_work_style(self):
        response = '{"work_style": "远程", "confidence": 0.8}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert result.work_style == "远程"

    def test_parse_work_challenge(self):
        response = '{"work_challenge": "项目进度紧张", "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert result.work_challenge == "项目进度紧张"

    def test_parse_lifestyle(self):
        response = '{"lifestyle": "夜猫子", "confidence": 0.8}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert result.lifestyle == "夜猫子"

    def test_parse_emotional_triggers(self):
        response = '{"emotional_triggers": ["被批评", "被忽视"], "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert len(result.emotional_triggers) == 2

    def test_parse_emotional_soothers(self):
        response = '{"emotional_soothers": {"音乐": "放松"}, "confidence": 0.6}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert "音乐" in result.emotional_soothers

    def test_parse_social_boundaries(self):
        response = '{"social_boundaries": {"政治": "不讨论"}, "confidence": 0.8}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert "政治" in result.social_boundaries

    def test_parse_personality(self):
        response = '{"personality": {"openness": 0.08, "neuroticism": -0.05}, "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert result.personality_openness_delta == 0.08
        assert result.personality_neuroticism_delta == -0.05

    def test_parse_personality_clamped(self):
        response = '{"personality": {"openness": 0.5}, "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        # Clamped to 0.1
        assert result.personality_openness_delta == 0.1

    def test_parse_directness(self):
        response = '{"directness": 0.5, "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        assert result.directness_adjustment == 0.5

    def test_parse_humor(self):
        response = '{"humor": -0.3, "confidence": 0.6}'
        result = LLMExtractor._parse_response(response)
        assert result.humor_adjustment == -0.3

    def test_parse_empathy(self):
        response = '{"empathy": 0.4, "confidence": 0.7}'
        result = LLMExtractor._parse_response(response)
        assert result.empathy_adjustment == 0.4

    def test_parse_proactive_reply(self):
        response = '{"proactive_reply": -0.08, "confidence": 0.6}'
        result = LLMExtractor._parse_response(response)
        assert result.proactive_reply_delta == -0.08

    def test_parse_full_v2_response(self):
        """解析包含所有 v2 维度的完整响应"""
        response = '''{
            "interests": {"摄影": 0.8},
            "social_style": "外向",
            "reply_preference": "brief",
            "formality": 0.3,
            "directness": 0.5,
            "humor": 0.2,
            "empathy": 0.1,
            "topic_blacklist": ["政治"],
            "work_info": "软件工程师",
            "work_style": "远程",
            "work_challenge": "deadline紧",
            "life_info": "爱旅行",
            "lifestyle": "夜猫子",
            "emotional_triggers": ["被忽视"],
            "emotional_soothers": {"音乐": "治愈"},
            "social_boundaries": {"收入": "不讨论"},
            "personality": {
                "openness": 0.08,
                "conscientiousness": 0.05,
                "extraversion": 0.07,
                "agreeableness": -0.03,
                "neuroticism": -0.05
            },
            "proactive_reply": 0.05,
            "confidence": 0.9
        }'''
        result = LLMExtractor._parse_response(response)
        assert result is not None
        assert result.work_style == "远程"
        assert result.lifestyle == "夜猫子"
        assert result.personality_openness_delta == 0.08
        assert result.directness_adjustment == 0.5
        assert result.proactive_reply_delta == 0.05
        assert result.confidence == 0.9


# ==============================================================
# apply_extraction_result 测试（LLM/Hybrid 模式入口）
# ==============================================================

class TestApplyExtractionResult_NewFields:
    """apply_extraction_result 新字段应用"""

    def test_work_style_applied(self, persona):
        result = ExtractionResult(source="llm", work_style="远程", confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.work_style == "远程"
        assert any("work_style" in c.field_name for c in changes)

    def test_work_challenge_applied(self, persona):
        result = ExtractionResult(source="llm", work_challenge="deadline紧",
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert "deadline紧" in persona.work_challenges

    def test_work_preferences_applied(self, persona):
        result = ExtractionResult(source="llm",
                                  work_preferences={"工具": "VSCode"},
                                  confidence=0.7)
        changes = persona.apply_extraction_result(result)
        assert "工具" in persona.work_preferences

    def test_lifestyle_applied(self, persona):
        result = ExtractionResult(source="llm", lifestyle="夜猫子",
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.lifestyle == "夜猫子"

    def test_life_preferences_applied(self, persona):
        result = ExtractionResult(source="llm",
                                  life_preferences={"饮食": "素食"},
                                  confidence=0.7)
        changes = persona.apply_extraction_result(result)
        assert "饮食" in persona.life_preferences

    def test_emotional_triggers_applied(self, persona):
        result = ExtractionResult(source="llm",
                                  emotional_triggers=["被批评", "被忽视"],
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert "被批评" in persona.emotional_triggers
        assert "被忽视" in persona.emotional_triggers

    def test_emotional_triggers_dedup(self, persona):
        persona.emotional_triggers = ["被批评"]
        result = ExtractionResult(source="llm",
                                  emotional_triggers=["被批评"],
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.emotional_triggers.count("被批评") == 1

    def test_emotional_soothers_applied(self, persona):
        result = ExtractionResult(source="llm",
                                  emotional_soothers={"音乐": "治愈"},
                                  confidence=0.7)
        changes = persona.apply_extraction_result(result)
        assert "音乐" in persona.emotional_soothers

    def test_social_boundaries_applied(self, persona):
        result = ExtractionResult(source="llm",
                                  social_boundaries={"政治": "不讨论"},
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert "政治" in persona.social_boundaries

    def test_personality_openness_delta(self, persona):
        result = ExtractionResult(source="llm",
                                  personality_openness_delta=0.08,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.personality_openness == pytest.approx(0.58, abs=0.01)

    def test_personality_neuroticism_decrease(self, persona):
        result = ExtractionResult(source="llm",
                                  personality_neuroticism_delta=-0.05,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.personality_neuroticism < 0.5

    def test_personality_capped_at_bounds(self, persona):
        persona.personality_openness = 0.98
        result = ExtractionResult(source="llm",
                                  personality_openness_delta=0.1,
                                  confidence=0.8)
        persona.apply_extraction_result(result)
        assert persona.personality_openness <= 1.0

    def test_directness_applied(self, persona):
        result = ExtractionResult(source="llm", directness_adjustment=0.5,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        # LLM 模式 delta * 0.2 = 0.1
        assert persona.communication_directness > 0.5

    def test_humor_applied(self, persona):
        result = ExtractionResult(source="llm", humor_adjustment=-0.5,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.communication_humor < 0.5

    def test_empathy_applied(self, persona):
        result = ExtractionResult(source="llm", empathy_adjustment=0.3,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.communication_empathy > 0.5

    def test_proactive_reply_increase(self, persona):
        result = ExtractionResult(source="llm", proactive_reply_delta=0.08,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.proactive_reply_preference > 0.5

    def test_proactive_reply_decrease(self, persona):
        result = ExtractionResult(source="llm", proactive_reply_delta=-0.08,
                                  confidence=0.8)
        changes = persona.apply_extraction_result(result)
        assert persona.proactive_reply_preference < 0.5

    def test_proactive_reply_capped(self, persona):
        persona.proactive_reply_preference = 0.02
        result = ExtractionResult(source="llm", proactive_reply_delta=-0.1,
                                  confidence=0.8)
        persona.apply_extraction_result(result)
        assert persona.proactive_reply_preference >= 0.0

    def test_rule_mode_directness(self, persona):
        """Rule模式 directness_adjustment 直接应用"""
        result = ExtractionResult(source="rule", directness_adjustment=0.1,
                                  confidence=0.5)
        persona.apply_extraction_result(result)
        assert persona.communication_directness == pytest.approx(0.6, abs=0.01)

    def test_all_dimensions_in_one_result(self, persona):
        """一次提取结果包含所有维度"""
        result = ExtractionResult(
            source="hybrid",
            interests={"摄影": 0.8},
            social_style="外向",
            reply_style_preference="brief",
            formality_adjustment=0.1,
            directness_adjustment=0.1,
            humor_adjustment=0.1,
            empathy_adjustment=0.1,
            topic_blacklist=["政治"],
            work_info="工程师",
            work_style="远程",
            work_challenge="进度紧",
            lifestyle="夜猫子",
            emotional_triggers=["被批评"],
            emotional_soothers={"音乐": "治愈"},
            social_boundaries={"收入": "不聊"},
            personality_openness_delta=0.05,
            personality_conscientiousness_delta=0.05,
            personality_extraversion_delta=0.05,
            personality_agreeableness_delta=0.05,
            personality_neuroticism_delta=-0.05,
            proactive_reply_delta=0.05,
            trust_delta=0.1,
            intimacy_delta=0.1,
            confidence=0.9,
        )
        changes = persona.apply_extraction_result(result)
        assert len(changes) > 10  # 应该有很多变更
        assert persona.work_style == "远程"
        assert persona.lifestyle == "夜猫子"
        assert "被批评" in persona.emotional_triggers
        assert persona.personality_openness > 0.5
        assert persona.communication_directness > 0.5
        assert persona.proactive_reply_preference > 0.5


# ==============================================================
# PersonaExtractor._merge_results 测试
# ==============================================================

class TestMergeResults_NewDimensions:
    """合并规则与LLM结果的新维度"""

    def test_merge_work_style_llm_priority(self):
        rule = ExtractionResult(source="rule", work_style="坐班")
        llm = ExtractionResult(source="llm", work_style="远程")
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.work_style == "远程"

    def test_merge_work_style_rule_fallback(self):
        rule = ExtractionResult(source="rule", work_style="坐班")
        llm = ExtractionResult(source="llm")
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.work_style == "坐班"

    def test_merge_lifestyle_llm_priority(self):
        rule = ExtractionResult(source="rule", lifestyle="早起")
        llm = ExtractionResult(source="llm", lifestyle="夜猫子")
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.lifestyle == "夜猫子"

    def test_merge_emotional_triggers_combined(self):
        rule = ExtractionResult(source="rule", emotional_triggers=["怕黑"])
        llm = ExtractionResult(source="llm", emotional_triggers=["被批评"])
        merged = PersonaExtractor._merge_results(rule, llm)
        assert "怕黑" in merged.emotional_triggers
        assert "被批评" in merged.emotional_triggers

    def test_merge_social_boundaries_combined(self):
        rule = ExtractionResult(source="rule",
                                social_boundaries={"政治": "不聊"})
        llm = ExtractionResult(source="llm",
                                social_boundaries={"收入": "不问"})
        merged = PersonaExtractor._merge_results(rule, llm)
        assert "政治" in merged.social_boundaries
        assert "收入" in merged.social_boundaries

    def test_merge_personality_llm_priority(self):
        rule = ExtractionResult(source="rule",
                                personality_openness_delta=0.05)
        llm = ExtractionResult(source="llm",
                                personality_openness_delta=0.08)
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.personality_openness_delta == 0.08

    def test_merge_personality_rule_fallback(self):
        rule = ExtractionResult(source="rule",
                                personality_openness_delta=0.05)
        llm = ExtractionResult(source="llm")  # delta=0.0
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.personality_openness_delta == 0.05

    def test_merge_directness_llm_priority(self):
        rule = ExtractionResult(source="rule", directness_adjustment=0.1)
        llm = ExtractionResult(source="llm", directness_adjustment=-0.5)
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.directness_adjustment == -0.5

    def test_merge_humor_llm_priority(self):
        rule = ExtractionResult(source="rule", humor_adjustment=0.1)
        llm = ExtractionResult(source="llm", humor_adjustment=0.3)
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.humor_adjustment == 0.3

    def test_merge_proactive_reply_llm_priority(self):
        rule = ExtractionResult(source="rule", proactive_reply_delta=0.1)
        llm = ExtractionResult(source="llm", proactive_reply_delta=-0.08)
        merged = PersonaExtractor._merge_results(rule, llm)
        assert merged.proactive_reply_delta == -0.08


# ==============================================================
# to_injection_view 新维度测试
# ==============================================================

class TestInjectionView_NewDimensions:
    """to_injection_view 新维度输出"""

    def test_emotional_triggers_in_view(self, persona):
        persona.emotional_baseline = "joy"
        persona.emotional_triggers = ["被批评", "被忽视"]
        view = persona.to_injection_view()
        assert "triggers" in view["emotional"]
        assert "被批评" in view["emotional"]["triggers"]

    def test_emotional_soothers_in_view(self, persona):
        persona.emotional_baseline = "joy"
        persona.emotional_soothers = {"音乐": "治愈"}
        view = persona.to_injection_view()
        assert "soothers" in view["emotional"]

    def test_work_dimensions_in_view(self, persona):
        persona.work_style = "远程"
        persona.work_goals = ["完成项目"]
        persona.work_challenges = ["deadline紧"]
        view = persona.to_injection_view()
        assert "work" in view
        assert view["work"]["style"] == "远程"
        assert "deadline紧" in view["work"]["challenges"]

    def test_life_dimensions_in_view(self, persona):
        persona.lifestyle = "夜猫子"
        persona.life_preferences = {"饮食": "素食"}
        view = persona.to_injection_view()
        assert "life" in view
        assert view["life"]["style"] == "夜猫子"

    def test_personality_in_view_only_deviated(self, persona):
        """只有偏离默认值的特质会出现在视图中"""
        persona.personality_openness = 0.8
        persona.personality_neuroticism = 0.2
        # 其他保持0.5
        view = persona.to_injection_view()
        assert "personality" in view
        assert "openness" in view["personality"]
        assert "neuroticism" in view["personality"]
        assert "conscientiousness" not in view["personality"]

    def test_personality_not_in_view_when_default(self, persona):
        """所有特质在默认值时不出现"""
        view = persona.to_injection_view()
        assert "personality" not in view

    def test_communication_empathy_in_view(self, persona):
        view = persona.to_injection_view()
        assert "empathy" in view["communication"]

    def test_social_boundaries_in_view(self, persona):
        persona.social_boundaries = {"政治": "不讨论"}
        view = persona.to_injection_view()
        assert "boundaries" in view["relationship"]

    def test_work_not_in_view_when_empty(self, persona):
        """工作维度全空时不出现 work 键"""
        view = persona.to_injection_view()
        assert "work" not in view

    def test_life_not_in_view_when_empty(self, persona):
        """生活维度全空时不出现 life 键"""
        view = persona.to_injection_view()
        assert "life" not in view


# ==============================================================
# 端到端集成测试
# ==============================================================

class TestIntegration_V2:
    """端到端：从消息到画像更新"""

    def test_rule_mode_full_flow(self, persona):
        """Rule模式完整流程：多条消息覆盖所有新维度"""
        msgs = [
            _make_memory("我在家远程办公", mem_type="fact"),
            _make_memory("最近项目压力很大", mem_type="fact",
                         summary="项目压力大"),
            _make_memory("我是夜猫子", mem_type="fact"),
            _make_memory("别聊政治", mem_type="relationship"),
            _make_memory("直说就行不用绕弯", mem_type="interaction"),
            _make_memory("哈哈哈太好笑了", mem_type="interaction"),
            _make_memory("欢迎多聊聊", mem_type="interaction"),
            _make_memory("我很好奇想探索", mem_type="interaction"),
            _make_memory("被老板骂了很伤心", mem_type="emotion",
                         subtype="sadness", weight=0.8),
        ]
        for m in msgs:
            persona.update_from_memory(m)

        assert persona.work_style == "远程"
        assert "项目压力大" in persona.work_challenges
        assert persona.lifestyle == "夜猫子"
        assert len(persona.social_boundaries) > 0
        assert persona.communication_directness > 0.5
        assert persona.communication_humor > 0.5
        assert persona.proactive_reply_preference > 0.5
        assert persona.personality_openness > 0.5
        assert len(persona.emotional_triggers) > 0

    def test_llm_mode_full_flow(self, persona):
        """LLM模式: apply_extraction_result 覆盖所有新维度"""
        result = ExtractionResult(
            source="llm",
            work_style="远程",
            work_challenge="deadline紧",
            lifestyle="早起",
            emotional_triggers=["被忽视"],
            emotional_soothers={"运动": "解压"},
            social_boundaries={"收入": "不问"},
            personality_openness_delta=0.08,
            personality_conscientiousness_delta=0.06,
            personality_extraversion_delta=-0.04,
            personality_agreeableness_delta=0.03,
            personality_neuroticism_delta=-0.05,
            directness_adjustment=0.3,
            humor_adjustment=-0.2,
            empathy_adjustment=0.4,
            proactive_reply_delta=-0.06,
            confidence=0.85,
        )
        changes = persona.apply_extraction_result(result)
        assert persona.work_style == "远程"
        assert "deadline紧" in persona.work_challenges
        assert persona.lifestyle == "早起"
        assert "被忽视" in persona.emotional_triggers
        assert "运动" in persona.emotional_soothers
        assert "收入" in persona.social_boundaries
        assert persona.personality_openness > 0.5
        assert persona.personality_neuroticism < 0.5
        assert persona.communication_directness > 0.5
        assert persona.communication_humor < 0.5
        assert persona.communication_empathy > 0.5
        assert persona.proactive_reply_preference < 0.5

    def test_hybrid_mode_merge_preserves_all(self, persona):
        """Hybrid模式: 合并后所有维度完整"""
        rule = ExtractionResult(
            source="rule",
            work_style="坐班",
            directness_adjustment=0.1,
            personality_openness_delta=0.05,
        )
        llm = ExtractionResult(
            source="llm",
            lifestyle="夜猫子",
            humor_adjustment=0.3,
            personality_neuroticism_delta=-0.05,
        )
        merged = PersonaExtractor._merge_results(rule, llm)
        changes = persona.apply_extraction_result(merged)

        # Rule 的维度被保留
        assert persona.work_style == "坐班"
        assert persona.personality_openness > 0.5
        # LLM 的维度被保留
        assert persona.lifestyle == "夜猫子"
        assert persona.personality_neuroticism < 0.5


class TestKeywordMapsFallback:
    """关键词配置默认回退加载行为"""

    def test_keyword_maps_loads_packaged_yaml_by_default(self):
        kw = KeywordMaps()
        kw.load()
        # package yaml 中存在，builtin defaults 中不保证完整
        assert "摄影" in kw.interests
