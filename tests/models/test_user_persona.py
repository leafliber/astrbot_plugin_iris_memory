"""
UserPersona测试
测试用户画像数据模型的核心功能
"""

import pytest
from datetime import datetime
from iris_memory.models.user_persona import UserPersona
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, EmotionType


@pytest.fixture
def sample_persona():
    """示例用户画像"""
    return UserPersona(
        user_id="user_123",
        version=1,
        work_style="创新",
        work_goals=["完成项目", "提升技能"],
        lifestyle="规律",
        interests={"编程": 0.9, "阅读": 0.7},
        emotional_baseline="joy",
        emotional_volatility=0.3,
        personality_openness=0.8,
        personality_conscientiousness=0.7,
        personality_extraversion=0.6,
        communication_formality=0.4
    )


class TestUserPersonaInit:
    """测试初始化功能"""
    
    def test_init_with_defaults(self):
        """测试使用默认值初始化"""
        persona = UserPersona()

        assert persona.user_id == ""
        assert persona.version == 1
        assert isinstance(persona.last_updated, datetime)
        assert persona.work_style is None
        assert len(persona.work_goals) == 0
        assert len(persona.habits) == 0
        assert persona.emotional_baseline == "neutral"
        assert persona.emotional_volatility == 0.5
        assert len(persona.emotional_patterns) == 0
    
    def test_init_with_values(self, sample_persona):
        """测试使用指定值初始化"""
        assert sample_persona.user_id == "user_123"
        assert sample_persona.version == 1
        assert sample_persona.work_style == "创新"
        assert "完成项目" in sample_persona.work_goals
        assert "提升技能" in sample_persona.work_goals
        assert sample_persona.lifestyle == "规律"
        assert sample_persona.interests["编程"] == 0.9
        assert sample_persona.interests["阅读"] == 0.7
        assert sample_persona.emotional_baseline == "joy"
        assert sample_persona.emotional_volatility == 0.3
        assert sample_persona.personality_openness == 0.8
    
    def test_init_big_five_personality(self):
        """测试Big Five人格维度初始化"""
        persona = UserPersona(
            personality_openness=0.9,
            personality_conscientiousness=0.8,
            personality_extraversion=0.7,
            personality_agreeableness=0.6,
            personality_neuroticism=0.2
        )
        
        assert persona.personality_openness == 0.9
        assert persona.personality_conscientiousness == 0.8
        assert persona.personality_extraversion == 0.7
        assert persona.personality_agreeableness == 0.6
        assert persona.personality_neuroticism == 0.2
    
    def test_init_communication_dimensions(self):
        """测试沟通维度初始化"""
        persona = UserPersona(
            communication_formality=0.8,
            communication_directness=0.7,
            communication_humor=0.6,
            communication_empathy=0.9
        )
        
        assert persona.communication_formality == 0.8
        assert persona.communication_directness == 0.7
        assert persona.communication_humor == 0.6
        assert persona.communication_empathy == 0.9
    
    def test_init_hourly_distribution(self):
        """测试24小时活跃度分布初始化"""
        persona = UserPersona()

        assert len(persona.hourly_distribution) == 24
        assert all(v >= 0.0 for v in persona.hourly_distribution)


class TestUserPersonaSerialization:
    """测试序列化功能"""
    
    def test_to_dict_basic(self, sample_persona):
        """测试基本序列化"""
        data = sample_persona.to_dict()
        
        assert data['user_id'] == "user_123"
        assert data['version'] == 1
        assert data['work_style'] == "创新"
        assert data['lifestyle'] == "规律"
        assert 'last_updated' in data
        assert isinstance(data['last_updated'], str)  # datetime被转换为字符串
    
    def test_to_dict_datetime_conversion(self, sample_persona):
        """测试datetime字段转换"""
        persona = UserPersona(user_id="test")
        
        data = persona.to_dict()
        
        # last_updated应该被转换为ISO格式字符串
        assert 'last_updated' in data
        datetime.fromisoformat(data['last_updated'])  # 验证可以解析
    
    def test_from_dict_basic(self):
        """测试基本反序列化"""
        data = {
            'user_id': 'user_456',
            'version': 2,
            'work_style': '严谨',
            'emotional_baseline': 'sadness',
            'emotional_volatility': 0.6
        }
        
        persona = UserPersona.from_dict(data)
        
        assert persona.user_id == 'user_456'
        assert persona.version == 2
        assert persona.work_style == '严谨'
        assert persona.emotional_baseline == 'sadness'
        assert persona.emotional_volatility == 0.6
    
    def test_from_dict_datetime_parsing(self):
        """测试datetime字段解析"""
        data = {
            'user_id': 'user_789',
            'last_updated': '2024-01-15T10:30:00'
        }
        
        persona = UserPersona.from_dict(data)
        
        assert isinstance(persona.last_updated, datetime)
        assert persona.last_updated.year == 2024
        assert persona.last_updated.month == 1
        assert persona.last_updated.day == 15
    
    def test_serialization_roundtrip(self, sample_persona):
        """测试序列化和反序列化的往返"""
        # 序列化
        data = sample_persona.to_dict()
        
        # 反序列化
        new_persona = UserPersona.from_dict(data)
        
        # 验证数据一致
        assert new_persona.user_id == sample_persona.user_id
        assert new_persona.version == sample_persona.version
        assert new_persona.work_style == sample_persona.work_style
        assert new_persona.lifestyle == sample_persona.lifestyle
        assert new_persona.emotional_baseline == sample_persona.emotional_baseline
        assert new_persona.interests == sample_persona.interests
    
    def test_from_dict_with_lists_and_dicts(self):
        """测试包含列表和字典的反序列化"""
        data = {
            'user_id': 'user_001',
            'work_goals': ['goal1', 'goal2', 'goal3'],
            'habits': ['habit1', 'habit2'],
            'interests': {'sports': 0.8, 'music': 0.9},
            'work_preferences': {'remote': True, 'flexible': True}
        }
        
        persona = UserPersona.from_dict(data)
        
        assert len(persona.work_goals) == 3
        assert 'goal1' in persona.work_goals
        assert len(persona.habits) == 2
        assert persona.interests['sports'] == 0.8
        assert persona.work_preferences['remote'] is True


class TestUserPersonaEvidenceTracking:
    """测试证据追踪功能"""
    
    def test_add_memory_evidence_confirmed(self, sample_persona):
        """测试添加确认证据"""
        sample_persona.add_memory_evidence("mem_001", "confirmed")
        
        assert "mem_001" in sample_persona.evidence_confirmed
        assert len(sample_persona.evidence_confirmed) == 1
    
    def test_add_memory_evidence_inferred(self, sample_persona):
        """测试添加推断证据"""
        sample_persona.add_memory_evidence("mem_002", "inferred")
        
        assert "mem_002" in sample_persona.evidence_inferred
        assert len(sample_persona.evidence_inferred) == 1
    
    def test_add_memory_evidence_contested(self, sample_persona):
        """测试添加争议证据"""
        sample_persona.add_memory_evidence("mem_003", "contested")
        
        assert "mem_003" in sample_persona.evidence_contested
        assert len(sample_persona.evidence_contested) == 1
    
    def test_add_memory_evidence_duplicate(self, sample_persona):
        """测试添加重复证据（应该被忽略）"""
        sample_persona.add_memory_evidence("mem_004", "confirmed")
        sample_persona.add_memory_evidence("mem_004", "confirmed")  # 重复添加
        
        assert sample_persona.evidence_confirmed.count("mem_004") == 1
    
    def test_add_memory_evidence_multiple_types(self, sample_persona):
        """测试添加多种类型的证据"""
        sample_persona.add_memory_evidence("mem_001", "confirmed")
        sample_persona.add_memory_evidence("mem_002", "inferred")
        sample_persona.add_memory_evidence("mem_003", "contested")
        sample_persona.add_memory_evidence("mem_004", "confirmed")
        
        assert len(sample_persona.evidence_confirmed) == 2
        assert len(sample_persona.evidence_inferred) == 1
        assert len(sample_persona.evidence_contested) == 1
        assert "mem_001" in sample_persona.evidence_confirmed
        assert "mem_002" in sample_persona.evidence_inferred
        assert "mem_003" in sample_persona.evidence_contested
        assert "mem_004" in sample_persona.evidence_confirmed


class TestUserPersonaUpdateFromMemory:
    """测试从记忆更新画像功能"""
    
    def test_update_from_memory_basic(self, sample_persona):
        """测试基本更新"""
        old_updated = sample_persona.last_updated

        memory = Memory(
            type=MemoryType.FACT,
            content="这是一条测试记忆",
            user_id="user_123"
        )

        sample_persona.update_from_memory(memory)

        assert sample_persona.last_updated > old_updated
    
    def test_update_from_emotional_memory(self, sample_persona):
        """测试从情感记忆更新"""
        memory = Memory(
            type=MemoryType.EMOTION,
            subtype="joy",
            emotional_weight=0.8,
            content="我感到很开心",
            user_id="user_123"
        )

        sample_persona.update_from_memory(memory)

        # 情感基线应该更新为joy
        assert sample_persona.emotional_baseline == "joy"
        # 情感模式统计应该增加
        assert sample_persona.emotional_patterns.get("joy", 0) == 1
    
    def test_update_from_emotional_memory_low_weight(self, sample_persona):
        """测试从低权重情感记忆更新（不应改变基线）"""
        original_baseline = sample_persona.emotional_baseline

        memory = Memory(
            type=MemoryType.EMOTION,
            subtype="sadness",
            emotional_weight=0.5,  # 低于0.7阈值
            content="有点难过",
            user_id="user_123"
        )

        sample_persona.update_from_memory(memory)

        # 情感基线不应该改变
        assert sample_persona.emotional_baseline == original_baseline
        # 但情感模式统计仍然应该更新
        assert sample_persona.emotional_patterns.get("sadness", 0) == 1
    
    def test_update_from_fact_memory_work(self, sample_persona):
        """测试从工作相关事实记忆更新"""
        memory = Memory(
            type=MemoryType.FACT,
            content="我在工作方面希望能够提升技能",
            summary="希望提升工作技能",
            user_id="user_123"
        )

        initial_count = len(sample_persona.work_goals)

        sample_persona.update_from_memory(memory)

        # 工作目标应该被添加
        assert len(sample_persona.work_goals) == initial_count + 1
        assert "希望提升工作技能" in sample_persona.work_goals
    
    def test_update_from_fact_memory_life(self, sample_persona):
        """测试从生活相关事实记忆更新"""
        memory = Memory(
            type=MemoryType.FACT,
            content="我喜欢阅读和运动",
            summary="喜欢阅读和运动",
            user_id="user_123"
        )

        initial_count = len(sample_persona.habits)

        sample_persona.update_from_memory(memory)

        # 生活习惯应该被添加
        assert len(sample_persona.habits) == initial_count + 1
        assert "喜欢阅读和运动" in sample_persona.habits
    
    def test_update_from_relationship_memory(self, sample_persona):
        """测试从关系记忆更新"""
        memory = Memory(
            type=MemoryType.RELATIONSHIP,
            summary="我很信任他",
            content="我对朋友非常信任",
            user_id="user_123"
        )

        original_trust = sample_persona.trust_level

        sample_persona.update_from_memory(memory)

        # 信任等级应该提升
        assert sample_persona.trust_level > original_trust
        assert sample_persona.trust_level <= 1.0
    
    def test_update_from_relationship_memory_intimacy(self, sample_persona):
        """测试从亲密关系记忆更新"""
        memory = Memory(
            type=MemoryType.RELATIONSHIP,
            summary="我们很亲密",
            content="我和家人关系很亲密",
            user_id="user_123"
        )

        original_intimacy = sample_persona.intimacy_level

        sample_persona.update_from_memory(memory)

        # 亲密程度应该提升
        assert sample_persona.intimacy_level > original_intimacy
        assert sample_persona.intimacy_level <= 1.0

    def test_update_from_memory_trust_cap(self, sample_persona):
        """测试信任等级上限"""
        sample_persona.trust_level = 0.95  # 接近上限

        memory = Memory(
            type=MemoryType.RELATIONSHIP,
            summary="非常信任",
            content="我完全信任",
            user_id="user_123"
        )

        sample_persona.update_from_memory(memory)

        # 不应该超过1.0
        assert sample_persona.trust_level <= 1.0


class TestUserPersonaEmotionalDimensions:
    """测试情感维度功能"""
    
    def test_emotional_baseline_valid_values(self):
        """测试情感基线有效值"""
        valid_emotions = ["joy", "sadness", "anger", "fear", "neutral", "anxiety"]
        
        for emotion in valid_emotions:
            persona = UserPersona(emotional_baseline=emotion)
            assert persona.emotional_baseline == emotion
    
    def test_emotional_volatility_range(self):
        """测试情感波动性范围"""
        # 测试边界值
        persona_low = UserPersona(emotional_volatility=0.0)
        persona_high = UserPersona(emotional_volatility=1.0)
        persona_mid = UserPersona(emotional_volatility=0.5)
        
        assert persona_low.emotional_volatility == 0.0
        assert persona_high.emotional_volatility == 1.0
        assert persona_mid.emotional_volatility == 0.5
    
    def test_emotional_triggers(self, sample_persona):
        """测试情感触发器"""
        triggers = ["批评", "失败", "压力"]
        for trigger in triggers:
            sample_persona.emotional_triggers.append(trigger)
        
        assert len(sample_persona.emotional_triggers) == 3
        assert "批评" in sample_persona.emotional_triggers
    
    def test_emotional_soothers(self, sample_persona):
        """测试情感缓解因素"""
        sample_persona.emotional_soothers = {
            "音乐": {"effectiveness": 0.8},
            "运动": {"effectiveness": 0.7},
            "休息": {"effectiveness": 0.9}
        }
        
        assert len(sample_persona.emotional_soothers) == 3
        assert sample_persona.emotional_soothers["音乐"]["effectiveness"] == 0.8
    
    def test_emotional_trajectory(self, sample_persona):
        """测试情感趋势"""
        valid_trajectories = ["improving", "deteriorating", "stable", "volatile"]
        
        for trajectory in valid_trajectories:
            sample_persona.emotional_trajectory = trajectory
            assert sample_persona.emotional_trajectory == trajectory
    
    def test_negative_ratio(self, sample_persona):
        """测试负面情感占比"""
        sample_persona.negative_ratio = 0.4
        
        assert sample_persona.negative_ratio == 0.4


class TestUserPersonaWorkDimensions:
    """测试工作维度功能"""
    
    def test_work_style(self, sample_persona):
        """测试工作风格"""
        styles = ["严谨", "创新", "高效", "灵活", "传统"]
        
        for style in styles:
            sample_persona.work_style = style
            assert sample_persona.work_style == style
    
    def test_work_goals(self, sample_persona):
        """测试工作目标"""
        goals = ["完成项目", "提升技能", "升职加薪", "团队协作"]
        
        for goal in goals:
            if goal not in sample_persona.work_goals:
                sample_persona.work_goals.append(goal)
        
        assert len(sample_persona.work_goals) >= len(goals)
    
    def test_work_challenges(self, sample_persona):
        """测试工作挑战"""
        challenges = ["时间管理", "技术难题", "团队沟通"]
        
        for challenge in challenges:
            sample_persona.work_challenges.append(challenge)
        
        assert len(sample_persona.work_challenges) == len(challenges)
    
    def test_work_preferences(self, sample_persona):
        """测试工作偏好"""
        sample_persona.work_preferences = {
            "work_environment": "办公室",
            "working_hours": "9-6",
            "team_size": "small"
        }
        
        assert sample_persona.work_preferences["work_environment"] == "办公室"
        assert sample_persona.work_preferences["working_hours"] == "9-6"


class TestUserPersonaLifeDimensions:
    """测试生活维度功能"""
    
    def test_lifestyle(self, sample_persona):
        """测试生活方式"""
        lifestyles = ["规律", "忙碌", "悠闲", "不规律", "健康"]
        
        for lifestyle in lifestyles:
            sample_persona.lifestyle = lifestyle
            assert sample_persona.lifestyle == lifestyle
    
    def test_interests(self, sample_persona):
        """测试兴趣领域"""
        interests = {
            "编程": 0.9,
            "阅读": 0.8,
            "运动": 0.7,
            "音乐": 0.6
        }
        
        sample_persona.interests = interests
        
        assert len(sample_persona.interests) == len(interests)
        assert sample_persona.interests["编程"] == 0.9
    
    def test_habits(self, sample_persona):
        """测试习惯"""
        habits = ["早起", "阅读", "运动", "早睡"]
        
        for habit in habits:
            if habit not in sample_persona.habits:
                sample_persona.habits.append(habit)
        
        assert len(sample_persona.habits) >= len(habits)


class TestUserPersonaSocialDimensions:
    """测试社交维度功能"""
    
    def test_social_style(self, sample_persona):
        """测试社交风格"""
        styles = ["外向", "内向", "温和", "直率"]
        
        for style in styles:
            sample_persona.social_style = style
            assert sample_persona.social_style == style
    
    def test_social_boundaries(self, sample_persona):
        """测试社交边界"""
        sample_persona.social_boundaries = {
            "personal_space": "moderate",
            "sharing_personal_info": "selective",
            "emotional_openness": "gradual"
        }
        
        assert len(sample_persona.social_boundaries) == 3
        assert sample_persona.social_boundaries["personal_space"] == "moderate"
    
    def test_trust_level_range(self, sample_persona):
        """测试信任等级范围"""
        for level in [0.0, 0.5, 1.0]:
            sample_persona.trust_level = level
            assert 0.0 <= sample_persona.trust_level <= 1.0
    
    def test_intimacy_level_range(self, sample_persona):
        """测试亲密程度范围"""
        for level in [0.0, 0.5, 1.0]:
            sample_persona.intimacy_level = level
            assert 0.0 <= sample_persona.intimacy_level <= 1.0


class TestUserPersonaBehaviorPatterns:
    """测试行为模式功能"""
    
    def test_hourly_distribution_complete(self, sample_persona):
        """测试24小时活跃度分布完整性"""
        assert len(sample_persona.hourly_distribution) == 24
    
    def test_hourly_distribution_values(self, sample_persona):
        """测试24小时活跃度分布值"""
        # 设置一些值
        sample_persona.hourly_distribution = [0.1 * i for i in range(24)]
        
        assert sample_persona.hourly_distribution[0] == 0.0
        assert sample_persona.hourly_distribution[12] == 1.2
        assert sample_persona.hourly_distribution[23] == 2.3
    
    def test_topic_sequences(self, sample_persona):
        """测试话题转换序列"""
        topics = ["天气", "工作", "生活", "情感", "学习"]
        sample_persona.topic_sequences.extend(topics)
        
        assert len(sample_persona.topic_sequences) == len(topics)
        assert sample_persona.topic_sequences[0] == "天气"
        assert sample_persona.topic_sequences[-1] == "学习"
    
    def test_memory_cooccurrence(self, sample_persona):
        """测试记忆共现关系"""
        sample_persona.memory_cooccurrence = {
            "mem_001": ["mem_002", "mem_003"],
            "mem_002": ["mem_001", "mem_004"],
            "mem_003": ["mem_001"]
        }
        
        assert "mem_002" in sample_persona.memory_cooccurrence["mem_001"]
        assert "mem_003" in sample_persona.memory_cooccurrence["mem_001"]
        assert len(sample_persona.memory_cooccurrence["mem_002"]) == 2


class TestUserPersonaBigFivePersonality:
    """测试Big Five人格维度"""
    
    def test_personality_openness_range(self, sample_persona):
        """测试开放性维度范围"""
        for value in [0.0, 0.5, 1.0]:
            sample_persona.personality_openness = value
            assert 0.0 <= sample_persona.personality_openness <= 1.0
    
    def test_personality_conscientiousness_range(self, sample_persona):
        """测试尽责性维度范围"""
        for value in [0.0, 0.5, 1.0]:
            sample_persona.personality_conscientiousness = value
            assert 0.0 <= sample_persona.personality_conscientiousness <= 1.0
    
    def test_personality_extraversion_range(self, sample_persona):
        """测试外向性维度范围"""
        for value in [0.0, 0.5, 1.0]:
            sample_persona.personality_extraversion = value
            assert 0.0 <= sample_persona.personality_extraversion <= 1.0
    
    def test_personality_agreeableness_range(self, sample_persona):
        """测试宜人性维度范围"""
        for value in [0.0, 0.5, 1.0]:
            sample_persona.personality_agreeableness = value
            assert 0.0 <= sample_persona.personality_agreeableness <= 1.0
    
    def test_personality_neuroticism_range(self, sample_persona):
        """测试神经质维度范围"""
        for value in [0.0, 0.5, 1.0]:
            sample_persona.personality_neuroticism = value
            assert 0.0 <= sample_persona.personality_neuroticism <= 1.0
    
    def test_personality_profile_complete(self):
        """测试完整的人格画像"""
        persona = UserPersona(
            personality_openness=0.8,
            personality_conscientiousness=0.7,
            personality_extraversion=0.6,
            personality_agreeableness=0.5,
            personality_neuroticism=0.3
        )
        
        assert all([
            0.0 <= persona.personality_openness <= 1.0,
            0.0 <= persona.personality_conscientiousness <= 1.0,
            0.0 <= persona.personality_extraversion <= 1.0,
            0.0 <= persona.personality_agreeableness <= 1.0,
            0.0 <= persona.personality_neuroticism <= 1.0
        ])


class TestUserPersonaEdgeCases:
    """测试边界情况"""
    
    def test_empty_user_id(self):
        """测试空用户ID"""
        persona = UserPersona(user_id="")
        assert persona.user_id == ""
    
    def test_version_zero(self):
        """测试版本号为0"""
        persona = UserPersona(version=0)
        assert persona.version == 0
    
    def test_negative_values(self):
        """测试负值（边界情况）"""
        # 虽然不应该有负值，但测试代码的健壮性
        persona = UserPersona(emotional_volatility=-0.1)
        assert persona.emotional_volatility == -0.1  # 应该接受并存储
    
    def test_very_large_values(self):
        """测试非常大的值"""
        persona = UserPersona(
            emotional_volatility=100.0,
            trust_level=1000.0,
            personality_openness=999.0
        )
        
        assert persona.emotional_volatility == 100.0
        assert persona.trust_level == 1000.0
        assert persona.personality_openness == 999.0
    
    def test_unicode_content(self):
        """测试Unicode内容"""
        persona = UserPersona(
            user_id="用户_123",
            work_style="创新",
            habits=["阅读", "运动", "编程"],
            emotional_baseline="😊"  # emoji
        )
        
        assert persona.user_id == "用户_123"
        assert "😊" in persona.emotional_baseline


class TestUserPersonaIntegration:
    """测试集成场景"""
    
    def test_full_persona_workflow(self):
        """测试完整的画像工作流"""
        # 1. 创建初始画像
        persona = UserPersona(user_id="user_001")
        
        # 2. 从多条记忆更新
        memories = [
            Mock(type="emotion", subtype="joy", emotional_weight=0.8, content="很开心"),
            Mock(type="fact", content="我希望在工作中提升技能", summary="提升工作技能"),
            Mock(type="fact", content="我喜欢阅读和运动", summary="阅读和运动"),
            Mock(type="relationship", summary="很信任他", content="信任朋友")
        ]
        
        for memory in memories:
            persona.update_from_memory(memory)
        
        # 3. 添加证据
        persona.add_memory_evidence("mem_001", "confirmed")
        persona.add_memory_evidence("mem_002", "inferred")
        
        # 4. 验证结果
        assert persona.emotional_baseline == "joy"
        assert "提升工作技能" in persona.work_goals
        assert "阅读和运动" in persona.habits
        assert persona.trust_level > 0.5
        assert len(persona.evidence_confirmed) == 1
        assert len(persona.evidence_inferred) == 1
        
        # 5. 序列化和反序列化
        data = persona.to_dict()
        new_persona = UserPersona.from_dict(data)
        
        assert new_persona.user_id == persona.user_id
        assert new_persona.emotional_baseline == persona.emotional_baseline
        assert len(new_persona.work_goals) == len(persona.work_goals)
