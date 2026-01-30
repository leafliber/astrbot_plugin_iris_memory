"""
测试核心类型定义
测试 iris_memory.core.types 中的所有枚举和DecayRate类
"""

import pytest
from iris_memory.core.types import (
    MemoryType,
    ModalityType,
    QualityLevel,
    SensitivityLevel,
    StorageLayer,
    EmotionType,
    VerificationMethod,
    DecayRate,
    RetrievalStrategy,
    TriggerType
)


class TestMemoryType:
    """测试MemoryType枚举"""

    def test_memory_type_values(self):
        """测试记忆类型的值"""
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.EMOTION.value == "emotion"
        assert MemoryType.RELATIONSHIP.value == "relationship"
        assert MemoryType.INTERACTION.value == "interaction"
        assert MemoryType.INFERRED.value == "inferred"

    def test_memory_type_count(self):
        """测试记忆类型的数量"""
        assert len(MemoryType) == 5

    def test_memory_type_is_string_enum(self):
        """测试MemoryType是字符串枚举"""
        assert issubclass(MemoryType, str)
        assert MemoryType.FACT == "fact"


class TestModalityType:
    """测试ModalityType枚举"""

    def test_modality_type_values(self):
        """测试模态类型的值"""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.VIDEO.value == "video"

    def test_modality_type_count(self):
        """测试模态类型的数量"""
        assert len(ModalityType) == 4


class TestQualityLevel:
    """测试QualityLevel枚举"""

    def test_quality_level_values(self):
        """测试质量等级的值"""
        assert QualityLevel.UNCERTAIN.value == 1
        assert QualityLevel.LOW_CONFIDENCE.value == 2
        assert QualityLevel.MODERATE.value == 3
        assert QualityLevel.HIGH_CONFIDENCE.value == 4
        assert QualityLevel.CONFIRMED.value == 5

    def test_quality_level_ordering(self):
        """测试质量等级的顺序"""
        assert QualityLevel.UNCERTAIN < QualityLevel.LOW_CONFIDENCE
        assert QualityLevel.LOW_CONFIDENCE < QualityLevel.MODERATE
        assert QualityLevel.MODERATE < QualityLevel.HIGH_CONFIDENCE
        assert QualityLevel.HIGH_CONFIDENCE < QualityLevel.CONFIRMED

    def test_quality_level_count(self):
        """测试质量等级的数量"""
        assert len(QualityLevel) == 5


class TestSensitivityLevel:
    """测试SensitivityLevel枚举"""

    def test_sensitivity_level_values(self):
        """测试敏感度等级的值"""
        assert SensitivityLevel.PUBLIC.value == 0
        assert SensitivityLevel.PERSONAL.value == 1
        assert SensitivityLevel.PRIVATE.value == 2
        assert SensitivityLevel.SENSITIVE.value == 3
        assert SensitivityLevel.CRITICAL.value == 4

    def test_sensitivity_level_ordering(self):
        """测试敏感度等级的顺序"""
        assert SensitivityLevel.PUBLIC < SensitivityLevel.PERSONAL
        assert SensitivityLevel.PERSONAL < SensitivityLevel.PRIVATE
        assert SensitivityLevel.PRIVATE < SensitivityLevel.SENSITIVE
        assert SensitivityLevel.SENSITIVE < SensitivityLevel.CRITICAL

    def test_sensitivity_level_count(self):
        """测试敏感度等级的数量"""
        assert len(SensitivityLevel) == 5


class TestStorageLayer:
    """测试StorageLayer枚举"""

    def test_storage_layer_values(self):
        """测试存储层的值"""
        assert StorageLayer.WORKING.value == "working"
        assert StorageLayer.EPISODIC.value == "episodic"
        assert StorageLayer.SEMANTIC.value == "semantic"

    def test_storage_layer_count(self):
        """测试存储层的数量"""
        assert len(StorageLayer) == 3


class TestEmotionType:
    """测试EmotionType枚举"""

    def test_emotion_type_values(self):
        """测试情感类型的值"""
        assert EmotionType.JOY.value == "joy"
        assert EmotionType.SADNESS.value == "sadness"
        assert EmotionType.ANGER.value == "anger"
        assert EmotionType.FEAR.value == "fear"
        assert EmotionType.DISGUST.value == "disgust"
        assert EmotionType.SURPRISE.value == "surprise"
        assert EmotionType.NEUTRAL.value == "neutral"
        assert EmotionType.ANXIETY.value == "anxiety"
        assert EmotionType.EXCITEMENT.value == "excitement"
        assert EmotionType.CALM.value == "calm"

    def test_emotion_type_count(self):
        """测试情感类型的数量"""
        assert len(EmotionType) == 10


class TestVerificationMethod:
    """测试VerificationMethod枚举"""

    def test_verification_method_values(self):
        """测试验证方法的值"""
        assert VerificationMethod.USER_EXPLICIT.value == "user_explicit"
        assert VerificationMethod.MULTIPLE_MENTIONS.value == "multiple_mentions"
        assert VerificationMethod.CROSS_VALIDATION.value == "cross_validation"
        assert VerificationMethod.SYSTEM_INFERRED.value == "system_inferred"
        assert VerificationMethod.UNVERIFIED.value == "unverified"

    def test_verification_method_count(self):
        """测试验证方法的数量"""
        assert len(VerificationMethod) == 5


class TestDecayRate:
    """测试DecayRate类"""

    def test_decay_rate_values(self):
        """测试衰减率常量值"""
        # 测试兴趣衰减率：30天半衰期，ln(0.5)/30 ≈ 0.023
        assert abs(DecayRate.INTEREST - 0.023) < 0.001

        # 测试习惯衰减率：90天半衰期，ln(0.5)/90 ≈ 0.008
        assert abs(DecayRate.HABIT - 0.008) < 0.001

        # 测试人格衰减率：365天半衰期，ln(0.5)/365 ≈ 0.002
        assert abs(DecayRate.PERSONALITY - 0.002) < 0.001

        # 测试价值观衰减率：730天半衰期，ln(0.5)/730 ≈ 0.001
        assert abs(DecayRate.VALUES - 0.001) < 0.001

    def test_get_decay_rate_fact(self):
        """测试事实类记忆的衰减率"""
        rate = DecayRate.get_decay_rate(MemoryType.FACT)
        assert rate == DecayRate.HABIT

    def test_get_decay_rate_emotion(self):
        """测试情感类记忆的衰减率"""
        rate = DecayRate.get_decay_rate(MemoryType.EMOTION)
        assert rate == DecayRate.INTEREST

    def test_get_decay_rate_relationship(self):
        """测试关系类记忆的衰减率"""
        rate = DecayRate.get_decay_rate(MemoryType.RELATIONSHIP)
        assert rate == DecayRate.PERSONALITY

    def test_get_decay_rate_interaction(self):
        """测试互动类记忆的衰减率"""
        rate = DecayRate.get_decay_rate(MemoryType.INTERACTION)
        assert rate == DecayRate.INTEREST

    def test_get_decay_rate_inferred(self):
        """测试推断类记忆的衰减率"""
        rate = DecayRate.get_decay_rate(MemoryType.INFERRED)
        assert rate == DecayRate.HABIT

    def test_get_decay_rate_all_memory_types(self):
        """测试所有记忆类型都有对应的衰减率"""
        for memory_type in MemoryType:
            rate = DecayRate.get_decay_rate(memory_type)
            assert rate in [
                DecayRate.INTEREST,
                DecayRate.HABIT,
                DecayRate.PERSONALITY
            ]

    def test_decay_rate_ordering(self):
        """测试衰减率的大小顺序（兴趣衰减最快，价值观最慢）"""
        assert DecayRate.INTEREST > DecayRate.HABIT
        assert DecayRate.HABIT > DecayRate.PERSONALITY
        assert DecayRate.PERSONALITY > DecayRate.VALUES


class TestRetrievalStrategy:
    """测试RetrievalStrategy枚举"""

    def test_retrieval_strategy_values(self):
        """测试检索策略的值"""
        assert RetrievalStrategy.VECTOR_ONLY.value == "vector_only"
        assert RetrievalStrategy.GRAPH_ONLY.value == "graph_only"
        assert RetrievalStrategy.TIME_AWARE.value == "time_aware"
        assert RetrievalStrategy.EMOTION_AWARE.value == "emotion_aware"
        assert RetrievalStrategy.HYBRID.value == "hybrid"

    def test_retrieval_strategy_count(self):
        """测试检索策略的数量"""
        assert len(RetrievalStrategy) == 5


class TestTriggerType:
    """测试TriggerType枚举"""

    def test_trigger_type_values(self):
        """测试触发器类型的值"""
        assert TriggerType.EXPLICIT.value == "explicit"
        assert TriggerType.PREFERENCE.value == "preference"
        assert TriggerType.EMOTION.value == "emotion"
        assert TriggerType.RELATIONSHIP.value == "relationship"
        assert TriggerType.FACT.value == "fact"
        assert TriggerType.BOUNDARY.value == "boundary"

    def test_trigger_type_count(self):
        """测试触发器类型的数量"""
        assert len(TriggerType) == 6


class TestTypeEnumConsistency:
    """测试枚举类型的一致性"""

    def test_all_string_enums(self):
        """测试所有基于字符串的枚举"""
        string_enums = [
            MemoryType,
            ModalityType,
            StorageLayer,
            EmotionType,
            VerificationMethod,
            RetrievalStrategy,
            TriggerType
        ]

        for enum_class in string_enums:
            assert issubclass(enum_class, str)

    def test_quality_level_is_int_enum(self):
        """测试QualityLevel是基于整数的枚举"""
        assert issubclass(QualityLevel, int)
        assert not issubclass(QualityLevel, str)

    def test_sensitivity_level_is_int_enum(self):
        """测试SensitivityLevel是基于整数的枚举"""
        assert issubclass(SensitivityLevel, int)
        assert not issubclass(SensitivityLevel, str)


class TestTypeEnumValues:
    """测试枚举值的合理性"""

    def test_memory_type_values_are_valid_strings(self):
        """测试记忆类型值都是有效的字符串"""
        for memory_type in MemoryType:
            assert isinstance(memory_type.value, str)
            assert memory_type.value.islower() or "_" in memory_type.value

    def test_emotion_type_values_are_valid_strings(self):
        """测试情感类型值都是有效的字符串"""
        for emotion_type in EmotionType:
            assert isinstance(emotion_type.value, str)
            assert emotion_type.value.islower()

    def test_quality_level_values_are_integers(self):
        """测试质量等级值都是整数"""
        for quality_level in QualityLevel:
            assert isinstance(quality_level.value, int)
            assert 1 <= quality_level.value <= 5

    def test_sensitivity_level_values_are_integers(self):
        """测试敏感度等级值都是整数"""
        for sensitivity_level in SensitivityLevel:
            assert isinstance(sensitivity_level.value, int)
            assert 0 <= sensitivity_level.value <= 4

    def test_decay_rates_are_positive(self):
        """测试所有衰减率都是正数"""
        assert DecayRate.INTEREST > 0
        assert DecayRate.HABIT > 0
        assert DecayRate.PERSONALITY > 0
        assert DecayRate.VALUES > 0
