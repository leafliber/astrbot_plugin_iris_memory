"""情感差异化衰减模型测试"""

import pytest
from datetime import datetime, timedelta

from iris_memory.analysis.emotion_decay import EmotionDecayProfile
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def joy_memory():
    return Memory(
        id="mem_joy",
        type=MemoryType.EMOTION,
        subtype="joy",
        content="今天好开心",
        user_id="user_1",
        storage_layer=StorageLayer.EPISODIC,
        last_access_time=datetime.now() - timedelta(days=30),
    )


@pytest.fixture
def sadness_memory():
    return Memory(
        id="mem_sad",
        type=MemoryType.EMOTION,
        subtype="sadness",
        content="今天好难过",
        user_id="user_1",
        storage_layer=StorageLayer.EPISODIC,
        last_access_time=datetime.now() - timedelta(days=30),
    )


class TestDecayRates:
    def test_positive_emotion_decay_rate(self):
        rate = EmotionDecayProfile.get_decay_rate("joy")
        assert rate == 0.012

    def test_negative_emotion_decay_rate(self):
        rate = EmotionDecayProfile.get_decay_rate("sadness")
        assert rate == 0.099

    def test_neutral_emotion_decay_rate(self):
        rate = EmotionDecayProfile.get_decay_rate("unknown_emotion")
        assert rate == EmotionDecayProfile.NEUTRAL_DECAY_RATE

    def test_none_emotion_gives_neutral(self):
        rate = EmotionDecayProfile.get_decay_rate(None)
        assert rate == EmotionDecayProfile.NEUTRAL_DECAY_RATE


class TestValence:
    def test_positive_valence(self):
        assert EmotionDecayProfile.get_valence("joy") == "positive"
        assert EmotionDecayProfile.get_valence("love") == "positive"

    def test_negative_valence(self):
        assert EmotionDecayProfile.get_valence("sadness") == "negative"
        assert EmotionDecayProfile.get_valence("anger") == "negative"

    def test_neutral_valence(self):
        assert EmotionDecayProfile.get_valence(None) == "neutral"
        assert EmotionDecayProfile.get_valence("something") == "neutral"


class TestEmotionTimeScore:
    def test_positive_decays_slowly(self, joy_memory):
        """正面情绪30天后仍保留较高分数"""
        score = EmotionDecayProfile.calculate_emotion_time_score(joy_memory)
        # e^(-0.012 * 30) ≈ 0.698
        assert score > 0.6

    def test_negative_decays_fast(self, sadness_memory):
        """负面情绪30天后分数应很低"""
        score = EmotionDecayProfile.calculate_emotion_time_score(sadness_memory)
        # e^(-0.099 * 30) ≈ 0.051
        assert score < 0.1

    def test_positive_vs_negative_after_same_period(self, joy_memory, sadness_memory):
        """同期正面应显著高于负面"""
        joy_score = EmotionDecayProfile.calculate_emotion_time_score(joy_memory)
        sad_score = EmotionDecayProfile.calculate_emotion_time_score(sadness_memory)
        assert joy_score > sad_score * 3

    def test_recent_memory_score_high(self):
        """刚访问过的记忆分数接近1"""
        mem = Memory(
            id="m1",
            type=MemoryType.EMOTION,
            subtype="joy",
            content="test",
            user_id="u1",
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=datetime.now(),
        )
        score = EmotionDecayProfile.calculate_emotion_time_score(mem)
        assert score > 0.99

    def test_score_in_valid_range(self, joy_memory):
        score = EmotionDecayProfile.calculate_emotion_time_score(joy_memory)
        assert 0.0 <= score <= 1.0

    def test_custom_decay_rate_on_memory(self):
        """使用记忆自身 emotion_decay_rate"""
        mem = Memory(
            id="m2",
            type=MemoryType.EMOTION,
            subtype="joy",
            content="test",
            user_id="u1",
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=datetime.now() - timedelta(days=10),
            emotion_decay_rate=0.2,  # 自定义快速衰减
        )
        score = EmotionDecayProfile.calculate_emotion_time_score(mem)
        # e^(-0.2 * 10) ≈ 0.135
        assert score < 0.2

    def test_protection_flag_bonus(self):
        """高情感保护标记应加成"""
        from iris_memory.models.protection import ProtectionFlag

        mem = Memory(
            id="m3",
            type=MemoryType.EMOTION,
            subtype="joy",
            content="test",
            user_id="u1",
            storage_layer=StorageLayer.EPISODIC,
            last_access_time=datetime.now() - timedelta(days=60),
        )
        score_without = EmotionDecayProfile.calculate_emotion_time_score(mem)

        mem.add_protection(ProtectionFlag.HIGH_EMOTION)
        score_with = EmotionDecayProfile.calculate_emotion_time_score(mem)

        assert score_with > score_without
