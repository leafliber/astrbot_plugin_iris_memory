"""快速通道评估器测试"""

import pytest
from datetime import datetime

from iris_memory.capture.fast_track import FastTrackEvaluator
from iris_memory.models.memory import Memory
from iris_memory.models.protection import ProtectionFlag
from iris_memory.core.types import MemoryType, QualityLevel, StorageLayer


@pytest.fixture
def evaluator():
    return FastTrackEvaluator(confidence_threshold=0.9)


@pytest.fixture
def identity_fact():
    return Memory(
        id="mem_1",
        type=MemoryType.FACT,
        content="我叫张三",
        user_id="user_1",
        confidence=0.95,
        storage_layer=StorageLayer.WORKING,
    )


@pytest.fixture
def low_confidence_fact():
    return Memory(
        id="mem_2",
        type=MemoryType.FACT,
        content="我叫李四",
        user_id="user_1",
        confidence=0.5,
        storage_layer=StorageLayer.WORKING,
    )


class TestFastTrackCondition1:
    """条件1: FACT + 高置信 + 身份关键词"""

    def test_identity_fact_fast_tracks(self, evaluator, identity_fact):
        result = evaluator.evaluate(identity_fact, is_user_requested=False)
        assert result == StorageLayer.SEMANTIC

    def test_identity_fact_sets_protection(self, evaluator, identity_fact):
        evaluator.evaluate(identity_fact, is_user_requested=False)
        assert identity_fact.has_protection(ProtectionFlag.CORE_IDENTITY)

    def test_low_confidence_not_fast_tracked(self, evaluator, low_confidence_fact):
        result = evaluator.evaluate(low_confidence_fact, is_user_requested=False)
        assert result is None

    def test_non_identity_content_not_fast_tracked(self, evaluator):
        mem = Memory(
            id="m1", type=MemoryType.FACT, content="天气不错",
            user_id="u1", confidence=0.95, storage_layer=StorageLayer.WORKING,
        )
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result is None


class TestFastTrackCondition2:
    """条件2: 用户请求 + confidence >= 0.85"""

    def test_user_requested_high_conf(self, evaluator):
        mem = Memory(
            id="m2", type=MemoryType.FACT, content="记住这个",
            user_id="u1", confidence=0.9, storage_layer=StorageLayer.WORKING,
        )
        result = evaluator.evaluate(mem, is_user_requested=True)
        assert result == StorageLayer.SEMANTIC

    def test_user_requested_low_conf(self, evaluator):
        mem = Memory(
            id="m3", type=MemoryType.FACT, content="记住这个",
            user_id="u1", confidence=0.5, storage_layer=StorageLayer.WORKING,
        )
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result is None


class TestFastTrackCondition3:
    """条件3: CONFIRMED quality"""

    def test_confirmed_quality_fast_tracks(self, evaluator):
        mem = Memory(
            id="m4", type=MemoryType.FACT, content="已确认信息",
            user_id="u1", confidence=0.6, storage_layer=StorageLayer.WORKING,
            quality_level=QualityLevel.CONFIRMED,
        )
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result == StorageLayer.SEMANTIC


class TestFastTrackCondition4:
    """条件4: 已有CORE_IDENTITY标记 + confidence >= 0.85"""

    def test_existing_flag_fast_tracks(self, evaluator):
        mem = Memory(
            id="m5", type=MemoryType.FACT, content="普通内容",
            user_id="u1", confidence=0.85, storage_layer=StorageLayer.WORKING,
        )
        mem.add_protection(ProtectionFlag.CORE_IDENTITY)
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result == StorageLayer.SEMANTIC

    def test_existing_flag_low_conf(self, evaluator):
        mem = Memory(
            id="m6", type=MemoryType.FACT, content="普通内容",
            user_id="u1", confidence=0.5, storage_layer=StorageLayer.WORKING,
        )
        mem.add_protection(ProtectionFlag.CORE_IDENTITY)
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result is None


class TestFastTrackCustomThreshold:
    def test_lower_threshold(self):
        evaluator = FastTrackEvaluator(confidence_threshold=0.8)
        mem = Memory(
            id="m7", type=MemoryType.FACT, content="我的名字是王五",
            user_id="u1", confidence=0.82, storage_layer=StorageLayer.WORKING,
        )
        result = evaluator.evaluate(mem, is_user_requested=False)
        assert result == StorageLayer.SEMANTIC
