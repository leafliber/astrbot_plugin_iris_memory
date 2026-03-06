"""保护标记系统测试"""

import pytest
from datetime import datetime, timedelta

from iris_memory.models.protection import ProtectionFlag, ProtectionMixin, ProtectionRules
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, StorageLayer


@pytest.fixture
def identity_memory():
    return Memory(
        id="mem_identity",
        type=MemoryType.FACT,
        content="我叫张三",
        user_id="user_1",
        confidence=0.9,
        storage_layer=StorageLayer.EPISODIC,
    )


@pytest.fixture
def emotion_memory():
    return Memory(
        id="mem_emotion",
        type=MemoryType.EMOTION,
        content="今天好开心",
        user_id="user_1",
        emotional_weight=0.9,
        confidence=0.7,
        storage_layer=StorageLayer.EPISODIC,
    )


@pytest.fixture
def relationship_memory():
    return Memory(
        id="mem_rel",
        type=MemoryType.RELATIONSHIP,
        content="你是我最好的朋友，我们是朋友",
        user_id="user_1",
        confidence=0.85,
        storage_layer=StorageLayer.EPISODIC,
    )


class TestProtectionFlag:
    def test_flag_values_are_powers_of_two(self):
        assert ProtectionFlag.CORE_IDENTITY == 0x01
        assert ProtectionFlag.USER_PINNED == 0x02
        assert ProtectionFlag.HIGH_EMOTION == 0x04
        assert ProtectionFlag.ANNIVERSARY == 0x08
        assert ProtectionFlag.RELATIONSHIP_KEY == 0x10

    def test_flags_can_be_combined(self):
        combined = ProtectionFlag.CORE_IDENTITY | ProtectionFlag.USER_PINNED
        assert combined == 0x03


class TestProtectionMixin:
    def test_add_protection(self, identity_memory):
        identity_memory.add_protection(ProtectionFlag.CORE_IDENTITY)
        assert identity_memory.has_protection(ProtectionFlag.CORE_IDENTITY)

    def test_remove_protection(self, identity_memory):
        identity_memory.add_protection(ProtectionFlag.CORE_IDENTITY)
        identity_memory.remove_protection(ProtectionFlag.CORE_IDENTITY)
        assert not identity_memory.has_protection(ProtectionFlag.CORE_IDENTITY)

    def test_is_protected_true(self, identity_memory):
        identity_memory.add_protection(ProtectionFlag.USER_PINNED)
        assert identity_memory.is_protected

    def test_is_protected_false(self, identity_memory):
        assert not identity_memory.is_protected

    def test_multiple_flags(self, identity_memory):
        identity_memory.add_protection(ProtectionFlag.CORE_IDENTITY)
        identity_memory.add_protection(ProtectionFlag.HIGH_EMOTION)
        assert identity_memory.has_protection(ProtectionFlag.CORE_IDENTITY)
        assert identity_memory.has_protection(ProtectionFlag.HIGH_EMOTION)
        assert not identity_memory.has_protection(ProtectionFlag.ANNIVERSARY)

    def test_is_deletable_when_protected(self, identity_memory):
        identity_memory.add_protection(ProtectionFlag.CORE_IDENTITY)
        assert not identity_memory.is_deletable


class TestProtectionRules:
    def test_evaluate_core_identity(self, identity_memory):
        flags = ProtectionRules.evaluate(identity_memory)
        assert flags & ProtectionFlag.CORE_IDENTITY

    def test_evaluate_high_emotion(self, emotion_memory):
        flags = ProtectionRules.evaluate(emotion_memory)
        assert flags & ProtectionFlag.HIGH_EMOTION

    def test_evaluate_relationship(self, relationship_memory):
        flags = ProtectionRules.evaluate(relationship_memory)
        assert flags & ProtectionFlag.RELATIONSHIP_KEY

    def test_evaluate_anniversary(self):
        mem = Memory(
            id="mem_anniv",
            type=MemoryType.FACT,
            content="今天是我们的纪念日",
            user_id="user_1",
            confidence=0.6,
            storage_layer=StorageLayer.EPISODIC,
        )
        flags = ProtectionRules.evaluate(mem)
        assert flags & ProtectionFlag.ANNIVERSARY

    def test_evaluate_user_pinned(self):
        mem = Memory(
            id="mem_pin",
            type=MemoryType.FACT,
            content="一些普通内容",
            user_id="user_1",
            is_user_requested=True,
            confidence=0.5,
            storage_layer=StorageLayer.EPISODIC,
        )
        flags = ProtectionRules.evaluate(mem)
        assert flags & ProtectionFlag.USER_PINNED

    def test_evaluate_no_flags(self):
        mem = Memory(
            id="mem_plain",
            type=MemoryType.INTERACTION,
            content="嗯嗯",
            user_id="user_1",
            confidence=0.3,
            storage_layer=StorageLayer.WORKING,
        )
        flags = ProtectionRules.evaluate(mem)
        assert flags == 0

    def test_protected_memory_not_downgraded(self, identity_memory):
        """受保护记忆不应被降级"""
        identity_memory.add_protection(ProtectionFlag.CORE_IDENTITY)
        identity_memory.storage_layer = StorageLayer.SEMANTIC
        assert not identity_memory.should_downgrade_to_episodic()
