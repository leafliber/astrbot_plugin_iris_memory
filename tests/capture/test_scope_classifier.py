"""记忆可见性分类器测试"""

import pytest

from iris_memory.capture.scope_classifier import ScopeClassifier
from iris_memory.core.memory_scope import MemoryScope
from iris_memory.core.types import SensitivityLevel


@pytest.fixture
def classifier():
    return ScopeClassifier()


class TestPrivateChat:
    @pytest.mark.asyncio
    async def test_private_chat_always_user_private(self, classifier):
        result = await classifier.classify("你好", context={"is_group": False})
        assert result == MemoryScope.USER_PRIVATE

    @pytest.mark.asyncio
    async def test_no_context_defaults_private(self, classifier):
        result = await classifier.classify("你好", context={})
        assert result == MemoryScope.USER_PRIVATE


class TestStrongPrivacy:
    @pytest.mark.asyncio
    async def test_secret_detected(self, classifier):
        result = await classifier.classify(
            "这是秘密，别告诉别人",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.USER_PRIVATE

    @pytest.mark.asyncio
    async def test_dont_tell_others(self, classifier):
        result = await classifier.classify(
            "别告诉其他人我喜欢她",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.USER_PRIVATE

    @pytest.mark.asyncio
    async def test_password_detected(self, classifier):
        result = await classifier.classify(
            "我的密码是123456",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.USER_PRIVATE


class TestSensitivity:
    @pytest.mark.asyncio
    async def test_high_sensitivity_is_private(self, classifier):
        result = await classifier.classify(
            "一些内容",
            context={
                "is_group": True,
                "group_id": "g1",
                "sensitivity_level": SensitivityLevel.PRIVATE,
            },
        )
        assert result == MemoryScope.USER_PRIVATE


class TestGroupShared:
    @pytest.mark.asyncio
    async def test_group_announcement(self, classifier):
        result = await classifier.classify(
            "群公告：明天聚餐",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.GROUP_SHARED

    @pytest.mark.asyncio
    async def test_at_all(self, classifier):
        result = await classifier.classify(
            "@全体成员 放假通知",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.GROUP_SHARED


class TestPersonalPatterns:
    @pytest.mark.asyncio
    async def test_personal_statement(self, classifier):
        result = await classifier.classify(
            "我是一个程序员",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.GROUP_PRIVATE

    @pytest.mark.asyncio
    async def test_my_hobby(self, classifier):
        result = await classifier.classify(
            "我的爱好是画画",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.GROUP_PRIVATE


class TestDefault:
    @pytest.mark.asyncio
    async def test_no_match_defaults_group_private(self, classifier):
        result = await classifier.classify(
            "今天天气不错",
            context={"is_group": True, "group_id": "g1"},
        )
        assert result == MemoryScope.GROUP_PRIVATE
