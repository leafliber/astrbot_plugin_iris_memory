"""
MemberIdentityService 和成员标识系统的测试

覆盖范围：
- MemberProfile 数据类
- MemberIdentityService 核心功能
- 名称变更追踪
- 活跃度计算
- 群成员管理
- 序列化/反序列化
- format_member_tag 集成
- Reranker sender/activity 权重
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from iris_memory.utils.member_identity_service import MemberIdentityService, MemberProfile
from iris_memory.utils.member_utils import (
    format_member_tag,
    short_member_id,
    set_identity_service,
    get_identity_service,
)
from iris_memory.retrieval.reranker import Reranker
from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, ModalityType, QualityLevel, StorageLayer
from iris_memory.core.memory_scope import MemoryScope


# ===== Fixtures =====

@pytest.fixture
def service():
    """创建一个干净的 MemberIdentityService 实例"""
    return MemberIdentityService()


@pytest.fixture
def populated_service():
    """创建并填充测试数据的服务"""
    svc = MemberIdentityService()

    async def _populate():
        await svc.resolve_tag("u001", "Alice", "g100")
        await svc.resolve_tag("u002", "Bob", "g100")
        await svc.resolve_tag("u003", "Charlie", "g100")
        await svc.resolve_tag("u001", "Alice", "g200")
        await svc.resolve_tag("u004", "Dave", None)  # 私聊用户
        return svc

    asyncio.get_event_loop().run_until_complete(_populate())
    return svc


@pytest.fixture
def make_memory():
    """创建测试记忆的工厂函数"""
    def _make(user_id="u001", sender_name="Alice", group_id="g100",
              scope=MemoryScope.GROUP_PRIVATE, content="test memory"):
        return Memory(
            user_id=user_id,
            sender_name=sender_name,
            group_id=group_id,
            scope=scope,
            type=MemoryType.FACT,
            modality=ModalityType.TEXT,
            content=content,
        )
    return _make


# ===== MemberProfile Tests =====

class TestMemberProfile:
    def test_display_tag_with_name(self):
        p = MemberProfile(user_id="user123456", preferred_name="Alice")
        tag = p.display_tag
        assert tag.startswith("Alice#")
        assert "123456" in tag

    def test_display_tag_without_name(self):
        p = MemberProfile(user_id="user123456", preferred_name="")
        tag = p.display_tag
        assert tag.startswith("成员#")

    def test_short_id(self):
        p = MemberProfile(user_id="abc_xyz_123456")
        assert p.short_id == "123456"

    def test_inactive_days(self):
        p = MemberProfile(user_id="u1")
        p.last_active = datetime.now() - timedelta(days=10)
        assert abs(p.inactive_days - 10.0) < 0.1

    def test_serialization_roundtrip(self):
        p = MemberProfile(
            user_id="u1",
            preferred_name="Alice",
            name_history=[{"old_name": "OldName", "new_name": "Alice",
                           "timestamp": "2026-01-01T00:00:00"}],
            groups={"g1", "g2"},
            message_count=42,
        )
        data = p.to_dict()
        p2 = MemberProfile.from_dict(data)
        assert p2.user_id == "u1"
        assert p2.preferred_name == "Alice"
        assert p2.message_count == 42
        assert set(p2.groups) == {"g1", "g2"}
        assert len(p2.name_history) == 1


# ===== MemberIdentityService Core Tests =====

class TestMemberIdentityService:
    @pytest.mark.asyncio
    async def test_resolve_tag_creates_profile(self, service):
        tag = await service.resolve_tag("u001", "Alice", "g100")
        assert "Alice" in tag
        assert "#" in tag

    @pytest.mark.asyncio
    async def test_resolve_tag_same_user_returns_consistent(self, service):
        tag1 = await service.resolve_tag("u001", "Alice")
        tag2 = await service.resolve_tag("u001", "Alice")
        assert tag1 == tag2

    @pytest.mark.asyncio
    async def test_resolve_tag_without_name(self, service):
        tag = await service.resolve_tag("u001", None)
        assert "成员" in tag or "#" in tag

    @pytest.mark.asyncio
    async def test_resolve_tag_empty_user_id(self, service):
        tag = await service.resolve_tag("", "Alice")
        assert tag == "Alice"

    def test_resolve_tag_sync(self, service):
        tag = service.resolve_tag_sync("u001", "Bob", "g100")
        assert "Bob" in tag
        assert "#" in tag

    @pytest.mark.asyncio
    async def test_name_change_tracking(self, service):
        await service.resolve_tag("u001", "Alice")
        await service.resolve_tag("u001", "AliceNewName")

        history = service.get_name_history("u001")
        assert len(history) == 1
        assert history[0]["old_name"] == "Alice"
        assert history[0]["new_name"] == "AliceNewName"

    @pytest.mark.asyncio
    async def test_name_change_reflected_in_tag(self, service):
        await service.resolve_tag("u001", "Alice")
        tag = await service.resolve_tag("u001", "AliceRenamed")
        assert "AliceRenamed" in tag

    @pytest.mark.asyncio
    async def test_get_all_known_names(self, service):
        await service.resolve_tag("u001", "Alice")
        await service.resolve_tag("u001", "Alice2")
        await service.resolve_tag("u001", "Alice3")

        names = service.get_all_known_names("u001")
        assert "Alice" in names
        assert "Alice2" in names
        assert "Alice3" in names

    @pytest.mark.asyncio
    async def test_name_history_limit(self, service):
        for i in range(15):
            await service.resolve_tag("u001", f"Name{i}")
        history = service.get_name_history("u001")
        assert len(history) <= MemberIdentityService._MAX_NAME_HISTORY

    @pytest.mark.asyncio
    async def test_first_set_name_not_in_history(self, service):
        """首次设置名称不应产生历史记录"""
        await service.resolve_tag("u001", "FirstName")
        history = service.get_name_history("u001")
        assert len(history) == 0


# ===== Group Member Tests =====

class TestGroupMembers:
    @pytest.mark.asyncio
    async def test_get_group_members(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u002", "Bob", "g100")

        members = service.get_group_members("g100")
        assert len(members) == 2
        assert any("Alice" in m for m in members)
        assert any("Bob" in m for m in members)

    @pytest.mark.asyncio
    async def test_get_group_member_count(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u002", "Bob", "g100")
        await service.resolve_tag("u003", "Charlie", "g200")

        assert service.get_group_member_count("g100") == 2
        assert service.get_group_member_count("g200") == 1
        assert service.get_group_member_count("g999") == 0

    @pytest.mark.asyncio
    async def test_user_in_multiple_groups(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u001", "Alice", "g200")

        assert service.get_group_member_count("g100") == 1
        assert service.get_group_member_count("g200") == 1

    def test_empty_group(self, service):
        assert service.get_group_members("nonexistent") == []


# ===== Activity Score Tests =====

class TestActivityScore:
    @pytest.mark.asyncio
    async def test_recently_active_high_score(self, service):
        await service.resolve_tag("u001", "Alice")
        score = service.get_activity_score("u001")
        assert score > 0.7

    @pytest.mark.asyncio
    async def test_inactive_user_low_score(self, service):
        await service.resolve_tag("u001", "Alice")
        # 模拟长时间不活跃
        service._profiles["u001"].last_active = (
            datetime.now() - timedelta(days=90)
        )
        score = service.get_activity_score("u001")
        assert score < 0.5

    def test_unknown_user_zero_score(self, service):
        assert service.get_activity_score("unknown") == 0.0

    @pytest.mark.asyncio
    async def test_is_active(self, service):
        await service.resolve_tag("u001", "Alice")
        assert service.is_active("u001")

        service._profiles["u001"].last_active = (
            datetime.now() - timedelta(days=365)
        )
        service._profiles["u001"].message_count = 0
        assert not service.is_active("u001")


# ===== Identity Comparison Tests =====

class TestIdentityComparison:
    @pytest.mark.asyncio
    async def test_is_same_member(self, service):
        tag1 = await service.resolve_tag("u001", "Alice")
        tag2 = await service.resolve_tag("u001", "AliceRenamed")
        assert service.is_same_member(tag1, tag2)

    @pytest.mark.asyncio
    async def test_different_members(self, service):
        tag1 = await service.resolve_tag("u001", "Alice")
        tag2 = await service.resolve_tag("u002", "Bob")
        assert not service.is_same_member(tag1, tag2)

    def test_is_same_member_no_hash(self, service):
        assert not service.is_same_member("Alice", "Alice")

    @pytest.mark.asyncio
    async def test_get_user_id_by_tag(self, service):
        tag = await service.resolve_tag("u001", "Alice")
        uid = service.get_user_id_by_tag(tag)
        assert uid == "u001"

    def test_get_user_id_by_tag_unknown(self, service):
        assert service.get_user_id_by_tag("Unknown#999999") is None


# ===== Serialization Tests =====

class TestSerialization:
    @pytest.mark.asyncio
    async def test_serialize_deserialize(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u002", "Bob", "g100")
        await service.resolve_tag("u001", "AliceRenamed")

        data = service.serialize()

        new_service = MemberIdentityService()
        new_service.deserialize(data)

        assert new_service.get_group_member_count("g100") == 2
        tag = new_service.resolve_tag_sync("u001")
        assert "AliceRenamed" in tag

        history = new_service.get_name_history("u001")
        assert len(history) == 1

    def test_deserialize_empty(self, service):
        service.deserialize({})
        assert service.get_stats()["total_profiles"] == 0

    def test_deserialize_none(self, service):
        service.deserialize(None)
        assert service.get_stats()["total_profiles"] == 0


# ===== Cleanup Tests =====

class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_inactive(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u002", "Bob", "g100")

        # 模拟u002长时间不活跃
        service._profiles["u002"].last_active = (
            datetime.now() - timedelta(days=200)
        )

        removed = service.cleanup_inactive(inactive_days=180)
        assert removed == 1
        assert service.get_group_member_count("g100") == 1
        assert service.get_activity_score("u002") == 0.0


# ===== format_member_tag Integration Tests =====

class TestFormatMemberTagIntegration:
    """测试 format_member_tag 与 MemberIdentityService 的集成"""

    def test_without_service_backward_compatible(self):
        """无服务时退回到纯函数逻辑"""
        old_service = get_identity_service()
        set_identity_service(None)
        try:
            tag = format_member_tag("Alice", "user123456")
            assert tag == "Alice#123456"

            tag = format_member_tag(None, "user123456")
            assert tag == "成员#123456"

            tag = format_member_tag(None, None)
            assert tag == ""
        finally:
            set_identity_service(old_service)

    def test_with_service_delegates(self):
        """有服务时委托给 resolve_tag_sync"""
        svc = MemberIdentityService()
        old_service = get_identity_service()
        set_identity_service(svc)
        try:
            tag = format_member_tag("Alice", "u001", "g100")
            assert "Alice" in tag
            assert "#" in tag

            # 确认服务已注册了这个成员
            assert svc.get_group_member_count("g100") == 1
        finally:
            set_identity_service(old_service)

    def test_with_service_name_change_tracked(self):
        """通过服务时名称变更被追踪"""
        svc = MemberIdentityService()
        old_service = get_identity_service()
        set_identity_service(svc)
        try:
            format_member_tag("Alice", "u001")
            format_member_tag("AliceNew", "u001")

            history = svc.get_name_history("u001")
            assert len(history) == 1
            assert history[0]["old_name"] == "Alice"
        finally:
            set_identity_service(old_service)


# ===== Reranker Sender Weight Tests =====

class TestRerankerSenderWeight:
    """测试 Reranker 中新增的 sender 和 activity 权重"""

    def test_sender_match_boosts_score(self, make_memory):
        reranker = Reranker()
        memory_same = make_memory(user_id="u001", content="memory from current user")
        memory_other = make_memory(user_id="u002", content="memory from other user")

        context = {"current_user_id": "u001"}

        score_same = reranker._calculate_rerank_score(memory_same, "test", context)
        score_other = reranker._calculate_rerank_score(memory_other, "test", context)

        # 同一用户的记忆应该得分更高
        assert score_same > score_other

    def test_no_user_id_context_neutral(self, make_memory):
        reranker = Reranker()
        memory = make_memory()

        score_with = reranker._calculate_rerank_score(memory, "test", {"current_user_id": "u001"})
        score_without = reranker._calculate_rerank_score(memory, "test", {})

        # 都应该产生合理的分数
        assert score_with > 0
        assert score_without > 0

    def test_activity_score_with_service(self, make_memory):
        svc = MemberIdentityService()
        svc.resolve_tag_sync("u001", "Alice")

        reranker = Reranker()
        memory = make_memory(user_id="u001")

        context = {"member_identity_service": svc}
        score = reranker._calculate_activity_score(memory, context)
        assert score > 0

    def test_activity_score_without_service(self, make_memory):
        reranker = Reranker()
        memory = make_memory()

        score = reranker._calculate_activity_score(memory, {})
        assert score == 0.5

    def test_rerank_order_prefers_current_user(self, make_memory):
        """验证整体 rerank 时当前用户记忆排名更靠前"""
        reranker = Reranker()

        m1 = make_memory(user_id="u001", content="Alice's memory")
        m2 = make_memory(user_id="u002", sender_name="Bob", content="Bob's memory")

        # 设置相同基础属性
        m1.rif_score = 0.5
        m2.rif_score = 0.5
        m1.access_count = 3
        m2.access_count = 3

        context = {"current_user_id": "u001"}
        result = reranker.rerank([m2, m1], "test query", context)

        # u001 的记忆应该排在前面
        assert result[0].user_id == "u001"


# ===== Stats Tests =====

class TestServiceStats:
    @pytest.mark.asyncio
    async def test_stats(self, service):
        await service.resolve_tag("u001", "Alice", "g100")
        await service.resolve_tag("u002", "Bob", "g100")

        stats = service.get_stats()
        assert stats["total_profiles"] == 2
        assert stats["total_groups"] == 1
        assert stats["active_members"] == 2
