"""画像存储组件测试"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from iris_memory.profile.storage import ProfileStorage
from iris_memory.profile.models import GroupProfile, UserProfile


class TestProfileStorage:
    """画像存储组件测试"""

    @pytest.fixture
    def mock_context(self):
        """创建模拟的 AstrBotContext"""
        context = MagicMock()
        context.get_kv_data = AsyncMock(return_value=[])
        context.put_kv_data = AsyncMock()
        return context

    @pytest.fixture
    def storage(self, mock_context):
        """创建 ProfileStorage 实例"""
        storage = ProfileStorage(mock_context)
        storage._is_available = True
        return storage

    @pytest.mark.asyncio
    async def test_get_group_profile_not_found(self, storage, mock_context):
        """测试获取不存在的群聊画像"""
        mock_context.get_kv_data.return_value = None

        profile = await storage.get_group_profile("group_123")

        assert profile is None
        mock_context.get_kv_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_group_profile_found(self, storage, mock_context):
        """测试获取存在的群聊画像"""
        profile_data = {
            "group_id": "group_123",
            "group_name": "测试群",
            "version": 1,
            "interests": ["技术"],
        }
        mock_context.get_kv_data.return_value = profile_data

        profile = await storage.get_group_profile("group_123")

        assert profile is not None
        assert profile.group_id == "group_123"
        assert profile.group_name == "测试群"
        assert profile.interests == ["技术"]

    @pytest.mark.asyncio
    async def test_save_group_profile(self, storage, mock_context):
        """测试保存群聊画像"""
        profile = GroupProfile(
            group_id="group_123", group_name="测试群", interests=["技术"]
        )

        await storage.save_group_profile(profile)

        assert profile.version == 2
        assert mock_context.put_kv_data.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self, storage, mock_context):
        """测试获取不存在的用户画像"""
        mock_context.get_kv_data.return_value = None

        profile = await storage.get_user_profile("user_456", "group_123")

        assert profile is None
        mock_context.get_kv_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_profile_found(self, storage, mock_context):
        """测试获取存在的用户画像"""
        profile_data = {
            "user_id": "user_456",
            "user_name": "小明",
            "version": 1,
            "personality_tags": ["外向"],
        }
        mock_context.get_kv_data.return_value = profile_data

        profile = await storage.get_user_profile("user_456", "group_123")

        assert profile is not None
        assert profile.user_id == "user_456"
        assert profile.user_name == "小明"
        assert profile.personality_tags == ["外向"]

    @pytest.mark.asyncio
    async def test_save_user_profile(self, storage, mock_context):
        """测试保存用户画像"""
        profile = UserProfile(
            user_id="user_456", user_name="小明", personality_tags=["外向"]
        )

        await storage.save_user_profile(profile, "group_123")

        assert profile.version == 2
        assert mock_context.put_kv_data.call_count >= 1

    @pytest.mark.asyncio
    async def test_storage_not_available(self, mock_context):
        """测试存储组件不可用"""
        storage = ProfileStorage(mock_context)
        storage._is_available = False

        profile = await storage.get_group_profile("group_123")
        assert profile is None

        # 不应该调用任何 KV 操作
        mock_context.get_kv_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_group_profile_acquires_lock(self, storage, mock_context):
        """回归：update_group_profile 必须持锁，防止与 update_from_analysis 并发 lost update"""
        mock_context.get_kv_data.return_value = None

        lock_acquired = False
        original_lock_group = storage.lock_group

        @asynccontextmanager
        async def tracking_lock(group_id, persona_id="default"):
            nonlocal lock_acquired
            lock_acquired = True
            async with original_lock_group(group_id, persona_id):
                yield

        storage.lock_group = tracking_lock

        result = await storage.update_group_profile(
            "group_123", {"group_name": "测试群"}
        )

        assert result is True
        assert lock_acquired, "update_group_profile 必须持 lock_group 锁"

    @pytest.mark.asyncio
    async def test_update_user_profile_acquires_lock(self, storage, mock_context):
        """回归：update_user_profile 必须持锁，防止与 update_from_analysis 并发 lost update"""
        mock_context.get_kv_data.return_value = None

        lock_acquired = False
        original_lock_user = storage.lock_user

        @asynccontextmanager
        async def tracking_lock(user_id, group_id="default", persona_id="default"):
            nonlocal lock_acquired
            lock_acquired = True
            async with original_lock_user(user_id, group_id, persona_id):
                yield

        storage.lock_user = tracking_lock

        result = await storage.update_user_profile(
            "user_456", "group_123", {"user_name": "小明"}
        )

        assert result is True
        assert lock_acquired, "update_user_profile 必须持 lock_user 锁"
