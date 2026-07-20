"""群聊画像管理器测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from iris_memory.profile.group_profile import GroupProfileManager
from iris_memory.profile.models import (
    GroupProfile,
    UpdateTier,
)
from iris_memory.profile.storage import ProfileStorage


class TestGroupProfileManager:
    """群聊画像管理器测试"""

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock(spec=ProfileStorage)
        storage.get_group_profile = AsyncMock()
        storage.save_group_profile = AsyncMock()
        return storage

    @pytest.fixture
    def manager(self, mock_storage):
        return GroupProfileManager(mock_storage)

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123", group_name="测试群")
        mock_storage.get_group_profile.return_value = existing_profile

        profile = await manager.get_or_create("group_123")

        assert profile.group_id == "group_123"
        assert profile.group_name == "测试群"
        mock_storage.get_group_profile.assert_called_once_with("group_123", "default")

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, manager, mock_storage):
        mock_storage.get_group_profile.return_value = None

        profile = await manager.get_or_create("group_123")

        assert profile.group_id == "group_123"
        assert profile.group_name == ""
        mock_storage.get_group_profile.assert_called_once_with("group_123", "default")

    @pytest.mark.asyncio
    async def test_update_group_name(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123")
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.update_group_name(group_id="group_123", group_name="新群名")

        assert existing_profile.group_name == "新群名"
        mock_storage.save_group_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_from_analysis_mid_tier(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123")
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.update_from_analysis(
            group_id="group_123",
            interests=["技术", "AI"],
            atmosphere_tags=["轻松", "技术范"],
            tier=UpdateTier.MID,
            confidence=0.7,
        )

        assert existing_profile.interests == ["技术", "AI"]
        assert existing_profile.atmosphere_tags == ["轻松", "技术范"]

        tracker = existing_profile.get_update_tracker()
        assert tracker.last_mid_update_time is not None
        mock_storage.save_group_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_from_analysis_merges_lists(self, manager, mock_storage):
        existing_profile = GroupProfile(
            group_id="group_123", interests=["技术"], atmosphere_tags=["轻松"]
        )
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.update_from_analysis(
            group_id="group_123",
            interests=["AI", "技术"],
            atmosphere_tags=["技术范", "轻松"],
            tier=UpdateTier.MID,
            confidence=0.7,
        )

        assert "技术" in existing_profile.interests
        assert "AI" in existing_profile.interests
        assert "轻松" in existing_profile.atmosphere_tags
        assert "技术范" in existing_profile.atmosphere_tags

    @pytest.mark.asyncio
    async def test_update_long_term_from_analysis(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123")
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.update_long_term_from_analysis(
            group_id="group_123",
            long_term_tags=["技术交流群"],
            blacklist_topics=["政治"],
            interests=["编程"],
            confidence=0.85,
        )

        assert "技术交流群" in existing_profile.long_term_tags
        assert "政治" in existing_profile.blacklist_topics
        assert "编程" in existing_profile.interests

        tracker = existing_profile.get_update_tracker()
        assert tracker.last_long_update_time is not None

    @pytest.mark.asyncio
    async def test_should_update_mid(self, manager, mock_storage):
        profile = GroupProfile(group_id="group_123")
        tracker = profile.get_update_tracker()
        tracker.summary_count_since_mid_update = 5
        tracker.last_mid_update_time = datetime.now() - timedelta(hours=25)
        profile.set_update_tracker(tracker)

        with patch("iris_memory.profile.group_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile_mid_update_interval_summaries": 5,
                "profile_mid_update_interval_hours": 24.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            assert manager.should_update_mid(profile) is True

    @pytest.mark.asyncio
    async def test_should_not_update_mid_too_soon(self, manager, mock_storage):
        profile = GroupProfile(group_id="group_123")
        tracker = profile.get_update_tracker()
        tracker.summary_count_since_mid_update = 1
        tracker.last_mid_update_time = datetime.now() - timedelta(hours=1)
        profile.set_update_tracker(tracker)

        with patch("iris_memory.profile.group_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile_mid_update_interval_summaries": 5,
                "profile_mid_update_interval_hours": 24.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            assert manager.should_update_mid(profile) is False

    @pytest.mark.asyncio
    async def test_should_update_long(self, manager, mock_storage):
        profile = GroupProfile(group_id="group_123")
        tracker = profile.get_update_tracker()
        tracker.last_long_update_time = datetime.now() - timedelta(hours=200)
        profile.set_update_tracker(tracker)

        with patch("iris_memory.profile.group_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile_long_update_interval_hours": 168.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            assert manager.should_update_long(profile) is True

    @pytest.mark.asyncio
    async def test_add_long_term_tag(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123")
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.add_long_term_tag("group_123", "技术交流群")

        assert "技术交流群" in existing_profile.long_term_tags
        meta = existing_profile.get_field_meta("long_term_tags")
        assert meta.confidence == 1.0
        assert meta.source == "manual"
        mock_storage.save_group_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_blacklist_topic(self, manager, mock_storage):
        existing_profile = GroupProfile(group_id="group_123")
        mock_storage.get_group_profile.return_value = existing_profile

        await manager.add_blacklist_topic("group_123", "政治")

        assert "政治" in existing_profile.blacklist_topics
        meta = existing_profile.get_field_meta("blacklist_topics")
        assert meta.confidence == 1.0
        assert meta.source == "manual"
        mock_storage.save_group_profile.assert_called_once()
