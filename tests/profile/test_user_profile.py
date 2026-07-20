"""用户画像管理器测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from iris_memory.profile.user_profile import UserProfileManager
from iris_memory.profile.models import (
    UserProfile,
    UpdateTier,
)
from iris_memory.profile.storage import ProfileStorage


class TestUserProfileManager:
    """用户画像管理器测试"""

    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock(spec=ProfileStorage)
        storage.get_user_profile = AsyncMock()
        storage.save_user_profile = AsyncMock()
        return storage

    @pytest.fixture
    def manager(self, mock_storage):
        return UserProfileManager(mock_storage)

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456", user_name="小明")
        mock_storage.get_user_profile.return_value = existing_profile

        profile = await manager.get_or_create("user_456", "group_123")

        assert profile.user_id == "user_456"
        assert profile.user_name == "小明"
        mock_storage.get_user_profile.assert_called_once_with(
            "user_456", "group_123", "default"
        )

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, manager, mock_storage):
        mock_storage.get_user_profile.return_value = None

        profile = await manager.get_or_create("user_456", "group_123")

        assert profile.user_id == "user_456"
        assert profile.user_name == ""
        mock_storage.get_user_profile.assert_called_once_with(
            "user_456", "group_123", "default"
        )

    @pytest.mark.asyncio
    async def test_update_user_name(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.update_user_name(
            user_id="user_456", group_id="group_123", user_name="小明"
        )

        assert existing_profile.user_name == "小明"
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_name_with_name_change(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456", user_name="旧昵称")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.update_user_name(
            user_id="user_456", group_id="group_123", user_name="新昵称"
        )

        assert existing_profile.user_name == "新昵称"
        assert "旧昵称" in existing_profile.historical_names
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_from_analysis_mid_tier(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.update_from_analysis(
            user_id="user_456",
            group_id="group_123",
            personality_tags=["外向", "幽默"],
            interests=["编程", "游戏"],
            language_style="简洁",
            tier=UpdateTier.MID,
            confidence=0.7,
        )

        assert existing_profile.personality_tags == ["外向", "幽默"]
        assert existing_profile.interests == ["编程", "游戏"]
        assert existing_profile.language_style == "简洁"

        tracker = existing_profile.get_update_tracker()
        assert tracker.summary_count_since_mid_update == 0
        assert tracker.last_mid_update_time is not None
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_from_analysis_merges_lists(self, manager, mock_storage):
        existing_profile = UserProfile(
            user_id="user_456", personality_tags=["外向"], interests=["编程"]
        )
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.update_from_analysis(
            user_id="user_456",
            group_id="group_123",
            personality_tags=["幽默", "外向"],
            interests=["游戏", "编程"],
            tier=UpdateTier.MID,
            confidence=0.7,
        )

        assert "外向" in existing_profile.personality_tags
        assert "幽默" in existing_profile.personality_tags
        assert "编程" in existing_profile.interests
        assert "游戏" in existing_profile.interests

    @pytest.mark.asyncio
    async def test_update_favorability_delta_positive(self, manager, mock_storage):
        """正向 delta 增加好感度"""
        existing_profile = UserProfile(user_id="user_456", favorability=50.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": True,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                favorability_delta=10,
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        assert existing_profile.favorability == 60.0

    @pytest.mark.asyncio
    async def test_update_favorability_delta_negative(self, manager, mock_storage):
        """负向 delta 降低好感度"""
        existing_profile = UserProfile(user_id="user_456", favorability=50.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": True,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                favorability_delta=-15,
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        assert existing_profile.favorability == 35.0

    @pytest.mark.asyncio
    async def test_update_favorability_delta_clamped(self, manager, mock_storage):
        """delta 超过 max_delta 被夹紧"""
        existing_profile = UserProfile(user_id="user_456", favorability=50.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": True,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                favorability_delta=100,  # 远超 max_delta=20
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        # 被夹紧为 +20
        assert existing_profile.favorability == 70.0

    @pytest.mark.asyncio
    async def test_update_favorability_clamps_to_100(self, manager, mock_storage):
        """好感度上限 100"""
        existing_profile = UserProfile(user_id="user_456", favorability=95.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": True,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                favorability_delta=20,
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        assert existing_profile.favorability == 100.0

    @pytest.mark.asyncio
    async def test_update_favorability_disabled(self, manager, mock_storage):
        """favorability_enable=False 时不更新好感度"""
        existing_profile = UserProfile(user_id="user_456", favorability=50.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": False,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                favorability_delta=10,
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        assert existing_profile.favorability == 50.0

    @pytest.mark.asyncio
    async def test_update_favorability_none_delta_no_change(
        self, manager, mock_storage
    ):
        """favorability_delta=None 时不更新好感度"""
        existing_profile = UserProfile(user_id="user_456", favorability=50.0)
        mock_storage.get_user_profile.return_value = existing_profile

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile.favorability_enable": True,
                "profile_favorability_max_delta": 20.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            await manager.update_from_analysis(
                user_id="user_456",
                group_id="group_123",
                tier=UpdateTier.MID,
                confidence=0.7,
            )

        assert existing_profile.favorability == 50.0

    @pytest.mark.asyncio
    async def test_update_long_term_from_analysis(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.update_long_term_from_analysis(
            user_id="user_456",
            group_id="group_123",
            occupation="软件工程师",
            bot_relationship="朋友",
            important_events=["入职新公司"],
            taboo_topics=["个人隐私"],
            important_dates=["2024-01-15: 入职纪念日"],
            confidence=0.85,
        )

        assert existing_profile.occupation == "软件工程师"
        assert existing_profile.bot_relationship == "朋友"
        assert "入职新公司" in existing_profile.important_events
        assert "个人隐私" in existing_profile.taboo_topics
        assert len(existing_profile.important_dates) > 0

        tracker = existing_profile.get_update_tracker()
        assert tracker.last_long_update_time is not None

    @pytest.mark.asyncio
    async def test_should_update_mid(self, manager, mock_storage):
        profile = UserProfile(user_id="user_456")
        tracker = profile.get_update_tracker()
        tracker.summary_count_since_mid_update = 5
        tracker.last_mid_update_time = datetime.now() - timedelta(hours=25)
        profile.set_update_tracker(tracker)

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile_mid_update_interval_summaries": 5,
                "profile_mid_update_interval_hours": 24.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            assert manager.should_update_mid(profile) is True

    @pytest.mark.asyncio
    async def test_should_not_update_mid_too_soon(self, manager, mock_storage):
        profile = UserProfile(user_id="user_456")
        tracker = profile.get_update_tracker()
        tracker.summary_count_since_mid_update = 1
        tracker.last_mid_update_time = datetime.now() - timedelta(hours=1)
        profile.set_update_tracker(tracker)

        with patch("iris_memory.profile.user_profile.get_config") as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.get.side_effect = lambda k, d=None: {
                "profile_mid_update_interval_summaries": 5,
                "profile_mid_update_interval_hours": 24.0,
            }.get(k, d)
            mock_config.return_value = mock_config_obj

            assert manager.should_update_mid(profile) is False

    @pytest.mark.asyncio
    async def test_set_bot_relationship(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.set_bot_relationship("user_456", "group_123", "小助手")

        assert existing_profile.bot_relationship == "小助手"
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_taboo_topic(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.add_taboo_topic("user_456", "group_123", "个人隐私")

        assert "个人隐私" in existing_profile.taboo_topics
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_important_date(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        await manager.add_important_date(
            "user_456", "group_123", "2024-01-15", "入职纪念日"
        )

        assert len(existing_profile.important_dates) > 0
        assert existing_profile.important_dates[0]["date"] == "2024-01-15"
        assert existing_profile.important_dates[0]["description"] == "入职纪念日"
        mock_storage.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_increment_summary_count(self, manager, mock_storage):
        existing_profile = UserProfile(user_id="user_456")
        mock_storage.get_user_profile.return_value = existing_profile

        tracker = existing_profile.get_update_tracker()
        tracker.increment_summary_count()
        existing_profile.set_update_tracker(tracker)

        assert tracker.summary_count_since_mid_update == 1
