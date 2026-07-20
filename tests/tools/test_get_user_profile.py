"""测试 GetProfileTool（用户画像模式）"""

import pytest
from unittest.mock import Mock, AsyncMock
from iris_memory.tools import GetProfileTool


@pytest.fixture
def tool():
    return GetProfileTool()


@pytest.fixture
def mock_context():
    context = Mock()
    event = Mock()
    inner_context = Mock()
    inner_context.event = event
    context.context = inner_context
    return context


@pytest.mark.asyncio
async def test_tool_initialization(tool):
    assert tool.name == "get_profile"
    assert "画像" in tool.description


@pytest.mark.asyncio
async def test_get_user_profile(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_user_name = Mock(return_value="测试用户")
    mock_adapter.get_group_id = Mock(return_value="group_456")

    mock_profile = Mock()
    mock_profile.user_name = "测试用户"
    mock_profile.user_id = "user_123"
    mock_profile.personality_tags = ["外向"]
    mock_profile.interests = ["编程"]
    mock_profile.occupation = ""
    mock_profile.language_style = ""
    mock_profile.communication_style = ""
    mock_profile.emotional_baseline = ""
    mock_profile.bot_relationship = ""
    mock_profile.historical_names = []
    mock_profile.taboo_topics = []
    mock_profile.important_dates = []
    mock_profile.important_events = []

    mock_profile_storage = Mock()
    mock_profile_storage.is_available = True

    mock_user_manager = Mock()
    mock_user_manager.get_or_create = AsyncMock(return_value=mock_profile)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.get_component_manager",
        Mock(return_value=Mock(get_component=Mock(return_value=mock_profile_storage))),
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.UserProfileManager",
        Mock(return_value=mock_user_manager),
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.get_config",
        Mock(return_value=Mock(get=Mock(return_value=False))),
    )

    result = await tool.call(mock_context, target_type="user", target_id="user_123")

    assert result is not None


@pytest.mark.asyncio
async def test_get_user_profile_no_id(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value=None)
    mock_adapter.get_group_id = Mock(return_value=None)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )

    result = await tool.call(mock_context, target_type="user")

    assert "无法获取用户ID" in result
