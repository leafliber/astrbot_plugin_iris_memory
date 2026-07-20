"""测试 GetProfileTool（群聊画像模式）"""

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
async def test_get_group_profile(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_group_id = Mock(return_value="group_123")

    mock_profile = Mock()
    mock_profile.group_name = "测试群"
    mock_profile.group_id = "group_123"
    mock_profile.interests = ["技术"]
    mock_profile.atmosphere_tags = ["轻松"]
    mock_profile.long_term_tags = []
    mock_profile.blacklist_topics = []

    mock_profile_storage = Mock()
    mock_profile_storage.is_available = True

    mock_group_manager = Mock()
    mock_group_manager.get_or_create = AsyncMock(return_value=mock_profile)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.get_component_manager",
        Mock(return_value=Mock(get_component=Mock(return_value=mock_profile_storage))),
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.GroupProfileManager",
        Mock(return_value=mock_group_manager),
    )
    monkeypatch.setattr(
        "iris_memory.tools.get_profile.get_config",
        Mock(return_value=Mock(get=Mock(return_value=False))),
    )

    result = await tool.call(mock_context, target_type="group", target_id="group_123")

    assert result is not None


@pytest.mark.asyncio
async def test_get_group_profile_no_id(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_group_id = Mock(return_value=None)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )

    result = await tool.call(mock_context, target_type="group")

    assert "无法获取群聊ID" in result
