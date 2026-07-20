"""测试 SaveMemoryTool"""

import pytest
from unittest.mock import Mock, AsyncMock
from iris_memory.tools import SaveMemoryTool


@pytest.fixture
def tool():
    return SaveMemoryTool()


@pytest.fixture
def mock_context():
    context = Mock()
    event = Mock()
    event.user_id = "test_user_123"
    inner_context = Mock()
    inner_context.event = event
    context.context = inner_context
    return context


@pytest.mark.asyncio
async def test_tool_initialization(tool):
    assert tool.name == "save_memory"
    assert "记忆" in tool.description
    assert "content" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_save_memory_success(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_test123")

    mock_manager = Mock()
    mock_manager.get_component = Mock(return_value=mock_l2)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )
    monkeypatch.setattr("iris_memory.config.get_config", Mock(return_value=mock_config))
    monkeypatch.setattr(
        "iris_memory.tools.save_memory.get_component_manager",
        Mock(return_value=mock_manager),
    )

    result = await tool.call(mock_context, content="测试记忆内容", confidence=0.9)

    assert result is not None
    assert "成功" in result or "已保存" in result


@pytest.mark.asyncio
async def test_save_memory_empty_content(tool, mock_context):
    result = await tool.call(mock_context, content="")
    assert "不能为空" in result


@pytest.mark.asyncio
async def test_save_memory_l2_unavailable(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = False

    mock_manager = Mock()
    mock_manager.get_component = Mock(return_value=mock_l2)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )
    monkeypatch.setattr("iris_memory.config.get_config", Mock(return_value=mock_config))
    monkeypatch.setattr(
        "iris_memory.tools.save_memory.get_component_manager",
        Mock(return_value=mock_manager),
    )

    result = await tool.call(mock_context, content="测试内容")
    assert "不可用" in result
