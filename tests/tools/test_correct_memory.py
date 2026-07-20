"""测试 CorrectMemoryTool"""

import pytest
from unittest.mock import Mock
from iris_memory.tools import CorrectMemoryTool


@pytest.fixture
def tool():
    return CorrectMemoryTool()


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
    assert tool.name == "correct_memory"
    assert "修正" in tool.description or "纠正" in tool.description
    assert "memory_id" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_correct_memory_missing_params(tool, mock_context):
    result = await tool.call(mock_context, memory_id="mem_123")
    assert "参数不完整" in result


@pytest.mark.asyncio
async def test_correct_memory_l2_unavailable(tool, mock_context, monkeypatch):
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")

    mock_l2 = Mock()
    mock_l2._is_available = False

    mock_manager = Mock()
    mock_manager.get_component = Mock(return_value=mock_l2)

    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_adapter)
    )
    monkeypatch.setattr(
        "iris_memory.tools.correct_memory.get_component_manager",
        Mock(return_value=mock_manager),
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    result = await tool.call(
        mock_context, memory_id="mem_123", correction="修正内容", reason="修正原因"
    )

    assert "不可用" in result
