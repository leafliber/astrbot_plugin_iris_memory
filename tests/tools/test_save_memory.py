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
async def test_save_memory_writes_importance(tool, mock_context, monkeypatch):
    """importance 参数映射为元数据 importance 与 importance_level"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_imp123")

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

    result = await tool.call(
        mock_context, content="重要记忆", confidence=0.9, importance="high"
    )

    assert "已保存" in result
    metadata = mock_l2.add_memory.call_args[0][1]
    assert metadata["importance"] == 0.8
    assert metadata["importance_level"] == "high"


@pytest.mark.asyncio
async def test_save_memory_writes_active_users(tool, mock_context, monkeypatch):
    """工具写入的记忆带 active_users,与 L1 总结形态对齐,供用户级清理命中"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_au123")

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

    result = await tool.call(mock_context, content="带主体的记忆")

    assert "已保存" in result
    metadata = mock_l2.add_memory.call_args[0][1]
    assert metadata["user_id"] == "user_123"
    assert metadata["active_users"] == "user_123"


@pytest.mark.asyncio
async def test_save_memory_invalid_importance_defaults(tool, mock_context, monkeypatch):
    """非法 importance 档位回退 medium"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_imp456")

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

    await tool.call(mock_context, content="记忆", confidence=0.9, importance="bogus")

    metadata = mock_l2.add_memory.call_args[0][1]
    assert metadata["importance"] == 0.5
    assert metadata["importance_level"] == "medium"


@pytest.mark.asyncio
async def test_save_memory_ttl_writes_expires_at(tool, mock_context, monkeypatch):
    """ttl_hours 合法时写入 expires_at"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_ttl123")

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

    result = await tool.call(
        mock_context, content="明天要考试", confidence=0.9, ttl_hours=24
    )

    assert "过期时间" in result
    metadata = mock_l2.add_memory.call_args[0][1]
    assert "expires_at" in metadata
    assert metadata["expires_at"] > metadata["timestamp"]


@pytest.mark.asyncio
async def test_save_memory_invalid_ttl_ignored(tool, mock_context, monkeypatch):
    """ttl_hours 非法（0/负/非数）时不写 expires_at"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_ttl456")

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

    for bad_ttl in (0, -5, "abc"):
        await tool.call(mock_context, content="记忆", confidence=0.9, ttl_hours=bad_ttl)
        metadata = mock_l2.add_memory.call_args[0][1]
        assert "expires_at" not in metadata


@pytest.mark.asyncio
async def test_save_memory_scope_global(tool, mock_context, monkeypatch):
    """scope=global 时元数据带全局标记"""
    mock_adapter = Mock()
    mock_adapter.get_user_id = Mock(return_value="user_123")
    mock_adapter.get_group_id = Mock(return_value="group_456")
    mock_adapter.get_user_name = Mock(return_value="测试用户")
    mock_context.context.event.is_admin = Mock(return_value=True)

    mock_config = Mock()
    mock_config.get = Mock(return_value=True)

    mock_l2 = Mock()
    mock_l2.is_available = True
    mock_l2.add_memory = AsyncMock(return_value="mem_scope1")

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

    await tool.call(mock_context, content="全局事实", confidence=0.9, scope="global")
    metadata = mock_l2.add_memory.call_args[0][1]
    assert metadata.get("scope") == "global"

    mock_context.context.event.is_admin.return_value = False
    result = await tool.call(
        mock_context, content="普通成员请求的全局事实", confidence=0.9, scope="global"
    )
    metadata = mock_l2.add_memory.call_args[0][1]
    assert "scope" not in metadata
    assert "自动降级" in result

    await tool.call(mock_context, content="群内事实", confidence=0.9)
    metadata = mock_l2.add_memory.call_args[0][1]
    assert "scope" not in metadata

    await tool.call(mock_context, content="非法值", confidence=0.9, scope="bogus")
    metadata = mock_l2.add_memory.call_args[0][1]
    assert "scope" not in metadata


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
