"""SaveKnowledgeTool 测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from iris_memory.tools import SaveKnowledgeTool
from iris_memory.l3_kg import L3KGAdapter


@pytest.fixture
def tool():
    return SaveKnowledgeTool()


@pytest.fixture
def mock_adapter():
    adapter = MagicMock(spec=L3KGAdapter)
    adapter._is_available = True
    adapter.add_node = AsyncMock(return_value=True)
    adapter.add_edge = AsyncMock(return_value=True)
    return adapter


@pytest.fixture
def mock_component_manager(mock_adapter):
    manager = MagicMock()
    manager.get_component = MagicMock(return_value=mock_adapter)
    return manager


@pytest.fixture
def mock_context():
    context = Mock()
    event = Mock()
    inner_context = Mock()
    inner_context.event = event
    context.context = inner_context
    return context


@pytest.mark.asyncio
async def test_save_knowledge_basic(
    tool, mock_context, mock_component_manager, monkeypatch
):
    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager",
        lambda: mock_component_manager,
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    mock_platform_adapter = Mock()
    mock_platform_adapter.get_group_id = Mock(return_value="group_1")
    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_platform_adapter)
    )
    mock_config = Mock()
    mock_config.get = Mock(return_value=False)
    monkeypatch.setattr("iris_memory.config.get_config", lambda: mock_config)

    nodes = [
        {
            "label": "Person",
            "name": "Alice",
            "content": "Alice is a software engineer",
            "confidence": 0.9,
        }
    ]

    result = await tool.call(mock_context, nodes=nodes, edges=[])

    assert "成功保存" in result
    assert "1 个节点" in result


@pytest.mark.asyncio
async def test_save_knowledge_with_edges(
    tool, mock_context, mock_component_manager, monkeypatch
):
    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager",
        lambda: mock_component_manager,
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    mock_platform_adapter = Mock()
    mock_platform_adapter.get_group_id = Mock(return_value="group_1")
    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_platform_adapter)
    )
    mock_config = Mock()
    mock_config.get = Mock(return_value=False)
    monkeypatch.setattr("iris_memory.config.get_config", lambda: mock_config)

    nodes = [
        {
            "label": "Person",
            "name": "Alice",
            "content": "Alice is a software engineer",
            "confidence": 0.9,
        },
        {
            "label": "Event",
            "name": "Conference",
            "content": "AI Conference 2024",
            "confidence": 0.8,
        },
    ]

    edges = [
        {
            "source_name": "Alice",
            "target_name": "Conference",
            "relation_type": "ATTENDED",
            "confidence": 0.85,
        }
    ]

    result = await tool.call(mock_context, nodes=nodes, edges=edges)

    assert "成功保存" in result
    assert "2 个节点" in result
    assert "1 条边" in result


@pytest.mark.asyncio
async def test_save_knowledge_empty_nodes(
    tool, mock_context, mock_component_manager, monkeypatch
):
    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager",
        lambda: mock_component_manager,
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    result = await tool.call(mock_context, nodes=[], edges=[])

    assert "未提供任何节点" in result


@pytest.mark.asyncio
async def test_save_knowledge_adapter_unavailable(tool, mock_context, monkeypatch):
    mock_manager = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter._is_available = False
    mock_manager.get_component = MagicMock(return_value=mock_adapter)

    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager", lambda: mock_manager
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    result = await tool.call(
        mock_context,
        nodes=[{"label": "Person", "name": "Alice", "content": "Test"}],
        edges=[],
    )

    assert "知识图谱不可用" in result


@pytest.mark.asyncio
async def test_save_knowledge_clamps_confidence(
    tool, mock_context, mock_adapter, mock_component_manager, monkeypatch
):
    """回归：节点 confidence 越界时应钳制到 [0.0, 1.0]

    历史 bug：LLM 返回的 confidence（如 1.5 或 -0.3）原样写入 GraphNode，
    经 max() 合并后被永久固化，破坏遗忘评分语义。修复后使用
    ``max(0.0, min(1.0, float(raw_conf)))`` 钳制。
    """
    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager",
        lambda: mock_component_manager,
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    mock_platform_adapter = Mock()
    mock_platform_adapter.get_group_id = Mock(return_value="group_1")
    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_platform_adapter)
    )
    mock_config = Mock()
    mock_config.get = Mock(return_value=False)
    monkeypatch.setattr("iris_memory.config.get_config", lambda: mock_config)

    nodes = [
        {
            "label": "Person",
            "name": "OverConfident",
            "content": "confidence above 1.0",
            "confidence": 1.5,
        },
        {
            "label": "Person",
            "name": "UnderConfident",
            "content": "confidence below 0.0",
            "confidence": -0.3,
        },
    ]

    result = await tool.call(mock_context, nodes=nodes, edges=[])

    assert "成功保存" in result
    assert mock_adapter.add_node.call_count == 2

    stored_nodes = [call.args[0] for call in mock_adapter.add_node.call_args_list]
    # 1.5 应被钳制为 1.0
    assert stored_nodes[0].confidence == 1.0
    # -0.3 应被钳制为 0.0
    assert stored_nodes[1].confidence == 0.0


@pytest.mark.asyncio
async def test_save_knowledge_skips_subjectless_preference(
    tool, mock_context, mock_adapter, mock_component_manager, monkeypatch
):
    """回归：save_knowledge 降级无 Person 关联的主体绑定节点

    Preference/Trait/Belief/Goal/Skill 节点缺少 Person 关联边时，
    不再硬阻止保存（避免误伤有用信息），而是降级置信度并标记，
    交由梦境遗忘清洗按综合评分处理。
    """
    monkeypatch.setattr(
        "iris_memory.tools.save_knowledge.get_component_manager",
        lambda: mock_component_manager,
    )
    monkeypatch.setattr("iris_memory.utils.sanitize_input", lambda x, source="": x)

    mock_platform_adapter = Mock()
    mock_platform_adapter.get_group_id = Mock(return_value="group_1")
    monkeypatch.setattr(
        "iris_memory.platform.get_adapter", Mock(return_value=mock_platform_adapter)
    )
    mock_config = Mock()
    mock_config.get = Mock(return_value=False)
    monkeypatch.setattr("iris_memory.config.get_config", lambda: mock_config)

    # Person 节点 + Preference 节点（无 Person 边，应被降级但仍保存）
    nodes = [
        {
            "label": "Person",
            "name": "Alice",
            "content": "Alice 是一名工程师",
            "confidence": 0.9,
        },
        {
            "label": "Preference",
            "name": "角色偏好",
            "content": "有特定角色偏好",
            "confidence": 0.8,
        },
    ]

    result = await tool.call(mock_context, nodes=nodes, edges=[])

    # 返回信息包含"降级"
    assert "降级" in result

    # 两个节点都被保存（不阻止保存）
    assert mock_adapter.add_node.call_count == 2

    # Preference 节点被降级置信度
    saved_nodes = [call.args[0] for call in mock_adapter.add_node.call_args_list]
    pref_node = next(n for n in saved_nodes if n.label == "Preference")
    assert pref_node.confidence <= 0.4
    assert pref_node.properties.get("orphaned_subject") == "true"

    # Person 节点保持原置信度
    person_node = next(n for n in saved_nodes if n.label == "Person")
    assert person_node.confidence == 0.9
