"""learning remember 手令测试：双写 few_shot + expression_pattern 并直接通过"""

from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.commands.base import ParsedArgs
from iris_memory.commands.learning_handler import LearningCommandHandler
from iris_memory.learning.storage import LearningStorage


@pytest.fixture
def learning_env(tmp_path, monkeypatch):
    storage = LearningStorage(tmp_path / "learning.db")
    storage.init_schema()

    component = Mock()
    component.is_available = True
    component.storage = storage

    manager = Mock()
    manager.get_component = Mock(return_value=component)

    monkeypatch.setattr(
        "iris_memory.commands.learning_handler.get_component_manager",
        lambda: manager,
    )

    adapter = Mock()
    adapter.get_group_id = Mock(return_value="group_1")
    adapter.get_user_id = Mock(return_value="user_1")
    monkeypatch.setattr(
        "iris_memory.commands.learning_handler.get_adapter",
        Mock(return_value=adapter),
    )

    import iris_memory.core.persona as persona_mod

    monkeypatch.setattr(
        persona_mod, "resolve_persona", AsyncMock(return_value="default")
    )

    yield storage
    storage.close()


@pytest.mark.asyncio
async def test_remember_writes_pair_and_pattern_approved(learning_env):
    handler = LearningCommandHandler()
    args = ParsedArgs(
        raw_args=["remember", "用户：今晚吃什么", "=>", "吃火锅怎么样，热乎乎的！"]
    )

    result = await handler.handle(Mock(), args, "remember")

    assert result.success is True
    assert result.details["pair_id"] > 0
    assert result.details["pattern_id"] > 0

    pairs = learning_env.get_approved_few_shots("group_1", 10, "default")
    assert len(pairs) == 1
    assert pairs[0]["user_text"] == "用户：今晚吃什么"
    assert pairs[0]["bot_text"] == "吃火锅怎么样，热乎乎的！"

    patterns = learning_env.get_approved_patterns("group_1", 10, "default")
    assert len(patterns) == 1
    assert patterns[0]["expression"] == "吃火锅怎么样，热乎乎的！"
    assert patterns[0]["source_pair_id"] == result.details["pair_id"]
    assert patterns[0]["scene"] == "用户：今晚吃什么"

    assert not learning_env.get_pending_pairs(10)
    assert not learning_env.get_pending_patterns(10)


@pytest.mark.asyncio
async def test_remember_requires_arrow(learning_env):
    handler = LearningCommandHandler()
    args = ParsedArgs(raw_args=["remember", "没有分隔符的输入"])

    result = await handler.handle(Mock(), args, "remember")

    assert result.success is False
    assert "格式错误" in result.message


@pytest.mark.asyncio
async def test_remember_rejects_empty_parts(learning_env):
    handler = LearningCommandHandler()
    args = ParsedArgs(raw_args=["remember", "=>", "只有表达"])

    result = await handler.handle(Mock(), args, "remember")

    assert result.success is False
    assert "不能为空" in result.message


@pytest.mark.asyncio
async def test_remember_truncates_scene_to_20_chars(learning_env):
    long_context = "这是一个非常非常非常非常长的上下文内容用来测试场景截断行为是否生效"
    handler = LearningCommandHandler()
    args = ParsedArgs(raw_args=["remember", long_context, "=>", "短回复"])

    result = await handler.handle(Mock(), args, "remember")

    assert result.success is True
    patterns = learning_env.get_approved_patterns("group_1", 10, "default")
    assert len(patterns[0]["scene"]) <= 20
