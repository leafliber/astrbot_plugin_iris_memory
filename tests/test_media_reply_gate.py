"""纯媒体消息（贴纸/图片）回应判断的单元测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_iris_memory.main import IrisMemoryPlugin


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[贴纸 😍]", True),
        ("[图片]", True),
        ("[贴纸]", True),
        ("", True),
        ("   ", True),
        ("[图片] 看这个", False),
        ("[语音] 听到没", False),
        ("在吗", False),
        ("这个贴纸像不像我", False),
    ],
)
def test_is_media_only_message(text, expected):
    event = SimpleNamespace(message_str=text)
    assert IrisMemoryPlugin._is_media_only_message(event) is expected


@pytest.mark.asyncio
async def test_gate_returns_false_without_component_manager():
    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = None
    event = SimpleNamespace(message_str="[贴纸 😍]")
    assert await plugin._should_reply_to_media(event) is False


@pytest.mark.asyncio
async def test_gate_returns_false_when_llm_manager_missing():
    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = Mock()
    plugin.component_manager.get_available_component = Mock(return_value=None)
    event = SimpleNamespace(message_str="[贴纸 😍]")
    assert await plugin._should_reply_to_media(event) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_output", "expected"),
    [
        ('{"reply": true}', True),
        ('{"reply":true}', True),
        ('{"reply": false}', False),
        ("我觉得不用回", False),
    ],
)
async def test_gate_follows_model_decision(model_output, expected):
    plugin = object.__new__(IrisMemoryPlugin)
    manager = Mock()
    manager.generate = AsyncMock(return_value=model_output)
    plugin.component_manager = Mock()
    plugin.component_manager.get_available_component = Mock(
        side_effect=lambda name: manager if name == "llm_manager" else None
    )
    plugin._get_provider_id = AsyncMock(return_value="deepseek")
    event = SimpleNamespace(message_str="[贴纸 😍]")
    assert await plugin._should_reply_to_media(event) is expected


@pytest.mark.asyncio
async def test_gate_returns_false_when_llm_call_fails():
    plugin = object.__new__(IrisMemoryPlugin)
    manager = Mock()
    manager.generate = AsyncMock(side_effect=RuntimeError("boom"))
    plugin.component_manager = Mock()
    plugin.component_manager.get_available_component = Mock(
        side_effect=lambda name: manager if name == "llm_manager" else None
    )
    plugin._get_provider_id = AsyncMock(return_value=None)
    event = SimpleNamespace(message_str="[贴纸 😍]")
    assert await plugin._should_reply_to_media(event) is False
