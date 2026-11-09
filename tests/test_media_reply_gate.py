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


@pytest.mark.asyncio
async def test_media_gate_is_off_unless_continuity_enabled():
    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = None
    plugin.config = {}
    plugin._should_reply_to_media = AsyncMock(return_value=False)
    event = SimpleNamespace(
        message_str="[贴纸 😍]",
        get_extra=lambda *_args: None,
        get_group_id=lambda: "",
        stop_event=Mock(),
    )

    await plugin.on_all_message(event)

    plugin._should_reply_to_media.assert_not_awaited()
    event.stop_event.assert_not_called()


@pytest.mark.asyncio
async def test_sticker_understanding_is_off_unless_continuity_enabled(monkeypatch):
    monkeypatch.setattr(
        "astrbot_plugin_iris_memory.main.extract_sticker_meta", lambda _raw: object()
    )
    plugin = object.__new__(IrisMemoryPlugin)
    plugin.config = {}
    plugin._sticker_understanding = Mock(describe=AsyncMock(return_value="描述"))
    plugin._is_self_message = lambda _event: False
    plugin._sticker_inbound_allowed = lambda _event: True
    event = SimpleNamespace(message_obj=SimpleNamespace(raw_message=object()))

    await plugin.on_sticker_understanding(event)

    plugin._sticker_understanding.describe.assert_not_awaited()
