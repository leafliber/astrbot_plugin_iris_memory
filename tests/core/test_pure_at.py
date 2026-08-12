"""Pure @ takeover tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot.api.message_components import At, Plain
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.entities import ProviderRequest


def _event_with_messages(messages):
    event = MagicMock()
    event.get_messages.return_value = messages
    event.get_self_id.return_value = "bot-1"
    return event


def test_pure_at_self_detection():
    from main import _is_pure_at_self

    assert _is_pure_at_self(_event_with_messages([At(qq="bot-1", name="Chito")]))
    assert not _is_pure_at_self(
        _event_with_messages([At(qq="other", name="群友")])
    )
    assert not _is_pure_at_self(
        _event_with_messages([At(qq="bot-1", name="Chito"), Plain("你好")])
    )


@pytest.mark.asyncio
async def test_whitespace_transport_prompt_is_not_sent_to_model():
    request = ProviderRequest(prompt=" ")
    request.extra_user_content_parts.append(TextPart(text="L1 only"))

    assembled = await request.assemble_context()

    assert assembled == {
        "role": "user",
        "content": [{"type": "text", "text": "L1 only"}],
    }


@pytest.mark.asyncio
async def test_pure_at_uses_standard_request_without_semantic_prompt():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    buffer = MagicMock()
    buffer.get_context.return_value = [SimpleNamespace(content="推荐旧世界")]
    plugin.component_manager = MagicMock()
    plugin.component_manager.get_available_component.return_value = buffer
    plugin.config = MagicMock()
    plugin.config.get.return_value = True

    conversation = SimpleNamespace(id="conv-1", persona_id="default", history="[]")
    conv_mgr = MagicMock()
    conv_mgr.get_curr_conversation_id = AsyncMock(return_value="conv-1")
    conv_mgr.get_conversation = AsyncMock(return_value=conversation)
    plugin.context = MagicMock(conversation_manager=conv_mgr)
    plugin.context.get_config.return_value = {
        "platform_settings": {"empty_mention_waiting": False}
    }

    event = _event_with_messages([At(qq="bot-1", name="Chito")])
    event.unified_msg_origin = "umo-1"
    request = SimpleNamespace(prompt=" ", extra_user_content_parts=[])
    event.request_llm.return_value = request

    adapter = MagicMock()
    adapter.get_session_id.return_value = "group-1"

    extras = {}
    event.set_extra.side_effect = lambda key, value: extras.__setitem__(key, value)

    with patch("iris_memory.platform.get_adapter", return_value=adapter):
        handled = await plugin._prepare_pure_at_request(event)

    assert handled is True
    event.request_llm.assert_called_once_with(
        prompt=" ",
        session_id="conv-1",
        contexts=[],
        system_prompt="",
        conversation=conversation,
    )
    assert request.extra_user_content_parts == []
    assert extras["provider_request"] is request
    assert extras["iris_pure_at"] is True


@pytest.mark.asyncio
async def test_pure_at_is_not_taken_over_while_astrbot_waiter_is_enabled():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = MagicMock()
    plugin.config = MagicMock()
    plugin.config.get.return_value = True
    plugin.context = MagicMock()
    plugin.context.get_config.return_value = {
        "platform_settings": {"empty_mention_waiting": True}
    }
    event = _event_with_messages([At(qq="bot-1", name="Chito")])
    event.unified_msg_origin = "umo-1"

    assert await plugin._prepare_pure_at_request(event) is False
    event.request_llm.assert_not_called()


@pytest.mark.asyncio
async def test_pure_at_without_l1_context_keeps_default_empty_behavior():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    buffer = MagicMock()
    buffer.get_context.return_value = []
    plugin.component_manager = MagicMock()
    plugin.component_manager.get_available_component.return_value = buffer
    plugin.config = MagicMock()
    plugin.config.get.return_value = True
    plugin.context = MagicMock()
    plugin.context.get_config.return_value = {
        "platform_settings": {"empty_mention_waiting": False}
    }

    event = _event_with_messages([At(qq="bot-1", name="Chito")])
    event.unified_msg_origin = "umo-1"
    adapter = MagicMock()
    adapter.get_session_id.return_value = "group-1"

    with patch("iris_memory.platform.get_adapter", return_value=adapter):
        handled = await plugin._prepare_pure_at_request(event)

    assert handled is False
    event.request_llm.assert_not_called()


@pytest.mark.asyncio
async def test_pure_at_feature_switch_disables_takeover():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = MagicMock()
    plugin.config = MagicMock()
    plugin.config.get.return_value = False
    plugin.context = MagicMock()

    event = _event_with_messages([At(qq="bot-1", name="Chito")])
    event.unified_msg_origin = "umo-1"

    assert await plugin._prepare_pure_at_request(event) is False
    event.request_llm.assert_not_called()
