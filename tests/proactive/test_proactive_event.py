"""ProactiveMessageEvent 单元测试"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from iris_memory.proactive.proactive_event import ProactiveMessageEvent


@pytest.mark.asyncio
async def test_proactive_event_send_uses_context_session():
    context = SimpleNamespace(send_message=AsyncMock())

    event = ProactiveMessageEvent(
        context=context,
        umo="aiocqhttp:GroupMessage:12345",
        trigger_prompt="你好，最近怎么样？",
        user_id="u100",
        sender_name="Alice",
        group_id="12345",
        proactive_context={"reason": "test"},
    )

    with patch("astrbot.core.platform.astr_message_event.AstrMessageEvent.send", new=AsyncMock()) as super_send:
        await event.send(event.make_result().chain)

    context.send_message.assert_awaited_once_with(event.session, event.make_result().chain)
    super_send.assert_awaited_once()


def test_proactive_event_sets_extras_and_identity():
    context = SimpleNamespace(send_message=AsyncMock())

    event = ProactiveMessageEvent(
        context=context,
        umo="bad-format-umo",
        trigger_prompt="主动问候",
        user_id="u200",
        sender_name="Bob",
        group_id=None,
        proactive_context={"reason": "emotion"},
    )

    assert event.get_extra("iris_proactive", False) is True
    assert event.get_extra("iris_proactive_context", {}).get("reason") == "emotion"
    assert event.get_sender_id() == "u200"
    assert event.message_str == "主动问候"
