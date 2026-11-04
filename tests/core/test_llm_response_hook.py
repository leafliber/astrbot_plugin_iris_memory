"""LLM 响应写入 L1 的回归测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.core.llm_response_hook import (
    handle_llm_response,
)


@pytest.mark.asyncio
async def test_identical_response_in_same_event_is_written_once():
    extras: dict = {}
    event = MagicMock()
    event.get_extra.side_effect = lambda key: extras.get(key)
    event.set_extra.side_effect = lambda key, value: extras.__setitem__(key, value)

    response = MagicMock()
    response.completion_text = "这是一条会因为重复响应钩子而再次写入的回复"

    buffer = MagicMock()
    buffer.add_message = AsyncMock(return_value=True)

    component_manager = MagicMock()
    component_manager.get_available_component.side_effect = lambda name: (
        buffer if name == "l1_buffer" else None
    )

    adapter = MagicMock()
    adapter.get_session_id.return_value = "private:user-1"

    with (
        patch(
            "astrbot_plugin_iris_memory.iris_memory.platform.get_adapter",
            return_value=adapter,
        ),
        patch(
            "astrbot_plugin_iris_memory.iris_memory.core.persona.resolve_persona",
            new_callable=AsyncMock,
            return_value="default",
        ),
    ):
        await handle_llm_response(event, response, component_manager)
        await handle_llm_response(event, response, component_manager)

    assert buffer.add_message.await_count == 1

