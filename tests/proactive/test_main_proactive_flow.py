"""main.py 主动回复分支测试"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from main import IrisMemoryPlugin


class _FakeEvent:
    def __init__(self, *, proactive: bool, message: str = "hello"):
        self.message_str = message
        self.message_obj = SimpleNamespace(message=[])
        self.unified_msg_origin = "test:FriendMessage:u1"
        self._proactive = proactive
        self.persona = None  # persona isolation support

    def get_sender_id(self):
        return "u1"

    def get_extra(self, key, default=None):
        if key == "iris_proactive":
            return self._proactive
        if key == "iris_proactive_context":
            return {"reason": "unit-test"} if self._proactive else default
        return default

    def request_llm(self, prompt):
        return {"kind": "llm_request", "prompt": prompt}


@pytest.fixture
def plugin_stub():
    plugin = IrisMemoryPlugin.__new__(IrisMemoryPlugin)
    plugin._service = SimpleNamespace(
        cfg=SimpleNamespace(
            enable_inject=True,
            enable_memory=True,
            get_persona_id_for_storage=Mock(return_value="default"),
            get_persona_id_for_query=Mock(return_value=None),
        ),
        is_initialized=True,
        is_embedding_ready=Mock(return_value=True),
        member_identity=SimpleNamespace(resolve_tag=AsyncMock()),
        activate_session=AsyncMock(),
        image_analyzer=object(),
        analyze_images=AsyncMock(return_value=("IMG_CTX", "IMG_MEM")),
        prepare_llm_context=AsyncMock(return_value="CTX_BLOCK"),
        logger=SimpleNamespace(info=Mock(), warning=Mock(), debug=Mock()),
        record_chat_message=AsyncMock(),
        update_session_activity=Mock(),
        capture_and_store_memory=AsyncMock(),
    )
    return plugin


@pytest.mark.asyncio
async def test_on_all_messages_proactive_yields_llm_request(plugin_stub):
    event = _FakeEvent(proactive=True, message="trigger")

    with patch("main.get_group_id", return_value="g1"), patch("main.get_sender_name", return_value="Alice"):
        results = [item async for item in plugin_stub.on_all_messages(event)]

    assert results == [{"kind": "llm_request", "prompt": "trigger"}]
    plugin_stub._service.record_chat_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_llm_request_injects_proactive_directive_and_skips_identity_image(plugin_stub):
    event = _FakeEvent(proactive=True, message="trigger")
    req = SimpleNamespace(system_prompt="BASE")

    with patch("main.get_group_id", return_value="g1"), patch("main.get_sender_name", return_value="Alice"), patch.object(plugin_stub, "_build_proactive_directive", return_value="DIR_BLOCK"):
        await plugin_stub.on_llm_request(event, req)

    assert "CTX_BLOCK" in req.system_prompt
    assert "DIR_BLOCK" in req.system_prompt
    plugin_stub._service.member_identity.resolve_tag.assert_not_awaited()
    plugin_stub._service.analyze_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_llm_request_embedding_not_ready_adds_hint_and_returns(plugin_stub):
    event = _FakeEvent(proactive=False, message="hello")
    req = SimpleNamespace(system_prompt="BASE")
    plugin_stub._service.is_embedding_ready.return_value = False

    with patch("main.get_group_id", return_value="g1"), patch("main.get_sender_name", return_value="Alice"):
        await plugin_stub.on_llm_request(event, req)

    assert "记忆系统正在初始化" in req.system_prompt
    plugin_stub._service.prepare_llm_context.assert_not_awaited()
    plugin_stub._service.activate_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_llm_response_proactive_skips_memory_capture(plugin_stub):
    event = _FakeEvent(proactive=True, message="not-real-user-message")
    resp = SimpleNamespace(completion_text="bot proactive reply")

    with patch("main.get_group_id", return_value="g1"), patch("main.get_sender_name", return_value="Alice"):
        await plugin_stub.on_llm_response(event, resp)

    plugin_stub._service.record_chat_message.assert_awaited_once()
    plugin_stub._service.update_session_activity.assert_called_once_with("u1", "g1")
    plugin_stub._service.capture_and_store_memory.assert_not_awaited()
