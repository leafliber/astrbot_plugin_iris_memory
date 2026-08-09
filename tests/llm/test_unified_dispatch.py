"""插件业务模块不得绕过 LLMManager，以及主管线回复的统计衔接测试。"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_business_code_has_no_raw_llm_provider_calls():
    """防止后续模块重新直接调用 context.llm_generate/provider.text_chat。"""
    violations: list[str] = []
    for path in [ROOT / "main.py", *(ROOT / "iris_memory").rglob("*.py")]:
        if path == ROOT / "iris_memory/llm/manager.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"llm_generate", "text_chat"}
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


@pytest.mark.asyncio
async def test_framework_reply_is_marked_on_request_and_settled_on_response():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    plugin.component_manager = None
    plugin._llm_manager = SimpleNamespace(
        record_framework_attempt=AsyncMock(),
        record_framework_response=AsyncMock()
    )
    plugin._handle_reply_decision = AsyncMock(return_value=False)
    plugin._get_provider_id = AsyncMock(return_value="provider")
    plugin._reply_on_llm_response = AsyncMock()

    extras = {"iris_mode": "chime_in"}
    event = Mock()
    event.get_extra.side_effect = lambda key: extras.get(key)
    event.set_extra.side_effect = lambda key, value: extras.__setitem__(key, value)
    req = SimpleNamespace(
        prompt="hello",
        extra_user_content_parts=[],
    )

    await plugin.on_llm_request(event, req)
    plugin._llm_manager.record_framework_attempt.assert_awaited_once_with(
        "proactive_reply_chime_in"
    )
    assert extras["iris_llm_tracking"]["module"] == "proactive_reply_chime_in"

    response = SimpleNamespace(completion_text="world", usage=None)
    await plugin.on_llm_response(event, response)

    plugin._llm_manager.record_framework_response.assert_awaited_once()
    kwargs = plugin._llm_manager.record_framework_response.await_args.kwargs
    assert kwargs["module"] == "proactive_reply_chime_in"
    assert kwargs["provider_id"] == "provider"
    assert extras["iris_llm_tracking"] is None
