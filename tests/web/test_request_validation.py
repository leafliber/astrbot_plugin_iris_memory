"""Web 参数缺失与旧空键会话兼容性回归。"""

from unittest.mock import Mock

import pytest
from quart import Quart

from astrbot_plugin_iris_memory.iris_memory.web.routes.memory import list_l1_buffer
from astrbot_plugin_iris_memory.iris_memory.web.routes.profile import update_group_profile


@pytest.mark.asyncio
async def test_l1_missing_session_returns_400():
    app = Quart(__name__)
    async with app.test_request_context("/"):
        response, status = await list_l1_buffer()
        assert status == 400
        assert "group_id" in (await response.get_json())["error"]


@pytest.mark.asyncio
async def test_l1_explicit_empty_session_is_still_accessible(monkeypatch):
    app = Quart(__name__)
    buffer = Mock(is_available=True)
    buffer.get_context.return_value = []
    manager = Mock()
    manager.get_component.return_value = buffer
    monkeypatch.setattr(
        "astrbot_plugin_iris_memory.iris_memory.web.routes.memory.get_component_manager", lambda: manager
    )
    async with app.test_request_context("/?group_id="):
        await list_l1_buffer()
        buffer.get_context.assert_called_once_with("")


@pytest.mark.asyncio
async def test_profile_null_json_returns_400():
    app = Quart(__name__)
    async with app.test_request_context(
        "/?group_id=g1",
        method="POST",
        data="null",
        headers={"Content-Type": "application/json"},
    ):
        response, status = await update_group_profile()
        assert status == 400
        assert "请求体不能为空" in (await response.get_json())["error"]
