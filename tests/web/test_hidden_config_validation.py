"""隐藏配置更新必须在写入前拒绝非对象和无效数值。"""

from unittest.mock import Mock

import pytest
from quart import Quart

from astrbot_plugin_iris_memory.iris_memory.web.routes import hidden_config_routes as routes


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    '["bad"]', '123', '{',
    '{"updates": ["forgetting_lambda"]}',
    '{"updates": "forgetting_lambda"}',
    '{"updates": {"forgetting_lambda": true}}',
    '{"updates": {"forgetting_lambda": NaN}}',
    '{"updates": {"forgetting_lambda": Infinity}}',
    '{"updates": {"forgetting_lambda": -Infinity}}',
])
async def test_invalid_updates_never_reach_config(body, monkeypatch):
    config = Mock()
    monkeypatch.setattr(routes, "get_config", lambda: config)
    app = Quart(__name__)
    async with app.test_request_context(
        "/", method="POST", data=body, headers={"Content-Type": "application/json"}
    ):
        response, status = await routes.update_hidden_config()
        assert status == 400
        assert (await response.get_json())["success"] is False
    config.update_hidden.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 1, 0.5])
async def test_valid_float_updates_remain_supported(value, monkeypatch):
    config = Mock()
    monkeypatch.setattr(routes, "get_config", lambda: config)
    updates = {"forgetting_lambda": value}
    app = Quart(__name__)
    async with app.test_request_context("/", method="POST", json={"updates": updates}):
        response = await routes.update_hidden_config()
        assert (await response.get_json())["success"] is True
    config.update_hidden.assert_called_once_with(updates)
