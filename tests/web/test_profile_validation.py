"""画像写接口的 JSON 校验及人格路由回归。"""

from unittest.mock import AsyncMock, Mock

import pytest
from quart import Quart

from astrbot_plugin_iris_memory.iris_memory.web.routes import profile


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [
    profile.update_group_profile, profile.update_user_profile,
    profile.delete_group_profile, profile.delete_user_profile,
])
@pytest.mark.parametrize("body", ['["bad"]', '"bad"', '123', '{'])
async def test_invalid_profile_json_returns_400(handler, body, monkeypatch):
    storage_getter = Mock()
    monkeypatch.setattr(profile, "get_profile_storage", storage_getter)
    app = Quart(__name__)
    async with app.test_request_context(
        "/", method="POST", data=body, headers={"Content-Type": "application/json"}
    ):
        response, status = await handler()
        assert status == 400
        assert (await response.get_json())["success"] is False
    storage_getter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [profile.delete_group_profile, profile.delete_user_profile])
@pytest.mark.parametrize("body", ['{', 'null'])
async def test_delete_does_not_ignore_invalid_body_when_query_has_id(handler, body, monkeypatch):
    storage_getter = Mock()
    monkeypatch.setattr(profile, "get_profile_storage", storage_getter)
    app = Quart(__name__)
    async with app.test_request_context(
        "/?group_id=g1&user_id=u1", method="POST", data=body,
        headers={"Content-Type": "application/json"},
    ):
        response, status = await handler()
        assert status == 400
        assert (await response.get_json())["success"] is False
    storage_getter.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_persona", [
    ("", "iris"), ("?persona=other", "other"),
])
async def test_group_delete_uses_body_persona(query, expected_persona, monkeypatch):
    storage = Mock(delete_group_profile=AsyncMock(return_value=True))
    monkeypatch.setattr(profile, "get_profile_storage", lambda: (storage, None))
    app = Quart(__name__)
    async with app.test_request_context(
        f"/{query}", method="POST", json={"group_id": "g1", "persona": "iris"}
    ):
        response = await profile.delete_group_profile()
        assert (await response.get_json())["success"] is True
    storage.delete_group_profile.assert_awaited_once_with("g1", expected_persona)


@pytest.mark.asyncio
async def test_group_delete_still_supports_query_only(monkeypatch):
    storage = Mock(delete_group_profile=AsyncMock(return_value=True))
    monkeypatch.setattr(profile, "get_profile_storage", lambda: (storage, None))
    app = Quart(__name__)
    async with app.test_request_context("/?group_id=g1&persona=iris", method="POST"):
        await profile.delete_group_profile()
    storage.delete_group_profile.assert_awaited_once_with("g1", "iris")


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", [
    "update_group_profile", "update_user_profile", "delete_user_profile",
])
async def test_valid_profile_body_retains_routing(handler_name, monkeypatch):
    operation = AsyncMock(return_value=True)
    storage = Mock(**{handler_name: operation})
    monkeypatch.setattr(profile, "get_profile_storage", lambda: (storage, None))
    body = {"user_id": "u1", "group_id": "g1", "persona": "iris"}
    app = Quart(__name__)
    async with app.test_request_context("/", method="POST", json=body):
        response = await getattr(profile, handler_name)()
        assert (await response.get_json())["success"] is True
    if handler_name == "update_group_profile":
        operation.assert_awaited_once_with("g1", body, "iris")
    elif handler_name == "update_user_profile":
        operation.assert_awaited_once_with(
            user_id="u1", group_id="g1", updates=body, persona_id="iris"
        )
    else:
        operation.assert_awaited_once_with("u1", "g1", "iris")
