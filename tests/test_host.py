"""Real target-host loader, Web API auth and Page service in an isolated root.

This does not start AstrBot's platform pipeline or replace final Dashboard browser
qualification. Set IRIS_HOST_CHECKOUT and ASTRBOT_ROOT to explicit test paths.
"""

# Imports follow the required isolated-root guard to prevent host side effects.
# ruff: noqa: E402
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HOST = os.environ.get("IRIS_HOST_CHECKOUT")
if not HOST:
    pytest.skip("Real host source path not configured", allow_module_level=True)
if not os.environ.get("ASTRBOT_ROOT"):
    raise RuntimeError("An isolated ASTRBOT_ROOT is required")
sys.path.insert(0, HOST)
sys.path.insert(0, os.environ["ASTRBOT_ROOT"])
# Bootstrap the isolated namespace before host imports, as main.py does for
# its runtime root. A newly created test directory must be discoverable too.
Path(os.environ["ASTRBOT_ROOT"], "data", "plugins").mkdir(parents=True, exist_ok=True)

import httpx
import jwt
from astrbot.api.star import Context
from astrbot.api.web import PluginRequest, bind_request_context
from astrbot.core import astrbot_config
from astrbot.core.star.star import star_registry
from astrbot.core.star.star_manager import PluginManager
from astrbot.dashboard.api.plugins import router
from astrbot.dashboard.responses import ApiError
from astrbot.dashboard.services.plugin_page_service import PluginPageService
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from iris_memory import PLUGIN_NAME

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
async def loaded():
    root = Path(os.environ["ASTRBOT_ROOT"])
    destination = root / "data/plugins" / PLUGIN_NAME
    source = SOURCE_ROOT
    control_data = root / "data/plugin_data" / PLUGIN_NAME
    if control_data.exists():
        shutil.rmtree(control_data)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "docs", "__pycache__", ".pytest_cache", ".ruff_cache", "tests"
        ),
    )
    importlib.invalidate_caches()
    context = object.__new__(Context)
    context.registered_web_apis = []
    manager = PluginManager(context, astrbot_config)
    success, error = await manager.load(specified_dir_name=PLUGIN_NAME)
    assert success, error
    metadata = next(s for s in star_registry if s.name == PLUGIN_NAME)
    try:
        yield manager, metadata, context
    finally:
        current = next((s for s in star_registry if s.name == PLUGIN_NAME), None)
        if current:
            await manager._terminate_plugin(current)
            await manager._unbind_plugin(current.name, current.module_path)


def request_context(
    username="test-admin", plugin_name=PLUGIN_NAME, body=b"{}", method="GET"
):
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    raw = Request(
        {
            "type": "http",
            "method": method,
            "path": "/test",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "scheme": "http",
            "server": ("localhost", 80),
        },
        receive,
    )
    return bind_request_context(
        PluginRequest(raw, plugin_name=plugin_name, username=username)
    )


async def test_real_loader_reload_old_handlers_and_store_recovery(loaded):
    manager, metadata, context = loaded
    first = metadata.star_cls
    assert first.application.state == "ready"
    assert len(context.registered_web_apis) == 9
    old_handler = context.registered_web_apis[0][1]
    with request_context():
        assert (await old_handler()).status_code == 200
    await first.application.store.intent(0, "plugin.enabled", True)
    for _ in range(3):
        success, error = await manager.reload(PLUGIN_NAME)
        assert success, error
        assert len(context.registered_web_apis) == 9
    assert first.application.state == "closed"
    assert first.application.store.db is None and first.application.transport is None
    with request_context():
        assert (await old_handler()).status_code == 503
        response = await context.registered_web_apis[0][1]()
    assert json.loads(response.body)["settings"]["intents"]["plugin.enabled"]
    current = next(s for s in star_registry if s.name == PLUGIN_NAME)
    assert [
        p.name
        for p in await PluginPageService(
            manager, config=astrbot_config
        ).discover_plugin_pages(current)
    ] == ["iris"]


async def test_real_public_context_authorization_and_closed_inputs(loaded):
    _, metadata, context = loaded
    overview = context.registered_web_apis[0][1]
    for username, plugin, status in [(None, PLUGIN_NAME, 401), ("alice", "other", 403)]:
        with request_context(username, plugin):
            assert (await overview()).status_code == status
    save = next(
        row[1]
        for row in context.registered_web_apis
        if row[0].endswith("/connection/save")
    )
    with request_context(
        body=b'{"expected_revision":0,"origin":"http://localhost:12345","path":"/api/admin"}',
        method="POST",
    ):
        assert (await save()).status_code == 400
    assert (await metadata.star_cls.application.store.settings())["revision"] == 0


async def test_real_host_web_auth_routes_and_native_page_service(loaded):
    manager, metadata, context = loaded
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    signing = "synthetic-isolated-host-signing-key-at-least-32-bytes"
    app.state.jwt_secret = signing
    app.state.core_lifecycle = SimpleNamespace(star_context=context)

    @app.exception_handler(ApiError)
    async def failure(request, error):
        return JSONResponse({"message": str(error)}, status_code=error.status_code)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        endpoint = f"/api/v1/plugins/extensions/{PLUGIN_NAME}/overview"
        assert (await client.get(endpoint)).status_code == 401
        token = jwt.encode(
            {"username": "alice", "exp": time.time() + 60}, signing, algorithm="HS256"
        )
        response = await client.get(
            endpoint, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200 and response.json()["lifecycle"] == "ready"
        expired = jwt.encode(
            {"username": "alice", "exp": time.time() - 60}, signing, algorithm="HS256"
        )
        assert (
            await client.get(endpoint, headers={"Authorization": f"Bearer {expired}"})
        ).status_code == 401
    service = PluginPageService(manager, config=astrbot_config)
    pages = await service.discover_plugin_pages(metadata)
    assert [page.name for page in pages] == ["iris"]
    response = await service.serve_page_content(
        plugin_name=PLUGIN_NAME,
        page_name="iris",
        asset_path="",
        asset_token="",
        jwt_secret=signing,
        username="alice",
        locale="zh-CN",
        theme="dark",
    )
    assert "bridge-sdk.js" in response.content and "asset_token=" in response.content
    assert 'data-theme="dark"' in response.content
    with pytest.raises((ValueError, FileNotFoundError)):
        await service.resolve_plugin_page_file(metadata, "iris", "../main.py")


async def test_partial_route_registration_failure_leaves_inert_handlers(loaded):
    manager, metadata, _ = loaded
    await manager._terminate_plugin(metadata)

    class FailingContext(Context):
        def register_web_api(self, route, handler, methods, description):
            if len(self.registered_web_apis) == 2:
                raise RuntimeError("synthetic registration failure")
            super().register_web_api(route, handler, methods, description)

    context = object.__new__(FailingContext)
    context.registered_web_apis = []
    plugin = metadata.star_cls_type(context)
    with pytest.raises(RuntimeError, match="registration failure"):
        await plugin.initialize()
    assert plugin.application.state == "closed"
    assert plugin.application.store.db is None and plugin.application.transport is None
    with request_context():
        assert (await context.registered_web_apis[0][1]()).status_code == 503
    await plugin.terminate()
