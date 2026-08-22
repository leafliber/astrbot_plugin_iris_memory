"""L2 归档 Web 路由与 CLI restore 指令测试"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from quart import Quart

from iris_memory.commands.base import ParsedArgs
from iris_memory.commands.l2_handler import L2CommandHandler
from iris_memory.web.routes import memory as memory_routes

PREFIX = "/astrbot_plugin_iris_memory/memory"


class FakeWebContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakeManager:
    def __init__(self, component):
        self.component = component

    def get_component(self, name, type_=None):
        if name == "l2_memory":
            return self.component
        return None


class FakeArchiveComponent:
    def __init__(self):
        self.is_available = True
        self.list_archived_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem_arch1",
                    "content": "被归档的记忆",
                    "metadata": {},
                    "group_id": "g1",
                    "user_id": "u1",
                    "timestamp": "2026-01-01T00:00:00",
                    "persona_id": "default",
                    "archived_at": "2026-01-02T00:00:00",
                    "archive_reason": "dream_eviction",
                    "has_vector": True,
                }
            ]
        )
        self.get_archived_count = AsyncMock(return_value=1)
        self.restore_archived_memory = AsyncMock(return_value=True)
        self.delete_archived_memory = AsyncMock(return_value=True)


@pytest.fixture
def archive_env(monkeypatch):
    component = FakeArchiveComponent()
    manager = FakeManager(component)
    monkeypatch.setattr(memory_routes, "get_component_manager", lambda: manager)

    context = FakeWebContext()
    memory_routes.register_memory_routes(context)
    app = Quart("test_l2_archive_routes")
    for index, (route, handler, methods, _desc) in enumerate(context.routes):
        app.add_url_rule(route, f"memory_{index}", handler, methods=methods)

    yield SimpleNamespace(app=app, component=component, routes=context.routes)


def test_archive_routes_registered(archive_env):
    paths = {route for route, *_ in archive_env.routes}
    assert f"{PREFIX}/l2/archive/list" in paths
    assert f"{PREFIX}/l2/archive/restore" in paths
    assert f"{PREFIX}/l2/archive/delete" in paths


@pytest.mark.asyncio
async def test_archive_list(archive_env):
    response = await archive_env.app.test_client().get(f"{PREFIX}/l2/archive/list")
    data = await response.get_json()

    assert data["success"] is True
    assert data["total_count"] == 1
    assert data["results"][0]["id"] == "mem_arch1"
    assert data["results"][0]["archive_reason"] == "dream_eviction"


@pytest.mark.asyncio
async def test_archive_restore_success(archive_env):
    response = await archive_env.app.test_client().post(
        f"{PREFIX}/l2/archive/restore", json={"memory_id": "mem_arch1"}
    )
    data = await response.get_json()

    assert data["success"] is True
    archive_env.component.restore_archived_memory.assert_called_once_with("mem_arch1")


@pytest.mark.asyncio
async def test_archive_restore_conflict(archive_env):
    archive_env.component.restore_archived_memory = AsyncMock(return_value=False)
    response = await archive_env.app.test_client().post(
        f"{PREFIX}/l2/archive/restore", json={"memory_id": "mem_missing"}
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_archive_restore_requires_id(archive_env):
    response = await archive_env.app.test_client().post(
        f"{PREFIX}/l2/archive/restore", json={}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_archive_delete_success(archive_env):
    response = await archive_env.app.test_client().post(
        f"{PREFIX}/l2/archive/delete", json={"memory_id": "mem_arch1"}
    )
    data = await response.get_json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_archive_delete_not_found(archive_env):
    archive_env.component.delete_archived_memory = AsyncMock(return_value=False)
    response = await archive_env.app.test_client().post(
        f"{PREFIX}/l2/archive/delete", json={"memory_id": "mem_missing"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cli_restore_success(monkeypatch):
    component = FakeArchiveComponent()
    monkeypatch.setattr(
        "iris_memory.commands.l2_handler.get_component_manager",
        lambda: FakeManager(component),
    )

    handler = L2CommandHandler()
    args = ParsedArgs(raw_args=["restore", "mem_arch1"])
    result = await handler.handle(Mock(), args, "restore")

    assert result.success is True
    assert "mem_arch1" in result.message
    component.restore_archived_memory.assert_called_once_with("mem_arch1")


@pytest.mark.asyncio
async def test_cli_restore_missing_id(monkeypatch):
    component = FakeArchiveComponent()
    monkeypatch.setattr(
        "iris_memory.commands.l2_handler.get_component_manager",
        lambda: FakeManager(component),
    )

    handler = L2CommandHandler()
    result = await handler.handle(Mock(), ParsedArgs(raw_args=["restore"]), "restore")

    assert result.success is False
    assert "记忆 ID" in result.message


@pytest.mark.asyncio
async def test_cli_restore_failure(monkeypatch):
    component = FakeArchiveComponent()
    component.restore_archived_memory = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "iris_memory.commands.l2_handler.get_component_manager",
        lambda: FakeManager(component),
    )

    handler = L2CommandHandler()
    args = ParsedArgs(raw_args=["restore", "mem_missing"])
    result = await handler.handle(Mock(), args, "restore")

    assert result.success is False
    assert "恢复失败" in result.message
