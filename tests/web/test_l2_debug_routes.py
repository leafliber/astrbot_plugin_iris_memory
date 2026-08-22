"""L2 召回调试与 FTS 运维路由测试"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from quart import Quart

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


class FakeL2Component:
    def __init__(self):
        self.is_available = True
        self.retrieve_debug = AsyncMock(
            return_value={
                "query": "苹果",
                "group_id": None,
                "top_k": 10,
                "persona_id": "default",
                "relevance_threshold": 0.3,
                "rrf_k": 60,
                "vector": [{"id": "mem_1", "content": "喜欢苹果", "score": 0.8, "group_id": None}],
                "keyword": [{"id": "mem_2", "content": "苹果好吃", "score": 1.2, "group_id": None}],
                "fused": [{"id": "mem_1", "content": "喜欢苹果", "score": 0.03, "group_id": None}],
                "fts": {"available": True, "memory_rows": 2},
            }
        )
        self.get_fts_status = lambda: {"available": True, "memory_rows": 2}
        self.rebuild_fts_index = AsyncMock(return_value=True)
        self.update_content = AsyncMock(return_value=True)
        self.set_memory_scope = AsyncMock(return_value=True)


@pytest.fixture
def debug_env(monkeypatch):
    component = FakeL2Component()
    manager = FakeManager(component)
    monkeypatch.setattr(memory_routes, "get_component_manager", lambda: manager)

    context = FakeWebContext()
    memory_routes.register_memory_routes(context)
    app = Quart("test_l2_debug_routes")
    for index, (route, handler, methods, _desc) in enumerate(context.routes):
        app.add_url_rule(route, f"memory_{index}", handler, methods=methods)

    yield SimpleNamespace(app=app, component=component, routes=context.routes)


def test_debug_routes_registered(debug_env):
    paths = {route for route, *_ in debug_env.routes}
    assert f"{PREFIX}/l2/retrieval-debug" in paths
    assert f"{PREFIX}/l2/fts/status" in paths
    assert f"{PREFIX}/l2/fts/rebuild" in paths


@pytest.mark.asyncio
async def test_retrieval_debug_returns_lanes(debug_env):
    response = await debug_env.app.test_client().post(
        f"{PREFIX}/l2/retrieval-debug", json={"query": "苹果"}
    )
    data = await response.get_json()

    assert data["success"] is True
    assert len(data["vector"]) == 1
    assert len(data["keyword"]) == 1
    assert len(data["fused"]) == 1
    assert data["rrf_k"] == 60
    debug_env.component.retrieve_debug.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_debug_empty_query_rejected(debug_env):
    response = await debug_env.app.test_client().post(
        f"{PREFIX}/l2/retrieval-debug", json={"query": ""}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_fts_status_and_rebuild(debug_env):
    response = await debug_env.app.test_client().get(f"{PREFIX}/l2/fts/status")
    data = await response.get_json()
    assert data["success"] is True and data["available"] is True

    response = await debug_env.app.test_client().post(f"{PREFIX}/l2/fts/rebuild")
    data = await response.get_json()
    assert data["success"] is True
    debug_env.component.rebuild_fts_index.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_debug_unavailable_component(monkeypatch):
    monkeypatch.setattr(memory_routes, "get_component_manager", lambda: FakeManager(None))
    response_handler = memory_routes.debug_l2_retrieval

    from quart import Quart

    app = Quart("test_l2_debug_unavailable")
    app.add_url_rule(f"{PREFIX}/l2/retrieval-debug", "dbg", response_handler, methods=["POST"])
    response = await app.test_client().post(
        f"{PREFIX}/l2/retrieval-debug", json={"query": "苹果"}
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_update_l2_entry_with_scope(debug_env):
    response = await debug_env.app.test_client().post(
        f"{PREFIX}/l2/update",
        json={"id": "mem_1", "content": "新内容", "scope": "global"},
    )
    data = await response.get_json()

    assert data["success"] is True
    debug_env.component.update_content.assert_called_once_with("mem_1", "新内容")
    debug_env.component.set_memory_scope.assert_called_once_with("mem_1", "global")


@pytest.mark.asyncio
async def test_update_l2_entry_without_scope_skips_scope_call(debug_env):
    response = await debug_env.app.test_client().post(
        f"{PREFIX}/l2/update", json={"id": "mem_1", "content": "新内容"}
    )
    data = await response.get_json()

    assert data["success"] is True
    debug_env.component.set_memory_scope.assert_not_called()


@pytest.mark.asyncio
async def test_update_l2_entry_invalid_scope_ignored(debug_env):
    response = await debug_env.app.test_client().post(
        f"{PREFIX}/l2/update",
        json={"id": "mem_1", "content": "新内容", "scope": "bogus"},
    )
    data = await response.get_json()

    assert data["success"] is True
    debug_env.component.set_memory_scope.assert_not_called()
