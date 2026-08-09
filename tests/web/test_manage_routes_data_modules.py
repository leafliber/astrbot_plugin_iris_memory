"""数据管理页新增模块清理接口测试。"""

from types import SimpleNamespace

import pytest
from quart import Quart

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.learning.storage import LearningStorage
from iris_memory.persona_evolution import PersonaEvolutionStorage
from iris_memory.web.routes import manage_routes
from tests.persona_evolution.conftest import make_job, seed_samples


class FakeWebContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakeManager:
    def __init__(self, components):
        self.components = components

    def get_component(self, name, type_=None):
        return self.components.get(name)


class FakeComponent:
    def __init__(self, storage):
        self.storage = storage
        self.is_available = True


@pytest.fixture
def manage_env(tmp_path, monkeypatch):
    init_config({}, tmp_path)
    learning = LearningStorage(tmp_path / "learning.db")
    learning.init_schema()
    evolution = PersonaEvolutionStorage(tmp_path / "evolution.db")
    evolution.init_schema()
    manager = FakeManager(
        {
            "learning": FakeComponent(learning),
            "persona_evolution": FakeComponent(evolution),
        }
    )
    monkeypatch.setattr(manage_routes, "get_component_manager", lambda: manager)

    context = FakeWebContext()
    manage_routes.register_manage_routes(context)
    app = Quart("test_manage_data_modules")
    for index, (route, handler, methods, _desc) in enumerate(context.routes):
        app.add_url_rule(route, f"manage_{index}", handler, methods=methods)

    yield SimpleNamespace(app=app, learning=learning, evolution=evolution)
    learning.close()
    evolution.close()
    reset_config()


@pytest.mark.asyncio
async def test_delete_learning_data(manage_env):
    manage_env.learning.insert_pair("g", "u", "问", "答")
    manage_env.learning.insert_jargon("g", "yyds", "永远的神", 0.9)

    response = await manage_env.app.test_client().post(
        "/astrbot_plugin_iris_memory/manage/learning/delete"
    )
    data = await response.get_json()

    assert data["success"] is True
    assert data["deleted_count"] == 2
    assert manage_env.learning.get_stats()["few_shot"]["total"] == 0


@pytest.mark.asyncio
async def test_delete_persona_evolution_data(manage_env):
    make_job(manage_env.evolution, "p1")
    seed_samples(manage_env.evolution, 2)

    response = await manage_env.app.test_client().post(
        "/astrbot_plugin_iris_memory/manage/persona-evolution/delete"
    )
    data = await response.get_json()

    assert data["success"] is True
    assert data["deleted_count"] == 3
    assert manage_env.evolution.list_jobs() == []
    assert manage_env.evolution.count_samples() == 0
