"""全量备份 1.1 测试：persona_evolution section 导出/导入与旧版 1.0 兼容（§19）"""

import json
from types import SimpleNamespace

import pytest
from quart import Quart

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.persona_evolution import PersonaEvolutionStorage
from iris_memory.web.routes import data_routes
from tests.persona_evolution.conftest import make_job, seed_samples

DATA_PREFIX = "/astrbot_plugin_iris_memory/data"


class FakeWebContext:
    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakeManager:
    """按名字分发组件（data_routes 用 get_component(name, Type) 探测）"""

    def __init__(self, components):
        self._components = components

    def get_component(self, name, type_=None):
        return self._components.get(name)


class FakeComponent:
    def __init__(self, storage, available=True):
        self.storage = storage
        self.is_available = available


class BrokenStorage:
    """export_all 抛错的存储（验证导出降级不阻断）"""

    def export_all(self, include_samples: bool = False):
        raise RuntimeError("模拟导出故障")


@pytest.fixture
def backup_env(tmp_path, monkeypatch):
    """装配 data 路由 + 只含 persona_evolution 的组件管理器"""
    init_config({"persona_evolution": {"enable": True}}, tmp_path)
    storage = PersonaEvolutionStorage(tmp_path / "pe.db")
    storage.init_schema()
    component = FakeComponent(storage)

    def set_components(components):
        monkeypatch.setattr(
            data_routes, "get_component_manager", lambda: FakeManager(components)
        )

    set_components({"persona_evolution": component})

    web_context = FakeWebContext()
    data_routes.register_data_routes(web_context)
    app = Quart("test_backup")
    for i, (route, handler, methods, desc) in enumerate(web_context.routes):
        app.add_url_rule(route, f"data_{i}", handler, methods=methods)

    yield SimpleNamespace(
        app=app, storage=storage, component=component, set_components=set_components
    )
    reset_config()


class TestFullBackupV11:
    @pytest.mark.asyncio
    async def test_export_contains_persona_evolution_section(self, backup_env):
        make_job(backup_env.storage, "p1")
        seed_samples(backup_env.storage, 3)

        resp = await backup_env.app.test_client().get(f"{DATA_PREFIX}/all/export")
        data = json.loads(await resp.get_data(as_text=True))

        assert data["version"] == "1.1"
        section = data["persona_evolution"]
        assert section is not None
        assert section["version"] == "1.1"
        assert len(section["jobs"]) == 1
        # 全量备份不含语料原文（独立导出默认同样不含）
        assert section["samples"] == []
        # 其他组件未注册时保持 None
        assert data["l2_memory"] is None
        assert data["profiles"] is None

    @pytest.mark.asyncio
    async def test_export_disabled_component_section_none(self, backup_env):
        """功能未启用时 section 为 None 不报错"""
        backup_env.set_components(
            {"persona_evolution": FakeComponent(backup_env.storage, available=False)}
        )

        resp = await backup_env.app.test_client().get(f"{DATA_PREFIX}/all/export")
        data = json.loads(await resp.get_data(as_text=True))

        assert data["version"] == "1.1"
        assert data["persona_evolution"] is None

    @pytest.mark.asyncio
    async def test_export_error_degrades_not_blocks(self, backup_env):
        """persona_evolution 导出异常降级为 {"error": ...}，不阻断整体导出"""
        backup_env.set_components(
            {"persona_evolution": FakeComponent(BrokenStorage())}
        )

        resp = await backup_env.app.test_client().get(f"{DATA_PREFIX}/all/export")

        assert resp.status_code == 200
        data = json.loads(await resp.get_data(as_text=True))
        assert data["version"] == "1.1"
        assert "error" in data["persona_evolution"]

    @pytest.mark.asyncio
    async def test_import_legacy_v10_skips_section(self, backup_env):
        """导入旧版 1.0 备份（无 persona_evolution 字段）正常跳过"""
        legacy = {
            "version": "1.0",
            "export_time": "2025-01-01T00:00:00",
            "l2_memory": None,
            "l3_kg": None,
            "profiles": None,
        }

        resp = await backup_env.app.test_client().post(
            f"{DATA_PREFIX}/all/import", json={"data": legacy}
        )
        data = await resp.get_json()

        assert data["success"] is True, data
        assert data["result"]["persona_evolution"] is None

    @pytest.mark.asyncio
    async def test_import_v11_dispatches_section(self, backup_env):
        """备份 1.1 导入分发 persona_evolution section（不改 AstrBot Persona）"""
        make_job(backup_env.storage, "p1")
        exported = backup_env.storage.export_all()
        payload = {
            "version": "1.1",
            "persona_evolution": exported,
        }

        resp = await backup_env.app.test_client().post(
            f"{DATA_PREFIX}/all/import", json={"data": payload}
        )
        data = await resp.get_json()

        assert data["success"] is True, data
        # 同库存重复导入 → 跳过而非报错
        assert data["result"]["persona_evolution"]["skipped_jobs"] == 1

    @pytest.mark.asyncio
    async def test_import_section_component_unavailable(self, backup_env):
        """导入含 persona_evolution 但功能未启用：section 报不可用，不阻断"""
        backup_env.set_components(
            {"persona_evolution": FakeComponent(backup_env.storage, available=False)}
        )
        payload = {"version": "1.1", "persona_evolution": {"jobs": []}}

        resp = await backup_env.app.test_client().post(
            f"{DATA_PREFIX}/all/import", json={"data": payload}
        )
        data = await resp.get_json()

        assert data["success"] is True
        assert "error" in data["result"]["persona_evolution"]
