"""Web 人格自迭代路由测试：端点参数校验、错误码、jobs CRUD、语料统计不含原文

范本 tests/web/test_learning_routes.py；通过 Quart test_client +
FakeWebContext（捕获 register_web_api）+ FakeManager（替换组件管理器）
直接驱动全部端点。
"""

import json
from types import SimpleNamespace

import pytest
from quart import Quart

from iris_memory.config import init_config
from iris_memory.config.config import reset_config
from iris_memory.persona_evolution import PersonaEvolutionStorage
from iris_memory.persona_evolution.service import PersonaEvolutionService
from iris_memory.web.routes import persona_evolution as pe_routes
from tests.persona_evolution.conftest import make_job, seed_samples
from tests.persona_evolution.fakes import (
    ANALYSIS_MODULE,
    GENERATION_MODULE,
    REVIEW_MODULE,
    FakeContext,
    FakeLLMManager,
    FakePersonaManager,
    good_analysis_json,
    good_generation_json,
    good_review_json,
)

PREFIX = "/astrbot_plugin_iris_memory/persona-evolution"

BASE = "你是 Iris，一个群聊助手。"
CANDIDATE = BASE + "\n\n<!-- IRIS_EVOLUTION:BEGIN v1 -->\n旧风格\n<!-- IRIS_EVOLUTION:END -->\n"
# 受控模式首轮自动补标记：有效候选 = BASE + 标记区块（改 inner）
NEW_INNER = "新风格：短句为主"


class FakeWebContext:
    """捕获 register_web_api 调用的 AstrBot Context fake"""

    def __init__(self):
        self.routes = []

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


class FakeManager:
    """组件管理器 fake：只认识 persona_evolution"""

    def __init__(self, component):
        self._component = component

    def get_component(self, name, type_=None):
        if name == "persona_evolution":
            return self._component
        return None


class FakeComponent:
    def __init__(self, storage, service, context, available=True):
        self.storage = storage
        self.service = service
        self.context = context
        self.is_available = available


def _full_candidate(base: str, inner: str = NEW_INNER) -> str:
    return (
        f"{base}\n\n<!-- IRIS_EVOLUTION:BEGIN v1 -->\n{inner}\n"
        "<!-- IRIS_EVOLUTION:END -->\n"
    )


@pytest.fixture
def web_env(tmp_path, monkeypatch):
    """装配：真实 storage/service + fake PersonaManager/LLM + Quart 测试应用"""
    init_config({"persona_evolution": {"enable": True}}, tmp_path)
    storage = PersonaEvolutionStorage(tmp_path / "pe.db")
    storage.init_schema()
    pm = FakePersonaManager()
    llm = FakeLLMManager()
    context = FakeContext(pm)
    service = PersonaEvolutionService(storage, context, llm_manager=llm)
    component = FakeComponent(storage, service, context)
    monkeypatch.setattr(
        pe_routes, "get_component_manager", lambda: FakeManager(component)
    )

    web_context = FakeWebContext()
    pe_routes.register_persona_evolution_routes(web_context)
    app = Quart("test_pe")
    for i, (route, handler, methods, desc) in enumerate(web_context.routes):
        app.add_url_rule(route, f"pe_{i}", handler, methods=methods)

    yield SimpleNamespace(
        app=app, storage=storage, pm=pm, llm=llm, service=service, component=component
    )
    reset_config()


def _setup_llm(llm, candidate):
    llm.set_default(ANALYSIS_MODULE, good_analysis_json())
    llm.set_default(GENERATION_MODULE, good_generation_json(candidate))
    llm.set_default(REVIEW_MODULE, good_review_json())


class TestPersonas:
    @pytest.mark.asyncio
    async def test_list_goals(self, web_env):
        resp = await web_env.app.test_client().get(f"{PREFIX}/goals")
        data = await resp.get_json()

        assert resp.status_code == 200
        assert data["success"] is True
        by_id = {goal["preset_id"]: goal for goal in data["goals"]}
        assert by_id["natural"]["display_name"] == "自然拟人"
        assert by_id["custom"]["text"] == ""

    @pytest.mark.asyncio
    async def test_list_personas_marks_default(self, web_env):
        web_env.pm.add_persona("default", "默认人格")
        web_env.pm.add_persona("p1", "人格一")
        make_job(web_env.storage, "p1")

        resp = await web_env.app.test_client().get(f"{PREFIX}/personas")
        data = await resp.get_json()

        assert resp.status_code == 200
        assert data["success"] is True
        assert data["degraded"] is False
        by_id = {p["persona_id"]: p for p in data["personas"]}
        assert by_id["default"]["is_default"] is True
        assert by_id["default"]["iterable"] is False
        assert by_id["p1"]["iterable"] is True
        assert by_id["p1"]["has_job"] is True

    @pytest.mark.asyncio
    async def test_list_personas_degraded_without_manager(self, web_env):
        web_env.context_persona_backup = web_env.component.context.persona_manager
        web_env.component.context.persona_manager = None

        resp = await web_env.app.test_client().get(f"{PREFIX}/personas")
        data = await resp.get_json()

        assert data["success"] is True
        assert data["personas"] == []
        assert data["degraded"] is True

    @pytest.mark.asyncio
    async def test_clone_default(self, web_env):
        web_env.pm.add_persona("default", "默认人格内容")

        resp = await web_env.app.test_client().post(
            f"{PREFIX}/personas/clone-default", json={"persona_id": "p2"}
        )
        data = await resp.get_json()

        assert data["success"] is True, data
        assert web_env.pm.get_prompt("p2") == "默认人格内容"
        # 只创建，不修改 default
        assert web_env.pm.get_prompt("default") == "默认人格内容"

    @pytest.mark.asyncio
    async def test_clone_default_validation(self, web_env):
        web_env.pm.add_persona("default", "默认人格内容")
        client = web_env.app.test_client()

        # 缺少 persona_id
        resp = await client.post(f"{PREFIX}/personas/clone-default", json={})
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "invalid_params"
        # persona_id 不能为 default
        resp = await client.post(
            f"{PREFIX}/personas/clone-default", json={"persona_id": "default"}
        )
        assert resp.status_code == 400
        # 重复 id（create_persona 抛错 → 400）
        resp = await client.post(
            f"{PREFIX}/personas/clone-default", json={"persona_id": "p2"}
        )
        assert (await resp.get_json())["success"] is True
        resp = await client.post(
            f"{PREFIX}/personas/clone-default", json={"persona_id": "p2"}
        )
        assert resp.status_code == 400


class TestJobsCrud:
    @pytest.mark.asyncio
    async def test_create_job_validation(self, web_env):
        web_env.pm.add_persona("p1", BASE)
        client = web_env.app.test_client()

        # 缺 persona_id
        resp = await client.post(f"{PREFIX}/jobs", json={})
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "invalid_params"
        # default 不可直接迭代
        resp = await client.post(f"{PREFIX}/jobs", json={"persona_id": "default"})
        assert resp.status_code == 400
        # 非法枚举
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "edit_mode": "bogus"}
        )
        assert resp.status_code == 400
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "approval_mode": "bogus"}
        )
        assert resp.status_code == 400
        # 非法目标预设
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "goal_preset_id": "bogus"}
        )
        assert resp.status_code == 400
        # custom 预设缺 custom_goal
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "goal_preset_id": "custom"}
        )
        assert resp.status_code == 400
        # 范围参数类型错误
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "source_group_ids": "g1"}
        )
        assert resp.status_code == 400
        # 数值越界
        resp = await client.post(
            f"{PREFIX}/jobs", json={"persona_id": "p1", "trigger_sample_count": 0}
        )
        assert resp.status_code == 400
        # Persona 不存在
        resp = await client.post(f"{PREFIX}/jobs", json={"persona_id": "ghost"})
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "persona_not_found"

    @pytest.mark.asyncio
    async def test_create_get_update_job(self, web_env):
        web_env.pm.add_persona("p1", BASE)
        seed_samples(web_env.storage, 5)
        client = web_env.app.test_client()

        resp = await client.post(
            f"{PREFIX}/jobs",
            json={
                "persona_id": "p1",
                "name": "测试任务",
                "edit_mode": "managed_block",
                "approval_mode": "manual",
                "trigger_sample_count": 50,
                "source_group_ids": ["g1"],
            },
        )
        data = await resp.get_json()
        assert data["success"] is True, data
        job = data["job"]
        job_id = job["id"]
        assert job["approval_mode"] == "manual"
        # 创建基线 = 创建时最新 Sample ID
        assert job["last_sample_cursor"] == web_env.storage.get_latest_sample_id()

        # 重复创建 → 409 job_already_exists
        resp = await client.post(f"{PREFIX}/jobs", json={"persona_id": "p1"})
        assert resp.status_code == 409
        assert (await resp.get_json())["error_code"] == "job_already_exists"

        # 列表
        resp = await client.get(f"{PREFIX}/jobs")
        jobs = (await resp.get_json())["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["sample_total"] == 5

        # 详情
        resp = await client.get(f"{PREFIX}/jobs/{job_id}")
        data = await resp.get_json()
        assert data["job"]["name"] == "测试任务"
        assert data["runs"] == [] and data["revisions"] == []

        # 更新
        resp = await client.put(
            f"{PREFIX}/jobs/{job_id}",
            json={"name": "改名", "approval_mode": "auto", "min_interval_hours": 12},
        )
        data = await resp.get_json()
        assert data["success"] is True, data
        assert data["job"]["name"] == "改名"
        assert data["job"]["approval_mode"] == "auto"
        assert data["job"]["min_interval_hours"] == 12

        # 插件页桥接层使用 POST 更新别名，与 PUT 契约一致。
        resp = await client.post(
            f"{PREFIX}/jobs/{job_id}/update",
            json={"name": "桥接更新"},
        )
        data = await resp.get_json()
        assert data["success"] is True, data
        assert data["job"]["name"] == "桥接更新"

        # 不允许的字段
        resp = await client.put(f"{PREFIX}/jobs/{job_id}", json={"persona_id": "x"})
        assert resp.status_code == 400
        resp = await client.put(f"{PREFIX}/jobs/{job_id}", json={"status": "paused"})
        assert resp.status_code == 400
        # 空更新
        resp = await client.put(f"{PREFIX}/jobs/{job_id}", json={})
        assert resp.status_code == 400

        # 不存在 → 404
        resp = await client.get(f"{PREFIX}/jobs/9999")
        assert resp.status_code == 404
        assert (await resp.get_json())["error_code"] == "not_found"
        resp = await client.put(f"{PREFIX}/jobs/9999", json={"name": "x"})
        assert resp.status_code == 404


class TestJobActions:
    @pytest.mark.asyncio
    async def test_pause_resume(self, web_env):
        job_id = make_job(web_env.storage, "p1")
        client = web_env.app.test_client()

        resp = await client.post(f"{PREFIX}/jobs/{job_id}/pause")
        assert (await resp.get_json())["success"] is True
        assert web_env.storage.get_job(job_id).status == "paused"

        # 重复暂停 → 400 invalid_params
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/pause")
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "invalid_params"

        resp = await client.post(f"{PREFIX}/jobs/{job_id}/resume")
        assert (await resp.get_json())["success"] is True
        assert web_env.storage.get_job(job_id).status == "active"

        # 不存在 → 404
        resp = await client.post(f"{PREFIX}/jobs/9999/pause")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_insufficient_samples(self, web_env):
        web_env.pm.add_persona("p1", BASE)
        job_id = make_job(web_env.storage, "p1")
        client = web_env.app.test_client()

        resp = await client.post(f"{PREFIX}/jobs/{job_id}/run")

        assert resp.status_code == 400
        data = await resp.get_json()
        assert data["success"] is False
        assert data["error_code"] == "insufficient_samples"


class TestRevisionEndpoints:
    async def _make_candidate(self, web_env, client):
        """手动审批 Job 跑一轮停在 candidate，返回 (job_id, revision_id)"""
        web_env.pm.add_persona("p1", BASE)
        seed_samples(web_env.storage, 25)
        job_id = make_job(web_env.storage, "p1", approval_mode="manual")
        _setup_llm(web_env.llm, _full_candidate(BASE))
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/run")
        data = await resp.get_json()
        assert data["success"] is True, data
        return job_id, data["revision_id"]

    @pytest.mark.asyncio
    async def test_list_and_get_revision(self, web_env):
        client = web_env.app.test_client()
        job_id, rev_id = await self._make_candidate(web_env, client)

        resp = await client.get(f"{PREFIX}/jobs/{job_id}/revisions")
        data = await resp.get_json()
        assert data["success"] is True
        assert len(data["revisions"]) == 1
        revision = data["revisions"][0]
        assert revision["status"] == "candidate"
        assert revision["result_prompt"]  # 时间线含完整快照供 Diff

        # 状态过滤
        resp = await client.get(f"{PREFIX}/jobs/{job_id}/revisions?status=applied")
        assert (await resp.get_json())["revisions"] == []
        # 非法状态过滤 → 400
        resp = await client.get(f"{PREFIX}/jobs/{job_id}/revisions?status=bogus")
        assert resp.status_code == 400
        # 非法 limit → 400
        resp = await client.get(f"{PREFIX}/jobs/{job_id}/revisions?limit=0")
        assert resp.status_code == 400
        # Job 不存在 → 404
        resp = await client.get(f"{PREFIX}/jobs/9999/revisions")
        assert resp.status_code == 404

        # 单个 Revision
        resp = await client.get(f"{PREFIX}/revisions/{rev_id}")
        data = await resp.get_json()
        assert data["revision"]["id"] == rev_id
        assert data["revision"]["validation"]["passed"] is True
        resp = await client.get(f"{PREFIX}/revisions/9999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_and_reject(self, web_env):
        client = web_env.app.test_client()
        job_id, rev_id = await self._make_candidate(web_env, client)

        # 批准 → 发布
        resp = await client.post(f"{PREFIX}/revisions/{rev_id}/approve")
        data = await resp.get_json()
        assert data["success"] is True, data
        assert web_env.pm.get_prompt("p1") == _full_candidate(BASE)
        # 已发布不能再批准
        resp = await client.post(f"{PREFIX}/revisions/{rev_id}/approve")
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "invalid_params"

        # 再生成一个 candidate 用于拒绝
        rev2 = (await self._make_candidate_round2(web_env, client, job_id))
        resp = await client.post(
            f"{PREFIX}/revisions/{rev2}/reject", json={"reason": "改动太激进"}
        )
        assert (await resp.get_json())["success"] is True
        revision = web_env.storage.get_revision(rev2)
        assert revision.status == "rejected"
        assert revision.decision_reason == "改动太激进"

    async def _make_candidate_round2(self, web_env, client, job_id):
        current = web_env.pm.get_prompt("p1")
        web_env.llm.push(
            GENERATION_MODULE,
            good_generation_json(current.replace(NEW_INNER, "再改")),
        )
        web_env.storage.update_job(job_id, {"approval_mode": "manual"})
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/run")
        data = await resp.get_json()
        assert data["success"] is True, data
        return data["revision_id"]

    @pytest.mark.asyncio
    async def test_approve_conflict_returns_409(self, web_env):
        client = web_env.app.test_client()
        job_id, rev_id = await self._make_candidate(web_env, client)
        web_env.pm.external_edit("p1", "外部编辑")

        resp = await client.post(f"{PREFIX}/revisions/{rev_id}/approve")

        assert resp.status_code == 409
        data = await resp.get_json()
        assert data["error_code"] == "external_change"
        assert web_env.storage.get_job(job_id).status == "conflict"

    @pytest.mark.asyncio
    async def test_rollback_endpoint(self, web_env):
        web_env.pm.add_persona("p1", BASE)
        seed_samples(web_env.storage, 100)
        job_id = make_job(web_env.storage, "p1")
        _setup_llm(web_env.llm, _full_candidate(BASE))
        client = web_env.app.test_client()

        r1 = await (await client.post(f"{PREFIX}/jobs/{job_id}/run")).get_json()
        assert r1["success"] is True, r1
        v1 = r1["revision_id"]
        web_env.llm.push(
            GENERATION_MODULE,
            good_generation_json(
                web_env.pm.get_prompt("p1").replace(NEW_INNER, "二版")
            ),
        )
        r2 = await (await client.post(f"{PREFIX}/jobs/{job_id}/run")).get_json()
        assert r2["success"] is True, r2

        resp = await client.post(f"{PREFIX}/revisions/{v1}/rollback")
        data = await resp.get_json()
        assert data["success"] is True, data
        assert web_env.pm.get_prompt("p1") == _full_candidate(BASE)
        new_rev = web_env.storage.get_revision(data["revision_id"])
        assert new_rev.status == "rollback"
        assert new_rev.trigger_type == "rollback"

    @pytest.mark.asyncio
    async def test_adopt_current_endpoint(self, web_env):
        web_env.pm.add_persona("p1", BASE)
        seed_samples(web_env.storage, 100)
        job_id = make_job(web_env.storage, "p1")
        _setup_llm(web_env.llm, _full_candidate(BASE))
        client = web_env.app.test_client()

        r1 = await (await client.post(f"{PREFIX}/jobs/{job_id}/run")).get_json()
        assert r1["success"] is True, r1

        # 非冲突状态 → 400
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/conflict/adopt-current")
        assert resp.status_code == 400
        assert (await resp.get_json())["error_code"] == "invalid_params"

        # 制造冲突
        web_env.pm.external_edit("p1", "外部版本")
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/run")
        assert resp.status_code == 409
        assert (await resp.get_json())["error_code"] == "external_change"

        # 采纳外部版本为新基线
        resp = await client.post(f"{PREFIX}/jobs/{job_id}/conflict/adopt-current")
        data = await resp.get_json()
        assert data["success"] is True, data
        assert web_env.storage.get_job(job_id).status == "active"


class TestSamplesEndpoints:
    @pytest.mark.asyncio
    async def test_stats_contains_no_raw_text(self, web_env):
        seed_samples(web_env.storage, 5, text_prefix="独特语料文本xyz")
        seed_samples(web_env.storage, 3, group_id="g2", user_id="u2", start=100)

        resp = await web_env.app.test_client().get(f"{PREFIX}/samples/stats")
        data = await resp.get_json()
        body = await resp.get_data(as_text=True)

        assert data["success"] is True
        stats = data["stats"]
        assert stats["total"] == 8
        assert len(stats["by_group"]) == 2
        assert stats["by_day"]
        # 不返回任何原文
        assert "独特语料文本xyz" not in body
        assert "normalized_text" not in body

    @pytest.mark.asyncio
    async def test_clear_samples(self, web_env):
        seed_samples(web_env.storage, 5, group_id="g1")
        seed_samples(web_env.storage, 3, group_id="g2", start=100)
        client = web_env.app.test_client()

        # 按群清除
        resp = await client.post(f"{PREFIX}/samples/clear", json={"group_id": "g1"})
        data = await resp.get_json()
        assert data["deleted"] == 5
        assert web_env.storage.count_samples() == 3

        # 全部清除
        resp = await client.post(f"{PREFIX}/samples/clear", json={})
        assert (await resp.get_json())["deleted"] == 3
        assert web_env.storage.count_samples() == 0


class TestExportImportEndpoints:
    @pytest.mark.asyncio
    async def test_export_default_excludes_samples(self, web_env):
        make_job(web_env.storage, "p1")
        seed_samples(web_env.storage, 5)
        client = web_env.app.test_client()

        resp = await client.get(f"{PREFIX}/export")
        assert resp.status_code == 200
        data = json.loads(await resp.get_data(as_text=True))
        assert data["version"] == "1.1"
        assert len(data["jobs"]) == 1
        assert data["samples"] == []

        # include_samples=true 含脱敏语料
        resp = await client.get(f"{PREFIX}/export?include_samples=true")
        data = json.loads(await resp.get_data(as_text=True))
        assert len(data["samples"]) == 5

    @pytest.mark.asyncio
    async def test_import_endpoint(self, web_env):
        make_job(web_env.storage, "p1")
        seed_samples(web_env.storage, 3)
        exported = web_env.storage.export_all(include_samples=True)
        client = web_env.app.test_client()

        # 缺 data → 400
        resp = await client.post(f"{PREFIX}/import", json={})
        assert resp.status_code == 400

        # 重复导入（同库存）→ 跳过
        resp = await client.post(
            f"{PREFIX}/import", json={"data": exported, "skip_duplicates": True}
        )
        data = await resp.get_json()
        assert data["success"] is True, data
        assert data["stats"]["skipped_jobs"] == 1
        # 语料按 dedupe_hash 去重
        assert data["stats"]["imported_samples"] == 0


class TestUnavailable:
    @pytest.mark.asyncio
    async def test_component_unavailable_returns_503(self, web_env, monkeypatch):
        monkeypatch.setattr(
            pe_routes, "get_component_manager", lambda: FakeManager(None)
        )
        client = web_env.app.test_client()

        for method, path, kwargs in [
            ("get", f"{PREFIX}/jobs", {}),
            ("get", f"{PREFIX}/personas", {}),
            ("post", f"{PREFIX}/jobs", {"json": {"persona_id": "p1"}}),
            ("get", f"{PREFIX}/samples/stats", {}),
            ("get", f"{PREFIX}/export", {}),
        ]:
            resp = await getattr(client, method)(path, **kwargs)
            assert resp.status_code == 503, path
            assert (await resp.get_json())["success"] is False
