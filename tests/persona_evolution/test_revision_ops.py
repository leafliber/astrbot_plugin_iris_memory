"""阶段 3 测试：审批、拒绝、回滚、冲突解决、暂停恢复、导出导入、schema v2 迁移"""

import sqlite3

import pytest

from iris_memory.persona_evolution import PersonaEvolutionStorage
from iris_memory.persona_evolution.models import (
    ErrorCode,
    EvolutionJob,
    JobStatus,
    PersonaRevision,
    RevisionStatus,
    TriggerType,
)
from iris_memory.persona_evolution.publisher import (
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    append_managed_block,
)

from .conftest import make_job, seed_samples
from .fakes import (
    ANALYSIS_MODULE,
    GENERATION_MODULE,
    REVIEW_MODULE,
    FakeContext,
    FakePersonaManager,
    good_analysis_json,
    good_generation_json,
    good_review_json,
)

BASE = (
    "你是 Iris，一个群聊助手。\n\n"
    f"{MANAGED_BLOCK_BEGIN}\n旧风格\n{MANAGED_BLOCK_END}\n"
)
CANDIDATE = BASE.replace("旧风格", "新风格：短句为主")
CANDIDATE2 = BASE.replace("旧风格", "新风格：多用语气词")

APPLIED = RevisionStatus.APPLIED.value
CANDIDATE_S = RevisionStatus.CANDIDATE.value


def _setup_llm(llm, candidate=CANDIDATE):
    llm.set_default(ANALYSIS_MODULE, good_analysis_json())
    llm.set_default(GENERATION_MODULE, good_generation_json(candidate))
    llm.set_default(REVIEW_MODULE, good_review_json())


async def _run_to_candidate(storage, persona_manager, llm, service, candidate=CANDIDATE):
    """手动审批 Job 跑一轮，停在 candidate；返回 (svc, job_id, revision_id)"""
    persona_manager.add_persona("p1", BASE)
    seed_samples(storage, 25)
    job_id = make_job(storage, "p1", approval_mode="manual")
    _setup_llm(llm, candidate)
    svc = service()
    result = await svc.run_job(job_id, "manual")
    assert result["ok"], result
    revision_id = result["revision_id"]
    assert revision_id
    assert storage.get_revision(revision_id).status == CANDIDATE_S
    return svc, job_id, revision_id


async def _publish_two_versions(storage, persona_manager, llm, service):
    """自动 Job 连发两个版本；返回 (svc, job_id, v1_id, v2_id)"""
    persona_manager.add_persona("p1", BASE)
    seed_samples(storage, 100)
    job_id = make_job(storage, "p1")
    _setup_llm(llm, CANDIDATE)
    svc = service()
    r1 = await svc.run_job(job_id, "manual")
    assert r1["ok"] and not r1["no_change"], r1
    llm.push(GENERATION_MODULE, good_generation_json(CANDIDATE2))
    r2 = await svc.run_job(job_id, "manual")
    assert r2["ok"] and not r2["no_change"], r2
    assert persona_manager.get_prompt("p1") == CANDIDATE2
    return svc, job_id, r1["revision_id"], r2["revision_id"]


class TestApproveRevision:
    """手动审批（文档 §11.2）"""

    @pytest.mark.asyncio
    async def test_approve_publishes_candidate(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )

        result = await svc.approve_revision(rev_id)

        assert result["ok"], result
        assert persona_manager.get_prompt("p1") == CANDIDATE
        revision = storage.get_revision(rev_id)
        assert revision.status == APPLIED
        job = storage.get_job(job_id)
        assert job.last_applied_revision_id == rev_id
        assert job.last_success_at  # 发布刷新成功冷却

    @pytest.mark.asyncio
    async def test_approve_after_external_edit_goes_conflict(
        self, storage, persona_manager, llm, service
    ):
        """候选生成后 Persona 被外部编辑：不能批准，转 conflict（§11.2）"""
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )
        persona_manager.external_edit("p1", "外部编辑的内容")

        result = await svc.approve_revision(rev_id)

        assert not result["ok"]
        assert result["error_code"] == ErrorCode.EXTERNAL_CHANGE.value
        # 不覆盖外部编辑
        assert persona_manager.get_prompt("p1") == "外部编辑的内容"
        assert storage.get_job(job_id).status == JobStatus.CONFLICT.value
        assert storage.get_revision(rev_id).status == CANDIDATE_S
        # 外部修改快照已记录
        external = storage.list_revisions(
            job_id, status=RevisionStatus.EXTERNAL_CHANGE.value
        )
        assert len(external) == 1
        assert external[0].result_prompt == "外部编辑的内容"
        # 冲突状态下不能再批准
        again = await svc.approve_revision(rev_id)
        assert again["error_code"] == ErrorCode.CONFLICT_UNRESOLVED.value

    @pytest.mark.asyncio
    async def test_approve_rejects_non_candidate(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )
        svc.reject_revision(rev_id, "先拒绝")

        result = await svc.approve_revision(rev_id)

        assert not result["ok"]
        assert result["error_code"] == ErrorCode.INVALID_PARAMS.value
        assert persona_manager.get_prompt("p1") == BASE

    @pytest.mark.asyncio
    async def test_approve_not_found(self, storage, service):
        svc = service()
        result = await svc.approve_revision(9999)
        assert result["error_code"] == ErrorCode.NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_reject_stores_reason(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )

        result = svc.reject_revision(rev_id, "改动太激进")

        assert result["ok"], result
        revision = storage.get_revision(rev_id)
        assert revision.status == RevisionStatus.REJECTED.value
        assert revision.decision_reason == "改动太激进"
        # 重复拒绝报错
        again = svc.reject_revision(rev_id)
        assert again["error_code"] == ErrorCode.INVALID_PARAMS.value


class TestRollback:
    """回滚（文档 §13.3 git revert 语义）"""

    @pytest.mark.asyncio
    async def test_rollback_produces_new_version_keeps_history(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, v1, v2 = await _publish_two_versions(
            storage, persona_manager, llm, service
        )
        job_before = storage.get_job(job_id)

        result = await svc.rollback_to_revision(v1)

        assert result["ok"], result
        assert persona_manager.get_prompt("p1") == CANDIDATE
        new_rev = storage.get_revision(result["revision_id"])
        # 新建 rollback Revision：父=当前版本，修改前=当前快照，修改后=目标内容
        assert new_rev.version == 3
        assert new_rev.status == RevisionStatus.ROLLBACK.value
        assert new_rev.trigger_type == TriggerType.ROLLBACK.value
        assert new_rev.parent_revision_id == v2
        assert new_rev.base_prompt == CANDIDATE2
        assert new_rev.result_prompt == CANDIDATE
        # 历史不丢（不提供破坏性删除版本）
        assert storage.get_revision(v1).status == APPLIED
        assert storage.get_revision(v2).status == APPLIED
        # 冷却刷新，语料游标不动
        job_after = storage.get_job(job_id)
        assert job_after.last_applied_revision_id == new_rev.id
        assert job_after.last_success_at >= job_before.last_success_at
        assert job_after.last_sample_cursor == job_before.last_sample_cursor

    @pytest.mark.asyncio
    async def test_rollback_to_oldest_applied_and_chain(
        self, storage, persona_manager, llm, service
    ):
        """可回滚到任意 applied / rollback 版本（回滚链）"""
        svc, job_id, v1, v2 = await _publish_two_versions(
            storage, persona_manager, llm, service
        )
        r1 = await svc.rollback_to_revision(v1)
        assert r1["ok"], r1
        v3 = r1["revision_id"]

        # 再回滚到 v2（链式）
        r2 = await svc.rollback_to_revision(v2)
        assert r2["ok"], r2
        assert persona_manager.get_prompt("p1") == CANDIDATE2
        # rollback 状态的版本同样可作为回滚目标
        r3 = await svc.rollback_to_revision(v3)
        assert r3["ok"], r3
        assert persona_manager.get_prompt("p1") == CANDIDATE
        versions = [
            r.version for r in storage.list_revisions(job_id, limit=10)
        ]
        assert sorted(versions) == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_rollback_no_change_target(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, v1, v2 = await _publish_two_versions(
            storage, persona_manager, llm, service
        )

        result = await svc.rollback_to_revision(v2)

        assert not result["ok"]
        assert result["error_code"] == ErrorCode.NO_CHANGE.value
        assert result["no_change"] is True

    @pytest.mark.asyncio
    async def test_rollback_requires_applied_status(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )

        result = await svc.rollback_to_revision(rev_id)

        assert not result["ok"]
        assert result["error_code"] == ErrorCode.INVALID_PARAMS.value

    @pytest.mark.asyncio
    async def test_rollback_external_edit_then_resolve(
        self, storage, persona_manager, llm, service
    ):
        """外部修改后回滚：先转 conflict 提示，确认后回滚即冲突解决（§12.1）"""
        svc, job_id, v1, v2 = await _publish_two_versions(
            storage, persona_manager, llm, service
        )
        persona_manager.external_edit("p1", "外部版本")

        first = await svc.rollback_to_revision(v1)
        assert not first["ok"]
        assert first["error_code"] == ErrorCode.EXTERNAL_CHANGE.value
        assert storage.get_job(job_id).status == JobStatus.CONFLICT.value
        assert persona_manager.get_prompt("p1") == "外部版本"

        # 冲突状态下回滚是文档 §12.1 的解决路径：覆盖外部版本并恢复 active
        second = await svc.rollback_to_revision(v1)
        assert second["ok"], second
        assert persona_manager.get_prompt("p1") == CANDIDATE
        assert storage.get_job(job_id).status == JobStatus.ACTIVE.value


class TestAdoptCurrent:
    """冲突解决：采纳外部版本为新基线（文档 §12.1）"""

    async def _make_conflict(self, storage, persona_manager, llm, service):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm, CANDIDATE)
        svc = service()
        r1 = await svc.run_job(job_id, "manual")
        assert r1["ok"], r1
        persona_manager.external_edit("p1", "外部改的新内容")
        r2 = await svc.run_job(job_id, "manual")
        assert r2["error_code"] == ErrorCode.EXTERNAL_CHANGE.value
        assert storage.get_job(job_id).status == JobStatus.CONFLICT.value
        return svc, job_id

    @pytest.mark.asyncio
    async def test_adopt_current_resolves_conflict(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id = await self._make_conflict(
            storage, persona_manager, llm, service
        )

        result = await svc.adopt_current_for_conflict(job_id)

        assert result["ok"], result
        job = storage.get_job(job_id)
        assert job.status == JobStatus.ACTIVE.value
        baseline = storage.get_revision(result["revision_id"])
        assert baseline.status == APPLIED
        assert baseline.base_prompt == "外部改的新内容"
        assert baseline.result_prompt == "外部改的新内容"
        assert job.last_applied_revision_id == baseline.id
        # 采纳只改基线，不写 PersonaManager（外部内容保持不变）
        assert persona_manager.get_prompt("p1") == "外部改的新内容"

        # 采纳后再次运行不再误报外部修改，可正常迭代
        new_candidate = append_managed_block("外部改的新内容").replace(
            "[自动迭代生成的表达风格、语气、长度、节奏与互动习惯]", "新风格"
        )
        llm.push(GENERATION_MODULE, good_generation_json(new_candidate))
        r3 = await svc.run_job(job_id, "manual")
        assert r3["ok"], r3
        assert persona_manager.get_prompt("p1") == new_candidate

    @pytest.mark.asyncio
    async def test_adopt_requires_conflict_status(self, storage, service):
        make_job(storage, "p1")
        svc = service()

        result = await svc.adopt_current_for_conflict(1)

        assert not result["ok"]
        assert result["error_code"] == ErrorCode.INVALID_PARAMS.value


class TestPauseResume:
    """Job 暂停/恢复（文档 §8.3）"""

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, storage, service):
        job_id = make_job(storage, "p1")
        svc = service()

        assert svc.pause_job(job_id)["ok"]
        assert storage.get_job(job_id).status == JobStatus.PAUSED.value
        # 重复暂停报错
        assert svc.pause_job(job_id)["error_code"] == ErrorCode.INVALID_PARAMS.value
        # 暂停时不能运行
        run = await svc.run_job(job_id, "manual")
        assert run["error_code"] == ErrorCode.JOB_NOT_ACTIVE.value

        assert svc.resume_job(job_id)["ok"]
        assert storage.get_job(job_id).status == JobStatus.ACTIVE.value

    def test_resume_clears_paused_error(self, storage, service):
        job_id = make_job(storage, "p1")
        storage.update_job(
            job_id, {"status": JobStatus.PAUSED_ERROR.value, "consecutive_failures": 3}
        )
        svc = service()

        result = svc.resume_job(job_id)

        assert result["ok"], result
        job = storage.get_job(job_id)
        assert job.status == JobStatus.ACTIVE.value
        assert job.consecutive_failures == 0

    def test_resume_conflict_rejected(self, storage, service):
        job_id = make_job(storage, "p1")
        storage.update_job(job_id, {"status": JobStatus.CONFLICT.value})
        svc = service()

        result = svc.resume_job(job_id)

        assert result["error_code"] == ErrorCode.CONFLICT_UNRESOLVED.value

    def test_pause_not_found(self, storage, service):
        svc = service()
        assert svc.pause_job(9999)["error_code"] == ErrorCode.NOT_FOUND.value
        assert svc.resume_job(9999)["error_code"] == ErrorCode.NOT_FOUND.value


class TestApprovalModeSwitch:
    """手动切回自动：已有 candidate 不追溯发布（文档 §11.2）"""

    @pytest.mark.asyncio
    async def test_manual_to_auto_no_retroactive_publish(
        self, storage, persona_manager, llm, service
    ):
        svc, job_id, rev_id = await _run_to_candidate(
            storage, persona_manager, llm, service
        )
        assert persona_manager.get_prompt("p1") == BASE

        # 切换为自动审批
        storage.update_job(job_id, {"approval_mode": "auto"})

        # 旧 candidate 不被追溯发布
        assert persona_manager.get_prompt("p1") == BASE

        # 新一轮运行只发布本轮新候选
        llm.push(GENERATION_MODULE, good_generation_json(CANDIDATE2))
        r2 = await svc.run_job(job_id, "manual")
        assert r2["ok"], r2
        assert persona_manager.get_prompt("p1") == CANDIDATE2
        # 旧 candidate 保持 candidate 状态
        assert storage.get_revision(rev_id).status == CANDIDATE_S


class TestExportImport:
    """独立导出导入（文档 §19）"""

    async def _build_export_data(self, storage, persona_manager, llm, service):
        """造一份含 applied + candidate 的完整数据"""
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 30, group_id="g1", user_id="u1")
        seed_samples(storage, 30, group_id="g2", user_id="u2", start=100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm, CANDIDATE)
        svc = service()
        r1 = await svc.run_job(job_id, "manual")
        assert r1["ok"], r1
        storage.update_job(job_id, {"approval_mode": "manual"})
        llm.push(GENERATION_MODULE, good_generation_json(CANDIDATE2))
        r2 = await svc.run_job(job_id, "manual")
        assert r2["ok"], r2
        return svc, job_id, r1["revision_id"], r2["revision_id"]

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(
        self, storage, persona_manager, llm, service, tmp_path
    ):
        await self._build_export_data(storage, persona_manager, llm, service)

        data = storage.export_all()

        assert data["version"] == "1.1"
        assert data["samples"] == []  # 默认不含语料原文
        assert len(data["jobs"]) == 1
        assert len(data["revisions"]) == 2
        assert len(data["runs"]) == 2

        s2 = PersonaEvolutionStorage(tmp_path / "import.db")
        s2.init_schema()
        try:
            stats = s2.import_from_data(data)
            assert stats["error_count"] == 0
            assert stats["imported_jobs"] == 1
            assert stats["imported_runs"] == 2
            assert stats["imported_revisions"] == 2

            job2 = s2.get_job_by_persona("p1")
            assert job2 is not None
            by_version = {
                r.version: r for r in s2.list_revisions(job2.id, limit=10)
            }
            assert set(by_version) == {1, 2}
            # 完整快照保留（风格画像/校验/审查结果）
            v1 = by_version[1]
            assert v1.status == APPLIED
            assert v1.result_prompt == CANDIDATE
            assert v1.base_prompt == BASE
            assert v1.validation.get("passed") is True
            assert v1.style_profile.get("verbosity") == "short"
            assert v1.goal_snapshot
            # 版本号保留、父版本与发布基线重映射
            assert by_version[2].status == CANDIDATE_S
            assert by_version[2].parent_revision_id == v1.id
            assert job2.last_applied_revision_id == v1.id
            # 语料引用重映射：未导出语料时 sample_id 置空、hash 保留
            refs = s2.export_all()["revision_refs"]
            assert refs
            assert all(r["sample_id"] is None for r in refs)
            assert all(r["sample_hash"] for r in refs)
        finally:
            s2.close()

    @pytest.mark.asyncio
    async def test_export_include_samples(
        self, storage, persona_manager, llm, service, tmp_path
    ):
        seed_samples(storage, 5)
        data = storage.export_all(include_samples=True)
        assert len(data["samples"]) == 5
        assert data["include_samples"] is True

        s2 = PersonaEvolutionStorage(tmp_path / "import.db")
        s2.init_schema()
        try:
            stats = s2.import_from_data(data)
            assert stats["imported_samples"] == 5
            assert s2.count_samples() == 5
            # 重复导入：dedupe_hash 天然去重
            again = s2.import_from_data(data)
            assert again["imported_samples"] == 0
            assert s2.count_samples() == 5
        finally:
            s2.close()

    @pytest.mark.asyncio
    async def test_import_does_not_touch_persona(
        self, storage, persona_manager, llm, service, tmp_path
    ):
        """导入 Revision 历史绝不自动修改 AstrBot Persona（§19）"""
        await self._build_export_data(storage, persona_manager, llm, service)
        data = storage.export_all()

        s2 = PersonaEvolutionStorage(tmp_path / "import.db")
        s2.init_schema()
        pm2 = FakePersonaManager()
        pm2.add_persona("p1", BASE)
        try:
            stats = s2.import_from_data(data)
            assert stats["imported_revisions"] == 2
            # 导入全程不调用 PersonaManager 写入
            assert pm2.update_calls == []
            assert pm2.get_prompt("p1") == BASE
        finally:
            s2.close()

    @pytest.mark.asyncio
    async def test_import_duplicate_jobs_skipped(
        self, storage, persona_manager, llm, service, tmp_path
    ):
        await self._build_export_data(storage, persona_manager, llm, service)
        data = storage.export_all()

        s2 = PersonaEvolutionStorage(tmp_path / "import.db")
        s2.init_schema()
        try:
            first = s2.import_from_data(data)
            assert first["imported_jobs"] == 1
            second = s2.import_from_data(data)
            assert second["imported_jobs"] == 0
            assert second["skipped_jobs"] == 1
            assert second["imported_revisions"] == 0
            assert second["skipped_revisions"] == 2
            # 重复导入不产生重复数据
            assert len(s2.list_jobs()) == 1
            assert len(s2.list_revisions(s2.list_jobs()[0].id, limit=10)) == 2
        finally:
            s2.close()

    def test_import_rejects_bad_shape(self, storage):
        with pytest.raises(ValueError):
            storage.import_from_data(["不是字典"])

    def test_import_missing_sections_ok(self, storage):
        """空数据/缺字段不报错"""
        stats = storage.import_from_data({"version": "1.1"})
        assert stats["error_count"] == 0
        assert stats["imported_jobs"] == 0


class TestSchemaV2Migration:
    """schema v1 → v2 迁移（decision_reason 列）"""

    def test_migrates_v1_to_v2(self, tmp_path):
        from iris_memory.persona_evolution.storage import _SCHEMA_V1

        db_path = tmp_path / "pe.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(_SCHEMA_V1)
        conn.execute("PRAGMA user_version=1")
        # v1 结构插入一行（无 decision_reason 列）
        conn.execute(
            "INSERT INTO persona_revisions"
            " (job_id, version, status, result_prompt, created_at)"
            " VALUES (1, 1, 'applied', '旧内容', 1.0)"
        )
        conn.commit()
        conn.close()

        storage = PersonaEvolutionStorage(db_path)
        storage.init_schema()
        try:
            assert storage.get_schema_version() == 2
            revision = storage.get_revision_by_version(1, 1)
            assert revision.result_prompt == "旧内容"
            assert revision.decision_reason == ""  # 迁移默认空
            # 迁移后可正常写入 decision_reason
            job_id = storage.create_job(EvolutionJob(persona_id="p1"))
            rev_id = storage.create_revision(
                PersonaRevision(
                    job_id=job_id, version=0, result_prompt="x", decision_reason="拒绝理由"
                )
            )
            assert storage.get_revision(rev_id).decision_reason == "拒绝理由"
        finally:
            storage.close()
