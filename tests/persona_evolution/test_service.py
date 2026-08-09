"""service.py 集成测试：全链路发布、触发、冲突、恢复、重试熔断"""

import asyncio

import pytest

from iris_memory.persona_evolution.models import (
    EditMode,
    ErrorCode,
    JobStatus,
    PersonaRevision,
    RevisionStatus,
)
from iris_memory.persona_evolution.publisher import (
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    persona_hash,
    split_managed_block,
)

from .conftest import make_job, seed_samples
from .fakes import (
    ANALYSIS_MODULE,
    GENERATION_MODULE,
    REVIEW_MODULE,
    good_analysis_json,
    good_generation_json,
    good_review_json,
)

BASE = (
    "你是 Iris，一个群聊助手。\n\n"
    f"{MANAGED_BLOCK_BEGIN}\n旧风格\n{MANAGED_BLOCK_END}\n"
)
CANDIDATE = BASE.replace("旧风格", "新风格：短句为主")


def _setup_llm(llm, candidate=CANDIDATE):
    llm.set_default(ANALYSIS_MODULE, good_analysis_json())
    llm.set_default(GENERATION_MODULE, good_generation_json(candidate))
    llm.set_default(REVIEW_MODULE, good_review_json())


class TestManagedBlockAutoPublish:
    @pytest.mark.asyncio
    async def test_full_pipeline_auto_publish(self, storage, persona_manager, llm, service):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        svc = service()

        result = await svc.run_job(job_id, "auto")

        assert result["ok"], result
        assert result["error_code"] is None
        # 发布到 PersonaManager，且只传 persona_id + system_prompt
        assert persona_manager.get_prompt("p1") == CANDIDATE
        args, kwargs = persona_manager.update_calls[0]
        assert args == ("p1",) and set(kwargs) == {"system_prompt"}
        # 区块外逐字节不变
        base_split = split_managed_block(BASE)
        cand_split = split_managed_block(persona_manager.get_prompt("p1"))
        assert cand_split.before == base_split.before
        assert cand_split.after == base_split.after
        # Revision applied，校验快照通过
        revision = storage.get_revision(result["revision_id"])
        assert revision.status == RevisionStatus.APPLIED.value
        assert revision.validation["passed"] is True
        assert revision.style_profile["verbosity"] == "short"
        # 游标推进 + 冷却刷新
        job = storage.get_job(job_id)
        assert job.last_sample_cursor == 100
        assert job.last_success_at is not None
        assert job.last_applied_revision_id == revision.id
        # Run 审计
        run = storage.get_run(result["run_id"])
        assert run.status == "success"
        assert run.eligible_count == 100

    @pytest.mark.asyncio
    async def test_first_run_appends_block(self, storage, persona_manager, llm, service):
        raw = "你是 Iris，一个群聊助手。"
        persona_manager.add_persona("p1", raw)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        # 候选 = 追加区块后的有效基线替换区块内容
        from iris_memory.persona_evolution.publisher import append_managed_block

        effective = append_managed_block(raw)
        candidate = effective.replace(
            "[自动迭代生成的表达风格、语气、长度、节奏与互动习惯]", "新风格"
        )
        _setup_llm(llm, candidate)
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"], result
        published = persona_manager.get_prompt("p1")
        assert MANAGED_BLOCK_BEGIN in published
        assert published.startswith(raw)
        assert "新风格" in published


class TestFullPromptReview:
    @pytest.mark.asyncio
    async def test_review_failed_never_publishes(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1", edit_mode=EditMode.FULL_PROMPT.value)
        _setup_llm(llm)
        llm.set_default(REVIEW_MODULE, good_review_json(identity_consistency=0.5))
        svc = service()

        result = await svc.run_job(job_id, "auto")

        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.REVIEW_FAILED.value
        assert persona_manager.get_prompt("p1") == BASE  # 绝不发布
        assert persona_manager.update_calls == []
        revision = storage.get_revision(result["revision_id"])
        assert revision.status == RevisionStatus.FAILED_VALIDATION.value
        assert revision.review["identity_consistency"] == 0.5
        # 审查失败计入连续失败（熔断计数）
        assert storage.get_job(job_id).consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_review_passed_publishes(self, storage, persona_manager, llm, service):
        base = "你是 Iris，一个群聊助手，回答简洁直接，身份稳定不变。" * 3
        candidate = base.replace("简洁直接", "更加简洁", 1)
        persona_manager.add_persona("p1", base)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1", edit_mode=EditMode.FULL_PROMPT.value)
        _setup_llm(llm, candidate)
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"], result
        assert persona_manager.get_prompt("p1") == candidate
        # 审查走了独立 module
        assert any(c["module"] == REVIEW_MODULE for c in llm.calls)


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_trigger_runs_once(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        llm.delay = 0.15  # 拉长 LLM 调用制造重叠窗口
        svc = service()

        first = asyncio.create_task(svc.run_job(job_id, "auto"))
        await asyncio.sleep(0.05)
        second = await svc.run_job(job_id, "auto")
        await first

        assert second["ok"] is False
        assert second["error_code"] == ErrorCode.TRIGGER_CONDITIONS_NOT_MET.value
        analysis_calls = [c for c in llm.calls if c["module"] == ANALYSIS_MODULE]
        assert len(analysis_calls) == 1  # 流水线只跑了一次


class TestTriggerConditions:
    @pytest.mark.asyncio
    async def test_threshold_and_cooldown_gate(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")  # 默认 100 条 / 24h
        _setup_llm(llm)
        svc = service()

        # 语料不足：不触发
        seed_samples(storage, 99)
        assert await svc.run_trigger_scan() == 0
        assert not any(c["module"] == ANALYSIS_MODULE for c in llm.calls)

        # 凑满 100 条：触发一次
        seed_samples(storage, 1, start=99)
        assert await svc.run_trigger_scan() == 1
        assert persona_manager.get_prompt("p1") == CANDIDATE

        # 再补 100 条：冷却期内不重复触发
        seed_samples(storage, 100, start=100)
        llm.calls.clear()
        assert await svc.run_trigger_scan() == 0
        assert not any(c["module"] == ANALYSIS_MODULE for c in llm.calls)

    @pytest.mark.asyncio
    async def test_manual_bypasses_threshold(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 25)  # 低于 100 条自动门槛
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        svc = service()

        result = await svc.run_job(job_id, "manual")
        assert result["ok"], result
        assert persona_manager.get_prompt("p1") == CANDIDATE

    @pytest.mark.asyncio
    async def test_manual_min_samples_deficit(self, storage, persona_manager, llm, service):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 15)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        svc = service()

        result = await svc.run_job(job_id, "manual")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.INSUFFICIENT_SAMPLES.value
        assert "还差 5 条" in result["message"]
        assert not any(c["module"] == ANALYSIS_MODULE for c in llm.calls)


class TestExternalChange:
    @pytest.mark.asyncio
    async def test_external_edit_blocks_and_preserves(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        svc = service()

        # 先成功发布一次建立基线
        result = await svc.run_job(job_id, "auto")
        assert result["ok"]

        # 外部编辑 Persona
        persona_manager.external_edit("p1", BASE + "\n外部新增设定\n")

        # 下一轮检测到外部修改：不覆盖，转 conflict
        result = await svc.run_job(job_id, "manual")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.EXTERNAL_CHANGE.value
        assert persona_manager.get_prompt("p1") == BASE + "\n外部新增设定\n"
        job = storage.get_job(job_id)
        assert job.status == JobStatus.CONFLICT.value
        # 记录了 external_change 快照 Revision
        revisions = storage.list_revisions(
            job_id, status=RevisionStatus.EXTERNAL_CHANGE.value
        )
        assert len(revisions) == 1
        assert revisions[0].result_prompt == BASE + "\n外部新增设定\n"

        # conflict 状态下拒绝运行
        result = await svc.run_job(job_id, "manual")
        assert result["error_code"] == ErrorCode.CONFLICT_UNRESOLVED.value

    @pytest.mark.asyncio
    async def test_marker_conflict_blocks(self, storage, persona_manager, llm, service):
        # 标记重复 → conflict，禁止自动修复
        persona_manager.add_persona("p1", BASE + BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.MARKER_INVALID.value
        assert storage.get_job(job_id).status == JobStatus.CONFLICT.value
        assert persona_manager.update_calls == []


class TestPublishingReconcile:
    def _make_publishing_revision(self, storage, job_id, base, result):
        revision = PersonaRevision(
            job_id=job_id,
            version=0,
            status=RevisionStatus.PUBLISHING.value,
            base_prompt=base,
            result_prompt=result,
            base_hash=persona_hash(base),
            result_hash=persona_hash(result),
        )
        storage.create_revision(revision)
        return revision

    @pytest.mark.asyncio
    async def test_hash_matches_candidate_becomes_applied(
        self, storage, persona_manager, llm, service
    ):
        # 中断发生在"已更新 Persona 但未写回"之后：补记 applied
        persona_manager.add_persona("p1", CANDIDATE)
        job_id = make_job(storage, "p1")
        revision = self._make_publishing_revision(storage, job_id, BASE, CANDIDATE)
        svc = service()

        summary = await svc.reconcile_publishing()
        assert summary["applied"] == 1
        assert storage.get_revision(revision.id).status == RevisionStatus.APPLIED.value
        job = storage.get_job(job_id)
        assert job.last_applied_revision_id == revision.id

    @pytest.mark.asyncio
    async def test_hash_matches_base_becomes_publish_failed(
        self, storage, persona_manager, llm, service
    ):
        # 中断发生在 update_persona 之前：补记 publish_failed，允许重试
        persona_manager.add_persona("p1", BASE)
        job_id = make_job(storage, "p1")
        revision = self._make_publishing_revision(storage, job_id, BASE, CANDIDATE)
        svc = service()

        summary = await svc.reconcile_publishing()
        assert summary["publish_failed"] == 1
        assert (
            storage.get_revision(revision.id).status
            == RevisionStatus.PUBLISH_FAILED.value
        )
        assert storage.get_job(job_id).status == JobStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_hash_matches_neither_becomes_conflict(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", "完全未知的第三版本")
        job_id = make_job(storage, "p1")
        revision = self._make_publishing_revision(storage, job_id, BASE, CANDIDATE)
        svc = service()

        summary = await svc.reconcile_publishing()
        assert summary["conflict"] == 1
        assert storage.get_job(job_id).status == JobStatus.CONFLICT.value
        assert (
            storage.get_revision(revision.id).status
            == RevisionStatus.PUBLISH_FAILED.value
        )


class TestRetryAndCircuitBreaker:
    @pytest.mark.asyncio
    async def test_provider_error_retries_with_backoff(
        self, config, storage, persona_manager, llm, service
    ):
        config.set_hidden("persona_evolution_retry_intervals_minutes", "0.01,0.01,0.01")
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm)
        # 第一次分析调用 Provider 故障，队列耗尽后用默认好 JSON
        llm.push(ANALYSIS_MODULE, RuntimeError("provider down"))
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.PROVIDER_ERROR.value
        assert "重试" in result["message"]
        # 失败不推进游标不刷新冷却
        job = storage.get_job(job_id)
        assert job.last_sample_cursor == 0
        assert job.last_success_at is None
        assert job.consecutive_failures == 0

        # 退避后自动重试成功
        await asyncio.sleep(1.5)
        assert persona_manager.get_prompt("p1") == CANDIDATE
        assert storage.get_job(job_id).last_sample_cursor == 100
        svc.cancel_pending_retries()

    @pytest.mark.asyncio
    async def test_circuit_breaker_after_parse_failures(
        self, config, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        llm.set_default(ANALYSIS_MODULE, "这不是 JSON")
        llm.set_default(GENERATION_MODULE, good_generation_json(CANDIDATE))
        svc = service()

        for i in range(3):
            result = await svc.run_job(job_id, "manual")
            assert result["error_code"] == ErrorCode.ANALYSIS_PARSE_FAILED.value
            assert storage.get_job(job_id).consecutive_failures == i + 1

        # 连续 3 次解析失败 → paused_error
        assert storage.get_job(job_id).status == JobStatus.PAUSED_ERROR.value
        result = await svc.run_job(job_id, "manual")
        assert result["error_code"] == ErrorCode.CIRCUIT_OPEN.value
        svc.cancel_pending_retries()

    @pytest.mark.asyncio
    async def test_validation_failure_keeps_revision_never_republishes(
        self, storage, persona_manager, llm, service
    ):
        # 候选改了区块外 → 校验失败保留 failed_validation，同一候选不重试
        bad_candidate = CANDIDATE.replace("你是 Iris，一个群聊助手。", "你是 Moca。")
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        llm.set_default(ANALYSIS_MODULE, good_analysis_json())
        llm.set_default(GENERATION_MODULE, good_generation_json(bad_candidate))
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.BLOCK_OUTSIDE_MODIFIED.value
        assert persona_manager.get_prompt("p1") == BASE
        revision = storage.get_revision(result["revision_id"])
        assert revision.status == RevisionStatus.FAILED_VALIDATION.value
        assert revision.validation["passed"] is False
        assert storage.get_job(job_id).consecutive_failures == 1


class TestNoChangeAndLowConfidence:
    @pytest.mark.asyncio
    async def test_no_change_not_published(self, storage, persona_manager, llm, service):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        _setup_llm(llm, candidate=BASE)  # 候选与当前一致
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"] is True
        assert result["no_change"] is True
        assert persona_manager.update_calls == []
        revision = storage.get_revision(result["revision_id"])
        assert revision.status == RevisionStatus.NO_CHANGE.value
        # no_change 算成功：推进游标 + 刷新冷却
        job = storage.get_job(job_id)
        assert job.last_sample_cursor == 100
        assert job.last_success_at is not None

    @pytest.mark.asyncio
    async def test_low_confidence_stops_without_circuit(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1")
        llm.set_default(ANALYSIS_MODULE, good_analysis_json(confidence=0.3))
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"] is False
        assert result["error_code"] == ErrorCode.ANALYSIS_LOW_CONFIDENCE.value
        # 低置信度是良性停轮：不计熔断
        assert storage.get_job(job_id).consecutive_failures == 0


class TestManualApprovalMode:
    @pytest.mark.asyncio
    async def test_manual_mode_stops_at_candidate(
        self, storage, persona_manager, llm, service
    ):
        persona_manager.add_persona("p1", BASE)
        seed_samples(storage, 100)
        job_id = make_job(storage, "p1", approval_mode="manual")
        _setup_llm(llm)
        svc = service()

        result = await svc.run_job(job_id, "auto")
        assert result["ok"], result
        assert persona_manager.update_calls == []  # 不自动发布
        revision = storage.get_revision(result["revision_id"])
        assert revision.status == RevisionStatus.CANDIDATE.value
        assert revision.validation["passed"] is True
