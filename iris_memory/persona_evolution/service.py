"""
Iris Chat Memory - 人格自迭代用例编排

PersonaEvolutionService：三阶段流水线 + 确定性闸门 + 发布/冲突/恢复的
用例层，供组件触发、后续指令层与 Web 路由共用：
- run_job：per-Job 运行锁防并发；A→B→(C)→校验→自动发布或停 candidate；
- 手动执行跳过 100 条/24h 门槛但要求 >= manual_min_samples 条；
- 自动触发条件判定（文档 §8.1）与每小时兜底扫描；
- 外部修改检测（§12.1）与启动恢复对账（§12.2）；
- 失败重试（30/120/360 分钟退避，默认最多 3 次，不推进游标不刷新冷却）
  与熔断（连续解析/校验失败达阈值 → paused_error）；
- 手动审批 approve/reject（§11.2，批准前复核校验与 base_hash）；
- 回滚（§13.3 git revert 语义，新版本+冷却刷新+游标不动）；
- 冲突解决 adopt-current（§12.1）与 Job 暂停/恢复（§8.3）。
- 手动切回自动审批不追溯发布已有 candidate（只影响后续运行）。

LLM await 一律在组件写锁外（learning 模式）：存储读写短临界区持
db_lock，阶段 A/B/C 与发布调用不持锁。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.config import get_config
from iris_memory.core import get_component_manager, get_logger
from .analyzer import StyleAnalyzer
from .generator import CandidateGenerator
from .goals import build_goal_snapshot
from .models import (
    ApprovalMode,
    EditMode,
    ErrorCode,
    EvolutionRun,
    JobStatus,
    PersonaRevision,
    RevisionStatus,
    RunStatus,
    TriggerType,
)
from .publisher import (
    MARKERS_ABSENT,
    MARKERS_INVALID,
    PersonaPublisher,
    append_managed_block,
    persona_hash,
    split_managed_block,
)
from .reviewer import PromptReviewer
from .sampler import stratified_sample
from .validator import validate_candidate

logger = get_logger("persona_evolution.service")


class PersonaEvolutionService:
    """人格自迭代用例编排服务

    Args:
        storage: PersonaEvolutionStorage 实例
        context: AstrBot Context（PersonaManager 来源，可为 None 降级）
        db_lock: 组件级单写者锁（存储读写临界区共用）
        llm_manager: 可注入的 LLMManager（测试用）；
            None 时调用期从组件管理器解析
    """

    def __init__(
        self,
        storage: Any,
        context: Any = None,
        db_lock: Optional[asyncio.Lock] = None,
        llm_manager: Any = None,
    ):
        self._storage = storage
        self._db_lock = db_lock or asyncio.Lock()
        self._publisher = PersonaPublisher(context, storage)
        self._llm_manager = llm_manager
        self._run_locks: Dict[int, asyncio.Lock] = {}
        self._retry_attempts: Dict[int, int] = {}
        self._retry_tasks: Dict[int, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # 配置与依赖解析
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg(key: str, default: Any) -> Any:
        value = get_config().get(key, default)
        return default if value is None else value

    def _get_llm_manager(self) -> Any:
        if self._llm_manager is not None:
            return self._llm_manager
        try:
            manager = get_component_manager()
            llm_manager = manager.get_component("llm_manager")
            if llm_manager and getattr(llm_manager, "is_available", False):
                return llm_manager
        except Exception as e:
            logger.debug(f"解析 LLMManager 失败：{e}")
        return None

    def _retry_intervals_minutes(self) -> List[float]:
        raw = str(self._cfg("persona_evolution_retry_intervals_minutes", "30,120,360"))
        intervals: List[float] = []
        for part in raw.split(","):
            try:
                value = float(part.strip())
            except ValueError:
                continue
            if value > 0:
                intervals.append(value)
        return intervals or [30.0, 120.0, 360.0]

    def _run_lock(self, job_id: int) -> asyncio.Lock:
        lock = self._run_locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._run_locks[job_id] = lock
        return lock

    def is_job_running(self, job_id: int) -> bool:
        """Job 是否有运行中的迭代（运行锁被持有）"""
        lock = self._run_locks.get(job_id)
        return lock is not None and lock.locked()

    # ------------------------------------------------------------------
    # 自动触发条件判定（文档 §8.1）
    # ------------------------------------------------------------------

    def check_auto_trigger(self, job: Any) -> Tuple[bool, str]:
        """自动触发条件判定（只查本地状态，不读 PersonaManager）

        Persona 哈希基线一致性在 run_job 开头复核（外部修改检测）。

        Returns:
            (是否满足, 不满足原因)
        """
        if job.status != JobStatus.ACTIVE.value:
            return False, f"Job 状态为 {job.status}，非 active"
        if self.is_job_running(job.id):
            return False, "Job 有运行中的迭代"
        count = self._storage.count_samples(
            job.source_group_ids, job.source_user_ids, job.last_sample_cursor
        )
        if count < job.trigger_sample_count:
            return False, f"新增语料 {count} 条，未达门槛 {job.trigger_sample_count}"
        if job.last_success_at:
            elapsed = time.time() - job.last_success_at
            if elapsed < job.min_interval_hours * 3600:
                return False, f"距上次成功迭代不足 {job.min_interval_hours} 小时"
        return True, ""

    async def run_trigger_scan(self) -> int:
        """兜底扫描：对满足自动触发条件的 Job 逐个执行（每小时周期任务）

        Returns:
            实际启动运行的 Job 数
        """
        started = 0
        try:
            jobs = self._storage.list_jobs()
        except Exception as e:
            logger.warning(f"兜底扫描读取 Job 失败：{e}")
            return 0
        for job in jobs:
            try:
                ok, _ = self.check_auto_trigger(job)
                if ok:
                    await self.run_job(job.id, TriggerType.AUTO.value)
                    started += 1
            except Exception as e:
                logger.warning(f"兜底扫描执行 Job {job.id} 失败：{e}")
        return started

    # ------------------------------------------------------------------
    # 主流程：run_job
    # ------------------------------------------------------------------

    async def run_job(self, job_id: int, trigger_type: str) -> Dict[str, Any]:
        """执行一轮迭代（per-Job 运行锁防并发）

        Args:
            job_id: Job id
            trigger_type: auto / manual（rollback 在阶段 3 接入）

        Returns:
            结果字典：{ok, job_id, run_id, error_code, message,
            revision_id, no_change}
        """
        result: Dict[str, Any] = {
            "ok": False,
            "job_id": job_id,
            "run_id": None,
            "error_code": None,
            "message": "",
            "revision_id": None,
            "no_change": False,
        }
        lock = self._run_lock(job_id)
        if lock.locked():
            result["error_code"] = ErrorCode.TRIGGER_CONDITIONS_NOT_MET.value
            result["message"] = "Job 已有运行中的迭代"
            return result

        async with lock:
            return await self._run_job_locked(job_id, trigger_type, result)

    async def _run_job_locked(
        self, job_id: int, trigger_type: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        job = self._storage.get_job(job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {job_id} 不存在"
            return result

        run_id = self._storage.create_run(
            EvolutionRun(
                job_id=job.id,
                trigger_type=trigger_type,
                sample_cursor_from=job.last_sample_cursor,
            )
        )
        result["run_id"] = run_id

        def fail(code: ErrorCode, message: str) -> Dict[str, Any]:
            result["error_code"] = code.value
            result["message"] = message
            self._finish_run(run_id, RunStatus.FAILED, code.value, message)
            return result

        try:
            # ---- 状态检查 ----
            if job.status == JobStatus.CONFLICT.value:
                return fail(ErrorCode.CONFLICT_UNRESOLVED, "存在未解决冲突，需管理员处理")
            if job.status == JobStatus.PAUSED_ERROR.value:
                return fail(ErrorCode.CIRCUIT_OPEN, "连续失败已熔断，需管理员恢复")
            if job.status != JobStatus.ACTIVE.value:
                return fail(ErrorCode.JOB_NOT_ACTIVE, f"Job 状态为 {job.status}")

            manual = trigger_type == TriggerType.MANUAL.value

            # ---- 语料门槛（§8.1/§8.2）----
            since_id = 0 if manual else job.last_sample_cursor
            eligible = self._storage.count_samples(
                job.source_group_ids, job.source_user_ids, since_id
            )
            if manual:
                minimum = int(
                    self._cfg("persona_evolution_manual_min_samples", 20)
                )
                if eligible < minimum:
                    return fail(
                        ErrorCode.INSUFFICIENT_SAMPLES,
                        f"有效语料不足：需要至少 {minimum} 条，"
                        f"当前 {eligible} 条，还差 {minimum - eligible} 条",
                    )
            else:
                # 自动门槛复核（运行锁已持有，不做运行中自检）
                if eligible < job.trigger_sample_count:
                    return fail(
                        ErrorCode.TRIGGER_CONDITIONS_NOT_MET,
                        f"新增语料 {eligible} 条，未达门槛 {job.trigger_sample_count}",
                    )
                if job.last_success_at:
                    elapsed = time.time() - job.last_success_at
                    if elapsed < job.min_interval_hours * 3600:
                        return fail(
                            ErrorCode.TRIGGER_CONDITIONS_NOT_MET,
                            f"距上次成功迭代不足 {job.min_interval_hours} 小时",
                        )

            # ---- 读取当前 Persona ----
            current_prompt, err = await self._publisher.read_current_prompt(
                job.persona_id
            )
            if err is not None or current_prompt is None:
                return fail(
                    ErrorCode.PERSONA_NOT_FOUND,
                    f"Persona {job.persona_id} 不存在或 PersonaManager 不可用",
                )
            current_hash = persona_hash(current_prompt)

            # ---- 外部修改检测（§12.1）----
            if job.last_applied_revision_id:
                applied = self._storage.get_revision(job.last_applied_revision_id)
                if (
                    applied is not None
                    and applied.result_hash
                    and current_hash != applied.result_hash
                ):
                    self._record_external_change(job, applied, current_prompt)
                    return fail(
                        ErrorCode.EXTERNAL_CHANGE,
                        "检测到 Persona 被外部修改，Job 已转 conflict",
                    )

            # ---- 受控区块标记（§6.1）----
            effective_base = current_prompt
            if job.edit_mode == EditMode.MANAGED_BLOCK.value:
                split = split_managed_block(current_prompt)
                if split.status == MARKERS_INVALID:
                    self._storage.update_job(
                        job.id, {"status": JobStatus.CONFLICT.value}
                    )
                    return fail(
                        ErrorCode.MARKER_INVALID,
                        "受控区块标记重复/单侧缺失/嵌套，Job 已转 conflict",
                    )
                if split.status == MARKERS_ABSENT:
                    effective_base = append_managed_block(current_prompt)

            # ---- 抽样（§7.4）----
            samples = self._storage.fetch_samples(
                job.source_group_ids, job.source_user_ids, since_id
            )
            selected = stratified_sample(
                samples,
                max_count=int(self._cfg("persona_evolution_analysis_sample_size", 60)),
                group_max_ratio=float(
                    self._cfg("persona_evolution_sample_group_max_ratio", 0.35)
                ),
                user_max_ratio=float(
                    self._cfg("persona_evolution_sample_user_max_ratio", 0.20)
                ),
                single_group_scope=len(job.source_group_ids) == 1,
                single_user_scope=len(job.source_user_ids) == 1,
            )
            prompt_chars = int(self._cfg("persona_evolution_sample_prompt_chars", 240))
            corpus_texts = [
                (s.get("normalized_text") or "")[:prompt_chars] for s in selected
            ]
            self._storage.update_run(
                run_id,
                {
                    "eligible_count": eligible,
                    "selected_count": len(selected),
                    "sample_cursor_to": max((s.get("id") or 0 for s in samples), default=0),
                },
            )

            llm_manager = self._get_llm_manager()
            if llm_manager is None:
                return fail(ErrorCode.PROVIDER_ERROR, "LLMManager 不可用")

            goal_snapshot = build_goal_snapshot(job.goal_preset_id, job.custom_goal)

            # ---- 阶段 A：风格归纳（LLM await 在写锁外）----
            analyzer = StyleAnalyzer(
                min_confidence=float(self._cfg("persona_evolution_min_confidence", 0.65))
            )
            profile, a_err = await analyzer.analyze(
                llm_manager,
                corpus_texts,
                str(goal_snapshot.get("text") or ""),
                provider_id=job.provider_id or None,
            )
            if a_err is not None:
                return self._handle_stage_failure(
                    job, run_id, a_err, "阶段 A 风格归纳", trigger_type, result
                )

            # ---- 阶段 B：候选生成 ----
            generator = CandidateGenerator()
            generation, b_err = await generator.generate(
                llm_manager,
                current_prompt=effective_base,
                edit_mode=job.edit_mode,
                goal_snapshot=goal_snapshot,
                style_profile=profile,
                protected_fragments=job.protected_fragments,
                block_max_chars=int(self._cfg("persona_evolution_block_max_chars", 1500)),
                max_change_ratio=float(
                    self._cfg("persona_evolution_full_prompt_max_change_ratio", 0.20)
                ),
                full_max_growth_ratio=float(
                    self._cfg("persona_evolution_full_max_growth_ratio", 1.25)
                ),
                full_max_length=int(self._cfg("persona_evolution_full_max_length", 20000)),
                provider_id=job.provider_id or None,
            )
            if b_err is not None:
                return self._handle_stage_failure(
                    job, run_id, b_err, "阶段 B 候选生成", trigger_type, result
                )

            candidate_prompt = generation["candidate_prompt"]

            # ---- 阶段 C：完整人格独立审查（仅 full_prompt）----
            review_dict: Dict[str, Any] = {}
            if job.edit_mode == EditMode.FULL_PROMPT.value:
                reviewer = PromptReviewer()
                review, c_err = await reviewer.review(
                    llm_manager,
                    base_prompt=effective_base,
                    candidate_prompt=candidate_prompt,
                    goal_snapshot=goal_snapshot,
                    style_profile=profile,
                    provider_id=job.reviewer_provider_id or job.provider_id or None,
                )
                review_dict = review or {}
                if c_err is not None:
                    revision_id = self._save_candidate_revision(
                        job, trigger_type, effective_base, generation,
                        goal_snapshot, profile, current_hash, selected,
                        status=RevisionStatus.FAILED_VALIDATION,
                        review=review_dict,
                    )
                    result["revision_id"] = revision_id
                    return self._handle_stage_failure(
                        job, run_id, c_err, "阶段 C 独立审查", trigger_type, result
                    )

            # ---- 存 candidate Revision（校验/发布的事实来源）----
            revision_id = self._save_candidate_revision(
                job, trigger_type, effective_base, generation,
                goal_snapshot, profile, current_hash, selected,
                review=review_dict,
            )
            result["revision_id"] = revision_id

            # ---- 确定性发布闸门（§10）----
            known_names = sorted(
                {s.get("user_name") or "" for s in samples if s.get("user_name")}
            )
            outcome = validate_candidate(
                candidate_prompt=candidate_prompt,
                job=job,
                persona_id=job.persona_id,
                base_prompt=effective_base,
                base_hash=current_hash,
                current_prompt=current_prompt,
                corpus_texts=corpus_texts,
                known_user_names=known_names,
                block_max_chars=int(self._cfg("persona_evolution_block_max_chars", 1500)),
                full_max_change_ratio=float(
                    self._cfg("persona_evolution_full_prompt_max_change_ratio", 0.20)
                ),
                full_max_growth_ratio=float(
                    self._cfg("persona_evolution_full_max_growth_ratio", 1.25)
                ),
                full_max_length=int(self._cfg("persona_evolution_full_max_length", 20000)),
                max_reuse_chars=int(self._cfg("persona_evolution_max_reuse_chars", 16)),
            )
            self._storage.update_revision(
                revision_id, {"validation": outcome.to_snapshot()}
            )

            if outcome.no_change:
                self._storage.update_revision(
                    revision_id, {"status": RevisionStatus.NO_CHANGE.value}
                )
                self._advance_success_baseline(job, samples)
                self._finish_run(run_id, RunStatus.SUCCESS)
                result.update(ok=True, no_change=True, message="候选与当前一致，未发布")
                return result

            if not outcome.passed:
                # 校验失败候选保留 failed_validation 历史，同一候选永不自动重试
                first = outcome.failures[0]
                self._storage.update_revision(
                    revision_id, {"status": RevisionStatus.FAILED_VALIDATION.value}
                )
                return self._handle_circuit_failure(
                    job, run_id, first.code, first.message, result
                )

            # ---- 审批模式 ----
            if job.approval_mode == ApprovalMode.MANUAL.value:
                # 停在 candidate 待审批；推进游标但不刷新成功冷却（未发布）
                self._advance_cursor_only(job, samples)
                self._finish_run(run_id, RunStatus.SUCCESS)
                result.update(ok=True, message="候选已生成，等待管理员审批")
                return result

            # ---- 自动发布（§11.1）----
            revision = self._storage.get_revision(revision_id)
            p_err = await self._publisher.publish(job, revision)
            if p_err is not None:
                if p_err == ErrorCode.BASE_HASH_MISMATCH:
                    # 生成后 Persona 被外部编辑：不覆盖，转冲突（§11.2/§12.1）
                    self._record_external_change(job, revision, current_prompt=None)
                    return fail(
                        ErrorCode.EXTERNAL_CHANGE,
                        "发布前检测到外部修改，Job 已转 conflict",
                    )
                return fail(p_err, f"发布失败：{p_err.value}")

            # ---- 成功：推进游标 + 刷新冷却（§8.3 失败才不动）----
            self._advance_success_baseline(job, samples)
            self._finish_run(run_id, RunStatus.SUCCESS)
            result.update(ok=True, message="已自动发布")
            return result

        except Exception as e:
            logger.error(f"Job {job_id} 迭代执行异常：{e}", exc_info=True)
            return fail(ErrorCode.INTERNAL_ERROR, f"内部错误：{e}")

    # ------------------------------------------------------------------
    # 失败处理：Provider 退避重试 / 解析校验熔断
    # ------------------------------------------------------------------

    def _handle_stage_failure(
        self,
        job: Any,
        run_id: int,
        code: ErrorCode,
        stage: str,
        trigger_type: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """阶段 A/B/C 失败分流：Provider 类走退避重试，解析/校验类走熔断"""
        if code in (ErrorCode.PROVIDER_ERROR, ErrorCode.PROVIDER_TIMEOUT):
            message = self._schedule_retry(job, trigger_type, stage)
            result["error_code"] = code.value
            result["message"] = f"{stage}失败：{message}"
            self._finish_run(run_id, RunStatus.FAILED, code.value, message)
            return result
        if code == ErrorCode.ANALYSIS_LOW_CONFIDENCE:
            # 低置信度是良性停轮：不推进游标，不计熔断
            result["error_code"] = code.value
            result["message"] = f"{stage}置信度不足，停止本轮"
            self._finish_run(
                run_id, RunStatus.FAILED, code.value, "置信度低于阈值，停止本轮"
            )
            return result
        return self._handle_circuit_failure(
            job, run_id, code, f"{stage}失败：{code.value}", result
        )

    def _handle_circuit_failure(
        self,
        job: Any,
        run_id: int,
        code: ErrorCode,
        message: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """解析/校验失败：累计连续失败，达阈值熔断转 paused_error"""
        failures = job.consecutive_failures + 1
        threshold = int(self._cfg("persona_evolution_circuit_breaker_threshold", 3))
        fields: Dict[str, Any] = {"consecutive_failures": failures}
        if failures >= threshold:
            fields["status"] = JobStatus.PAUSED_ERROR.value
            message += f"；连续失败 {failures} 次，已熔断转 paused_error"
            logger.warning(f"Job {job.id} {message}")
        self._storage.update_job(job.id, fields)
        self._cancel_retry(job.id)
        result["error_code"] = code.value
        result["message"] = message
        self._finish_run(run_id, RunStatus.FAILED, code.value, message)
        return result

    def _schedule_retry(self, job: Any, trigger_type: str, stage: str) -> str:
        """Provider/网络失败退避重试（30/120/360 分钟，默认最多 3 次）

        失败不推进成功语料游标，也不刷新 24 小时成功冷却。
        """
        intervals = self._retry_intervals_minutes()
        attempts = self._retry_attempts.get(job.id, 0) + 1
        self._retry_attempts[job.id] = attempts
        if attempts > len(intervals):
            self._retry_attempts.pop(job.id, None)
            return (
                f"Provider 调用失败，已达最大自动重试次数 {len(intervals)}，"
                "等待下次触发"
            )
        delay_minutes = intervals[attempts - 1]
        self._cancel_retry(job.id)
        self._retry_tasks[job.id] = asyncio.create_task(
            self._retry_later(job.id, delay_minutes * 60, trigger_type),
            name=f"persona_evolution_retry_{job.id}",
        )
        return f"Provider 调用失败，将在 {delay_minutes:g} 分钟后第 {attempts} 次重试"

    async def _retry_later(self, job_id: int, delay_seconds: float, trigger_type: str) -> None:
        """退避等待后重新执行（不经过消息链，不占调度队列）"""
        try:
            await asyncio.sleep(delay_seconds)
            self._retry_tasks.pop(job_id, None)
            logger.info(f"Job {job_id} 退避重试开始")
            await self.run_job(job_id, trigger_type)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Job {job_id} 退避重试执行失败：{e}")

    def _cancel_retry(self, job_id: int) -> None:
        task = self._retry_tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()

    def cancel_pending_retries(self) -> None:
        """组件关闭时取消全部待重试任务"""
        for job_id in list(self._retry_tasks):
            self._cancel_retry(job_id)
        self._retry_attempts.clear()

    # ------------------------------------------------------------------
    # 外部修改与恢复
    # ------------------------------------------------------------------

    def _record_external_change(
        self, job: Any, reference_revision: Any, current_prompt: Optional[str]
    ) -> None:
        """外部修改处理（§12.1）：建 external_change Revision + Job 转 conflict

        Args:
            reference_revision: 作为基线参照的 Revision（最近 applied 或候选）
            current_prompt: 当前 Persona 快照；None 时发布路径已由
                Revision 记录基线，仅转 Job 状态
        """
        if current_prompt is not None:
            snapshot = PersonaRevision(
                job_id=job.id,
                version=0,
                parent_revision_id=job.last_applied_revision_id,
                status=RevisionStatus.EXTERNAL_CHANGE.value,
                trigger_type=TriggerType.AUTO.value,
                edit_mode=job.edit_mode,
                approval_mode=job.approval_mode,
                base_prompt=reference_revision.result_prompt,
                result_prompt=current_prompt,
                base_hash=reference_revision.result_hash,
                result_hash=persona_hash(current_prompt),
            )
            self._storage.create_revision(snapshot)
        self._storage.update_job(job.id, {"status": JobStatus.CONFLICT.value})
        self._cancel_retry(job.id)
        logger.warning(f"Job {job.id} 检测到外部修改，已转 conflict")

    async def reconcile_publishing(self) -> Dict[str, int]:
        """启动恢复对账（§12.2）：publishing 状态 Revision 三分支处理

        - 当前 Persona 哈希 == 候选哈希：补记 applied；
        - == base 哈希：补记 publish_failed（允许重试）；
        - 两者都不等：Revision 补记 publish_failed，Job 转 conflict。

        Returns:
            {applied, publish_failed, conflict, skipped} 计数
        """
        summary = {"applied": 0, "publish_failed": 0, "conflict": 0, "skipped": 0}
        try:
            revisions = self._storage.list_revisions_by_status(
                RevisionStatus.PUBLISHING.value
            )
        except Exception as e:
            logger.warning(f"恢复对账读取 publishing Revision 失败：{e}")
            return summary

        for revision in revisions:
            try:
                if self.is_job_running(revision.job_id):
                    # 发布进行中，留给本次发布/下次启动对账
                    summary["skipped"] += 1
                    continue
                job = self._storage.get_job(revision.job_id)
                if job is None:
                    summary["skipped"] += 1
                    continue
                current, err = await self._publisher.read_current_prompt(
                    job.persona_id
                )
                if err is not None or current is None:
                    summary["skipped"] += 1
                    continue
                current_hash = persona_hash(current)
                if revision.result_hash and current_hash == revision.result_hash:
                    self._storage.mark_revision_applied(
                        revision.id, job.id, time.time()
                    )
                    summary["applied"] += 1
                    logger.info(
                        f"恢复对账：Job {job.id} Revision v{revision.version}"
                        " 补记 applied"
                    )
                elif revision.base_hash and current_hash == revision.base_hash:
                    self._storage.update_revision(
                        revision.id,
                        {"status": RevisionStatus.PUBLISH_FAILED.value},
                    )
                    summary["publish_failed"] += 1
                    logger.info(
                        f"恢复对账：Job {job.id} Revision v{revision.version}"
                        " 补记 publish_failed（可重试）"
                    )
                else:
                    self._storage.update_revision(
                        revision.id,
                        {"status": RevisionStatus.PUBLISH_FAILED.value},
                    )
                    self._storage.update_job(
                        job.id, {"status": JobStatus.CONFLICT.value}
                    )
                    summary["conflict"] += 1
                    logger.warning(
                        f"恢复对账：Job {job.id} Revision v{revision.version}"
                        " 哈希两不等，Job 转 conflict"
                    )
            except Exception as e:
                summary["skipped"] += 1
                logger.warning(f"恢复对账处理 Revision {revision.id} 失败：{e}")
        return summary

    # ------------------------------------------------------------------
    # 审批 / 拒绝 / 回滚 / 冲突解决 / 暂停恢复（文档 §11.2/§12.1/§13.3/§8.3）
    # ------------------------------------------------------------------

    async def approve_revision(self, revision_id: int) -> Dict[str, Any]:
        """批准 candidate Revision（文档 §11.2）

        批准前重新执行全部确定性校验并重读 Persona 核 base_hash；
        候选生成后 Persona 被外部编辑则不能批准，Job 转 conflict。
        通过后走 publisher 发布流程。
        """
        result: Dict[str, Any] = {
            "ok": False,
            "revision_id": revision_id,
            "job_id": None,
            "error_code": None,
            "message": "",
        }
        revision = self._storage.get_revision(revision_id)
        if revision is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Revision {revision_id} 不存在"
            return result
        result["job_id"] = revision.job_id
        job = self._storage.get_job(revision.job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {revision.job_id} 不存在"
            return result
        if revision.status != RevisionStatus.CANDIDATE.value:
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = (
                f"Revision 状态为 {revision.status}，仅 candidate 可批准"
            )
            return result
        if job.status == JobStatus.CONFLICT.value:
            result["error_code"] = ErrorCode.CONFLICT_UNRESOLVED.value
            result["message"] = "存在未解决冲突，需先处理冲突"
            return result
        if job.status == JobStatus.PAUSED_ERROR.value:
            result["error_code"] = ErrorCode.CIRCUIT_OPEN.value
            result["message"] = "连续失败已熔断，需先恢复 Job"
            return result

        lock = self._run_lock(job.id)
        if lock.locked():
            result["error_code"] = ErrorCode.TRIGGER_CONDITIONS_NOT_MET.value
            result["message"] = "Job 有运行中的迭代，稍后再试"
            return result
        async with lock:
            # ---- 重读 Persona 核 base_hash（§11.2）----
            current, err = await self._publisher.read_current_prompt(job.persona_id)
            if err is not None or current is None:
                result["error_code"] = ErrorCode.PERSONA_NOT_FOUND.value
                result["message"] = f"Persona {job.persona_id} 不存在或不可用"
                return result
            current_hash = persona_hash(current)
            if revision.base_hash and current_hash != revision.base_hash:
                # 候选生成后 Persona 被外部编辑：不能批准，转 conflict
                self._record_external_change(job, revision, current)
                result["error_code"] = ErrorCode.EXTERNAL_CHANGE.value
                result["message"] = (
                    "检测到 Persona 被外部编辑，不能批准，Job 已转 conflict"
                )
                return result

            # ---- 重新执行全部确定性校验（§11.2）----
            sample_ids = self._storage.list_revision_sample_ids(revision.id)
            samples = self._storage.fetch_samples_by_ids(sample_ids)
            prompt_chars = int(self._cfg("persona_evolution_sample_prompt_chars", 240))
            corpus_texts = [
                (s.get("normalized_text") or "")[:prompt_chars] for s in samples
            ]
            known_names = sorted(
                {s.get("user_name") or "" for s in samples if s.get("user_name")}
            )
            outcome = validate_candidate(
                candidate_prompt=revision.result_prompt,
                job=job,
                persona_id=job.persona_id,
                base_prompt=revision.base_prompt or "",
                base_hash=revision.base_hash or "",
                current_prompt=current,
                corpus_texts=corpus_texts,
                known_user_names=known_names,
                block_max_chars=int(self._cfg("persona_evolution_block_max_chars", 1500)),
                full_max_change_ratio=float(
                    self._cfg("persona_evolution_full_prompt_max_change_ratio", 0.20)
                ),
                full_max_growth_ratio=float(
                    self._cfg("persona_evolution_full_max_growth_ratio", 1.25)
                ),
                full_max_length=int(self._cfg("persona_evolution_full_max_length", 20000)),
                max_reuse_chars=int(self._cfg("persona_evolution_max_reuse_chars", 16)),
            )
            self._storage.update_revision(
                revision.id, {"validation": outcome.to_snapshot()}
            )
            if outcome.no_change:
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.NO_CHANGE.value}
                )
                result.update(ok=True, no_change=True, message="候选与当前一致，无需发布")
                return result
            if not outcome.passed:
                # 管理员手动批准不触发熔断，仅更新状态与校验快照
                first = outcome.failures[0]
                self._storage.update_revision(
                    revision.id, {"status": RevisionStatus.FAILED_VALIDATION.value}
                )
                result["error_code"] = first.code.value
                result["message"] = f"复核未通过：{first.message}"
                return result

            # ---- 发布（发布锁内再次核 base_hash）----
            p_err = await self._publisher.publish(job, revision)
            if p_err is not None:
                if p_err == ErrorCode.BASE_HASH_MISMATCH:
                    self._record_external_change(job, revision, current_prompt=None)
                    result["error_code"] = ErrorCode.EXTERNAL_CHANGE.value
                    result["message"] = "发布前检测到外部修改，Job 已转 conflict"
                    return result
                result["error_code"] = p_err.value
                result["message"] = f"发布失败：{p_err.value}"
                return result

            result.update(ok=True, message=f"已批准并发布 v{revision.version}")
            logger.info(f"Job {job.id} Revision v{revision.version} 经管理员批准发布")
            return result

    def reject_revision(self, revision_id: int, reason: str = "") -> Dict[str, Any]:
        """拒绝 candidate Revision（文档 §11.2）：保存理由，状态 rejected"""
        result: Dict[str, Any] = {
            "ok": False,
            "revision_id": revision_id,
            "job_id": None,
            "error_code": None,
            "message": "",
        }
        revision = self._storage.get_revision(revision_id)
        if revision is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Revision {revision_id} 不存在"
            return result
        result["job_id"] = revision.job_id
        if revision.status != RevisionStatus.CANDIDATE.value:
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = (
                f"Revision 状态为 {revision.status}，仅 candidate 可拒绝"
            )
            return result
        reason = (reason or "").strip()[:500]
        self._storage.update_revision(
            revision.id,
            {
                "status": RevisionStatus.REJECTED.value,
                "decision_reason": reason or "管理员拒绝",
            },
        )
        result.update(ok=True, message=f"已拒绝 v{revision.version}")
        logger.info(f"Revision {revision_id} 被管理员拒绝：{reason[:50]}")
        return result

    async def rollback_to_revision(self, revision_id: int) -> Dict[str, Any]:
        """回滚到指定已发布 Revision（文档 §13.3 git revert 语义）

        以当前版本为父新建 rollback Revision（修改前=当前 Persona 快照、
        修改后=目标内容），执行哈希冲突检查与发布；成功后刷新 24h 冷却
        （防自动任务立刻改回），不动语料游标。冲突状态下回滚是文档
        §12.1 的冲突解决路径之一，成功后 Job 恢复 active。
        不提供破坏性删除版本。
        """
        result: Dict[str, Any] = {
            "ok": False,
            "revision_id": None,
            "job_id": None,
            "error_code": None,
            "message": "",
            "no_change": False,
        }
        target = self._storage.get_revision(revision_id)
        if target is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Revision {revision_id} 不存在"
            return result
        result["job_id"] = target.job_id
        job = self._storage.get_job(target.job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {target.job_id} 不存在"
            return result
        if (
            target.status
            not in (RevisionStatus.APPLIED.value, RevisionStatus.ROLLBACK.value)
            or not target.result_prompt
            or not target.result_hash
        ):
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = (
                f"Revision 状态为 {target.status}，仅可回滚到已发布版本"
            )
            return result

        lock = self._run_lock(job.id)
        if lock.locked():
            result["error_code"] = ErrorCode.TRIGGER_CONDITIONS_NOT_MET.value
            result["message"] = "Job 有运行中的迭代，稍后再试"
            return result
        async with lock:
            current, err = await self._publisher.read_current_prompt(job.persona_id)
            if err is not None or current is None:
                result["error_code"] = ErrorCode.PERSONA_NOT_FOUND.value
                result["message"] = f"Persona {job.persona_id} 不存在或不可用"
                return result
            current_hash = persona_hash(current)
            if current_hash == target.result_hash:
                result["error_code"] = ErrorCode.NO_CHANGE.value
                result["message"] = "目标版本与当前内容一致，无需回滚"
                result["no_change"] = True
                return result

            # ---- 哈希冲突检查（§13.3-4）：非冲突状态当前须等于发布基线 ----
            if job.status != JobStatus.CONFLICT.value and job.last_applied_revision_id:
                baseline = self._storage.get_revision(job.last_applied_revision_id)
                if (
                    baseline is not None
                    and baseline.result_hash
                    and current_hash != baseline.result_hash
                ):
                    self._record_external_change(job, baseline, current)
                    result["error_code"] = ErrorCode.EXTERNAL_CHANGE.value
                    result["message"] = (
                        "检测到 Persona 被外部修改，Job 已转 conflict，"
                        "请确认后再次执行回滚"
                    )
                    return result

            # ---- 以当前版本为父新建 rollback Revision（§13.3-2/3）----
            reason = f"回滚到 v{target.version}"
            rollback = PersonaRevision(
                job_id=job.id,
                version=0,
                parent_revision_id=job.last_applied_revision_id,
                status=RevisionStatus.CANDIDATE.value,
                trigger_type=TriggerType.ROLLBACK.value,
                edit_mode=job.edit_mode,
                approval_mode=job.approval_mode,
                base_prompt=current,
                result_prompt=target.result_prompt,
                base_hash=current_hash,
                result_hash=target.result_hash,
                rationale=reason,
                decision_reason=reason,
            )
            new_id = self._storage.create_revision(rollback)
            result["revision_id"] = new_id
            new_revision = self._storage.get_revision(new_id)

            p_err = await self._publisher.publish(
                job, new_revision, final_status=RevisionStatus.ROLLBACK
            )
            if p_err is not None:
                if p_err == ErrorCode.BASE_HASH_MISMATCH:
                    self._record_external_change(job, new_revision, current_prompt=None)
                    result["error_code"] = ErrorCode.EXTERNAL_CHANGE.value
                    result["message"] = "发布前检测到外部修改，Job 已转 conflict"
                    return result
                result["error_code"] = p_err.value
                result["message"] = f"回滚发布失败：{p_err.value}"
                return result

            # 成功：mark_revision_applied 已刷新 24h 冷却（§13.3-5）；
            # 语料游标不动（§13.3-6）；冲突经回滚解决后恢复 active（§12.1）
            fields: Dict[str, Any] = {"consecutive_failures": 0}
            if job.status == JobStatus.CONFLICT.value:
                fields["status"] = JobStatus.ACTIVE.value
            self._storage.update_job(job.id, fields)
            self._cancel_retry(job.id)
            result.update(
                ok=True,
                message=f"已回滚到 v{target.version}（生成新版本 v{new_revision.version}）",
            )
            logger.info(
                f"Job {job.id} 回滚到 v{target.version}，"
                f"新 rollback Revision v{new_revision.version}"
            )
            return result

    async def adopt_current_for_conflict(self, job_id: int) -> Dict[str, Any]:
        """冲突解决：采纳外部版本为新基线（文档 §12.1）

        把当前 Persona 快照存为新基线 Revision（不调用 PersonaManager
        写入），Job 恢复 active。
        """
        result: Dict[str, Any] = {
            "ok": False,
            "revision_id": None,
            "job_id": job_id,
            "error_code": None,
            "message": "",
        }
        job = self._storage.get_job(job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {job_id} 不存在"
            return result
        if job.status != JobStatus.CONFLICT.value:
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = f"Job 状态为 {job.status}，仅 conflict 可采纳基线"
            return result

        lock = self._run_lock(job.id)
        if lock.locked():
            result["error_code"] = ErrorCode.TRIGGER_CONDITIONS_NOT_MET.value
            result["message"] = "Job 有运行中的迭代，稍后再试"
            return result
        async with lock:
            current, err = await self._publisher.read_current_prompt(job.persona_id)
            if err is not None or current is None:
                result["error_code"] = ErrorCode.PERSONA_NOT_FOUND.value
                result["message"] = f"Persona {job.persona_id} 不存在或不可用"
                return result
            current_hash = persona_hash(current)
            baseline = PersonaRevision(
                job_id=job.id,
                version=0,
                parent_revision_id=job.last_applied_revision_id,
                status=RevisionStatus.CANDIDATE.value,
                trigger_type=TriggerType.MANUAL.value,
                edit_mode=job.edit_mode,
                approval_mode=job.approval_mode,
                base_prompt=current,
                result_prompt=current,
                base_hash=current_hash,
                result_hash=current_hash,
                rationale="采纳外部版本为新基线",
                decision_reason="采纳外部版本为新基线",
            )
            new_id = self._storage.create_revision(baseline)
            result["revision_id"] = new_id
            # 原子置 applied：新基线 + 刷新成功冷却，内容已在 Persona 中
            self._storage.mark_revision_applied(new_id, job.id, time.time())
            self._storage.update_job(
                job.id,
                {
                    "status": JobStatus.ACTIVE.value,
                    "consecutive_failures": 0,
                },
            )
            self._cancel_retry(job.id)
            new_revision = self._storage.get_revision(new_id)
            result.update(
                ok=True,
                message=f"已采纳当前 Persona 为新基线 v{new_revision.version}，Job 恢复运行",
            )
            logger.info(f"Job {job.id} 采纳外部版本为新基线 v{new_revision.version}")
            return result

    def pause_job(self, job_id: int) -> Dict[str, Any]:
        """暂停 Job 自动迭代（管理员操作）"""
        result: Dict[str, Any] = {
            "ok": False, "job_id": job_id, "error_code": None, "message": "",
        }
        job = self._storage.get_job(job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {job_id} 不存在"
            return result
        if job.status != JobStatus.ACTIVE.value:
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = f"Job 状态为 {job.status}，仅 active 可暂停"
            return result
        self._storage.update_job(job.id, {"status": JobStatus.PAUSED.value})
        self._cancel_retry(job.id)
        result.update(ok=True, message=f"Job {job_id} 已暂停")
        logger.info(f"Job {job_id} 被管理员暂停")
        return result

    def resume_job(self, job_id: int) -> Dict[str, Any]:
        """恢复 Job（文档 §8.3：管理员查看原因后恢复）

        paused → active；paused_error → active 并清零连续失败；
        conflict 不可直接恢复，需先处理冲突。
        """
        result: Dict[str, Any] = {
            "ok": False, "job_id": job_id, "error_code": None, "message": "",
        }
        job = self._storage.get_job(job_id)
        if job is None:
            result["error_code"] = ErrorCode.NOT_FOUND.value
            result["message"] = f"Job {job_id} 不存在"
            return result
        if job.status == JobStatus.PAUSED.value:
            self._storage.update_job(job.id, {"status": JobStatus.ACTIVE.value})
            result.update(ok=True, message=f"Job {job_id} 已恢复运行")
        elif job.status == JobStatus.PAUSED_ERROR.value:
            self._storage.update_job(
                job.id,
                {"status": JobStatus.ACTIVE.value, "consecutive_failures": 0},
            )
            result.update(ok=True, message=f"Job {job_id} 已恢复运行（连续失败已清零）")
        elif job.status == JobStatus.CONFLICT.value:
            result["error_code"] = ErrorCode.CONFLICT_UNRESOLVED.value
            result["message"] = "存在未解决冲突，请先采纳基线或回滚后再恢复"
            return result
        else:
            result["error_code"] = ErrorCode.INVALID_PARAMS.value
            result["message"] = f"Job 状态为 {job.status}，无需恢复"
            return result
        logger.info(f"Job {job_id} 被管理员恢复")
        return result

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _save_candidate_revision(
        self,
        job: Any,
        trigger_type: str,
        effective_base: str,
        generation: Dict[str, Any],
        goal_snapshot: Dict[str, Any],
        profile: Dict[str, Any],
        base_hash: str,
        selected_samples: List[Dict[str, Any]],
        status: RevisionStatus = RevisionStatus.CANDIDATE,
        review: Optional[Dict[str, Any]] = None,
    ) -> int:
        """落库 candidate/failed_validation Revision 与语料引用"""
        candidate = generation["candidate_prompt"]
        revision = PersonaRevision(
            job_id=job.id,
            version=0,
            parent_revision_id=job.last_applied_revision_id,
            status=status.value,
            trigger_type=trigger_type,
            edit_mode=job.edit_mode,
            approval_mode=job.approval_mode,
            base_prompt=effective_base,
            result_prompt=candidate,
            base_hash=base_hash,
            result_hash=persona_hash(candidate),
            goal_snapshot=goal_snapshot,
            style_profile=profile,
            change_summary=generation.get("change_summary", []),
            rationale=generation.get("rationale", ""),
            confidence=generation.get("confidence"),
            review=review or {},
            provider_snapshot={
                "provider_id": job.provider_id,
                "reviewer_provider_id": job.reviewer_provider_id,
            },
        )
        revision_id = self._storage.create_revision(revision)
        self._storage.insert_revision_samples(revision_id, selected_samples)
        return revision_id

    def _advance_success_baseline(
        self, job: Any, samples: List[Dict[str, Any]]
    ) -> None:
        """成功后推进语料游标并刷新成功冷却、清零连续失败

        （发布成功路径的 last_success_at 由 mark_revision_applied 写入，
        此处补齐游标；no_change 路径由本方法一并写冷却）
        """
        cursor = max((s.get("id") or 0 for s in samples), default=0)
        fields: Dict[str, Any] = {"consecutive_failures": 0}
        if cursor > job.last_sample_cursor:
            fields["last_sample_cursor"] = cursor
        latest = self._storage.get_job(job.id)
        if latest is not None and latest.last_success_at == job.last_success_at:
            # no_change 路径：发布流程未写冷却，这里刷新
            fields["last_success_at"] = time.time()
        self._storage.update_job(job.id, fields)
        self._retry_attempts.pop(job.id, None)

    def _advance_cursor_only(self, job: Any, samples: List[Dict[str, Any]]) -> None:
        """手动审批模式生成候选后推进游标（未发布不刷新成功冷却）"""
        cursor = max((s.get("id") or 0 for s in samples), default=0)
        fields: Dict[str, Any] = {"consecutive_failures": 0}
        if cursor > job.last_sample_cursor:
            fields["last_sample_cursor"] = cursor
        self._storage.update_job(job.id, fields)
        self._retry_attempts.pop(job.id, None)

    def _finish_run(
        self,
        run_id: int,
        status: RunStatus,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """收尾 Run 记录（永不抛出）"""
        try:
            self._storage.update_run(
                run_id,
                {
                    "status": status.value,
                    "finished_at": time.time(),
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
        except Exception as e:
            logger.warning(f"Run {run_id} 收尾写库失败：{e}")
