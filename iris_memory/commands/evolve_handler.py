"""
Iris Chat Memory - 人格自迭代指令处理器

处理人格自迭代的管理指令（/iris_mem evolve）：
- status [job_id]：查看迭代任务状态（默认列出全部 Job 摘要）；
- run <job_id>：立即执行一轮迭代（手动触发，跳过门槛要求 >=20 条）；
- pause <job_id>：暂停自动迭代；
- resume <job_id>：恢复迭代（含熔断恢复，conflict 需先处理冲突）；
- rollback <job_id> <revision_id>：回滚到指定已发布版本。

Job 创建、复杂过滤与 Diff 查看以 Web UI 为主（文档 §18）。
"""

import time
from typing import Optional, TYPE_CHECKING

from iris_memory.core import get_logger, get_component_manager
from iris_memory.persona_evolution import PersonaEvolutionComponent
from .base import CommandHandler, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("commands.evolve")

_UNAVAILABLE_MSG = "人格自迭代不可用或未启用（persona_evolution.enable）"

_JOB_STATUS_LABELS = {
    "active": "正常运行",
    "paused": "已暂停",
    "conflict": "冲突待处理",
    "paused_error": "错误熔断",
}

_REVISION_STATUS_LABELS = {
    "candidate": "待审批",
    "publishing": "发布中",
    "applied": "已发布",
    "rejected": "已拒绝",
    "failed_validation": "校验未过",
    "publish_failed": "发布失败",
    "external_change": "外部修改",
    "rollback": "回滚",
    "no_change": "无变化",
}

_TRIGGER_LABELS = {"auto": "自动", "manual": "手动", "rollback": "回滚"}

# status 详情列出的最近版本/运行条数
_STATUS_REVISION_LIMIT = 5
_STATUS_RUN_LIMIT = 3


def _fmt_ts(ts: Optional[float]) -> str:
    """时间戳格式化（None 显示为 -）"""
    if not ts:
        return "-"
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _parse_int(raw: Optional[str], label: str) -> tuple[Optional[int], Optional[str]]:
    """解析整数参数，失败返回错误消息"""
    if raw is None:
        return None, f"缺少参数：{label}"
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, f"{label} 必须是数字：{raw}"
    if value <= 0:
        return None, f"{label} 必须是正整数：{raw}"
    return value, None


class EvolutionCommandHandler(CommandHandler):
    """人格自迭代指令处理器

    支持的子指令：
    - status [job_id]: 查看迭代任务状态（默认）
    - run <job_id>: 立即执行一轮迭代
    - pause <job_id>: 暂停自动迭代
    - resume <job_id>: 恢复迭代
    - rollback <job_id> <revision_id>: 回滚到指定版本
    """

    @property
    def name(self) -> str:
        return "evolve"

    @property
    def description(self) -> str:
        return "人格自迭代管理（状态/执行/暂停/恢复/回滚）"

    @property
    def sub_commands(self) -> dict[str, str]:
        return {
            "status [job_id]": "查看迭代任务状态（默认列出全部）",
            "run <job_id>": "立即执行一轮迭代",
            "pause <job_id>": "暂停自动迭代",
            "resume <job_id>": "恢复迭代（含熔断恢复）",
            "rollback <job_id> <revision_id>": "回滚到指定已发布版本",
        }

    async def handle(
        self,
        event: "AstrMessageEvent",
        args: ParsedArgs,
        sub_command: Optional[str] = None,
    ) -> CommandResult:
        """处理人格自迭代指令"""
        if sub_command == "status" or sub_command is None:
            return await self._handle_status(args)
        elif sub_command == "run":
            return await self._handle_run(args)
        elif sub_command == "pause":
            return await self._handle_pause_resume(args, pause=True)
        elif sub_command == "resume":
            return await self._handle_pause_resume(args, pause=False)
        elif sub_command == "rollback":
            return await self._handle_rollback(args)
        elif sub_command == "help":
            return CommandResult(success=True, message=self.get_help_text())
        else:
            return CommandResult(
                success=False,
                message=f"未知的子指令: {sub_command}\n{self.get_help_text()}",
            )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get_component(self) -> Optional[PersonaEvolutionComponent]:
        """取人格自迭代组件（不可用返回 None）"""
        try:
            manager = get_component_manager()
        except RuntimeError:
            return None
        if not manager:
            return None
        component = manager.get_component("persona_evolution", PersonaEvolutionComponent)
        if not component or not component.is_available or not component.storage:
            return None
        return component

    async def _handle_status(self, args: ParsedArgs) -> CommandResult:
        """处理状态查询：无 job_id 列出全部摘要，有则给详情"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)
        storage = component.storage

        # raw_args 含子指令本身（["status", job_id?]），job_id 取下标 1
        raw_job_id = args.raw_args[1] if len(args.raw_args) > 1 else None
        if raw_job_id is None:
            jobs = storage.list_jobs()
            if not jobs:
                return CommandResult(
                    success=True,
                    message="🧬 人格自迭代暂无任务\n可在 Web 管理页创建迭代 Job",
                )
            lines = ["🧬 人格自迭代任务列表"]
            for job in jobs:
                total = storage.count_samples(
                    job.source_group_ids, job.source_user_ids, 0
                )
                new_count = storage.count_samples(
                    job.source_group_ids, job.source_user_ids, job.last_sample_cursor
                )
                status_label = _JOB_STATUS_LABELS.get(job.status, job.status)
                lines.append(
                    f"- #{job.id} {job.name or job.persona_id}"
                    f"（{job.persona_id}）｜{status_label}"
                    f"｜语料 {total}（新增 {new_count}）"
                    f"｜最近成功 {_fmt_ts(job.last_success_at)}"
                )
            return CommandResult(success=True, message="\n".join(lines))

        job_id, err = _parse_int(raw_job_id, "job_id")
        if err:
            return CommandResult(success=False, message=err)
        job = storage.get_job(job_id)
        if not job:
            return CommandResult(success=False, message=f"Job {job_id} 不存在")

        total = storage.count_samples(job.source_group_ids, job.source_user_ids, 0)
        new_count = storage.count_samples(
            job.source_group_ids, job.source_user_ids, job.last_sample_cursor
        )
        status_label = _JOB_STATUS_LABELS.get(job.status, job.status)
        lines = [
            f"🧬 Job #{job.id} {job.name or job.persona_id}",
            f"目标 Persona：{job.persona_id}",
            f"状态：{status_label}"
            + (f"（连续失败 {job.consecutive_failures} 次）" if job.consecutive_failures else ""),
            f"语料：匹配 {total} 条，新增 {new_count} 条"
            f"（自动门槛 {job.trigger_sample_count} 条 / 间隔 {job.min_interval_hours}h）",
            f"模式：{job.edit_mode} / 审批 {job.approval_mode}",
            f"最近成功：{_fmt_ts(job.last_success_at)}",
        ]

        revisions = storage.list_revisions(job.id, limit=_STATUS_REVISION_LIMIT)
        if revisions:
            lines.append("最近版本：")
            for rev in revisions:
                label = _REVISION_STATUS_LABELS.get(rev.status, rev.status)
                trigger = _TRIGGER_LABELS.get(rev.trigger_type, rev.trigger_type)
                lines.append(
                    f"- v{rev.version}（#{rev.id}）{label}｜{trigger}"
                    f"｜{_fmt_ts(rev.created_at)}"
                )

        runs = storage.list_runs(job.id, limit=_STATUS_RUN_LIMIT)
        if runs:
            lines.append("最近运行：")
            for run in runs:
                if run.status == "success":
                    lines.append(
                        f"- {_fmt_ts(run.started_at)} 成功"
                        f"（语料 {run.eligible_count} 选 {run.selected_count}）"
                    )
                else:
                    lines.append(
                        f"- {_fmt_ts(run.started_at)} 失败"
                        f"｜{run.error_code or '-'}：{(run.error_message or '')[:40]}"
                    )
        return CommandResult(success=True, message="\n".join(lines))

    async def _handle_run(self, args: ParsedArgs) -> CommandResult:
        """处理立即迭代：run <job_id>"""
        component = self._get_component()
        if not component:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)
        raw_job_id = args.raw_args[1] if len(args.raw_args) > 1 else None
        job_id, err = _parse_int(raw_job_id, "job_id")
        if err:
            return CommandResult(
                success=False, message=f"{err}\n用法: iris_mem evolve run <job_id>"
            )

        result = await component.run_job(job_id, "manual")
        if result.get("ok"):
            message = f"✅ Job {job_id} 迭代完成：{result.get('message', '')}"
            if result.get("revision_id"):
                message += f"（Revision #{result['revision_id']}）"
            return CommandResult(success=True, message=message, details=result)
        return CommandResult(
            success=False,
            message=f"❌ Job {job_id} 迭代失败"
            f"｜{result.get('error_code') or '-'}：{result.get('message', '')}",
            details=result,
        )

    async def _handle_pause_resume(self, args: ParsedArgs, pause: bool) -> CommandResult:
        """处理暂停/恢复：pause|resume <job_id>"""
        component = self._get_component()
        if not component or not component.service:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)
        action = "pause" if pause else "resume"
        raw_job_id = args.raw_args[1] if len(args.raw_args) > 1 else None
        job_id, err = _parse_int(raw_job_id, "job_id")
        if err:
            return CommandResult(
                success=False,
                message=f"{err}\n用法: iris_mem evolve {action} <job_id>",
            )

        service = component.service
        result = service.pause_job(job_id) if pause else service.resume_job(job_id)
        if result.get("ok"):
            return CommandResult(
                success=True, message=f"✅ {result['message']}", details=result
            )
        return CommandResult(
            success=False,
            message=f"❌ {result.get('message', '')}",
            details=result,
        )

    async def _handle_rollback(self, args: ParsedArgs) -> CommandResult:
        """处理回滚：rollback <job_id> <revision_id>"""
        component = self._get_component()
        if not component or not component.service:
            return CommandResult(success=False, message=_UNAVAILABLE_MSG)
        # raw_args 含子指令本身（["rollback", job_id, revision_id]）
        raw_job_id = args.raw_args[1] if len(args.raw_args) > 1 else None
        raw_rev_id = args.raw_args[2] if len(args.raw_args) > 2 else None
        job_id, err = _parse_int(raw_job_id, "job_id")
        if err:
            return CommandResult(
                success=False,
                message=f"{err}\n用法: iris_mem evolve rollback <job_id> <revision_id>",
            )
        revision_id, err = _parse_int(raw_rev_id, "revision_id")
        if err:
            return CommandResult(
                success=False,
                message=f"{err}\n用法: iris_mem evolve rollback <job_id> <revision_id>",
            )

        revision = component.storage.get_revision(revision_id)
        if not revision or revision.job_id != job_id:
            return CommandResult(
                success=False,
                message=f"Revision {revision_id} 不存在或不属于 Job {job_id}",
            )

        result = await component.service.rollback_to_revision(revision_id)
        if result.get("ok"):
            return CommandResult(
                success=True, message=f"✅ {result['message']}", details=result
            )
        return CommandResult(
            success=False,
            message=f"❌ 回滚失败｜{result.get('error_code') or '-'}："
            f"{result.get('message', '')}",
            details=result,
        )
