"""
人格自迭代管理 API 路由（文档 §17）

统一前缀 /astrbot_plugin_iris_memory/persona-evolution：
- Persona：列表（default 标记为不可直接迭代）、克隆 default；
- Job：CRUD、暂停/恢复/立即迭代、冲突采纳基线；
- Revision：时间线查询、批准/拒绝/回滚；
- 语料：分布统计（不返回原文）、按群/用户/全部清除；
- 数据：独立导出（默认不含语料原文，include_samples=true 含脱敏语料）
  与导入（导入 Revision 历史绝不自动修改 AstrBot Persona，§19）。

服务端重新校验所有 ID/枚举/长度/状态；错误响应含稳定 error_code +
人类可读信息；内部错误不暴露堆栈。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from quart import Response, jsonify, request

from iris_memory.core import get_component_manager, get_logger
from iris_memory.persona_evolution import PersonaEvolutionComponent
from iris_memory.persona_evolution.goals import (
    CUSTOM_PRESET_ID,
    GOAL_PRESETS,
    list_goal_presets,
)
from iris_memory.persona_evolution.models import (
    ApprovalMode,
    EditMode,
    ErrorCode,
    EvolutionJob,
    RevisionStatus,
)

logger = get_logger("web.persona_evolution")

PLUGIN_NAME = "astrbot_plugin_iris_memory"

DEBUG_MODE = os.environ.get("IRIS_DEBUG", "").lower() in ("true", "1", "yes")

# 不可直接迭代的内置 Persona（§2.1，需先克隆为具名 Persona）
DEFAULT_PERSONA_ID = "default"

# 字段长度上限
_MAX_NAME_LEN = 64
_MAX_ID_LEN = 128
_MAX_GOAL_LEN = 500
_MAX_SCOPE_ITEMS = 100
_MAX_SCOPE_ID_LEN = 64
_MAX_FRAGMENTS = 20
_MAX_FRAGMENT_LEN = 200
_MAX_REASON_LEN = 500

# service 结果 error_code → HTTP 状态
_ERROR_HTTP_STATUS = {
    ErrorCode.NOT_FOUND.value: 404,
    ErrorCode.EXTERNAL_CHANGE.value: 409,
    ErrorCode.CONFLICT_UNRESOLVED.value: 409,
    ErrorCode.JOB_ALREADY_EXISTS.value: 409,
    ErrorCode.CIRCUIT_OPEN.value: 409,
}

# PUT /jobs/<id> 允许更新的字段
_JOB_EDITABLE_FIELDS = (
    "name",
    "goal_preset_id",
    "custom_goal",
    "source_group_ids",
    "source_user_ids",
    "edit_mode",
    "approval_mode",
    "trigger_sample_count",
    "min_interval_hours",
    "provider_id",
    "reviewer_provider_id",
    "protected_fragments",
)


# ----------------------------------------------------------------------
# 通用辅助
# ----------------------------------------------------------------------


def get_pe_component() -> Tuple[Optional[Any], Optional[Tuple]]:
    """取人格自迭代组件（不可用返回 503 错误响应）"""
    manager = get_component_manager()
    component = manager.get_component("persona_evolution", PersonaEvolutionComponent)
    if not component or not component.is_available or not component.storage:
        return None, (
            jsonify(
                {
                    "success": False,
                    "error": "人格自迭代不可用或未启用",
                    "error_code": ErrorCode.INTERNAL_ERROR.value,
                }
            ),
            503,
        )
    return component, None


def handle_exception(e: Exception, operation: str) -> Tuple[Any, int]:
    logger.error(f"{operation}失败：{e}", exc_info=True)
    error_msg = str(e) if DEBUG_MODE else "内部错误，详见服务日志"
    return (
        jsonify(
            {
                "success": False,
                "error": error_msg,
                "error_code": ErrorCode.INTERNAL_ERROR.value,
            }
        ),
        500,
    )


def _bad_request(msg: str, code: str = ErrorCode.INVALID_PARAMS.value) -> Tuple[Any, int]:
    return jsonify({"success": False, "error": msg, "error_code": code}), 400


def _not_found(msg: str) -> Tuple[Any, int]:
    return (
        jsonify(
            {"success": False, "error": msg, "error_code": ErrorCode.NOT_FOUND.value}
        ),
        404,
    )


def _service_response(result: Dict[str, Any]) -> Tuple[Any, int]:
    """把 service 结果字典映射为统一响应（含稳定 error_code）"""
    if result.get("ok"):
        return jsonify(
            {
                "success": True,
                "message": result.get("message", ""),
                "job_id": result.get("job_id"),
                "revision_id": result.get("revision_id"),
                "no_change": result.get("no_change", False),
            }
        )
    code = result.get("error_code") or ErrorCode.INTERNAL_ERROR.value
    status = _ERROR_HTTP_STATUS.get(code, 400)
    return (
        jsonify(
            {
                "success": False,
                "error": result.get("message", ""),
                "error_code": code,
                "job_id": result.get("job_id"),
                "revision_id": result.get("revision_id"),
            }
        ),
        status,
    )


# ----------------------------------------------------------------------
# 参数校验（服务端不信任前端，全部重新校验）
# ----------------------------------------------------------------------


def _require_str(value: Any, field: str, max_len: int = _MAX_ID_LEN) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串")
    value = value.strip()
    if not value:
        raise ValueError(f"缺少必填字段：{field}")
    if len(value) > max_len:
        raise ValueError(f"{field} 超长（最大 {max_len} 字符）")
    return value


def _optional_str(value: Any, field: str, max_len: int = _MAX_ID_LEN) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串")
    value = value.strip()
    if len(value) > max_len:
        raise ValueError(f"{field} 超长（最大 {max_len} 字符）")
    return value


def _str_list(
    value: Any,
    field: str,
    max_items: int = _MAX_SCOPE_ITEMS,
    max_len: int = _MAX_SCOPE_ID_LEN,
) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须是数组")
    if len(value) > max_items:
        raise ValueError(f"{field} 条目过多（最大 {max_items} 条）")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} 元素必须是字符串")
        item = item.strip()
        if not item or len(item) > max_len:
            raise ValueError(f"{field} 含空值或超长元素（最大 {max_len} 字符）")
        result.append(item)
    return result


def _enum(value: Any, field: str, allowed: List[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{field} 非法：{value}（允许：{', '.join(allowed)}）")
    return value


def _int_range(value: Any, field: str, lo: int, hi: int, default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 必须是整数") from None
    if not (lo <= number <= hi):
        raise ValueError(f"{field} 超出范围（{lo}~{hi}）")
    return number


def _validate_goal(goal_preset_id: Any, custom_goal: Any) -> Tuple[str, str]:
    """校验目标预设与自定义目标"""
    preset = _optional_str(goal_preset_id, "goal_preset_id") or "natural"
    known = list(GOAL_PRESETS) + [CUSTOM_PRESET_ID]
    _enum(preset, "goal_preset_id", known)
    goal = _optional_str(custom_goal, "custom_goal", _MAX_GOAL_LEN)
    if preset == CUSTOM_PRESET_ID and not goal:
        raise ValueError("自定义目标 preset=custom 时 custom_goal 必填")
    return preset, goal


def _validate_edit_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """校验 Job 编辑字段（POST 创建与 PUT 更新共用）

    只校验 data 中存在的键；返回可直接传给 storage 的字段字典。
    """
    fields: Dict[str, Any] = {}
    if "name" in data:
        fields["name"] = _optional_str(data.get("name"), "name", _MAX_NAME_LEN)
    if "goal_preset_id" in data or "custom_goal" in data:
        preset, goal = _validate_goal(data.get("goal_preset_id"), data.get("custom_goal"))
        fields["goal_preset_id"] = preset
        fields["custom_goal"] = goal
    if "source_group_ids" in data:
        fields["source_group_ids"] = _str_list(data.get("source_group_ids"), "source_group_ids")
    if "source_user_ids" in data:
        fields["source_user_ids"] = _str_list(data.get("source_user_ids"), "source_user_ids")
    if "edit_mode" in data:
        fields["edit_mode"] = _enum(
            data.get("edit_mode"), "edit_mode", [e.value for e in EditMode]
        )
    if "approval_mode" in data:
        fields["approval_mode"] = _enum(
            data.get("approval_mode"), "approval_mode", [e.value for e in ApprovalMode]
        )
    if "trigger_sample_count" in data:
        fields["trigger_sample_count"] = _int_range(
            data.get("trigger_sample_count"), "trigger_sample_count", 1, 100000, 100
        )
    if "min_interval_hours" in data:
        fields["min_interval_hours"] = _int_range(
            data.get("min_interval_hours"), "min_interval_hours", 1, 720, 24
        )
    if "provider_id" in data:
        fields["provider_id"] = _optional_str(data.get("provider_id"), "provider_id")
    if "reviewer_provider_id" in data:
        fields["reviewer_provider_id"] = _optional_str(
            data.get("reviewer_provider_id"), "reviewer_provider_id"
        )
    if "protected_fragments" in data:
        fields["protected_fragments"] = _str_list(
            data.get("protected_fragments"),
            "protected_fragments",
            max_items=_MAX_FRAGMENTS,
            max_len=_MAX_FRAGMENT_LEN,
        )
    return fields


# ----------------------------------------------------------------------
# 序列化
# ----------------------------------------------------------------------


def job_to_dict(job: Any, storage: Any = None) -> Dict[str, Any]:
    """Job 序列化（可选附带语料计数）"""
    result = {
        "id": job.id,
        "persona_id": job.persona_id,
        "name": job.name,
        "goal_preset_id": job.goal_preset_id,
        "custom_goal": job.custom_goal,
        "source_group_ids": job.source_group_ids,
        "source_user_ids": job.source_user_ids,
        "edit_mode": job.edit_mode,
        "approval_mode": job.approval_mode,
        "status": job.status,
        "trigger_sample_count": job.trigger_sample_count,
        "min_interval_hours": job.min_interval_hours,
        "provider_id": job.provider_id,
        "reviewer_provider_id": job.reviewer_provider_id,
        "protected_fragments": job.protected_fragments,
        "last_success_at": job.last_success_at,
        "last_sample_cursor": job.last_sample_cursor,
        "last_applied_revision_id": job.last_applied_revision_id,
        "consecutive_failures": job.consecutive_failures,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if storage is not None:
        result["sample_total"] = storage.count_samples(
            job.source_group_ids, job.source_user_ids, 0
        )
        result["sample_new"] = storage.count_samples(
            job.source_group_ids, job.source_user_ids, job.last_sample_cursor
        )
    return result


def run_to_dict(run: Any) -> Dict[str, Any]:
    """Run 序列化"""
    return {
        "id": run.id,
        "job_id": run.job_id,
        "trigger_type": run.trigger_type,
        "status": run.status,
        "sample_cursor_from": run.sample_cursor_from,
        "sample_cursor_to": run.sample_cursor_to,
        "eligible_count": run.eligible_count,
        "selected_count": run.selected_count,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error_code": run.error_code,
        "error_message": run.error_message,
        "analysis_tokens": run.analysis_tokens,
        "generation_tokens": run.generation_tokens,
        "review_tokens": run.review_tokens,
    }


def revision_to_dict(rev: Any, include_prompts: bool = True) -> Dict[str, Any]:
    """Revision 序列化（时间线/Diff 用，含完整快照）"""
    result = {
        "id": rev.id,
        "job_id": rev.job_id,
        "version": rev.version,
        "parent_revision_id": rev.parent_revision_id,
        "status": rev.status,
        "trigger_type": rev.trigger_type,
        "edit_mode": rev.edit_mode,
        "approval_mode": rev.approval_mode,
        "base_hash": rev.base_hash,
        "result_hash": rev.result_hash,
        "goal_snapshot": rev.goal_snapshot,
        "style_profile": rev.style_profile,
        "change_summary": rev.change_summary,
        "rationale": rev.rationale,
        "decision_reason": rev.decision_reason,
        "confidence": rev.confidence,
        "validation": rev.validation,
        "review": rev.review,
        "provider_snapshot": rev.provider_snapshot,
        "created_at": rev.created_at,
        "applied_at": rev.applied_at,
    }
    if include_prompts:
        result["base_prompt"] = rev.base_prompt
        result["result_prompt"] = rev.result_prompt
    else:
        result["base_prompt_length"] = len(rev.base_prompt or "")
        result["result_prompt_length"] = len(rev.result_prompt or "")
    return result


def _get_persona_manager(component: Any) -> Any:
    """从组件持有的 AstrBot Context 探测 PersonaManager"""
    return getattr(getattr(component, "context", None), "persona_manager", None)


# ----------------------------------------------------------------------
# Persona
# ----------------------------------------------------------------------


async def list_goals():
    """GET /goals：返回版本化目标预设，供 Web UI 动态渲染。"""
    return jsonify({"success": True, "goals": list_goal_presets()})


async def list_personas():
    """GET /personas：列出全部 Persona，标记 default 不可直接迭代"""
    try:
        component, error = get_pe_component()
        if error:
            return error

        persona_manager = _get_persona_manager(component)
        get_all = (
            getattr(persona_manager, "get_all_personas", None)
            if persona_manager is not None
            else None
        )
        if get_all is None:
            # PersonaManager 不可用：降级返回空列表，不报错
            return jsonify({"success": True, "personas": [], "degraded": True})

        try:
            personas = await get_all()
        except Exception as e:
            logger.warning(f"get_all_personas 失败：{e}")
            return jsonify({"success": True, "personas": [], "degraded": True})

        job_personas = {j.persona_id for j in component.storage.list_jobs()}
        items = []
        for persona in personas or []:
            persona_id = getattr(persona, "persona_id", None) or getattr(
                persona, "name", ""
            )
            if not persona_id:
                continue
            prompt = getattr(persona, "system_prompt", "") or ""
            items.append(
                {
                    "persona_id": persona_id,
                    "prompt_length": len(prompt),
                    "is_default": persona_id == DEFAULT_PERSONA_ID,
                    "iterable": persona_id != DEFAULT_PERSONA_ID,
                    "has_job": persona_id in job_personas,
                }
            )
        return jsonify({"success": True, "personas": items, "degraded": False})

    except Exception as e:
        return handle_exception(e, "获取 Persona 列表")


async def clone_default_persona():
    """POST /personas/clone-default：克隆 default 为具名 Persona

    只创建新 Persona，不修改任何会话绑定。
    """
    try:
        data = await request.get_json(silent=True) or {}
        try:
            new_id = _require_str(data.get("persona_id"), "persona_id", _MAX_NAME_LEN)
            source_id = _optional_str(data.get("source_id"), "source_id", _MAX_NAME_LEN)
        except ValueError as e:
            return _bad_request(str(e))
        source_id = source_id or DEFAULT_PERSONA_ID
        if new_id == DEFAULT_PERSONA_ID:
            return _bad_request("persona_id 不能为 default")

        component, error = get_pe_component()
        if error:
            return error

        persona_manager = _get_persona_manager(component)
        create_persona = (
            getattr(persona_manager, "create_persona", None)
            if persona_manager is not None
            else None
        )
        if create_persona is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "当前 AstrBot 版本不支持创建 Persona，"
                        "请在 AstrBot 面板手动创建",
                        "error_code": "persona_create_unsupported",
                    }
                ),
                503,
            )

        # 读取源 Persona 内容（default 兼容 v3 默认人格）
        source_prompt: Optional[str] = None
        begin_dialogs: Optional[List[str]] = None
        get_persona = getattr(persona_manager, "get_persona", None)
        if get_persona is not None:
            try:
                source = await get_persona(source_id)
                source_prompt = getattr(source, "system_prompt", None)
                dialogs = getattr(source, "begin_dialogs", None)
                if isinstance(dialogs, list):
                    begin_dialogs = [str(d) for d in dialogs]
            except Exception:
                source_prompt = None
        if source_prompt is None and source_id == DEFAULT_PERSONA_ID:
            get_default_v3 = getattr(persona_manager, "get_default_persona_v3", None)
            if get_default_v3 is not None:
                try:
                    default_v3 = await get_default_v3()
                    if isinstance(default_v3, dict):
                        source_prompt = default_v3.get("prompt")
                    else:
                        source_prompt = getattr(default_v3, "prompt", None)
                except Exception as e:
                    logger.warning(f"读取 default 人格失败：{e}")
        if source_prompt is None:
            return _bad_request(
                f"源 Persona {source_id} 不存在或不可读",
                ErrorCode.PERSONA_NOT_FOUND.value,
            )

        try:
            await create_persona(
                new_id, system_prompt=source_prompt, begin_dialogs=begin_dialogs
            )
        except Exception as e:
            return _bad_request(f"克隆失败：{e}")

        logger.info(f"克隆 default Persona 为 {new_id}")
        return jsonify(
            {
                "success": True,
                "persona_id": new_id,
                "message": f"已克隆 {source_id} 为 {new_id}（会话绑定未变动）",
            }
        )

    except Exception as e:
        return handle_exception(e, "克隆 default Persona")


# ----------------------------------------------------------------------
# Job
# ----------------------------------------------------------------------


async def list_jobs():
    """GET /jobs：列出全部迭代 Job（附语料计数）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        storage = component.storage
        jobs = [job_to_dict(j, storage) for j in storage.list_jobs()]
        return jsonify({"success": True, "jobs": jobs})

    except Exception as e:
        return handle_exception(e, "获取迭代 Job 列表")


async def create_job():
    """POST /jobs：创建迭代 Job（语料游标基线=创建时最新 Sample ID）"""
    try:
        data = await request.get_json(silent=True) or {}
        try:
            persona_id = _require_str(data.get("persona_id"), "persona_id")
            if persona_id == DEFAULT_PERSONA_ID:
                raise ValueError("default Persona 不可直接迭代，请先克隆为具名 Persona")
            fields = _validate_edit_fields(data)
        except ValueError as e:
            return _bad_request(str(e))

        component, error = get_pe_component()
        if error:
            return error

        # Persona 存在性检查（PersonaManager 不可用时降级跳过）
        persona_manager = _get_persona_manager(component)
        get_persona = (
            getattr(persona_manager, "get_persona", None)
            if persona_manager is not None
            else None
        )
        if get_persona is not None:
            try:
                persona = await get_persona(persona_id)
                if persona is None:
                    raise ValueError(f"Persona {persona_id} 不存在")
            except ValueError:
                return _bad_request(
                    f"Persona {persona_id} 不存在", ErrorCode.PERSONA_NOT_FOUND.value
                )
            except Exception as e:
                logger.warning(f"创建 Job 时检查 Persona 失败（跳过）：{e}")

        storage = component.storage
        job = EvolutionJob(
            persona_id=persona_id,
            name=fields.get("name", ""),
            goal_preset_id=fields.get("goal_preset_id", "natural"),
            custom_goal=fields.get("custom_goal", ""),
            source_group_ids=fields.get("source_group_ids", []),
            source_user_ids=fields.get("source_user_ids", []),
            edit_mode=fields.get("edit_mode", EditMode.MANAGED_BLOCK.value),
            approval_mode=fields.get("approval_mode", ApprovalMode.AUTO.value),
            trigger_sample_count=fields.get("trigger_sample_count", 100),
            min_interval_hours=fields.get("min_interval_hours", 24),
            provider_id=fields.get("provider_id", ""),
            reviewer_provider_id=fields.get("reviewer_provider_id", ""),
            protected_fragments=fields.get("protected_fragments", []),
            last_sample_cursor=storage.get_latest_sample_id(),
        )
        try:
            job_id = storage.create_job(job)
        except ValueError as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(e),
                        "error_code": ErrorCode.JOB_ALREADY_EXISTS.value,
                    }
                ),
                409,
            )

        logger.info(f"创建迭代 Job {job_id}（persona={persona_id}）")
        return jsonify(
            {"success": True, "job": job_to_dict(storage.get_job(job_id), storage)}
        )

    except Exception as e:
        return handle_exception(e, "创建迭代 Job")


async def get_job(job_id: int):
    """GET /jobs/<id>：Job 详情 + 最近运行与版本摘要"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        storage = component.storage
        job = storage.get_job(job_id)
        if job is None:
            return _not_found(f"Job {job_id} 不存在")

        runs = [run_to_dict(r) for r in storage.list_runs(job_id, limit=10)]
        revisions = [
            revision_to_dict(r, include_prompts=False)
            for r in storage.list_revisions(job_id, limit=10)
        ]
        return jsonify(
            {
                "success": True,
                "job": job_to_dict(job, storage),
                "runs": runs,
                "revisions": revisions,
            }
        )

    except Exception as e:
        return handle_exception(e, "获取迭代 Job 详情")


async def update_job(job_id: int):
    """PUT /jobs/<id>：更新 Job 配置

    审批模式从手动切回自动只影响后续运行，已有 candidate
    不会被追溯自动发布（文档 §11.2）。
    """
    try:
        data = await request.get_json(silent=True) or {}
        unknown = set(data) - set(_JOB_EDITABLE_FIELDS)
        if unknown:
            return _bad_request(
                f"不允许更新的字段：{', '.join(sorted(unknown))}"
                f"（允许：{', '.join(_JOB_EDITABLE_FIELDS)}）"
            )
        try:
            fields = _validate_edit_fields(data)
        except ValueError as e:
            return _bad_request(str(e))
        if not fields:
            return _bad_request("没有需要更新的字段")

        component, error = get_pe_component()
        if error:
            return error
        storage = component.storage
        job = storage.get_job(job_id)
        if job is None:
            return _not_found(f"Job {job_id} 不存在")

        try:
            storage.update_job(job_id, fields)
        except ValueError as e:
            return _bad_request(str(e))

        logger.info(f"更新迭代 Job {job_id}：{sorted(fields)}")
        return jsonify(
            {"success": True, "job": job_to_dict(storage.get_job(job_id), storage)}
        )

    except Exception as e:
        return handle_exception(e, "更新迭代 Job")


async def pause_job(job_id: int):
    """POST /jobs/<id>/pause：暂停自动迭代"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        return _service_response(component.service.pause_job(job_id))

    except Exception as e:
        return handle_exception(e, "暂停迭代 Job")


async def resume_job(job_id: int):
    """POST /jobs/<id>/resume：恢复迭代（§8.3 管理员查看原因后恢复）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        return _service_response(component.service.resume_job(job_id))

    except Exception as e:
        return handle_exception(e, "恢复迭代 Job")


async def run_job(job_id: int):
    """POST /jobs/<id>/run：立即执行一轮迭代（手动触发）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        result = await component.service.run_job(job_id, "manual")
        if result.get("ok"):
            return jsonify(
                {
                    "success": True,
                    "message": result.get("message", ""),
                    "job_id": job_id,
                    "run_id": result.get("run_id"),
                    "revision_id": result.get("revision_id"),
                    "no_change": result.get("no_change", False),
                }
            )
        return _service_response(result)

    except Exception as e:
        return handle_exception(e, "手动执行迭代")


async def adopt_current(job_id: int):
    """POST /jobs/<id>/conflict/adopt-current：采纳外部版本为新基线（§12.1）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        return _service_response(
            await component.service.adopt_current_for_conflict(job_id)
        )

    except Exception as e:
        return handle_exception(e, "采纳外部版本为新基线")


# ----------------------------------------------------------------------
# Revision
# ----------------------------------------------------------------------


async def list_revisions(job_id: int):
    """GET /jobs/<id>/revisions：版本时间线（含完整快照供 Diff）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        storage = component.storage
        if storage.get_job(job_id) is None:
            return _not_found(f"Job {job_id} 不存在")

        status = request.args.get("status") or None
        if status is not None:
            try:
                _enum(status, "status", [e.value for e in RevisionStatus])
            except ValueError as e:
                return _bad_request(str(e))
        try:
            limit = _int_range(request.args.get("limit"), "limit", 1, 100, 50)
        except ValueError as e:
            return _bad_request(str(e))

        revisions = [
            revision_to_dict(r)
            for r in storage.list_revisions(job_id, limit=limit, status=status)
        ]
        return jsonify({"success": True, "revisions": revisions})

    except Exception as e:
        return handle_exception(e, "获取版本时间线")


async def get_revision(revision_id: int):
    """GET /revisions/<id>：单个 Revision 完整快照"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        revision = component.storage.get_revision(revision_id)
        if revision is None:
            return _not_found(f"Revision {revision_id} 不存在")
        return jsonify({"success": True, "revision": revision_to_dict(revision)})

    except Exception as e:
        return handle_exception(e, "获取 Revision 详情")


async def approve_revision(revision_id: int):
    """POST /revisions/<id>/approve：批准候选（§11.2 复核校验+哈希检查）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        return _service_response(
            await component.service.approve_revision(revision_id)
        )

    except Exception as e:
        return handle_exception(e, "批准 Revision")


async def reject_revision(revision_id: int):
    """POST /revisions/<id>/reject：拒绝候选（保存理由）"""
    try:
        data = await request.get_json(silent=True) or {}
        try:
            reason = _optional_str(data.get("reason"), "reason", _MAX_REASON_LEN)
        except ValueError as e:
            return _bad_request(str(e))

        component, error = get_pe_component()
        if error:
            return error
        return _service_response(
            component.service.reject_revision(revision_id, reason)
        )

    except Exception as e:
        return handle_exception(e, "拒绝 Revision")


async def rollback_revision(revision_id: int):
    """POST /revisions/<id>/rollback：回滚到该版本（§13.3 生成新版本）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        return _service_response(
            await component.service.rollback_to_revision(revision_id)
        )

    except Exception as e:
        return handle_exception(e, "回滚 Revision")


# ----------------------------------------------------------------------
# 语料与数据
# ----------------------------------------------------------------------


async def sample_stats():
    """GET /samples/stats：群/用户/时间分布与计数（不返回原文）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        stats = component.storage.get_sample_stats()
        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        return handle_exception(e, "获取语料统计")


async def clear_samples():
    """POST /samples/clear：按群/用户清除语料（都不传清空全部）"""
    try:
        data = await request.get_json(silent=True) or {}
        try:
            group_id = _optional_str(data.get("group_id"), "group_id")
            user_id = _optional_str(data.get("user_id"), "user_id")
        except ValueError as e:
            return _bad_request(str(e))

        component, error = get_pe_component()
        if error:
            return error
        deleted = component.storage.clear_samples(
            [group_id] if group_id else None,
            [user_id] if user_id else None,
        )
        scope = "全部"
        if group_id and user_id:
            scope = f"群 {group_id} 用户 {user_id}"
        elif group_id:
            scope = f"群 {group_id}"
        elif user_id:
            scope = f"用户 {user_id}"
        logger.info(f"清除人格自迭代语料：scope={scope}, deleted={deleted}")
        return jsonify({"success": True, "deleted": deleted, "scope": scope})

    except Exception as e:
        return handle_exception(e, "清除语料")


async def export_data():
    """GET /export：独立导出（默认不含语料原文，§19）"""
    try:
        component, error = get_pe_component()
        if error:
            return error
        include_samples = (request.args.get("include_samples") or "").lower() in (
            "true",
            "1",
            "yes",
        )
        data = component.storage.export_all(include_samples=include_samples)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iris_persona_evolution_{timestamp}.json"
        logger.info(f"人格自迭代导出成功（include_samples={include_samples}）")
        return Response(
            json.dumps(data, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return handle_exception(e, "导出人格自迭代数据")


async def import_data():
    """POST /import：独立导入（导入 Revision 历史不修改 AstrBot Persona，§19）"""
    try:
        body = await request.get_json(silent=True)
        if not body or "data" not in body:
            return _bad_request("请求体缺少 data 字段")
        skip_duplicates = bool(body.get("skip_duplicates", True))

        component, error = get_pe_component()
        if error:
            return error
        try:
            stats = component.storage.import_from_data(
                body["data"], skip_duplicates=skip_duplicates
            )
        except ValueError as e:
            return _bad_request(str(e))

        logger.info(f"人格自迭代导入完成：{stats}")
        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        return handle_exception(e, "导入人格自迭代数据")


# ----------------------------------------------------------------------
# 注册
# ----------------------------------------------------------------------


def register_persona_evolution_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/persona-evolution"

    routes = [
        (f"{prefix}/personas", list_personas, ["GET"], "获取 Persona 列表"),
        (f"{prefix}/goals", list_goals, ["GET"], "获取人格迭代目标预设"),
        (f"{prefix}/personas/clone-default", clone_default_persona, ["POST"], "克隆 default Persona"),
        (f"{prefix}/jobs", list_jobs, ["GET"], "获取迭代 Job 列表"),
        (f"{prefix}/jobs", create_job, ["POST"], "创建迭代 Job"),
        (f"{prefix}/jobs/<int:job_id>", get_job, ["GET"], "获取迭代 Job 详情"),
        (f"{prefix}/jobs/<int:job_id>", update_job, ["PUT"], "更新迭代 Job"),
        # AstrBot 插件页桥接层当前只提供 apiGet/apiPost；保留 PUT 的同时
        # 提供语义明确的 POST 别名，避免前端绕过宿主认证桥接直接 fetch。
        (f"{prefix}/jobs/<int:job_id>/update", update_job, ["POST"], "更新迭代 Job"),
        (f"{prefix}/jobs/<int:job_id>/pause", pause_job, ["POST"], "暂停迭代 Job"),
        (f"{prefix}/jobs/<int:job_id>/resume", resume_job, ["POST"], "恢复迭代 Job"),
        (f"{prefix}/jobs/<int:job_id>/run", run_job, ["POST"], "立即执行迭代"),
        (f"{prefix}/jobs/<int:job_id>/revisions", list_revisions, ["GET"], "获取版本时间线"),
        (f"{prefix}/jobs/<int:job_id>/conflict/adopt-current", adopt_current, ["POST"], "采纳外部版本为新基线"),
        (f"{prefix}/revisions/<int:revision_id>", get_revision, ["GET"], "获取 Revision 详情"),
        (f"{prefix}/revisions/<int:revision_id>/approve", approve_revision, ["POST"], "批准 Revision"),
        (f"{prefix}/revisions/<int:revision_id>/reject", reject_revision, ["POST"], "拒绝 Revision"),
        (f"{prefix}/revisions/<int:revision_id>/rollback", rollback_revision, ["POST"], "回滚到该版本"),
        (f"{prefix}/samples/stats", sample_stats, ["GET"], "获取语料统计"),
        (f"{prefix}/samples/clear", clear_samples, ["POST"], "清除语料"),
        (f"{prefix}/export", export_data, ["GET"], "导出人格自迭代数据"),
        (f"{prefix}/import", import_data, ["POST"], "导入人格自迭代数据"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)
