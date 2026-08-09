"""
学习模块管理 API 路由

提供学习产物（圈内暗语/表达模式/对话样例）的手动管理与统计：
- 列表查询：分页 + 群/状态筛选
- 增删改：手动新增、编辑字段、批量删除
- 审查：批量通过/禁用
- 统计：三表状态分布 + 暗语词频 Top
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from quart import jsonify, request

from iris_memory.core import get_component_manager, get_logger
from iris_memory.learning import LearningComponent
from iris_memory.learning.storage import (
    _TABLES,
    _VALID_STATUSES,
)

logger = get_logger("web.learning")

PLUGIN_NAME = "astrbot_plugin_iris_memory"

DEBUG_MODE = os.environ.get("IRIS_DEBUG", "").lower() in ("true", "1", "yes")

# /add 接口各表的必填字段
_REQUIRED_ADD_FIELDS = {
    "jargon": ("term",),
    "expression_pattern": ("expression",),
    "few_shot": ("user_text", "bot_text"),
}

MAX_PAGE_SIZE = 100


def validate_table(table: Optional[str]) -> str:
    """校验表名参数，非法抛 ValueError"""
    if not table or table not in _TABLES:
        raise ValueError(f"非法的表名：{table}")
    return table


def validate_table_status(table: str, status: Optional[str]) -> str:
    """校验目标状态对该表合法（暗语仅 active/disabled），非法抛 ValueError"""
    if not status or status not in _VALID_STATUSES[table]:
        allowed = ", ".join(_VALID_STATUSES[table])
        raise ValueError(f"表 {table} 非法的状态：{status}（允许：{allowed}）")
    return status


def parse_pagination(args: Any) -> Tuple[int, int]:
    """解析分页参数，返回 (page, page_size)，页码从 1 起"""
    try:
        page = max(1, int(args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(args.get("page_size", 20))
    except (TypeError, ValueError):
        page_size = 20
    page_size = min(MAX_PAGE_SIZE, max(1, page_size))
    return page, page_size


def get_learning_component() -> Tuple[Optional[Any], Optional[Tuple]]:
    manager = get_component_manager()
    component = manager.get_component("learning", LearningComponent)

    if not component or not component.is_available or not component.storage:
        return None, (jsonify({"success": False, "error": "学习模块不可用或未启用"}), 503)

    return component, None


def handle_exception(e: Exception, operation: str) -> Tuple[Any, int]:
    logger.error(f"{operation}失败：{e}", exc_info=True)

    if DEBUG_MODE:
        error_msg = str(e)
    else:
        error_msg = "服务器内部错误"

    return jsonify({"success": False, "error": error_msg}), 500


def _bad_request(msg: str) -> Tuple[Any, int]:
    return jsonify({"success": False, "error": msg}), 400


async def list_items():
    try:
        table = request.args.get("table")
        try:
            validate_table(table)
        except ValueError as e:
            return _bad_request(str(e))

        group_id = request.args.get("group_id") or None
        status = request.args.get("status") or None
        page, page_size = parse_pagination(request.args)

        component, error = get_learning_component()
        if error:
            return error

        storage = component.storage
        items = storage.list_rows(
            table,
            group_id=group_id,
            status=status,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        total = storage.count_rows(table, group_id=group_id, status=status)

        return jsonify({"success": True, "items": items, "total": total})

    except Exception as e:
        return handle_exception(e, "获取学习数据列表")


async def list_groups():
    try:
        component, error = get_learning_component()
        if error:
            return error

        return jsonify({"success": True, "groups": component.storage.list_groups()})

    except Exception as e:
        return handle_exception(e, "获取学习数据群列表")


async def get_stats():
    try:
        component, error = get_learning_component()
        if error:
            return error

        storage = component.storage
        stats = storage.get_stats()
        jargon_top = storage.list_rows("jargon", limit=10)
        usage = storage.get_jargon_usage(datetime.now().date().isoformat())

        return jsonify({
            "success": True, "stats": stats, "jargon_top": jargon_top,
            "jargon_llm_usage": usage,
        })

    except Exception as e:
        return handle_exception(e, "获取学习统计")


async def list_jargon_candidates():
    """只读查看自动暗语漏斗中的候选和判定原因。"""
    try:
        group_id = request.args.get("group_id") or None
        state = request.args.get("state") or None
        page, page_size = parse_pagination(request.args)
        component, error = get_learning_component()
        if error:
            return error
        items = component.storage.list_jargon_candidates(
            group_id, state, page_size, (page - 1) * page_size
        )
        total = component.storage.count_jargon_candidates(group_id, state)
        return jsonify({"success": True, "items": items, "total": total})
    except Exception as e:
        return handle_exception(e, "获取暗语候选")


async def add_item():
    try:
        data = await request.get_json(silent=True) or {}
        table = data.get("table")
        try:
            validate_table(table)
        except ValueError as e:
            return _bad_request(str(e))

        missing = [f for f in _REQUIRED_ADD_FIELDS[table] if not data.get(f)]
        if missing:
            return _bad_request(f"缺少必填字段：{', '.join(missing)}")

        component, error = get_learning_component()
        if error:
            return error

        storage = component.storage
        group_id = data.get("group_id") or ""

        if table == "jargon":
            row_id = storage.insert_jargon(
                group_id=group_id,
                term=data["term"],
                meaning=data.get("meaning") or None,
                confidence=data.get("confidence"),
            )
        elif table == "expression_pattern":
            row_id = storage.insert_pattern(
                group_id=group_id,
                scene=data.get("scene") or "",
                expression=data["expression"],
            )
        else:  # few_shot
            row_id = storage.insert_pair(
                group_id=group_id,
                user_id=data.get("user_id") or "",
                user_text=data["user_text"],
                bot_text=data["bot_text"],
            )

        logger.info(f"手动新增学习数据：table={table}, id={row_id}")
        return jsonify({"success": True, "id": row_id})

    except Exception as e:
        return handle_exception(e, "新增学习数据")


async def update_item():
    try:
        data = await request.get_json(silent=True) or {}
        table = data.get("table")
        try:
            validate_table(table)
        except ValueError as e:
            return _bad_request(str(e))

        row_id = data.get("id")
        if not isinstance(row_id, int):
            return _bad_request("缺少或非法的 id 参数")

        fields: Dict[str, Any] = data.get("fields") or {}

        component, error = get_learning_component()
        if error:
            return error

        try:
            updated = component.storage.update_row(table, row_id, fields)
        except ValueError as e:
            return _bad_request(str(e))

        if not updated:
            return jsonify({"success": False, "error": "条目不存在或未发生变化"}), 404

        logger.info(f"更新学习数据：table={table}, id={row_id}")
        return jsonify({"success": True, "message": "已更新"})

    except Exception as e:
        return handle_exception(e, "更新学习数据")


async def delete_items():
    try:
        data = await request.get_json(silent=True) or {}
        table = data.get("table")
        try:
            validate_table(table)
        except ValueError as e:
            return _bad_request(str(e))

        ids: List[int] = data.get("ids") or []
        if not ids:
            return _bad_request("缺少 ids 参数")

        component, error = get_learning_component()
        if error:
            return error

        deleted = component.storage.delete_rows(table, ids)

        logger.info(f"删除学习数据：table={table}, ids={ids}, deleted={deleted}")
        return jsonify({"success": True, "deleted": deleted})

    except Exception as e:
        return handle_exception(e, "删除学习数据")


async def set_status():
    try:
        data = await request.get_json(silent=True) or {}
        table = data.get("table")
        try:
            validate_table(table)
        except ValueError as e:
            return _bad_request(str(e))

        ids: List[int] = data.get("ids") or []
        if not ids:
            return _bad_request("缺少 ids 参数")

        status = data.get("status")
        try:
            validate_table_status(table, status)
        except ValueError as e:
            return _bad_request(str(e))

        component, error = get_learning_component()
        if error:
            return error

        component.storage.update_status(table, ids, status)

        logger.info(f"更新学习数据状态：table={table}, ids={ids}, status={status}")
        return jsonify({"success": True, "message": "状态已更新"})

    except Exception as e:
        return handle_exception(e, "更新学习数据状态")


def register_learning_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/learning"

    routes = [
        (f"{prefix}/list", list_items, ["GET"], "获取学习数据列表"),
        (f"{prefix}/groups", list_groups, ["GET"], "获取学习数据群列表"),
        (f"{prefix}/stats", get_stats, ["GET"], "获取学习统计"),
        (f"{prefix}/candidates", list_jargon_candidates, ["GET"], "获取暗语候选"),
        (f"{prefix}/add", add_item, ["POST"], "新增学习数据"),
        (f"{prefix}/update", update_item, ["POST"], "更新学习数据"),
        (f"{prefix}/delete", delete_items, ["POST"], "删除学习数据"),
        (f"{prefix}/status", set_status, ["POST"], "批量更新学习数据状态"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)
