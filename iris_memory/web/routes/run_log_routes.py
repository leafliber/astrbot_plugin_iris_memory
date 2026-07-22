"""
运行日志 API 路由

提供插件运行日志的查询和清理：
- 获取运行日志（支持按类型过滤：llm_call / injection / proactive）
- 清空运行日志（全部或指定类型）
"""

from quart import jsonify, request
from iris_memory.core import get_logger, get_run_log_manager
from iris_memory.core.run_log import LOG_TYPES, TYPE_LABELS

logger = get_logger("web.run_log")

PLUGIN_NAME = "astrbot_plugin_iris_memory"


async def get_run_logs():
    try:
        manager = get_run_log_manager()

        log_type = request.args.get("type") or None
        if log_type and log_type not in LOG_TYPES:
            return jsonify(
                {"success": False, "error": f"无效的日志类型: {log_type}"}
            ), 400

        limit_arg = request.args.get("limit")
        limit = None
        if limit_arg:
            try:
                limit = max(1, int(limit_arg))
            except ValueError:
                return jsonify({"success": False, "error": "limit 必须为整数"}), 400

        entries = manager.get_entries(log_type=log_type, limit=limit)
        counts = manager.get_counts()

        return jsonify(
            {
                "success": True,
                "entries": entries,
                "counts": counts,
                "types": [
                    {"key": t, "label": TYPE_LABELS[t]} for t in LOG_TYPES
                ],
            }
        )

    except Exception as e:
        logger.error(f"获取运行日志失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


async def clear_run_logs():
    try:
        manager = get_run_log_manager()

        log_type = None
        data = await request.get_json(silent=True)
        if data:
            log_type = data.get("type") or None
        if not log_type:
            log_type = request.args.get("type") or None
        if log_type and log_type not in LOG_TYPES:
            return jsonify(
                {"success": False, "error": f"无效的日志类型: {log_type}"}
            ), 400

        cleared = manager.clear(log_type)
        logger.info(f"已清空运行日志：type={log_type or 'all'}, count={cleared}")

        return jsonify({"success": True, "cleared": cleared})

    except Exception as e:
        logger.error(f"清空运行日志失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": "内部错误，详见服务日志"}), 500


def register_run_log_routes(context) -> None:
    prefix = f"/{PLUGIN_NAME}/run-log"

    routes = [
        (f"{prefix}", get_run_logs, ["GET"], "获取运行日志"),
        (f"{prefix}/clear", clear_run_logs, ["POST"], "清空运行日志"),
    ]

    for route, handler, methods, desc in routes:
        context.register_web_api(route, handler, methods, desc)
