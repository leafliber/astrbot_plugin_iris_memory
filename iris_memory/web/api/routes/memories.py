"""记忆管理路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.web.utils.helpers import safe_int
from iris_memory.utils.logger import get_logger

logger = get_logger("route_memories")


def register_memory_routes(app: Any, memory_service: Any) -> None:
    """注册记忆管理路由

    Args:
        app: Quart 应用实例
        memory_service: MemoryWebService 实例
    """
    from quart import request

    @app.route("/api/memories", methods=["GET"])
    async def memories_list():
        try:
            result = await memory_service.search_memories_web(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
                memory_type=request.args.get("type"),
                page=safe_int(request.args.get("page"), 1, 1, 10000),
                page_size=safe_int(request.args.get("page_size"), 20, 1, 100),
            )
            return success_response(result)
        except Exception as e:
            logger.error(f"Memories list error: {e}")
            return error_response("查询失败", 500)

    @app.route("/api/memories/search", methods=["GET"])
    async def memories_search():
        try:
            query = request.args.get("q", "")
            if not query:
                return error_response("缺少搜索关键词 q")

            result = await memory_service.search_memories_web(
                query=query,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
                memory_type=request.args.get("type"),
                page=safe_int(request.args.get("page"), 1, 1, 10000),
                page_size=safe_int(request.args.get("page_size"), 20, 1, 100),
            )
            return success_response(result)
        except Exception as e:
            logger.error(f"Memories search error: {e}")
            return error_response("搜索失败", 500)

    @app.route("/api/memories/detail", methods=["GET"])
    async def memory_detail():
        try:
            memory_id = request.args.get("id")
            if not memory_id:
                return error_response("缺少记忆 ID")

            detail = await memory_service.get_memory_detail(memory_id)
            if detail:
                return success_response(detail)
            return error_response("记忆不存在", 404)
        except Exception as e:
            logger.error(f"Memory detail error: {e}")
            return error_response("获取详情失败", 500)

    @app.route("/api/memories/update", methods=["POST"])
    async def memory_update():
        try:
            body = await request.get_json()
            if not body or not body.get("id"):
                return error_response("缺少记忆 ID")

            updates = body.get("updates", {})
            if not updates:
                return error_response("缺少更新内容")

            success, msg = await memory_service.update_memory_by_id(body["id"], updates)
            if success:
                return success_response(message=msg)
            return error_response(msg)
        except Exception as e:
            logger.error(f"Memory update error: {e}")
            return error_response("更新失败", 500)

    @app.route("/api/memories/delete", methods=["POST"])
    async def memory_delete():
        try:
            body = await request.get_json()
            if not body or not body.get("id"):
                return error_response("缺少记忆 ID")

            success, msg = await memory_service.delete_memory_by_id(body["id"])
            if success:
                return success_response(message=msg)
            return error_response(msg)
        except Exception as e:
            logger.error(f"Memory delete error: {e}")
            return error_response("删除失败", 500)

    @app.route("/api/memories/batch-delete", methods=["POST"])
    async def memory_batch_delete():
        try:
            body = await request.get_json()
            if not body or not body.get("ids"):
                return error_response("缺少记忆 ID 列表")

            ids = body["ids"]
            if not isinstance(ids, list) or len(ids) == 0:
                return error_response("ids 必须是非空数组")
            if len(ids) > 100:
                return error_response("单次最多删除 100 条")

            result = await memory_service.batch_delete_memories(ids)
            return success_response(result)
        except Exception as e:
            logger.error(f"Batch delete error: {e}")
            return error_response("批量删除失败", 500)
