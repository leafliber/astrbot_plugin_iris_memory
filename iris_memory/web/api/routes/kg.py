"""知识图谱路由蓝图"""

from __future__ import annotations

from typing import Any

from iris_memory.web.api.response import error_response, success_response
from iris_memory.web.utils.helpers import safe_int
from iris_memory.utils.logger import get_logger

logger = get_logger("route_kg")


def register_kg_routes(app: Any, kg_service: Any) -> None:
    """注册知识图谱路由

    Args:
        app: Quart 应用实例
        kg_service: KgWebService 实例
    """
    from quart import request

    @app.route("/api/kg/nodes", methods=["GET"])
    async def kg_nodes():
        try:
            nodes = await kg_service.search_kg_nodes(
                query=request.args.get("q", ""),
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                node_type=request.args.get("type"),
                limit=safe_int(request.args.get("limit"), 50, 1, 500),
            )
            return success_response(nodes)
        except Exception as e:
            logger.error(f"KG nodes error: {e}")
            return error_response("查询失败", 500)

    @app.route("/api/kg/edges", methods=["GET"])
    async def kg_edges():
        try:
            edges = await kg_service.list_kg_edges(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                relation_type=request.args.get("relation_type"),
                node_id=request.args.get("node_id"),
                limit=safe_int(request.args.get("limit"), 50, 1, 500),
            )
            return success_response(edges)
        except Exception as e:
            logger.error(f"KG edges error: {e}")
            return error_response("查询失败", 500)

    @app.route("/api/kg/graph", methods=["GET"])
    async def kg_graph():
        try:
            graph = await kg_service.get_kg_graph_data(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                center_node_id=request.args.get("center"),
                depth=safe_int(request.args.get("depth"), 2, 1, 5),
                max_nodes=safe_int(request.args.get("max_nodes"), 100, 1, 500),
            )
            return success_response(graph)
        except Exception as e:
            logger.error(f"KG graph error: {e}")
            return error_response("获取图谱失败", 500)

    @app.route("/api/kg/node/delete", methods=["POST"])
    async def kg_node_delete():
        try:
            body = await request.get_json()
            if not body or not body.get("id"):
                return error_response("缺少节点 ID")

            success, msg = await kg_service.delete_kg_node(body["id"])
            if success:
                return success_response(message=msg)
            return error_response(msg)
        except Exception as e:
            logger.error(f"KG node delete error: {e}")
            return error_response("删除失败", 500)

    @app.route("/api/kg/edge/delete", methods=["POST"])
    async def kg_edge_delete():
        try:
            body = await request.get_json()
            if not body or not body.get("id"):
                return error_response("缺少边 ID")

            success, msg = await kg_service.delete_kg_edge(body["id"])
            if success:
                return success_response(message=msg)
            return error_response(msg)
        except Exception as e:
            logger.error(f"KG edge delete error: {e}")
            return error_response("删除失败", 500)

    @app.route("/api/kg/maintenance", methods=["POST"])
    async def kg_maintenance():
        try:
            result = await kg_service.run_maintenance()
            if "error" in result:
                return error_response(result["error"])
            return success_response(result, message="维护完成")
        except Exception as e:
            logger.error(f"KG maintenance error: {e}")
            return error_response("维护执行失败", 500)

    @app.route("/api/kg/consistency", methods=["GET"])
    async def kg_consistency():
        try:
            result = await kg_service.check_consistency()
            if "error" in result:
                return error_response(result["error"])
            return success_response(result)
        except Exception as e:
            logger.error(f"KG consistency error: {e}")
            return error_response("一致性检查失败", 500)

    @app.route("/api/kg/quality", methods=["GET"])
    async def kg_quality():
        try:
            result = await kg_service.get_quality_report()
            if "error" in result:
                return error_response(result["error"])
            return success_response(result)
        except Exception as e:
            logger.error(f"KG quality error: {e}")
            return error_response("质量报告生成失败", 500)
