"""知识图谱路由 /api/v1/kg"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quart import request

from iris_memory.web.helpers import safe_int
from iris_memory.web.response import error_response, success_response

if TYPE_CHECKING:
    from quart import Quart
    from iris_memory.web.container import WebContainer


def register_kg_routes(app: "Quart", container: "WebContainer") -> None:

    @app.route("/api/v1/kg/nodes", methods=["GET"])
    async def api_list_kg_nodes():
        svc = container.get("kg_service")
        data = await svc.search_kg_nodes(
            query=request.args.get("query", ""),
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            node_type=request.args.get("node_type"),
            page=safe_int(request.args.get("page"), 1, min_val=1),
            page_size=safe_int(request.args.get("page_size"), 20, min_val=1, max_val=100),
        )
        return success_response(data)

    @app.route("/api/v1/kg/edges", methods=["GET"])
    async def api_list_kg_edges():
        svc = container.get("kg_service")
        data = await svc.list_kg_edges(
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            relation_type=request.args.get("relation_type"),
            node_id=request.args.get("node_id"),
            page=safe_int(request.args.get("page"), 1, min_val=1),
            page_size=safe_int(request.args.get("page_size"), 20, min_val=1, max_val=100),
        )
        return success_response(data)

    @app.route("/api/v1/kg/nodes/<node_id>", methods=["DELETE"])
    async def api_delete_kg_node(node_id: str):
        svc = container.get("kg_service")
        ok, msg = await svc.delete_kg_node(node_id)
        if not ok:
            return error_response(msg)
        return success_response(message=msg)

    @app.route("/api/v1/kg/edges/<edge_id>", methods=["DELETE"])
    async def api_delete_kg_edge(edge_id: str):
        svc = container.get("kg_service")
        ok, msg = await svc.delete_kg_edge(edge_id)
        if not ok:
            return error_response(msg)
        return success_response(message=msg)

    @app.route("/api/v1/kg/graph", methods=["GET"])
    async def api_kg_graph():
        svc = container.get("kg_service")
        data = await svc.get_kg_graph_data(
            user_id=request.args.get("user_id"),
            group_id=request.args.get("group_id"),
            center_node_id=request.args.get("center_node_id"),
            depth=safe_int(request.args.get("depth"), 2, min_val=1, max_val=5),
            max_nodes=safe_int(request.args.get("max_nodes"), 100, max_val=500),
        )
        return success_response(data)

    @app.route("/api/v1/kg/maintenance", methods=["POST"])
    async def api_kg_maintenance():
        svc = container.get("kg_service")
        data = await svc.run_maintenance()
        return success_response(data)

    @app.route("/api/v1/kg/consistency", methods=["GET"])
    async def api_kg_consistency():
        svc = container.get("kg_service")
        data = await svc.check_consistency()
        return success_response(data)

    @app.route("/api/v1/kg/quality", methods=["GET"])
    async def api_kg_quality():
        svc = container.get("kg_service")
        data = await svc.get_quality_report()
        return success_response(data)
