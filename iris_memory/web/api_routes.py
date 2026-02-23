"""
Web API 路由 - 通过 AstrBot Context.register_web_api() 注册

所有路由通过 /api/plug/iris/<path> 访问，受 AstrBot JWT 认证保护。

功能模块：
1. /iris/dashboard   - 统计面板
2. /iris/memories    - 记忆管理
3. /iris/kg          - 知识图谱
4. /iris/export      - 数据导出
5. /iris/import      - 数据导入
6. /iris/page        - 前端页面
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from iris_memory.utils.logger import get_logger
from iris_memory.web.web_service import WebService

logger = get_logger("web_api")

# 静态文件目录
_STATIC_DIR = Path(__file__).parent / "static"


def _json_response(data: Any, status: int = 200) -> Any:
    """构建 JSON 响应（兼容 Quart）"""
    try:
        from quart import jsonify, make_response
        resp = jsonify(data)
        return resp
    except ImportError:
        return data


def _error_response(message: str, status: int = 400) -> Any:
    """构建错误响应"""
    return _json_response({"status": "error", "message": message}, status)


def _success_response(data: Any = None, message: str = "success") -> Any:
    """构建成功响应"""
    result = {"status": "ok", "message": message}
    if data is not None:
        result["data"] = data
    return _json_response(result)


class IrisWebAPI:
    """Iris Memory Web API 管理器

    负责注册所有 Web 路由到 AstrBot Context。
    """

    def __init__(self, context: Any, memory_service: Any) -> None:
        """
        Args:
            context: AstrBot Context 对象
            memory_service: MemoryService 实例
        """
        self._context = context
        self._web_service = WebService(memory_service)
        self._registered = False

    def register_all_routes(self) -> None:
        """注册所有 Web API 路由"""
        if self._registered:
            return

        routes = [
            # 前端页面
            ("/iris/page", self._handle_page, ["GET"], "Iris Memory 管理页面"),

            # 统计面板
            ("/iris/api/dashboard", self._handle_dashboard, ["GET"], "仪表盘统计"),
            ("/iris/api/dashboard/trend", self._handle_trend, ["GET"], "记忆趋势"),

            # 记忆管理
            ("/iris/api/memories", self._handle_memories_list, ["GET"], "记忆列表"),
            ("/iris/api/memories/search", self._handle_memories_search, ["GET"], "记忆搜索"),
            ("/iris/api/memories/detail", self._handle_memory_detail, ["GET"], "记忆详情"),
            ("/iris/api/memories/update", self._handle_memory_update, ["POST"], "更新记忆"),
            ("/iris/api/memories/delete", self._handle_memory_delete, ["POST"], "删除记忆"),
            ("/iris/api/memories/batch-delete", self._handle_memory_batch_delete, ["POST"], "批量删除记忆"),

            # 知识图谱
            ("/iris/api/kg/nodes", self._handle_kg_nodes, ["GET"], "KG 节点列表"),
            ("/iris/api/kg/edges", self._handle_kg_edges, ["GET"], "KG 边列表"),
            ("/iris/api/kg/graph", self._handle_kg_graph, ["GET"], "KG 图谱数据"),
            ("/iris/api/kg/node/delete", self._handle_kg_node_delete, ["POST"], "删除 KG 节点"),
            ("/iris/api/kg/edge/delete", self._handle_kg_edge_delete, ["POST"], "删除 KG 边"),

            # 数据导出
            ("/iris/api/export/memories", self._handle_export_memories, ["GET"], "导出记忆"),
            ("/iris/api/export/kg", self._handle_export_kg, ["GET"], "导出知识图谱"),

            # 数据导入
            ("/iris/api/import/memories", self._handle_import_memories, ["POST"], "导入记忆"),
            ("/iris/api/import/kg", self._handle_import_kg, ["POST"], "导入知识图谱"),
            ("/iris/api/import/preview", self._handle_import_preview, ["POST"], "预览导入数据"),

            # 用户画像与情感
            ("/iris/api/personas", self._handle_personas_list, ["GET"], "用户画像列表"),
            ("/iris/api/personas/detail", self._handle_persona_detail, ["GET"], "用户画像详情"),
            ("/iris/api/emotions", self._handle_emotion_state, ["GET"], "情感状态"),
        ]

        for route, handler, methods, desc in routes:
            try:
                self._context.register_web_api(
                    route=route,
                    view_handler=handler,
                    methods=methods,
                    desc=desc,
                )
            except Exception as e:
                logger.warning(f"Failed to register route {route}: {e}")

        self._registered = True
        logger.info(f"Registered {len(routes)} web API routes")

    # ================================================================
    # 前端页面
    # ================================================================

    async def _handle_page(self) -> Any:
        """服务前端 SPA 页面"""
        try:
            from quart import Response
            html_path = _STATIC_DIR / "index.html"
            if html_path.exists():
                content = html_path.read_text(encoding="utf-8")
                return Response(content, content_type="text/html; charset=utf-8")
            return Response("<h1>Iris Memory Dashboard</h1><p>Frontend not found.</p>",
                          content_type="text/html; charset=utf-8")
        except ImportError:
            return {"error": "Quart not available"}

    # ================================================================
    # 统计面板 API
    # ================================================================

    async def _handle_dashboard(self) -> Any:
        """GET /api/plug/iris/api/dashboard"""
        try:
            stats = await self._web_service.get_dashboard_stats()
            return _success_response(stats)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return _error_response(f"获取统计失败: {e}", 500)

    async def _handle_trend(self) -> Any:
        """GET /api/plug/iris/api/dashboard/trend?days=30"""
        try:
            from quart import request
            days = int(request.args.get("days", "30"))
            days = max(1, min(days, 365))
            trend = await self._web_service.get_memory_trend(days)
            return _success_response(trend)
        except ImportError:
            trend = await self._web_service.get_memory_trend(30)
            return _success_response(trend)
        except Exception as e:
            logger.error(f"Trend error: {e}")
            return _error_response(f"获取趋势失败: {e}", 500)

    # ================================================================
    # 记忆管理 API
    # ================================================================

    async def _handle_memories_list(self) -> Any:
        """GET /api/plug/iris/api/memories?user_id=&group_id=&type=&layer=&page=&page_size="""
        try:
            from quart import request
            result = await self._web_service.search_memories_web(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
                memory_type=request.args.get("type"),
                page=int(request.args.get("page", "1")),
                page_size=int(request.args.get("page_size", "20")),
            )
            return _success_response(result)
        except ImportError:
            result = await self._web_service.search_memories_web()
            return _success_response(result)
        except Exception as e:
            logger.error(f"Memories list error: {e}")
            return _error_response(f"查询失败: {e}", 500)

    async def _handle_memories_search(self) -> Any:
        """GET /api/plug/iris/api/memories/search?q=&user_id=&group_id="""
        try:
            from quart import request
            query = request.args.get("q", "")
            if not query:
                return _error_response("缺少搜索关键词 q")

            result = await self._web_service.search_memories_web(
                query=query,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
                memory_type=request.args.get("type"),
                page=int(request.args.get("page", "1")),
                page_size=int(request.args.get("page_size", "20")),
            )
            return _success_response(result)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Memories search error: {e}")
            return _error_response(f"搜索失败: {e}", 500)

    async def _handle_memory_detail(self) -> Any:
        """GET /api/plug/iris/api/memories/detail?id=..."""
        try:
            from quart import request
            memory_id = request.args.get("id")
            if not memory_id:
                return _error_response("缺少记忆 ID")

            detail = await self._web_service.get_memory_detail(memory_id)
            if detail:
                return _success_response(detail)
            return _error_response("记忆不存在", 404)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Memory detail error: {e}")
            return _error_response(f"获取详情失败: {e}", 500)

    async def _handle_memory_update(self) -> Any:
        """POST /api/plug/iris/api/memories/update  body: {id: "...", updates: {...}}"""
        try:
            from quart import request
            body = await request.get_json()
            if not body or not body.get("id"):
                return _error_response("缺少记忆 ID")

            updates = body.get("updates", {})
            if not updates:
                return _error_response("缺少更新内容")

            success, msg = await self._web_service.update_memory_by_id(body["id"], updates)
            if success:
                return _success_response(message=msg)
            return _error_response(msg)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Memory update error: {e}")
            return _error_response(f"更新失败: {e}", 500)

    async def _handle_memory_delete(self) -> Any:
        """POST /api/plug/iris/api/memories/delete  body: {id: "..."}"""
        try:
            from quart import request
            body = await request.get_json()
            if not body or not body.get("id"):
                return _error_response("缺少记忆 ID")

            success, msg = await self._web_service.delete_memory_by_id(body["id"])
            if success:
                return _success_response(message=msg)
            return _error_response(msg)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Memory delete error: {e}")
            return _error_response(f"删除失败: {e}", 500)

    async def _handle_memory_batch_delete(self) -> Any:
        """POST /api/plug/iris/api/memories/batch-delete  body: {ids: ["...", ...]}"""
        try:
            from quart import request
            body = await request.get_json()
            if not body or not body.get("ids"):
                return _error_response("缺少记忆 ID 列表")

            ids = body["ids"]
            if not isinstance(ids, list) or len(ids) == 0:
                return _error_response("ids 必须是非空数组")
            if len(ids) > 100:
                return _error_response("单次最多删除 100 条")

            result = await self._web_service.batch_delete_memories(ids)
            return _success_response(result)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Batch delete error: {e}")
            return _error_response(f"批量删除失败: {e}", 500)

    # ================================================================
    # 知识图谱 API
    # ================================================================

    async def _handle_kg_nodes(self) -> Any:
        """GET /api/plug/iris/api/kg/nodes?q=&user_id=&type=&limit="""
        try:
            from quart import request
            nodes = await self._web_service.search_kg_nodes(
                query=request.args.get("q", ""),
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                node_type=request.args.get("type"),
                limit=int(request.args.get("limit", "50")),
            )
            return _success_response(nodes)
        except ImportError:
            nodes = await self._web_service.search_kg_nodes()
            return _success_response(nodes)
        except Exception as e:
            logger.error(f"KG nodes error: {e}")
            return _error_response(f"查询失败: {e}", 500)

    async def _handle_kg_edges(self) -> Any:
        """GET /api/plug/iris/api/kg/edges?user_id=&relation_type=&node_id=&limit="""
        try:
            from quart import request
            edges = await self._web_service.list_kg_edges(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                relation_type=request.args.get("relation_type"),
                node_id=request.args.get("node_id"),
                limit=int(request.args.get("limit", "50")),
            )
            return _success_response(edges)
        except ImportError:
            edges = await self._web_service.list_kg_edges()
            return _success_response(edges)
        except Exception as e:
            logger.error(f"KG edges error: {e}")
            return _error_response(f"查询失败: {e}", 500)

    async def _handle_kg_graph(self) -> Any:
        """GET /api/plug/iris/api/kg/graph?user_id=&center=&depth=&max_nodes="""
        try:
            from quart import request
            graph = await self._web_service.get_kg_graph_data(
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                center_node_id=request.args.get("center"),
                depth=int(request.args.get("depth", "2")),
                max_nodes=int(request.args.get("max_nodes", "100")),
            )
            return _success_response(graph)
        except ImportError:
            graph = await self._web_service.get_kg_graph_data()
            return _success_response(graph)
        except Exception as e:
            logger.error(f"KG graph error: {e}")
            return _error_response(f"获取图谱失败: {e}", 500)

    async def _handle_kg_node_delete(self) -> Any:
        """POST /api/plug/iris/api/kg/node/delete  body: {id: "..."}"""
        try:
            from quart import request
            body = await request.get_json()
            if not body or not body.get("id"):
                return _error_response("缺少节点 ID")

            success, msg = await self._web_service.delete_kg_node(body["id"])
            if success:
                return _success_response(message=msg)
            return _error_response(msg)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"KG node delete error: {e}")
            return _error_response(f"删除失败: {e}", 500)

    async def _handle_kg_edge_delete(self) -> Any:
        """POST /api/plug/iris/api/kg/edge/delete  body: {id: "..."}"""
        try:
            from quart import request
            body = await request.get_json()
            if not body or not body.get("id"):
                return _error_response("缺少边 ID")

            success, msg = await self._web_service.delete_kg_edge(body["id"])
            if success:
                return _success_response(message=msg)
            return _error_response(msg)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"KG edge delete error: {e}")
            return _error_response(f"删除失败: {e}", 500)

    # ================================================================
    # 数据导出 API
    # ================================================================

    async def _handle_export_memories(self) -> Any:
        """GET /api/plug/iris/api/export/memories?format=json&user_id=&group_id=&layer="""
        try:
            from quart import request, Response

            fmt = request.args.get("format", "json")
            if fmt not in ("json", "csv"):
                return _error_response("format 必须是 json 或 csv")

            data, content_type, filename = await self._web_service.export_memories(
                format=fmt,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
                storage_layer=request.args.get("layer"),
            )

            return Response(
                data,
                content_type=f"{content_type}; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Export memories error: {e}")
            return _error_response(f"导出失败: {e}", 500)

    async def _handle_export_kg(self) -> Any:
        """GET /api/plug/iris/api/export/kg?format=json&user_id=&group_id="""
        try:
            from quart import request, Response

            fmt = request.args.get("format", "json")
            if fmt not in ("json", "csv"):
                return _error_response("format 必须是 json 或 csv")

            data, content_type, filename = await self._web_service.export_kg(
                format=fmt,
                user_id=request.args.get("user_id"),
                group_id=request.args.get("group_id"),
            )

            return Response(
                data,
                content_type=f"{content_type}; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Export KG error: {e}")
            return _error_response(f"导出失败: {e}", 500)

    # ================================================================
    # 数据导入 API
    # ================================================================

    async def _handle_import_memories(self) -> Any:
        """POST /api/plug/iris/api/import/memories  body: {data: "...", format: "json"}"""
        try:
            from quart import request

            body = await request.get_json()
            if not body or not body.get("data"):
                return _error_response("缺少导入数据")

            fmt = body.get("format", "json")
            if fmt not in ("json", "csv"):
                return _error_response("format 必须是 json 或 csv")

            result = await self._web_service.import_memories(
                data=body["data"],
                format=fmt,
            )
            return _success_response(result)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Import memories error: {e}")
            return _error_response(f"导入失败: {e}", 500)

    async def _handle_import_kg(self) -> Any:
        """POST /api/plug/iris/api/import/kg  body: {data: "...", format: "json"}"""
        try:
            from quart import request

            body = await request.get_json()
            if not body or not body.get("data"):
                return _error_response("缺少导入数据")

            fmt = body.get("format", "json")
            if fmt not in ("json", "csv"):
                return _error_response("format 必须是 json 或 csv")

            result = await self._web_service.import_kg(
                data=body["data"],
                format=fmt,
            )
            return _success_response(result)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Import KG error: {e}")
            return _error_response(f"导入失败: {e}", 500)

    async def _handle_import_preview(self) -> Any:
        """POST /api/plug/iris/api/import/preview  body: {data: "...", format: "json", type: "memories|kg"}"""
        try:
            from quart import request

            body = await request.get_json()
            if not body or not body.get("data"):
                return _error_response("缺少导入数据")

            fmt = body.get("format", "json")
            import_type = body.get("type", "memories")

            result = await self._web_service.preview_import_data(
                data=body["data"],
                format=fmt,
                import_type=import_type,
            )
            return _success_response(result)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Import preview error: {e}")
            return _error_response(f"预览失败: {e}", 500)

    # ================================================================
    # 用户画像与情感状态 API
    # ================================================================

    async def _handle_personas_list(self) -> Any:
        """GET /api/plug/iris/api/personas"""
        try:
            personas = await self._web_service.get_user_personas_list()
            return _success_response(personas)
        except Exception as e:
            logger.error(f"Personas list error: {e}")
            return _error_response(f"获取画像列表失败: {e}", 500)

    async def _handle_persona_detail(self) -> Any:
        """GET /api/plug/iris/api/personas/detail?user_id=..."""
        try:
            from quart import request
            user_id = request.args.get("user_id")
            if not user_id:
                return _error_response("缺少 user_id")

            detail = await self._web_service.get_user_persona_detail(user_id)
            if detail:
                return _success_response(detail)
            return _error_response("用户画像不存在", 404)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Persona detail error: {e}")
            return _error_response(f"获取画像详情失败: {e}", 500)

    async def _handle_emotion_state(self) -> Any:
        """GET /api/plug/iris/api/emotions?user_id=..."""
        try:
            from quart import request
            user_id = request.args.get("user_id")
            if not user_id:
                return _error_response("缺少 user_id")

            state = await self._web_service.get_emotion_state(user_id)
            if state:
                return _success_response(state)
            return _error_response("情感状态不存在", 404)
        except ImportError:
            return _error_response("Quart not available")
        except Exception as e:
            logger.error(f"Emotion state error: {e}")
            return _error_response(f"获取情感状态失败: {e}", 500)
