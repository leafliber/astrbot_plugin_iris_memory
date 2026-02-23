"""
独立 Web 服务器 - 与 AstrBot 认证解耦的 Web 管理界面

提供独立的 Quart 应用，通过配置的端口和访问密钥进行认证。
"""

from __future__ import annotations

import asyncio
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("standalone_web")

_STATIC_DIR = Path(__file__).parent / "static"

_SESSION_TOKENS: Dict[str, float] = {}
_TOKEN_EXPIRE_SECONDS = 3600 * 24


def _json_response(data: Any, status: int = 200) -> Any:
    """构建 JSON 响应"""
    from quart import jsonify
    return jsonify(data), status


def _error_response(message: str, status: int = 400) -> Any:
    """构建错误响应"""
    return _json_response({"status": "error", "message": message}, status)


def _success_response(data: Any = None, message: str = "success") -> Any:
    """构建成功响应"""
    result = {"status": "ok", "message": message}
    if data is not None:
        result["data"] = data
    return _json_response(result)


class AuthMiddleware:
    """访问密钥认证中间件"""
    
    def __init__(self, access_key: str):
        self._access_key = access_key
        self._require_auth = bool(access_key)
    
    def check_auth(self, request: Any) -> bool:
        """检查请求是否通过认证
        
        支持两种方式：
        1. Authorization: Bearer <access_key>
        2. Query parameter: ?key=<access_key>
        """
        if not self._require_auth:
            return True
        
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == self._access_key:
                return True
            if token in _SESSION_TOKENS:
                expire_at = _SESSION_TOKENS[token]
                if time.time() < expire_at:
                    return True
                del _SESSION_TOKENS[token]
        
        query_key = request.args.get("key", "")
        if query_key and query_key == self._access_key:
            return True
        
        return False
    
    def create_session_token(self) -> str:
        """创建会话令牌"""
        token = secrets.token_urlsafe(32)
        _SESSION_TOKENS[token] = time.time() + _TOKEN_EXPIRE_SECONDS
        return token
    
    def cleanup_expired_tokens(self) -> None:
        """清理过期令牌"""
        now = time.time()
        expired = [t for t, exp in _SESSION_TOKENS.items() if now >= exp]
        for t in expired:
            del _SESSION_TOKENS[t]


class StandaloneWebServer:
    """独立 Web 服务器
    
    与 AstrBot 认证系统解耦，通过配置的端口和访问密钥进行认证。
    """
    
    def __init__(
        self,
        memory_service: Any,
        port: int = 8088,
        host: str = "127.0.0.1",
        access_key: str = "",
    ):
        self._service = memory_service
        self._port = port
        self._host = host
        self._access_key = access_key
        self._auth = AuthMiddleware(access_key)
        self._app: Optional[Any] = None
        self._running = False
    
    @property
    def port(self) -> int:
        return self._port
    
    @property
    def host(self) -> str:
        return self._host
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def create_app(self) -> Any:
        """创建 Quart 应用"""
        from quart import Quart, request, Response
        
        app = Quart(__name__, static_folder=None)
        app.name = "iris_memory_web"
        
        @app.before_request
        async def check_auth():
            if request.path == "/api/login":
                return None
            if request.path == "/" or request.path == "/index.html":
                return None
            if request.path.startswith("/static/"):
                return None
            
            if not self._auth.check_auth(request):
                return _error_response("Unauthorized", 401)
            return None
        
        @app.route("/api/login", methods=["POST"])
        async def login():
            data = await request.get_json()
            if not data:
                return _error_response("Missing request body")
            
            key = data.get("key", "")
            if not key:
                return _error_response("Missing access key")
            
            if key != self._access_key:
                return _error_response("Invalid access key", 401)
            
            token = self._auth.create_session_token()
            return _success_response({"token": token})
        
        @app.route("/api/check-auth", methods=["GET"])
        async def check_auth_status():
            if not self._auth._require_auth:
                return _success_response({"auth_required": False})
            return _success_response({"auth_required": True})
        
        self._register_api_routes(app)
        
        @app.route("/")
        async def index():
            return await self._serve_index()
        
        @app.route("/index.html")
        async def index_html():
            return await self._serve_index()
        
        @app.route("/static/<path:filename>")
        async def static_files(filename: str):
            return await self._serve_static(filename)
        
        self._app = app
        return app
    
    def _register_api_routes(self, app: Any) -> None:
        """注册 API 路由"""
        from quart import request, Response
        
        from iris_memory.web.web_service import WebService
        web_service = WebService(self._service)
        
        @app.route("/api/dashboard", methods=["GET"])
        async def dashboard():
            try:
                stats = await web_service.get_dashboard_stats()
                return _success_response(stats)
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                return _error_response(f"获取统计失败: {e}", 500)
        
        @app.route("/api/dashboard/trend", methods=["GET"])
        async def dashboard_trend():
            try:
                days = int(request.args.get("days", "30"))
                days = max(1, min(days, 365))
                trend = await web_service.get_memory_trend(days)
                return _success_response(trend)
            except Exception as e:
                logger.error(f"Trend error: {e}")
                return _error_response(f"获取趋势失败: {e}", 500)
        
        @app.route("/api/memories", methods=["GET"])
        async def memories_list():
            try:
                result = await web_service.search_memories_web(
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                    storage_layer=request.args.get("layer"),
                    memory_type=request.args.get("type"),
                    page=int(request.args.get("page", "1")),
                    page_size=int(request.args.get("page_size", "20")),
                )
                return _success_response(result)
            except Exception as e:
                logger.error(f"Memories list error: {e}")
                return _error_response(f"查询失败: {e}", 500)
        
        @app.route("/api/memories/search", methods=["GET"])
        async def memories_search():
            try:
                query = request.args.get("q", "")
                if not query:
                    return _error_response("缺少搜索关键词 q")
                
                result = await web_service.search_memories_web(
                    query=query,
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                    storage_layer=request.args.get("layer"),
                    memory_type=request.args.get("type"),
                    page=int(request.args.get("page", "1")),
                    page_size=int(request.args.get("page_size", "20")),
                )
                return _success_response(result)
            except Exception as e:
                logger.error(f"Memories search error: {e}")
                return _error_response(f"搜索失败: {e}", 500)
        
        @app.route("/api/memories/detail", methods=["GET"])
        async def memory_detail():
            try:
                memory_id = request.args.get("id")
                if not memory_id:
                    return _error_response("缺少记忆 ID")
                
                detail = await web_service.get_memory_detail(memory_id)
                if detail:
                    return _success_response(detail)
                return _error_response("记忆不存在", 404)
            except Exception as e:
                logger.error(f"Memory detail error: {e}")
                return _error_response(f"获取详情失败: {e}", 500)
        
        @app.route("/api/memories/update", methods=["POST"])
        async def memory_update():
            try:
                body = await request.get_json()
                if not body or not body.get("id"):
                    return _error_response("缺少记忆 ID")
                
                updates = body.get("updates", {})
                if not updates:
                    return _error_response("缺少更新内容")
                
                success, msg = await web_service.update_memory_by_id(body["id"], updates)
                if success:
                    return _success_response(message=msg)
                return _error_response(msg)
            except Exception as e:
                logger.error(f"Memory update error: {e}")
                return _error_response(f"更新失败: {e}", 500)
        
        @app.route("/api/memories/delete", methods=["POST"])
        async def memory_delete():
            try:
                body = await request.get_json()
                if not body or not body.get("id"):
                    return _error_response("缺少记忆 ID")
                
                success, msg = await web_service.delete_memory_by_id(body["id"])
                if success:
                    return _success_response(message=msg)
                return _error_response(msg)
            except Exception as e:
                logger.error(f"Memory delete error: {e}")
                return _error_response(f"删除失败: {e}", 500)
        
        @app.route("/api/memories/batch-delete", methods=["POST"])
        async def memory_batch_delete():
            try:
                body = await request.get_json()
                if not body or not body.get("ids"):
                    return _error_response("缺少记忆 ID 列表")
                
                ids = body["ids"]
                if not isinstance(ids, list) or len(ids) == 0:
                    return _error_response("ids 必须是非空数组")
                if len(ids) > 100:
                    return _error_response("单次最多删除 100 条")
                
                result = await web_service.batch_delete_memories(ids)
                return _success_response(result)
            except Exception as e:
                logger.error(f"Batch delete error: {e}")
                return _error_response(f"批量删除失败: {e}", 500)
        
        @app.route("/api/kg/nodes", methods=["GET"])
        async def kg_nodes():
            try:
                nodes = await web_service.search_kg_nodes(
                    query=request.args.get("q", ""),
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                    node_type=request.args.get("type"),
                    limit=int(request.args.get("limit", "50")),
                )
                return _success_response(nodes)
            except Exception as e:
                logger.error(f"KG nodes error: {e}")
                return _error_response(f"查询失败: {e}", 500)
        
        @app.route("/api/kg/edges", methods=["GET"])
        async def kg_edges():
            try:
                edges = await web_service.list_kg_edges(
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                    relation_type=request.args.get("relation_type"),
                    node_id=request.args.get("node_id"),
                    limit=int(request.args.get("limit", "50")),
                )
                return _success_response(edges)
            except Exception as e:
                logger.error(f"KG edges error: {e}")
                return _error_response(f"查询失败: {e}", 500)
        
        @app.route("/api/kg/graph", methods=["GET"])
        async def kg_graph():
            try:
                graph = await web_service.get_kg_graph_data(
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                    center_node_id=request.args.get("center"),
                    depth=int(request.args.get("depth", "2")),
                    max_nodes=int(request.args.get("max_nodes", "100")),
                )
                return _success_response(graph)
            except Exception as e:
                logger.error(f"KG graph error: {e}")
                return _error_response(f"获取图谱失败: {e}", 500)
        
        @app.route("/api/kg/node/delete", methods=["POST"])
        async def kg_node_delete():
            try:
                body = await request.get_json()
                if not body or not body.get("id"):
                    return _error_response("缺少节点 ID")
                
                success, msg = await web_service.delete_kg_node(body["id"])
                if success:
                    return _success_response(message=msg)
                return _error_response(msg)
            except Exception as e:
                logger.error(f"KG node delete error: {e}")
                return _error_response(f"删除失败: {e}", 500)
        
        @app.route("/api/kg/edge/delete", methods=["POST"])
        async def kg_edge_delete():
            try:
                body = await request.get_json()
                if not body or not body.get("id"):
                    return _error_response("缺少边 ID")
                
                success, msg = await web_service.delete_kg_edge(body["id"])
                if success:
                    return _success_response(message=msg)
                return _error_response(msg)
            except Exception as e:
                logger.error(f"KG edge delete error: {e}")
                return _error_response(f"删除失败: {e}", 500)
        
        @app.route("/api/export/memories", methods=["GET"])
        async def export_memories():
            try:
                fmt = request.args.get("format", "json")
                if fmt not in ("json", "csv"):
                    return _error_response("format 必须是 json 或 csv")
                
                data, content_type, filename = await web_service.export_memories(
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
            except Exception as e:
                logger.error(f"Export memories error: {e}")
                return _error_response(f"导出失败: {e}", 500)
        
        @app.route("/api/export/kg", methods=["GET"])
        async def export_kg():
            try:
                fmt = request.args.get("format", "json")
                if fmt not in ("json", "csv"):
                    return _error_response("format 必须是 json 或 csv")
                
                data, content_type, filename = await web_service.export_kg(
                    format=fmt,
                    user_id=request.args.get("user_id"),
                    group_id=request.args.get("group_id"),
                )
                
                return Response(
                    data,
                    content_type=f"{content_type}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                )
            except Exception as e:
                logger.error(f"Export KG error: {e}")
                return _error_response(f"导出失败: {e}", 500)
        
        @app.route("/api/import/memories", methods=["POST"])
        async def import_memories():
            try:
                body = await request.get_json()
                if not body or not body.get("data"):
                    return _error_response("缺少导入数据")
                
                fmt = body.get("format", "json")
                if fmt not in ("json", "csv"):
                    return _error_response("format 必须是 json 或 csv")
                
                result = await web_service.import_memories(
                    data=body["data"],
                    format=fmt,
                )
                return _success_response(result)
            except Exception as e:
                logger.error(f"Import memories error: {e}")
                return _error_response(f"导入失败: {e}", 500)
        
        @app.route("/api/import/kg", methods=["POST"])
        async def import_kg():
            try:
                body = await request.get_json()
                if not body or not body.get("data"):
                    return _error_response("缺少导入数据")
                
                fmt = body.get("format", "json")
                if fmt not in ("json", "csv"):
                    return _error_response("format 必须是 json 或 csv")
                
                result = await web_service.import_kg(
                    data=body["data"],
                    format=fmt,
                )
                return _success_response(result)
            except Exception as e:
                logger.error(f"Import KG error: {e}")
                return _error_response(f"导入失败: {e}", 500)
        
        @app.route("/api/import/preview", methods=["POST"])
        async def import_preview():
            try:
                body = await request.get_json()
                if not body or not body.get("data"):
                    return _error_response("缺少导入数据")
                
                fmt = body.get("format", "json")
                import_type = body.get("type", "memories")
                
                result = await web_service.preview_import_data(
                    data=body["data"],
                    format=fmt,
                    import_type=import_type,
                )
                return _success_response(result)
            except Exception as e:
                logger.error(f"Import preview error: {e}")
                return _error_response(f"预览失败: {e}", 500)
        
        @app.route("/api/personas", methods=["GET"])
        async def personas_list():
            try:
                personas = await web_service.list_personas(
                    page=int(request.args.get("page", "1")),
                    page_size=int(request.args.get("page_size", "20")),
                )
                return _success_response(personas)
            except Exception as e:
                logger.error(f"Personas list error: {e}")
                return _error_response(f"查询失败: {e}", 500)
        
        @app.route("/api/personas/detail", methods=["GET"])
        async def persona_detail():
            try:
                user_id = request.args.get("user_id")
                if not user_id:
                    return _error_response("缺少 user_id")
                
                detail = await web_service.get_persona_detail(user_id)
                if detail:
                    return _success_response(detail)
                return _error_response("用户画像不存在", 404)
            except Exception as e:
                logger.error(f"Persona detail error: {e}")
                return _error_response(f"获取详情失败: {e}", 500)
        
        @app.route("/api/emotions", methods=["GET"])
        async def emotion_state():
            try:
                user_id = request.args.get("user_id")
                group_id = request.args.get("group_id")
                
                state = await web_service.get_emotion_state(user_id, group_id)
                return _success_response(state)
            except Exception as e:
                logger.error(f"Emotion state error: {e}")
                return _error_response(f"获取情感状态失败: {e}", 500)
    
    async def _serve_index(self) -> Any:
        """服务前端页面"""
        from quart import Response
        html_path = _STATIC_DIR / "index.html"
        if html_path.exists():
            content = html_path.read_text(encoding="utf-8")
            return Response(content, content_type="text/html; charset=utf-8")
        return Response("<h1>Iris Memory Dashboard</h1><p>Frontend not found.</p>",
                       content_type="text/html; charset=utf-8")
    
    async def _serve_static(self, filename: str) -> Any:
        """服务静态文件"""
        from quart import Response
        file_path = _STATIC_DIR / filename
        if file_path.exists() and file_path.is_file():
            content = file_path.read_bytes()
            content_type = self._get_content_type(filename)
            return Response(content, content_type=content_type)
        return Response("Not Found", status=404)
    
    def _get_content_type(self, filename: str) -> str:
        """获取文件的 Content-Type"""
        ext = Path(filename).suffix.lower()
        types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
        }
        return types.get(ext, "application/octet-stream")
    
    async def start(self) -> None:
        """启动 Web 服务器"""
        if self._running:
            return
        
        try:
            from quart import Quart
            app = self.create_app()
            
            import asyncio
            from hypercorn.asyncio import serve
            from hypercorn.config import Config
            
            config = Config()
            config.bind = [f"{self._host}:{self._port}"]
            config.accesslog = None
            
            self._running = True
            logger.info(f"Web 管理界面已启动: http://{self._host}:{self._port}")
            
            await serve(app, config)
            
        except ImportError as e:
            logger.warning(f"无法启动 Web 服务器（缺少依赖）: {e}")
            logger.info("请安装 quart 和 hypercorn: pip install quart hypercorn")
        except Exception as e:
            logger.error(f"Web 服务器启动失败: {e}")
            self._running = False
    
    async def stop(self) -> None:
        """停止 Web 服务器"""
        self._running = False
        logger.info("Web 管理界面已停止")
