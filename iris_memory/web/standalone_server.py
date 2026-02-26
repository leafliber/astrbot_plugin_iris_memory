"""
独立 Web 服务器 - 与 AstrBot 认证解耦的 Web 管理界面

提供独立的 Quart 应用，通过配置的端口和访问密钥进行认证。
路由逻辑委托给 api/routes/ 下的蓝图模块。
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("standalone_web")

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR_RESOLVED = _STATIC_DIR.resolve()

_SESSION_TOKENS: Dict[str, float] = {}
_TOKEN_EXPIRE_SECONDS = 3600 * 24

# 登录限流
_LOGIN_ATTEMPTS: Dict[str, List[float]] = defaultdict(list)
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 60


def _check_login_rate_limit(client_ip: str) -> bool:
    """检查登录限流

    Returns:
        True 表示允许，False 表示被限流
    """
    now = time.time()
    attempts = _LOGIN_ATTEMPTS[client_ip]
    _LOGIN_ATTEMPTS[client_ip] = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    if len(_LOGIN_ATTEMPTS[client_ip]) >= _LOGIN_MAX_ATTEMPTS:
        return False
    return True


def _record_login_attempt(client_ip: str) -> None:
    """记录登录尝试"""
    _LOGIN_ATTEMPTS[client_ip].append(time.time())


class AuthMiddleware:
    """访问密钥认证中间件"""
    
    def __init__(self, access_key: str):
        self._access_key = access_key
        self._require_auth = bool(access_key)
    
    def check_auth(self, request: Any) -> bool:
        """Checks request authentication.
        Also cleans up expired tokens periodically.

        Supports:
        1. Authorization: Bearer <access_key>
        2. Query parameter: ?key=<access_key>
        """
        if not self._require_auth:
            return True

        # 定期清理过期令牌
        self.cleanup_expired_tokens()

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
        self._shutdown_event: Optional[asyncio.Event] = None
        self._server: Optional[Any] = None
    
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

        from iris_memory.web.api.response import error_response, success_response

        app = Quart(__name__, static_folder=None)
        app.name = "iris_memory_web"

        @app.before_request
        async def check_auth():
            if request.path == "/api/login":
                return None
            if request.path == "/api/check-auth":
                return None
            if request.path == "/" or request.path == "/index.html":
                return None
            if request.path.startswith("/static/"):
                return None

            if not self._auth.check_auth(request):
                return error_response("Unauthorized", 401)
            return None

        @app.route("/api/login", methods=["POST"])
        async def login():
            client_ip = request.remote_addr or "unknown"
            if not _check_login_rate_limit(client_ip):
                return error_response("登录尝试过于频繁，请稍后重试", 429)

            data = await request.get_json()
            if not data:
                return error_response("Missing request body")

            key = data.get("key", "")
            if not key:
                return error_response("Missing access key")

            _record_login_attempt(client_ip)
            if key != self._access_key:
                return error_response("Invalid access key", 401)

            token = self._auth.create_session_token()
            return success_response({"token": token})

        @app.route("/api/check-auth", methods=["GET"])
        async def check_auth_status():
            if not self._auth._require_auth:
                return success_response({"auth_required": False})
            return success_response({"auth_required": True})

        # 注册领域路由蓝图
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
        """注册领域 API 路由

        通过各蓝图模块注册路由，不再内联定义路由处理函数。
        """
        from iris_memory.web.api.routes.dashboard import register_dashboard_routes
        from iris_memory.web.api.routes.memories import register_memory_routes
        from iris_memory.web.api.routes.kg import register_kg_routes
        from iris_memory.web.api.routes.io import register_io_routes
        from iris_memory.web.api.routes.personas import register_persona_routes

        from iris_memory.web.service.dashboard_service import DashboardService
        from iris_memory.web.service.memory_web_service import MemoryWebService
        from iris_memory.web.service.kg_web_service import KgWebService
        from iris_memory.web.service.io_service import IoService
        from iris_memory.web.service.persona_web_service import PersonaWebService

        dashboard_svc = DashboardService(self._service)
        memory_svc = MemoryWebService(self._service)
        kg_svc = KgWebService(self._service)
        io_svc = IoService(self._service)
        persona_svc = PersonaWebService(self._service)

        register_dashboard_routes(app, dashboard_svc)
        register_memory_routes(app, memory_svc)
        register_kg_routes(app, kg_svc)
        register_io_routes(app, io_svc)
        register_persona_routes(app, persona_svc)
    
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
        """服务静态文件（包含路径穿越防护）"""
        from quart import Response
        file_path = (_STATIC_DIR / filename).resolve()
        # 路径穿越防护
        if not str(file_path).startswith(str(_STATIC_DIR_RESOLVED)):
            return Response("Forbidden", status=403)
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
            
            from hypercorn.asyncio import serve
            from hypercorn.config import Config
            
            config = Config()
            config.bind = [f"{self._host}:{self._port}"]
            config.accesslog = None
            config.reuse_address = True
            config.reuse_port = True
            
            self._shutdown_event = asyncio.Event()
            self._running = True
            logger.info(f"Web 管理界面已启动: http://{self._host}:{self._port}")
            
            await serve(app, config, shutdown_trigger=self._shutdown_event.wait)
            
        except ImportError as e:
            logger.warning(f"无法启动 Web 服务器（缺少依赖）: {e}")
            logger.info("请安装 quart 和 hypercorn: pip install quart hypercorn")
        except Exception as e:
            logger.error(f"Web 服务器启动失败: {e}")
        finally:
            self._running = False
    
    async def stop(self) -> None:
        """停止 Web 服务器"""
        if self._shutdown_event:
            self._shutdown_event.set()
            self._shutdown_event = None
        self._running = False
        logger.info("Web 管理界面已停止")
