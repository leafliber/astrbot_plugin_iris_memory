"""独立 Web 服务器

基于 Quart + Hypercorn 提供 Web UI 和 REST API。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from quart import Quart, request

from iris_memory.web.auth import AuthMiddleware
from iris_memory.web.container import WebContainer
from iris_memory.web.response import error_response, json_response
from iris_memory.utils.logger import get_logger

logger = get_logger("web_server")

try:
    from quart_cors import cors as quart_cors  # type: ignore
except ImportError:
    quart_cors = None


def create_app(
    memory_service: Any,
    access_key: str = "",
    static_folder: Optional[str] = None,
) -> Quart:
    """创建 Quart 应用并注册所有路由"""

    if static_folder is None:
        static_folder = os.path.join(os.path.dirname(__file__), "static")

    app = Quart(
        __name__,
        static_folder=static_folder,
        static_url_path="/static",
    )
    if quart_cors is not None:
        app = quart_cors(app, allow_origin="*")
    else:
        # 手动添加 CORS 头
        @app.after_request
        async def _add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

    # ── 认证中间件 ──
    auth = AuthMiddleware(access_key)
    app.config["AUTH_MIDDLEWARE"] = auth
    app.config["ACCESS_KEY"] = access_key

    # ── DI 容器 ──
    container = WebContainer(memory_service)
    app.config["CONTAINER"] = container

    # ── 全局 before_request 认证 ──
    @app.before_request
    async def _check_auth():
        path = request.path

        # 静态文件和前端页面无需认证
        if path.startswith("/static/") or path == "/" or path == "/index.html":
            return None
        # 认证端点自身无需认证
        if path in ("/api/v1/auth/login", "/api/v1/auth/check"):
            return None
        # 健康检查无需认证
        if path == "/api/v1/system/health":
            return None

        if not auth.check_auth(request):
            return error_response("未授权访问", 401)

        return None

    # ── 全局错误处理 ──
    @app.errorhandler(404)
    async def _not_found(e):
        if request.path.startswith("/api/"):
            return error_response("接口不存在", 404)
        # 非 API 请求返回前端页面（SPA fallback）
        return await app.send_static_file("index.html")

    @app.errorhandler(405)
    async def _method_not_allowed(e):
        return error_response("方法不允许", 405)

    @app.errorhandler(500)
    async def _internal_error(e):
        logger.error(f"Internal server error: {e}")
        return error_response("服务器内部错误", 500)

    @app.errorhandler(Exception)
    async def _unhandled_exception(e):
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return error_response("服务器内部错误", 500)

    # ── 注册 API 路由 ──
    from iris_memory.web.api import register_all_routes

    register_all_routes(app, container)

    # ── 前端首页 ──
    @app.route("/")
    async def _index():
        return await app.send_static_file("index.html")

    return app


class StandaloneWebServer:
    """独立 Web 服务器生命周期管理"""

    def __init__(
        self,
        memory_service: Any,
        host: str = "127.0.0.1",
        port: int = 8089,
        access_key: str = "",
    ) -> None:
        self._memory_service = memory_service
        self._host = host
        self._port = port
        self._access_key = access_key
        self._app: Optional[Quart] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def app(self) -> Optional[Quart]:
        return self._app

    async def start(self) -> None:
        """启动 Web 服务器"""
        self._app = create_app(
            self._memory_service,
            access_key=self._access_key,
        )
        self._shutdown_event = asyncio.Event()

        try:
            from hypercorn.asyncio import serve
            from hypercorn.config import Config as HyperConfig

            config = HyperConfig()
            config.bind = [f"{self._host}:{self._port}"]
            config.accesslog = None
            config.errorlog = "-"
            config.graceful_timeout = 3

            logger.info(f"Web UI 启动于 http://{self._host}:{self._port}")

            self._task = asyncio.create_task(
                serve(self._app, config, shutdown_trigger=self._shutdown_event.wait)  # type: ignore
            )

        except ImportError:
            logger.error("Hypercorn 未安装，无法启动 Web 服务器。请执行: pip install hypercorn")
        except Exception as e:
            logger.error(f"Web 服务器启动失败: {e}")

    async def stop(self) -> None:
        """停止 Web 服务器"""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Web 服务器关闭超时，强制取消")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Web 服务器停止异常: {e}")
            self._task = None

        logger.info("Web 服务器已停止")
