"""独立 Web 服务器

基于 Quart + Hypercorn 提供 Web UI 和 REST API。
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
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
    """独立 Web 服务器生命周期管理
    
    适配 AstrBot 热重启机制：
    1. 优雅关闭时释放端口
    2. 启动前等待端口可用
    3. 支持 SO_REUSEADDR 快速重启
    """

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
        self._started: bool = False

    @property
    def app(self) -> Optional[Quart]:
        return self._app

    def _is_port_in_use(self) -> bool:
        """检查端口是否被占用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((self._host, self._port))
                return result == 0
        except Exception:
            return False

    async def _wait_for_port(self, timeout: float = 5.0) -> bool:
        """等待端口变为可用
        
        AstrBot 热重启时使用：等待旧进程释放端口
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_port_in_use():
                return True
            await asyncio.sleep(0.1)
        return False

    async def start(self) -> None:
        """启动 Web 服务器
        
        适配 AstrBot 热重启流程：
        1. 先等待端口释放（防止旧进程未完全关闭）
        2. 使用 Hypercorn 标准方式启动
        """
        if self._started and self._task and not self._task.done():
            logger.warning("Web 服务器已在运行中，跳过启动")
            return

        # AstrBot 热重启时，等待旧进程释放端口
        if self._is_port_in_use():
            logger.info(f"端口 {self._port} 被占用，等待释放...")
            if not await self._wait_for_port(timeout=5.0):
                logger.warning(f"端口 {self._port} 未能在 5 秒内释放，尝试强制绑定...")

        self._app = create_app(
            self._memory_service,
            access_key=self._access_key,
        )
        self._shutdown_event = asyncio.Event()

        try:
            from hypercorn.asyncio.run import worker_serve
            from hypercorn.config import Config as HyperConfig
            from hypercorn.utils import wrap_app

            config = HyperConfig()
            # 使用标准 bind 格式，Hypercorn 会自动创建 socket
            config.bind = [f"{self._host}:{self._port}"]
            config.accesslog = None
            config.errorlog = "-"
            # 优雅关闭配置
            config.graceful_timeout = 1.0
            config.shutdown_timeout = 2.0

            app_wrapper = wrap_app(self._app, config.wsgi_max_body_size, "asgi")

            async def _serve_with_error_handling():
                try:
                    await worker_serve(
                        app_wrapper,
                        config,
                        shutdown_trigger=self._shutdown_event.wait,
                    )
                except OSError as e:
                    if "Address already in use" in str(e) or "address already in use" in str(e):
                        logger.error(f"端口 {self._port} 已被占用，Web 服务器启动失败")
                    else:
                        logger.error(f"Web 服务器运行错误: {e}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Web 服务器运行错误: {e}")

            self._task = asyncio.create_task(_serve_with_error_handling())
            self._started = True

            # 等待服务器真正启动
            await asyncio.sleep(1)
            
            # 检查任务是否还在运行
            if self._task.done():
                exc = self._task.exception()
                if exc:
                    logger.error(f"Web 服务器启动失败: {exc}")
                else:
                    logger.error("Web 服务器意外退出")
                self._started = False
                self._task = None
                return
            
            if self._is_port_in_use():
                logger.info(f"Web UI 启动于 http://{self._host}:{self._port}")
            else:
                logger.warning("Web 服务器可能未成功启动，端口未监听")

        except ImportError:
            logger.error("Hypercorn 未安装，无法启动 Web 服务器。请执行: pip install hypercorn")
        except Exception as e:
            logger.error(f"Web 服务器启动失败: {e}")

    async def stop(self) -> None:
        """停止 Web 服务器
        
        适配 AstrBot 热重启流程：
        1. 触发优雅关闭信号
        2. 等待活跃连接完成
        3. 强制关闭超时连接
        4. 确保端口释放
        """
        if not self._started:
            return

        self._started = False

        # 触发 Hypercorn 优雅关闭
        if self._shutdown_event:
            self._shutdown_event.set()

        # 等待任务完成
        if self._task:
            try:
                # 给优雅关闭 1 秒时间
                await asyncio.wait_for(self._task, timeout=1.0)
                logger.debug("Web 服务器优雅关闭完成")
            except asyncio.TimeoutError:
                logger.debug("Web 服务器优雅关闭超时，强制取消")
                self._task.cancel()
                try:
                    # 再给 0.5 秒强制关闭
                    await asyncio.wait_for(self._task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Web 服务器停止异常: {e}")
            finally:
                self._task = None

        self._app = None
        self._shutdown_event = None

        # 等待端口完全释放（最多 0.5 秒）
        for i in range(5):
            if not self._is_port_in_use():
                break
            await asyncio.sleep(0.1)

        logger.info("Web 服务器已停止")
