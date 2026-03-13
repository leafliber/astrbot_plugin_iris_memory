"""独立 Web 服务器

基于 Quart + Uvicorn 提供 Web UI 和 REST API。
支持 AstrBot 热重启，端口复用。
"""

from __future__ import annotations

import asyncio
import os
import socket
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
    1. 使用 SO_REUSEADDR/SO_REUSEPORT 支持端口复用
    2. 即使旧进程未正常关闭，也能重新绑定端口
    3. 优雅关闭时释放资源
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
        self._server: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None
        self._started: bool = False
        self._should_exit: bool = False

    @property
    def app(self) -> Optional[Quart]:
        return self._app

    def _create_reuse_socket(self) -> socket.socket:
        """创建支持端口复用的 socket
        
        设置 SO_REUSEADDR 和 SO_REUSEPORT，允许：
        1. 快速重启（TIME_WAIT 状态下也能绑定）
        2. 多进程绑定同一端口（某些平台支持）
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # SO_REUSEADDR: 允许重用处于 TIME_WAIT 状态的端口
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # SO_REUSEPORT: 允许多个 socket 绑定同一端口（Linux 3.9+, macOS）
        # 这对于热重启场景特别有用
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        sock.bind((self._host, self._port))
        sock.listen(128)
        
        return sock

    async def start(self) -> None:
        """启动 Web 服务器
        
        使用 Uvicorn + 端口复用，支持 AstrBot 热重启。
        """
        if self._started and self._task and not self._task.done():
            logger.warning("Web 服务器已在运行中，跳过启动")
            return

        self._app = create_app(
            self._memory_service,
            access_key=self._access_key,
        )
        self._should_exit = False

        try:
            import uvicorn
            from uvicorn import Config
            
            config = Config(
                app=self._app,
                host=self._host,
                port=self._port,
                access_log=False,
                log_level="warning",
            )
            
            sock = self._create_reuse_socket()
            
            async def _serve():
                server = uvicorn.Server(config=config)
                
                loop = asyncio.get_event_loop()
                
                async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
                    try:
                        proto = config.http_protocol_class(
                            config=config,
                            app=self._app,
                            logger=server.logger,
                        )
                        proto.connection_made(writer)
                        
                        while not self._should_exit and not server.should_exit:
                            try:
                                data = await asyncio.wait_for(reader.read(65536), timeout=0.5)
                                if not data:
                                    break
                                proto.data_received(data)
                            except asyncio.TimeoutError:
                                continue
                            except Exception:
                                break
                    except Exception as e:
                        logger.debug(f"连接处理错误: {e}")
                    finally:
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except Exception:
                            pass
                
                server_sock = await asyncio.start_server(
                    _handle_connection,
                    sock=sock,
                )
                server.servers = [server_sock]
                
                logger.info(f"Web UI 启动于 http://{self._host}:{self._port}")
                
                while not server.should_exit and not self._should_exit:
                    await asyncio.sleep(0.1)
                
                for srv in server.servers:
                    srv.close()
                    await srv.wait_closed()
            
            self._task = asyncio.create_task(_serve())
            self._started = True

        except ImportError:
            logger.error("Uvicorn 未安装，无法启动 Web 服务器。请执行: pip install uvicorn")
        except OSError as e:
            if "Address already in use" in str(e) or "address already in use" in str(e):
                logger.error(f"端口 {self._port} 已被占用，Web 服务器启动失败")
                logger.error("请检查是否有其他进程占用该端口")
            else:
                logger.error(f"Web 服务器启动失败: {e}")
        except Exception as e:
            logger.error(f"Web 服务器启动失败: {e}")

    async def stop(self) -> None:
        """停止 Web 服务器
        
        触发优雅关闭，等待连接处理完成。
        """
        if not self._started:
            return

        self._started = False
        self._should_exit = True

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
                logger.debug("Web 服务器优雅关闭完成")
            except asyncio.TimeoutError:
                logger.debug("Web 服务器优雅关闭超时，强制取消")
                self._task.cancel()
                try:
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
        self._server = None

        logger.info("Web 服务器已停止")
