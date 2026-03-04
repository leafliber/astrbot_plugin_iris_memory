"""
Web UI 模块

封装 Web 管理界面的初始化和生命周期管理。
"""

import asyncio
from typing import Any, Optional, TYPE_CHECKING

from astrbot.api import logger

if TYPE_CHECKING:
    from iris_memory.services.memory_service import MemoryService


class WebUIManager:
    """
    Web UI 管理器

    负责 Web 管理界面的初始化、启动和停止。

    配置项：
    - web_ui_enabled: 是否启用 Web UI
    - web_ui_port: 监听端口
    - web_ui_host: 监听地址
    - web_ui_access_key: 访问密钥（可选）
    """

    def __init__(self, service: "MemoryService") -> None:
        """
        初始化 Web UI 管理器

        Args:
            service: MemoryService 实例
        """
        self._service = service
        self._standalone_web: Optional[Any] = None
        self._server_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """
        初始化并启动 Web 管理界面

        Returns:
            bool: 是否成功启动
        """
        from iris_memory.core.config_manager import get_config_manager
        config_mgr = get_config_manager()

        if not config_mgr.web_ui_enabled:
            return False

        try:
            from iris_memory.web.standalone_server import StandaloneWebServer

            self._standalone_web = StandaloneWebServer(
                memory_service=self._service,
                port=config_mgr.web_ui_port,
                host=config_mgr.web_ui_host,
                access_key=config_mgr.web_ui_access_key,
            )

            self._server_task = asyncio.create_task(self._standalone_web.start())

            key_info = "需访问密钥" if config_mgr.web_ui_access_key else "无需认证"
            logger.info(
                f"Web 管理界面已启动: http://{config_mgr.web_ui_host}:{config_mgr.web_ui_port} ({key_info})"
            )
            return True

        except Exception as e:
            logger.warning(f"Web 管理界面启动失败: {e}")
            return False

    async def stop(self) -> None:
        """停止 Web 管理界面"""
        if self._standalone_web:
            await self._standalone_web.stop()
            self._standalone_web = None

        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("[Hot-Reload] Web server task did not stop in time, cancelling...")
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                logger.debug(f"[Hot-Reload] Error waiting for web server task: {e}")
            finally:
                self._server_task = None

    @property
    def is_running(self) -> bool:
        """检查 Web UI 是否正在运行"""
        return self._standalone_web is not None
