"""Web UI 管理器

由 main.py 的 IrisMemoryPlugin 调用，管理 Web 服务器的初始化和停止。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from iris_memory.config import get_store
from iris_memory.utils.logger import get_logger

logger = get_logger("web_ui")


class WebUIManager:
    """Web UI 生命周期管理器"""

    def __init__(self, memory_service: Any) -> None:
        self._memory_service = memory_service
        self._server: Optional[Any] = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """读取配置并启动 Web 服务器（如已启用）"""
        if self._initialized and self._server:
            logger.debug("Web UI 已初始化，跳过重复初始化")
            return

        store = get_store()
        enabled = store.get("web_ui.enable", False)
        if not enabled:
            logger.info("Web UI 已禁用")
            return

        host = store.get("web_ui.host", "127.0.0.1")
        port = store.get("web_ui.port", 8089)
        access_key = store.get("web_ui.access_key", "")

        try:
            from iris_memory.web.server import StandaloneWebServer

            self._server = StandaloneWebServer(
                memory_service=self._memory_service,
                host=host,
                port=port,
                access_key=access_key,
            )
            await self._server.start()
            self._initialized = True
        except Exception as e:
            logger.error(f"Web UI 启动失败: {e}")
            self._server = None

    async def stop(self) -> None:
        """停止 Web 服务器"""
        if not self._initialized:
            return

        self._initialized = False

        if self._server:
            try:
                await asyncio.wait_for(self._server.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Web UI 停止超时")
            except Exception as e:
                logger.warning(f"Web UI 停止异常: {e}")
            finally:
                self._server = None
