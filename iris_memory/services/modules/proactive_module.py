"""
主动回复模块 — 封装主动回复相关组件
使用 ProactiveManager（三级检测器 + 场景库 + 反馈闭环）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.proactive.manager import ProactiveManager

logger = get_logger("module.proactive")


class ProactiveModule:
    """主动回复模块"""

    def __init__(self) -> None:
        self._manager: Optional[ProactiveManager] = None

    @property
    def manager(self) -> Optional[ProactiveManager]:
        """主动回复管理器"""
        return self._manager

    # 向后兼容别名
    @property
    def new_manager(self) -> Optional[ProactiveManager]:
        return self._manager

    async def initialize(
        self,
        cfg: Any,
        plugin_data_path: Optional[Path] = None,
        chroma_manager: Any = None,
        embedding_manager: Any = None,
        shared_state: Any = None,
        llm_provider: Any = None,
        # 兼容占位符，不再使用
        context: Any = None,
        emotion_analyzer: Any = None,
        llm_proactive_reply_detector: Any = None,
    ) -> None:
        """初始化主动回复组件

        Args:
            cfg: 配置管理器
            plugin_data_path: 插件数据目录
            chroma_manager: ChromaDB 管理器
            embedding_manager: 嵌入管理器
            shared_state: SharedState 共享状态
            llm_provider: LLM 提供者
        """
        from iris_memory.core.constants import LogTemplates

        logger.debug(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))

        if not cfg.proactive_reply_enabled:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return

        if not plugin_data_path:
            logger.warning("ProactiveModule: plugin_data_path not provided, skipping initialization")
            return

        try:
            from iris_memory.proactive.manager import ProactiveManager

            quiet_start = getattr(cfg, "proactive_quiet_hours_start", 23)
            quiet_end = getattr(cfg, "proactive_quiet_hours_end", 7)
            personality = getattr(cfg, "proactive_personality", "balanced")
            max_history = getattr(cfg, "proactive_context_max_history", 10)
            max_text_tokens = getattr(cfg, "proactive_context_max_text_tokens", 150)

            self._manager = ProactiveManager(
                plugin_data_path=plugin_data_path,
                chroma_manager=chroma_manager,
                embedding_manager=embedding_manager,
                shared_state=shared_state,
                llm_provider=llm_provider,
                personality=personality,
                quiet_hours=[quiet_start, quiet_end],
                max_history=max_history,
                max_text_tokens=max_text_tokens,
            )
            await self._manager.initialize()
            logger.info("Proactive manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize proactive manager: {e}")
            self._manager = None

    async def stop(self) -> None:
        """停止主动回复管理器"""
        if self._manager:
            try:
                await self._manager.close()
                logger.debug("[Hot-Reload] Proactive manager closed")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error closing proactive manager: {e}")
            self._manager = None
