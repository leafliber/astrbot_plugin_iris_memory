"""
主动回复模块 — 封装主动回复相关组件（v3）
使用 ProactiveManager（SignalQueue + GroupScheduler + FollowUpPlanner）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.proactive.manager import ProactiveManager
    from iris_memory.proactive.reply_sender import ProactiveReplySender

logger = get_logger("module.proactive")


class ProactiveModule:
    """主动回复模块"""

    def __init__(self) -> None:
        self._manager: Optional[ProactiveManager] = None

    @property
    def manager(self) -> Optional[ProactiveManager]:
        """主动回复管理器"""
        return self._manager

    async def initialize(
        self,
        plugin_data_path: Optional[Path] = None,
        llm_provider: Any = None,
        context: Any = None,
    ) -> None:
        """初始化主动回复组件（v3）

        Args:
            plugin_data_path: 插件数据目录
            llm_provider: LLM 提供者（用于 hybrid 模式 LLM 确认）
            context: AstrBot 上下文（用于 LLM 确认等内部调用）
        """
        from iris_memory.config import get_store
        from iris_memory.core.constants import LogTemplates

        cfg = get_store()

        logger.debug(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))

        followup_after_all = cfg.get("proactive_reply.followup_after_all_replies", False)
        proactive_enabled = cfg.get("proactive_reply.enable", False)
        
        if not proactive_enabled and not followup_after_all:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return

        if not plugin_data_path:
            logger.warning("ProactiveModule: plugin_data_path not provided, skipping initialization")
            return

        try:
            from iris_memory.proactive.manager import ProactiveManager

            config = None  # 不再需要，ProactiveManager 内部直接使用 get_store()

            self._manager = ProactiveManager(
                plugin_data_path=plugin_data_path,
                config=config,
                llm_provider=llm_provider,
            )
            await self._manager.initialize()

            # 设置 AstrBot 上下文（用于 LLM 确认等内部调用）
            # 使用智能增强模块的 LLM 提供者（llm_providers.enhanced_provider_id）
            if context:
                llm_provider_id = cfg.get("llm_enhanced.enhanced_provider_id", None)
                self._manager.set_context(
                    astrbot_context=context,
                    llm_provider_id=llm_provider_id,
                )

            logger.info("Proactive manager v3 initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize proactive manager: {e}")
            self._manager = None

    def setup_reply_sender(self, sender: "ProactiveReplySender") -> None:
        """注入主动回复发送器

        在所有组件初始化完成后，由 MemoryService 调用。

        Args:
            sender: ProactiveReplySender 实例
        """
        if self._manager:
            self._manager.set_reply_sender(sender)
            logger.debug("Reply sender wired to ProactiveManager")

    async def stop(self) -> None:
        """停止主动回复管理器"""
        if self._manager:
            try:
                await self._manager.close()
                logger.debug("[Hot-Reload] Proactive manager closed")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error closing proactive manager: {e}")
            self._manager = None
