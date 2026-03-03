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
        """初始化主动回复组件（v3）

        Args:
            cfg: 配置管理器
            plugin_data_path: 插件数据目录
            chroma_manager: ChromaDB 管理器（v3 不再依赖）
            embedding_manager: 嵌入管理器（v3 不再依赖）
            shared_state: SharedState 共享状态（v3 不再依赖）
            llm_provider: LLM 提供者（用于 hybrid 模式 LLM 确认）
        """
        from iris_memory.core.constants import LogTemplates

        logger.debug(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))

        followup_after_all = cfg.proactive_followup_after_all_replies
        if not cfg.proactive_reply_enabled and not followup_after_all:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return

        if not plugin_data_path:
            logger.warning("ProactiveModule: plugin_data_path not provided, skipping initialization")
            return

        try:
            from iris_memory.proactive.manager import ProactiveManager
            from iris_memory.proactive.config import ProactiveConfig

            quiet_hours = cfg.proactive_reply_quiet_hours
            group_whitelist_mode = cfg.proactive_reply_group_whitelist_mode
            proactive_mode = cfg.proactive_mode

            config = ProactiveConfig(
                enabled=cfg.proactive_reply_enabled,
                signal_queue_enabled=True,
                followup_enabled=True,
                followup_after_all_replies=cfg.proactive_followup_after_all_replies,
                group_whitelist_mode=group_whitelist_mode,
                proactive_mode=proactive_mode,
                quiet_hours=quiet_hours,
                followup_window_seconds=cfg.proactive_followup_window_seconds,
                max_followup_count=cfg.proactive_max_followup_count,
                max_daily_replies=cfg.proactive_reply_max_daily,
            )

            self._manager = ProactiveManager(
                plugin_data_path=plugin_data_path,
                config=config,
                llm_provider=llm_provider,
            )
            await self._manager.initialize()

            # 设置 AstrBot 上下文（用于 LLM 确认等内部调用）
            if context:
                llm_provider_id = getattr(cfg, 'proactive_llm_provider_id', None) or ""
                self._manager.set_context(
                    astrbot_context=context,
                    llm_provider_id=llm_provider_id or None,
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
