"""
主动回复模块 — 聚合主动回复相关组件

包含：ProactiveReplyManager, ReplyDetector
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.proactive.proactive_manager import ProactiveReplyManager

logger = get_logger("module.proactive")


class ProactiveModule:
    """主动回复模块"""

    def __init__(self) -> None:
        self._proactive_manager: Optional[ProactiveReplyManager] = None
        self._reply_detector: Optional[Any] = None

    @property
    def proactive_manager(self) -> Optional[ProactiveReplyManager]:
        return self._proactive_manager

    @property
    def reply_detector(self) -> Optional[Any]:
        return self._reply_detector

    async def initialize(
        self,
        cfg: Any,
        context: Any,
        emotion_analyzer: Any,
        llm_proactive_reply_detector: Any = None,
    ) -> None:
        """初始化主动回复组件"""
        from iris_memory.core.defaults import DEFAULTS
        from iris_memory.core.constants import LogTemplates

        logger.debug(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))

        if not cfg.proactive_reply_enabled:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return

        # 选择检测器
        reply_detector = self._resolve_detector(
            emotion_analyzer, llm_proactive_reply_detector
        )

        event_queue = getattr(context, "_event_queue", None)
        if event_queue is None:
            logger.warning(
                "Cannot access context._event_queue, "
                "proactive reply event dispatch may not work"
            )

        # 创建管理器
        from iris_memory.proactive.proactive_manager import ProactiveReplyManager

        self._proactive_manager = ProactiveReplyManager(
            astrbot_context=context,
            reply_detector=reply_detector,
            event_queue=event_queue,
            config={
                "enable_proactive_reply": cfg.proactive_reply_enabled,
                "reply_cooldown": DEFAULTS.proactive_reply.cooldown_seconds,
                "max_daily_replies": cfg.proactive_reply_max_daily,
                "group_whitelist_mode": cfg.proactive_reply_group_whitelist_mode,
                # 智能增强配置
                "smart_boost_enabled": cfg.smart_boost_enabled,
                "smart_boost_window_seconds": cfg.smart_boost_window_seconds,
                "smart_boost_score_multiplier": cfg.smart_boost_score_multiplier,
                "smart_boost_reply_threshold": cfg.smart_boost_reply_threshold,
            },
            config_manager=cfg,
        )
        await self._proactive_manager.initialize()
        logger.debug("Proactive reply components initialized")

    def _resolve_detector(
        self,
        emotion_analyzer: Any,
        llm_detector: Any = None,
    ) -> Any:
        if llm_detector:
            logger.debug("Using LLM-enhanced proactive reply detector")
            return llm_detector

        from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
        from iris_memory.core.defaults import DEFAULTS

        self._reply_detector = ProactiveReplyDetector(
            emotion_analyzer=emotion_analyzer,
            config={
                "high_emotion_threshold": DEFAULTS.proactive_reply.high_emotion_threshold,
                "question_threshold": DEFAULTS.proactive_reply.question_threshold,
            },
        )
        return self._reply_detector

    async def stop(self) -> None:
        """停止主动回复管理器"""
        if self._proactive_manager:
            try:
                await self._proactive_manager.stop()
                logger.debug("[Hot-Reload] Proactive manager stopped")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error stopping proactive manager: {e}")
