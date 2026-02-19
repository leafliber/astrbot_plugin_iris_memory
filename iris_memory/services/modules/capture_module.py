"""
捕获模块 — 聚合记忆捕获相关组件

包含：CaptureEngine, BatchProcessor, MessageClassifier
"""
from __future__ import annotations

from typing import Optional, Any, Dict, Callable, TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import LogTemplates

if TYPE_CHECKING:
    from iris_memory.capture.capture_engine import MemoryCaptureEngine
    from iris_memory.capture.batch_processor import MessageBatchProcessor
    from iris_memory.capture.message_classifier import MessageClassifier

logger = get_logger("module.capture")


class CaptureModule:
    """捕获模块

    封装 CaptureEngine + BatchProcessor + MessageClassifier 的创建和生命周期。
    """

    def __init__(self) -> None:
        self._capture_engine: Optional[MemoryCaptureEngine] = None
        self._batch_processor: Optional[MessageBatchProcessor] = None
        self._message_classifier: Optional[MessageClassifier] = None

    @property
    def capture_engine(self) -> Optional[MemoryCaptureEngine]:
        return self._capture_engine

    @property
    def batch_processor(self) -> Optional[MessageBatchProcessor]:
        return self._batch_processor

    @property
    def message_classifier(self) -> Optional[MessageClassifier]:
        return self._message_classifier

    # ── 初始化 ──

    def init_capture_engine(
        self,
        chroma_manager: Any,
        emotion_analyzer: Any,
        rif_scorer: Any,
        llm_sensitivity_detector: Any = None,
        llm_trigger_detector: Any = None,
        llm_conflict_resolver: Any = None,
    ) -> None:
        """初始化记忆捕获引擎"""
        from iris_memory.capture.capture_engine import MemoryCaptureEngine

        self._capture_engine = MemoryCaptureEngine(
            chroma_manager=chroma_manager,
            emotion_analyzer=emotion_analyzer,
            rif_scorer=rif_scorer,
            llm_sensitivity_detector=llm_sensitivity_detector,
            llm_trigger_detector=llm_trigger_detector,
            llm_conflict_resolver=llm_conflict_resolver,
        )
        logger.info("CaptureEngine initialized")

    def init_message_classifier(
        self,
        emotion_analyzer: Any,
        llm_processor: Any = None,
    ) -> None:
        """初始化消息分类器"""
        from iris_memory.capture.message_classifier import MessageClassifier
        from iris_memory.core.defaults import DEFAULTS

        self._message_classifier = MessageClassifier(
            trigger_detector=(
                self._capture_engine.trigger_detector
                if self._capture_engine
                else None
            ),
            emotion_analyzer=emotion_analyzer,
            llm_processor=llm_processor,
            config={
                "llm_processing_mode": DEFAULTS.message_processing.llm_processing_mode,
                "immediate_trigger_confidence": DEFAULTS.message_processing.immediate_trigger_confidence,
                "immediate_emotion_intensity": DEFAULTS.message_processing.immediate_emotion_intensity,
            },
        )
        logger.info("MessageClassifier initialized")

    async def init_batch_processor(
        self,
        cfg: Any,
        llm_processor: Any = None,
        proactive_manager: Any = None,
        on_save_callback: Optional[Callable] = None,
    ) -> None:
        """初始化批量消息处理器"""
        from iris_memory.capture.batch_processor import MessageBatchProcessor
        from iris_memory.core.defaults import DEFAULTS

        batch_config = {
            "short_message_threshold": cfg.short_message_threshold,
            "merge_time_window": cfg.merge_time_window,
            "max_merge_count": cfg.max_merge_count,
            "llm_cooldown_seconds": 60,
            "summary_interval_seconds": 300,
        }

        threshold_count = cfg.batch_threshold_count
        use_llm = cfg.use_llm

        self._batch_processor = MessageBatchProcessor(
            capture_engine=self._capture_engine,
            llm_processor=llm_processor,
            proactive_manager=proactive_manager,
            threshold_count=threshold_count,
            threshold_interval=DEFAULTS.message_processing.batch_threshold_interval,
            processing_mode=DEFAULTS.message_processing.batch_processing_mode,
            use_llm_summary=use_llm and llm_processor is not None,
            on_save_callback=on_save_callback,
            config=batch_config,
            config_manager=cfg,
        )
        await self._batch_processor.start()
        logger.info(f"BatchProcessor initialized (threshold={threshold_count})")

    # ── 配置应用 ──

    def apply_config(self, cfg: Any) -> None:
        """应用配置到 CaptureEngine"""
        from iris_memory.core.defaults import DEFAULTS

        if self._capture_engine:
            self._capture_engine.set_config(
                {
                    "auto_capture": cfg.enable_memory,
                    "min_confidence": DEFAULTS.memory.min_confidence,
                    "rif_threshold": cfg.rif_threshold,
                }
            )

    # ── 生命周期 ──

    async def stop(self) -> None:
        """停止批量处理器"""
        if self._batch_processor:
            try:
                await self._batch_processor.stop()
                logger.debug("[Hot-Reload] Batch processor stopped")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error stopping batch processor: {e}")
