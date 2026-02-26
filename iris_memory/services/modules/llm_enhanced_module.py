"""
LLM增强模块 — 聚合所有 LLM 检测器和 LLM 处理器

包含 6 个 LLM 增强检测器 + LLMMessageProcessor，
使用注册表模式统一管理。
"""
from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.processing.llm_processor import LLMMessageProcessor

logger = get_logger("module.llm_enhanced")

# LLM 检测器名称常量
SENSITIVITY = "sensitivity"
TRIGGER = "trigger"
EMOTION = "emotion"
PROACTIVE = "proactive"
CONFLICT = "conflict"
RETRIEVAL = "retrieval"


class LLMEnhancedModule:
    """LLM 增强模块

    使用注册表模式替代 6 个独立字段，统一管理所有 LLM 检测器。
    """

    def __init__(self) -> None:
        self._detectors: Dict[str, Any] = {}
        self._llm_processor: Optional[LLMMessageProcessor] = None

    # ── 注册表访问 ──

    def get_detector(self, name: str) -> Optional[Any]:
        """获取指定名称的 LLM 检测器"""
        return self._detectors.get(name)

    def register_detector(self, name: str, detector: Any) -> None:
        """注册一个 LLM 检测器"""
        self._detectors[name] = detector
        logger.debug(f"LLM detector registered: {name}")

    def has_detector(self, name: str) -> bool:
        return name in self._detectors

    @property
    def registered_names(self) -> list[str]:
        return list(self._detectors.keys())

    # ── 便捷属性（兼容旧代码） ──

    @property
    def sensitivity_detector(self) -> Optional[Any]:
        return self._detectors.get(SENSITIVITY)

    @property
    def trigger_detector(self) -> Optional[Any]:
        return self._detectors.get(TRIGGER)

    @property
    def emotion_analyzer(self) -> Optional[Any]:
        return self._detectors.get(EMOTION)

    @property
    def proactive_reply_detector(self) -> Optional[Any]:
        return self._detectors.get(PROACTIVE)

    @property
    def conflict_resolver(self) -> Optional[Any]:
        return self._detectors.get(CONFLICT)

    @property
    def retrieval_router(self) -> Optional[Any]:
        return self._detectors.get(RETRIEVAL)

    # ── LLM Processor ──

    @property
    def llm_processor(self) -> Optional[LLMMessageProcessor]:
        return self._llm_processor

    # ── 初始化 ──

    async def initialize(self, cfg: Any, context: Any) -> None:
        """初始化所有 LLM 增强组件"""
        if not cfg.llm_enhanced_enabled:
            logger.debug("LLM enhanced: all modules using rule mode")
            return

        from iris_memory.core.detection.llm_enhanced_base import DetectionMode
        from iris_memory.capture.detector.llm_sensitivity_detector import LLMSensitivityDetector
        from iris_memory.capture.detector.llm_trigger_detector import LLMTriggerDetector
        from iris_memory.analysis.emotion.llm_emotion_analyzer import LLMEmotionAnalyzer
        from iris_memory.proactive.llm_proactive_reply_detector import LLMProactiveReplyDetector
        from iris_memory.capture.conflict.llm_conflict_resolver import LLMConflictResolver
        from iris_memory.retrieval.llm_retrieval_router import LLMRetrievalRouter

        provider_id = cfg.llm_enhanced_provider_id
        logger.debug(f"[DEBUG] enhanced_provider_id from config: {repr(provider_id)}")
        logger.debug(f"[DEBUG] raw config value: {repr(cfg.get('llm_providers.enhanced_provider_id'))}")
        modes: list[str] = []

        _MAPPING = [
            (SENSITIVITY, "sensitivity_mode", LLMSensitivityDetector),
            (TRIGGER, "trigger_mode", LLMTriggerDetector),
            (EMOTION, "emotion_mode", LLMEmotionAnalyzer),
            (PROACTIVE, "proactive_mode", LLMProactiveReplyDetector),
            (CONFLICT, "conflict_mode", LLMConflictResolver),
            (RETRIEVAL, "retrieval_mode", LLMRetrievalRouter),
        ]

        for name, cfg_attr, cls in _MAPPING:
            mode_str = getattr(cfg, cfg_attr)
            if mode_str != "rule":
                detector = cls(
                    astrbot_context=context,
                    provider_id=provider_id,
                    mode=DetectionMode(mode_str),
                )
                self.register_detector(name, detector)
                modes.append(f"{name}={mode_str}")

        if modes:
            logger.debug(f"LLM enhanced enabled: {', '.join(modes)}")
        else:
            logger.debug("LLM enhanced: all modules using rule mode")

    async def init_llm_processor(
        self,
        context: Any,
        cfg: Any,
        lifecycle_manager: Any = None,
    ) -> None:
        """初始化 LLM 消息处理器"""
        from iris_memory.processing.llm_processor import LLMMessageProcessor

        # 检查 context 是否为 None
        if context is None:
            logger.warning("AstrBot context is None, LLM features disabled")
            self._llm_processor = None
            return

        self._llm_processor = LLMMessageProcessor(
            astrbot_context=context,
            max_tokens=cfg.get(
                "message_processing.llm_max_tokens_for_summary",
                500,
            ),
            provider_id=cfg.llm_provider_id,
        )
        llm_ready = await self._llm_processor.initialize()
        if llm_ready and lifecycle_manager:
            lifecycle_manager.set_llm_provider(self._llm_processor)
            logger.debug("LLM processor ready")
        elif not llm_ready:
            logger.warning("LLM context not available, LLM features disabled")
            self._llm_processor = None
