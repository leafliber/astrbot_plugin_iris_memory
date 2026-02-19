"""
记忆捕获结构化日志器

为记忆捕获全生命周期提供统一的 DEBUG 日志。
所有日志事件均以 ``CAPTURE.`` 前缀标识，可通过 ``log_level=DEBUG`` 激活。

用法示例::

    from iris_memory.capture.capture_logger import capture_log
    capture_log.capture_start(user_id, message)
    capture_log.trigger_detected(user_id, triggers)
    capture_log.capture_ok(user_id, memory_id, memory_type)
"""

from typing import Any, Dict, List, Optional

from iris_memory.utils.logger import get_logger

_logger = get_logger("capture_logger")


class CaptureLogger:
    """记忆捕获结构化日志器 — 统一 DEBUG 输出"""

    def capture_start(
        self,
        user_id: str,
        message: str,
        is_user_requested: bool = False
    ) -> None:
        flag = "USER_REQUESTED" if is_user_requested else "AUTO"
        _logger.debug(
            f"CAPTURE.START user={user_id} flag={flag} msg_len={len(message)}"
        )

    def negative_sample(self, user_id: str, reason: str = "chat") -> None:
        _logger.debug(
            f"CAPTURE.NEGATIVE_SAMPLE user={user_id} reason={reason}"
        )

    def trigger_detected(
        self,
        user_id: str,
        triggers: list
    ) -> None:
        if not triggers:
            _logger.debug(f"CAPTURE.TRIGGER.NONE user={user_id}")
            return
        for t in triggers:
            _logger.debug(
                f"CAPTURE.TRIGGER.DETECT user={user_id} "
                f"type={getattr(t, 'type', '?')} "
                f"conf={getattr(t, 'confidence', 0):.2f} "
                f"pattern={_trunc(str(getattr(t, 'pattern', '')), 20)}"
            )

    def sensitivity_detected(
        self,
        user_id: str,
        level: str,
        entities: List[str]
    ) -> None:
        _logger.debug(
            f"CAPTURE.SENSITIVITY user={user_id} "
            f"level={level} "
            f"entities={_trunc(str(entities[:3]), 40)}"
        )

    def sensitivity_filtered(self, user_id: str, level: str) -> None:
        _logger.warning(
            f"CAPTURE.SENSITIVITY.FILTERED user={user_id} level={level}"
        )

    def emotion_analyzed(
        self,
        user_id: str,
        primary: str,
        intensity: float,
        confidence: float
    ) -> None:
        _logger.debug(
            f"CAPTURE.EMOTION user={user_id} "
            f"primary={primary} "
            f"intensity={intensity:.2f} "
            f"conf={confidence:.2f}"
        )

    def memory_created(
        self,
        user_id: str,
        memory_type: str,
        scope: str
    ) -> None:
        _logger.debug(
            f"CAPTURE.MEMORY.CREATE user={user_id} "
            f"type={memory_type} scope={scope}"
        )

    def quality_assessed(
        self,
        user_id: str,
        confidence: float,
        quality_level: str
    ) -> None:
        _logger.debug(
            f"CAPTURE.QUALITY user={user_id} "
            f"conf={confidence:.2f} level={quality_level}"
        )

    def rif_calculated(
        self,
        user_id: str,
        rif_score: float
    ) -> None:
        _logger.debug(
            f"CAPTURE.RIF user={user_id} "
            f"score={rif_score:.3f}"
        )

    def storage_determined(
        self,
        user_id: str,
        storage_layer: str,
        reason: str
    ) -> None:
        _logger.debug(
            f"CAPTURE.STORAGE user={user_id} "
            f"layer={storage_layer} reason={reason}"
        )

    def duplicate_found(self, user_id: str, memory_id: str) -> None:
        _logger.info(
            f"CAPTURE.DUPLICATE user={user_id} dup_of={memory_id[:8]}..."
        )

    def conflict_detected(
        self,
        user_id: str,
        conflict_count: int,
        resolved: bool
    ) -> None:
        _logger.debug(
            f"CAPTURE.CONFLICT user={user_id} "
            f"count={conflict_count} resolved={resolved}"
        )

    def capture_ok(
        self,
        user_id: str,
        memory_id: str,
        memory_type: str,
        confidence: float,
        rif_score: float,
        storage_layer: str
    ) -> None:
        _logger.info(
            f"CAPTURE.OK user={user_id} "
            f"id={memory_id[:8]}... "
            f"type={memory_type} "
            f"conf={confidence:.2f} "
            f"rif={rif_score:.3f} "
            f"layer={storage_layer}"
        )

    def capture_skip(self, user_id: str, reason: str) -> None:
        _logger.debug(
            f"CAPTURE.SKIP user={user_id} reason={reason}"
        )

    def capture_error(self, user_id: str, error: Exception) -> None:
        _logger.error(
            f"CAPTURE.ERROR user={user_id} error={error}"
        )

    def llm_trigger_detection_failed(self, user_id: str, error: str) -> None:
        _logger.debug(
            f"CAPTURE.LLM.TRIGGER.FALLBACK user={user_id} error={_trunc(error, 100)}"
        )

    def llm_sensitivity_detection_failed(self, user_id: str, error: str) -> None:
        _logger.debug(
            f"CAPTURE.LLM.SENSITIVITY.FALLBACK user={user_id} error={_trunc(error, 100)}"
        )


def _trunc(value: Any, max_len: int = 60) -> str:
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "..."


capture_log = CaptureLogger()
