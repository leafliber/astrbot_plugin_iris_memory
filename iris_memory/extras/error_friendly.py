"""错误消息友好化处理器

在消息发送前拦截框架错误消息，替换为友好提示。
（自 v2 processing/message_processor.py 摘出，独立保留）
"""

from typing import Any, Final

from astrbot.api.event import AstrMessageEvent


class ErrorFriendlyMessages:
    """错误消息友好化配置"""

    ERROR_PATTERNS: Final[tuple] = (
        "AstrBot 请求失败",
        "错误类型:",
        "错误信息:",
        "请在平台日志查看",
    )

    DEFAULT_FRIENDLY_MSG: Final[str] = "呜...遇到了一点问题，请稍后再试试吧~"
    NETWORK_ERROR_MSG: Final[str] = "网络好像不太稳定呢，稍后再试试？"
    RATE_LIMIT_MSG: Final[str] = "我需要休息一下，请稍后再来找我~"
    BAD_REQUEST_MSG: Final[str] = "请求出了点问题，稍后再试试吧~"


class ErrorFriendlyProcessor:
    """错误消息友好化处理器

    在消息发送前拦截框架错误消息，替换为友好提示。

    Args:
        config: 插件配置对象（iris_memory.config.Config）
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    def should_process(self, event: AstrMessageEvent) -> bool:
        """检查是否需要处理该事件"""
        return self._is_enabled()

    def process_result(self, result: Any) -> None:
        """处理消息结果，替换错误消息"""
        if not result:
            return

        text = self._get_result_plain_text(result)
        if not text:
            return

        if self._is_framework_error(text):
            friendly_msg = self._get_friendly_error_message(text)
            result.chain.clear()
            result.message(friendly_msg)

    def _is_enabled(self) -> bool:
        """检查错误消息友好化功能是否启用"""
        try:
            if hasattr(self._config, "get"):
                return bool(self._config.get("error_friendly.enable", True))
            return True
        except Exception:
            return True

    def _get_result_plain_text(self, result: Any) -> str:
        """获取消息结果的纯文本内容"""
        if hasattr(result, "get_plain_text"):
            return result.get_plain_text() or ""
        return ""

    def _is_framework_error(self, text: str) -> bool:
        """检测是否为 AstrBot 框架错误消息"""
        text_lower = text.lower()

        match_count = sum(
            1 for pattern in ErrorFriendlyMessages.ERROR_PATTERNS
            if pattern.lower() in text_lower
        )
        if match_count >= 2:
            return True

        if "400" in text or "bad request" in text_lower:
            if any(keyword in text_lower for keyword in ["请求", "request", "error", "failed"]):
                return True

        error_keywords = [
            "error", "failed", "exception", "traceback",
            "请求失败", "错误", "异常"
        ]
        framework_indicators = [
            "platform", "api", "http", "status code",
            "平台", "框架"
        ]

        has_error_keyword = any(kw in text_lower for kw in error_keywords)
        has_framework_indicator = any(ind in text_lower for ind in framework_indicators)

        return has_error_keyword and has_framework_indicator

    def _get_friendly_error_message(self, text: str) -> str:
        """根据错误内容返回合适的友好消息"""
        text_lower = text.lower()

        if "400" in text or "bad request" in text_lower:
            return ErrorFriendlyMessages.BAD_REQUEST_MSG

        if any(kw in text_lower for kw in ["network", "timeout", "连接", "网络"]):
            return ErrorFriendlyMessages.NETWORK_ERROR_MSG

        if any(kw in text_lower for kw in ["rate", "limit", "限流", "频率", "频繁"]):
            return ErrorFriendlyMessages.RATE_LIMIT_MSG

        return ErrorFriendlyMessages.DEFAULT_FRIENDLY_MSG
