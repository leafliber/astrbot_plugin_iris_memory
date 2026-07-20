"""
Iris Chat Memory - LLM 调用记录

定义 LLM 调用记录数据结构，用于追踪每次 LLM 调用。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class CallLog:
    """LLM 调用记录

    记录每次 LLM 调用的详细信息，用于统计和追踪。

    Attributes:
        call_id: 调用唯一ID
        timestamp: 调用时间
        module: 调用模块（如 "l1_summarizer"）
        provider_id: Provider ID
        prompt: 输入提示词（可能截断）
        response: 响应文本（可能截断）
        input_tokens: 输入 Token 数
        output_tokens: 输出 Token 数
        duration_ms: 调用耗时（毫秒）
        success: 是否成功
        error_message: 错误信息（失败时）
        metadata: 额外元数据

    Examples:
        >>> log = CallLog(
        ...     call_id="abc123",
        ...     timestamp=datetime.now(),
        ...     module="l1_summarizer",
        ...     provider_id="gpt-4o",
        ...     prompt="Summarize...",
        ...     response="Summary...",
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     duration_ms=1234,
        ...     success=True
        ... )
        >>> log.total_tokens
        150
    """

    call_id: str  # 调用唯一ID
    timestamp: datetime  # 调用时间
    module: str  # 调用模块（如 "l1_summarizer"）
    provider_id: str  # Provider ID
    prompt: str  # 输入提示词（可能截断）
    response: str  # 响应文本（可能截断）
    input_tokens: int  # 输入 Token 数
    output_tokens: int  # 输出 Token 数
    duration_ms: int  # 调用耗时（毫秒）
    success: bool  # 是否成功
    error_message: Optional[str] = None  # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    @property
    def total_tokens(self) -> int:
        """总 Token 数"""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）

        Returns:
            字典表示，timestamp 转换为 ISO 格式字符串

        Examples:
            >>> log_dict = log.to_dict()
            >>> log_dict["call_id"]
            'abc123'
        """
        data = asdict(self)
        # 转换 datetime 为 ISO 格式字符串
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallLog":
        """从字典创建实例

        Args:
            data: 字典数据

        Returns:
            CallLog 实例

        Examples:
            >>> log = CallLog.from_dict({
            ...     "call_id": "abc123",
            ...     "timestamp": "2024-01-01T12:00:00",
            ...     ...
            ... })
        """
        # 转换 ISO 字符串为 datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
