"""
检测器基类

所有三级检测器（Rule / Vector / LLM）都继承自此基类，
统一接口和生命周期管理。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from iris_memory.proactive.core.models import ProactiveContext
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.detector.base")


class BaseDetector(ABC):
    """检测器抽象基类

    所有检测器实现以下契约：
    1. `initialize()` — 异步初始化（加载模型、连接存储等）
    2. `detect(context)` — 执行检测，返回具体的 Result dataclass
    3. `close()` — 释放资源

    子类必须实现 `detect` 方法，`initialize` 和 `close` 可选。
    """

    def __init__(self, name: str = "base") -> None:
        self._name = name
        self._initialized = False

    @property
    def name(self) -> str:
        """检测器名称"""
        return self._name

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    async def initialize(self) -> None:
        """异步初始化

        子类可覆盖此方法来执行异步初始化（如加载模型）。
        基类提供默认的空实现。
        """
        self._initialized = True

    @abstractmethod
    async def detect(self, context: ProactiveContext) -> Any:
        """执行检测

        Args:
            context: 主动回复上下文

        Returns:
            检测器特定的结果 dataclass
        """
        ...

    async def close(self) -> None:
        """释放资源

        子类可覆盖此方法来执行清理操作。
        基类提供默认的空实现。
        """
        self._initialized = False
