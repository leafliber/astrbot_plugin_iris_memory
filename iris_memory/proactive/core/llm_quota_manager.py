"""
LLM 限流管理器

内存+SQLite 持久化的限流策略。
热路径只做内存操作，后台任务处理持久化。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.llm_quota")


class LLMQuotaManager:
    """LLM 限流管理器

    设计要点：
    1. 重启后从 SQLite 恢复当前小时配额
    2. 热路径只做内存操作，不阻塞
    3. SQLite 写入通过后台任务异步完成
    """

    def __init__(self, feedback_store: Optional[Any] = None) -> None:
        self._feedback_store = feedback_store
        self._hourly_counts: Dict[str, int] = {}
        self._current_hour: int = datetime.now().hour
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """从 SQLite 恢复当前小时配额"""
        if self._initialized:
            return

        if self._feedback_store:
            try:
                current_hour = datetime.now().hour
                self._current_hour = current_hour
                quotas = await self._feedback_store.get_llm_quotas_for_hour(
                    current_hour
                )
                for q in quotas:
                    self._hourly_counts[q["session_key"]] = q["count"]
                logger.info(
                    f"LLMQuotaManager initialized with "
                    f"{len(self._hourly_counts)} sessions"
                )
            except Exception as e:
                logger.warning(f"Failed to restore LLM quotas: {e}")

        self._initialized = True

    async def acquire(
        self, session_key: str, max_per_hour: int = 5
    ) -> bool:
        """尝试获取 LLM 调用配额

        热路径只做内存操作，SQLite 写入异步完成。

        Args:
            session_key: 会话键
            max_per_hour: 每小时最大调用数

        Returns:
            是否获取成功
        """
        if not self._initialized:
            await self.initialize()

        current_hour = datetime.now().hour

        # 跨小时重置
        if current_hour != self._current_hour:
            async with self._lock:
                if current_hour != self._current_hour:
                    self._current_hour = current_hour
                    self._hourly_counts.clear()

        async with self._lock:
            count = self._hourly_counts.get(session_key, 0)
            if count >= max_per_hour:
                return False

            self._hourly_counts[session_key] = count + 1

        # 异步持久化
        if self._feedback_store:
            try:
                asyncio.create_task(
                    self._feedback_store.update_llm_quota(
                        session_key, current_hour, count + 1
                    )
                )
            except Exception:
                pass

        return True

    def get_remaining(self, session_key: str, max_per_hour: int = 5) -> int:
        """获取剩余配额"""
        count = self._hourly_counts.get(session_key, 0)
        return max(0, max_per_hour - count)
