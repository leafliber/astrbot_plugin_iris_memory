"""
共享限流工具

提供两种限流器：
- ``CooldownTracker``  ─ 基于时间戳的冷却（per-key）
- ``DailyCallLimiter`` ─ 基于每日计数的限流
"""

from __future__ import annotations

import time
from datetime import date
from typing import Dict, Optional


class CooldownTracker:
    """基于 Dict[key, timestamp] 的冷却追踪器。

    用法::

        tracker = CooldownTracker(cooldown_seconds=60)
        if tracker.is_ready(session_key):
            do_work()
            tracker.record(session_key)
    """

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._last_call: Dict[str, float] = {}

    def is_ready(self, key: str) -> bool:
        """检查 key 是否已过冷却期"""
        last = self._last_call.get(key, 0.0)
        return time.time() - last >= self.cooldown_seconds

    def record(self, key: str) -> None:
        """记录一次调用"""
        self._last_call[key] = time.time()

    def clear(self, key: Optional[str] = None) -> None:
        """清除某个 key 或全部冷却记录"""
        if key is None:
            self._last_call.clear()
        else:
            self._last_call.pop(key, None)


class DailyCallLimiter:
    """基于每日调用计数的限流器。

    ``daily_limit <= 0`` 表示不限制。

    用法::

        limiter = DailyCallLimiter(daily_limit=200)
        if limiter.is_within_limit():
            do_work()
            limiter.increment()
    """

    def __init__(self, daily_limit: int = 0):
        self._daily_limit = daily_limit
        self._call_date: Optional[date] = None
        self._call_count: int = 0

    # ------ internal ------

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if self._call_date != today:
            self._call_date = today
            self._call_count = 0

    # ------ public API ------

    def is_within_limit(self) -> bool:
        """检查是否仍在每日限制内"""
        if self._daily_limit <= 0:
            return True
        self._reset_if_new_day()
        return self._call_count < self._daily_limit

    def increment(self) -> None:
        """记录一次成功调用"""
        self._reset_if_new_day()
        self._call_count += 1

    @property
    def remaining(self) -> int:
        """剩余可用次数。-1 = 无限制。"""
        self._reset_if_new_day()
        if self._daily_limit <= 0:
            return -1
        return max(0, self._daily_limit - self._call_count)
