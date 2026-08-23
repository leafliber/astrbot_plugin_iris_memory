"""Task-local hard budgets for bounded background LLM jobs."""

from __future__ import annotations

import asyncio
import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional


class LLMCallBudgetExceeded(RuntimeError):
    """Raised before a Provider call when a task-local budget is exhausted."""


@dataclass
class LLMCallBudget:
    """A concurrency-safe call/runtime budget shared by one background run."""

    max_calls: int
    max_runtime_seconds: float
    min_call_interval_seconds: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    calls: int = 0
    exhausted_reason: str = ""
    _last_call_at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def exhausted(self) -> bool:
        if self.exhausted_reason:
            return True
        if self.max_calls > 0 and self.calls >= self.max_calls:
            return True
        return bool(
            self.max_runtime_seconds > 0
            and time.monotonic() - self.started_at >= self.max_runtime_seconds
        )

    async def before_call(self, module: str) -> None:
        """Reserve one actual call attempt, enforcing spacing atomically."""

        async with self._lock:
            now = time.monotonic()
            if self.max_runtime_seconds > 0:
                remaining = self.max_runtime_seconds - (now - self.started_at)
                if remaining <= 0:
                    self.exhausted_reason = "runtime"
                    raise LLMCallBudgetExceeded(
                        f"LLM task runtime budget exhausted before {module}"
                    )
            else:
                remaining = None

            if self.max_calls > 0 and self.calls >= self.max_calls:
                self.exhausted_reason = "calls"
                raise LLMCallBudgetExceeded(
                    f"LLM task call budget exhausted before {module}"
                )

            delay = max(
                0.0,
                self.min_call_interval_seconds - (now - self._last_call_at),
            )
            if remaining is not None and delay >= remaining:
                self.exhausted_reason = "runtime"
                raise LLMCallBudgetExceeded(
                    f"LLM task runtime budget exhausted before {module}"
                )
            if delay:
                await asyncio.sleep(delay)

            self.calls += 1
            self._last_call_at = time.monotonic()


_CURRENT_BUDGET: contextvars.ContextVar[Optional[LLMCallBudget]] = (
    contextvars.ContextVar("iris_llm_call_budget", default=None)
)


def current_llm_call_budget() -> Optional[LLMCallBudget]:
    return _CURRENT_BUDGET.get()


@contextmanager
def use_llm_call_budget(budget: LLMCallBudget) -> Iterator[LLMCallBudget]:
    """Apply a budget to all LLMManager calls in the current task context."""

    token = _CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _CURRENT_BUDGET.reset(token)
