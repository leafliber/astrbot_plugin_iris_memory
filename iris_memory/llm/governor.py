"""插件级 LLM 请求治理器。

统一提供全局/Provider 并发、后台并发、RPM、最小调用间隔、优先级 aging、
有界排队、熔断以及可观测指标。该组件不创建执行任务，只发放有时限的 lease，
因此业务任务的生命周期仍由各自的有界工作队列管理。
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Deque, Dict, Optional

from .policy import CallPriority, coerce_priority


class LLMGovernorError(RuntimeError):
    """Governor 拒绝请求的基类。"""


class LLMQueueFullError(LLMGovernorError):
    """等待队列已达到上限。"""


class LLMQueueTimeoutError(LLMGovernorError, asyncio.TimeoutError):
    """等待调用资格超时。"""


class LLMCircuitOpenError(LLMGovernorError):
    """Provider 熔断器处于打开状态。"""


@dataclass(frozen=True)
class GovernorSettings:
    global_concurrency: int = 4
    provider_concurrency: int = 2
    provider_background_concurrency: int = 1
    provider_rpm: int = 30
    provider_min_interval_ms: int = 0
    queue_limit: int = 500
    interactive_reserved_slots: int = 1
    circuit_failure_threshold: int = 5
    circuit_open_seconds: float = 60.0
    aging_seconds: float = 30.0

    def normalized(self) -> "GovernorSettings":
        global_limit = max(1, int(self.global_concurrency))
        provider_limit = max(1, int(self.provider_concurrency))
        return GovernorSettings(
            global_concurrency=global_limit,
            provider_concurrency=provider_limit,
            provider_background_concurrency=max(
                1, min(provider_limit, int(self.provider_background_concurrency))
            ),
            provider_rpm=max(0, int(self.provider_rpm)),
            provider_min_interval_ms=max(0, int(self.provider_min_interval_ms)),
            queue_limit=max(1, int(self.queue_limit)),
            interactive_reserved_slots=max(
                0, min(global_limit - 1, int(self.interactive_reserved_slots))
            ),
            circuit_failure_threshold=max(0, int(self.circuit_failure_threshold)),
            circuit_open_seconds=max(0.0, float(self.circuit_open_seconds)),
            aging_seconds=max(0.1, float(self.aging_seconds)),
        )


@dataclass
class _Waiter:
    waiter_id: str
    provider_id: str
    module: str
    priority: CallPriority
    enqueued_at: float
    sequence: int
    queue_depth_at_start: int


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    open_until: float = 0.0


class LLMLease:
    """一次 LLM 调用资格；释放操作幂等。"""

    def __init__(
        self,
        governor: "LLMCallGovernor",
        *,
        lease_id: str,
        provider_id: str,
        module: str,
        priority: CallPriority,
        queue_wait_ms: int,
        queue_depth_at_start: int,
        in_flight_at_start: int,
    ) -> None:
        self._governor = governor
        self.lease_id = lease_id
        self.provider_id = provider_id
        self.module = module
        self.priority = priority
        self.queue_wait_ms = queue_wait_ms
        self.queue_depth_at_start = queue_depth_at_start
        self.in_flight_at_start = in_flight_at_start
        self._released = False
        self._watchdog: Optional[asyncio.Task] = None

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> bool:
        return await self._governor.release(self.lease_id)

    async def __aenter__(self) -> "LLMLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class LLMCallGovernor:
    """在一个事件循环内治理所有 Provider 调用。"""

    def __init__(self, settings: GovernorSettings | None = None) -> None:
        self.settings = (settings or GovernorSettings()).normalized()
        self._condition = asyncio.Condition()
        self._pending: list[_Waiter] = []
        self._leases: Dict[str, LLMLease] = {}
        self._active_global = 0
        self._active_provider: Dict[str, int] = defaultdict(int)
        self._active_background: Dict[str, int] = defaultdict(int)
        self._starts: Dict[str, Deque[float]] = defaultdict(deque)
        self._last_start: Dict[str, float] = {}
        self._circuits: Dict[str, _CircuitState] = defaultdict(_CircuitState)
        self._sequence = 0
        self._closed = False
        self._queue_wait_samples: Deque[int] = deque(maxlen=2000)
        self._calls_per_minute: Dict[tuple[str, str], Deque[float]] = defaultdict(deque)
        self._rejected_by_backpressure = 0
        self._watchdog_releases = 0
        self._background_budget_exhausted = 0
        self._singleflight_join_count = 0
        self._retry_count = 0

    @staticmethod
    def _provider_key(provider_id: str) -> str:
        return provider_id or "__default__"

    def _effective_priority(self, waiter: _Waiter, now: float) -> int:
        aged_levels = int((now - waiter.enqueued_at) / self.settings.aging_seconds)
        return max(int(CallPriority.INTERACTIVE), int(waiter.priority) - aged_levels)

    def _is_circuit_open(self, provider_id: str, now: float) -> bool:
        circuit = self._circuits[provider_id]
        if circuit.open_until <= 0:
            return False
        if now < circuit.open_until:
            return True
        circuit.open_until = 0.0
        circuit.consecutive_failures = 0
        return False

    def _prune_starts(self, provider_id: str, now: float) -> None:
        starts = self._starts[provider_id]
        cutoff = now - 60.0
        while starts and starts[0] <= cutoff:
            starts.popleft()

    def _rate_delay(self, provider_id: str, now: float) -> float:
        delay = 0.0
        rpm = self.settings.provider_rpm
        if rpm > 0:
            self._prune_starts(provider_id, now)
            starts = self._starts[provider_id]
            if len(starts) >= rpm:
                delay = max(delay, starts[0] + 60.0 - now)
        min_interval = self.settings.provider_min_interval_ms / 1000.0
        last_start = self._last_start.get(provider_id)
        if min_interval > 0 and last_start is not None:
            delay = max(delay, last_start + min_interval - now)
        return max(0.0, delay)

    def _has_capacity(self, waiter: _Waiter, now: float) -> bool:
        if self._active_global >= self.settings.global_concurrency:
            return False
        provider_active = self._active_provider[waiter.provider_id]
        if provider_active >= self.settings.provider_concurrency:
            return False

        if waiter.priority != CallPriority.INTERACTIVE:
            reserved = self.settings.interactive_reserved_slots
            global_noninteractive_limit = max(
                1, self.settings.global_concurrency - reserved
            )
            provider_noninteractive_limit = max(
                1, self.settings.provider_concurrency - min(reserved, 1)
            )
            if self._active_global >= global_noninteractive_limit:
                return False
            if provider_active >= provider_noninteractive_limit:
                return False

        if waiter.priority >= CallPriority.BACKGROUND:
            if (
                self._active_background[waiter.provider_id]
                >= self.settings.provider_background_concurrency
            ):
                return False
        return self._rate_delay(waiter.provider_id, now) <= 0

    def _next_eligible(self, now: float) -> Optional[_Waiter]:
        candidates = [w for w in self._pending if self._has_capacity(w, now)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda w: (self._effective_priority(w, now), w.sequence),
        )

    async def acquire(
        self,
        *,
        provider_id: str,
        module: str,
        priority: CallPriority | int | str | None = None,
        queue_timeout: Optional[float] = 300.0,
        lease_timeout: Optional[float] = None,
    ) -> LLMLease:
        """排队并取得调用资格。

        ``queue_timeout`` 只计算排队时间；``lease_timeout`` 用于框架钩子缺失时
        自动回收，不应替代 Provider 自身调用超时。
        """

        provider_key = self._provider_key(provider_id)
        resolved_priority = coerce_priority(priority, module)
        started = time.monotonic()
        deadline = None
        if queue_timeout is not None and queue_timeout > 0:
            deadline = started + queue_timeout

        async with self._condition:
            if self._closed:
                raise LLMGovernorError("LLM Governor 已关闭")
            if len(self._pending) >= self.settings.queue_limit:
                self._rejected_by_backpressure += 1
                raise LLMQueueFullError(
                    f"LLM 等待队列已满({self.settings.queue_limit})"
                )
            now = time.monotonic()
            if self._is_circuit_open(provider_key, now):
                circuit = self._circuits[provider_key]
                raise LLMCircuitOpenError(
                    f"Provider {provider_id or 'default'} 熔断中，"
                    f"约 {max(0.0, circuit.open_until - now):.1f}s 后恢复"
                )

            self._sequence += 1
            waiter = _Waiter(
                waiter_id=str(uuid.uuid4()),
                provider_id=provider_key,
                module=module,
                priority=resolved_priority,
                enqueued_at=started,
                sequence=self._sequence,
                queue_depth_at_start=len(self._pending),
            )
            self._pending.append(waiter)
            self._condition.notify_all()

            try:
                while True:
                    now = time.monotonic()
                    if self._closed:
                        raise LLMGovernorError("LLM Governor 已关闭")
                    if self._is_circuit_open(provider_key, now):
                        circuit = self._circuits[provider_key]
                        raise LLMCircuitOpenError(
                            f"Provider {provider_id or 'default'} 熔断中，"
                            f"约 {max(0.0, circuit.open_until - now):.1f}s 后恢复"
                        )

                    selected = self._next_eligible(now)
                    if selected is waiter:
                        self._pending.remove(waiter)
                        in_flight_at_start = self._active_global
                        self._active_global += 1
                        self._active_provider[provider_key] += 1
                        if resolved_priority >= CallPriority.BACKGROUND:
                            self._active_background[provider_key] += 1
                        self._starts[provider_key].append(now)
                        self._last_start[provider_key] = now
                        self._calls_per_minute[(provider_key, module)].append(now)
                        queue_wait_ms = int(max(0.0, now - started) * 1000)
                        self._queue_wait_samples.append(queue_wait_ms)
                        lease = LLMLease(
                            self,
                            lease_id=str(uuid.uuid4()),
                            provider_id=provider_key,
                            module=module,
                            priority=resolved_priority,
                            queue_wait_ms=queue_wait_ms,
                            queue_depth_at_start=waiter.queue_depth_at_start,
                            in_flight_at_start=in_flight_at_start,
                        )
                        self._leases[lease.lease_id] = lease
                        if lease_timeout is not None and lease_timeout > 0:
                            lease._watchdog = asyncio.create_task(
                                self._lease_watchdog(lease.lease_id, lease_timeout),
                                name=f"iris-llm-lease-{lease.lease_id[:8]}",
                            )
                        self._condition.notify_all()
                        return lease

                    if deadline is not None and now >= deadline:
                        raise LLMQueueTimeoutError(
                            f"LLM 排队超时({queue_timeout:.1f}s): module={module}"
                        )

                    wait_for = 1.0
                    if deadline is not None:
                        wait_for = min(wait_for, max(0.001, deadline - now))
                    rate_delay = self._rate_delay(provider_key, now)
                    if rate_delay > 0:
                        wait_for = min(wait_for, max(0.001, rate_delay))
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=wait_for)
                    except asyncio.TimeoutError:
                        pass
            finally:
                if waiter in self._pending:
                    self._pending.remove(waiter)
                    self._condition.notify_all()

    @asynccontextmanager
    async def lease(
        self,
        *,
        provider_id: str,
        module: str,
        priority: CallPriority | int | str | None = None,
        queue_timeout: Optional[float] = 300.0,
    ) -> AsyncIterator[LLMLease]:
        lease = await self.acquire(
            provider_id=provider_id,
            module=module,
            priority=priority,
            queue_timeout=queue_timeout,
        )
        try:
            yield lease
        finally:
            await lease.release()

    async def _lease_watchdog(self, lease_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            lease = self._leases.get(lease_id)
            provider_id = lease.provider_id if lease is not None else ""
            released = await self.release(lease_id, _from_watchdog=True)
            if released:
                self._watchdog_releases += 1
                await self.record_failure(provider_id)
        except asyncio.CancelledError:
            return

    async def release(self, lease_id: str, *, _from_watchdog: bool = False) -> bool:
        """按 ID 幂等释放 lease。"""

        current = asyncio.current_task()
        async with self._condition:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return False
            lease._released = True
            watchdog = lease._watchdog
            if watchdog is not None and watchdog is not current and not watchdog.done():
                watchdog.cancel()
            self._active_global = max(0, self._active_global - 1)
            self._active_provider[lease.provider_id] = max(
                0, self._active_provider[lease.provider_id] - 1
            )
            if lease.priority >= CallPriority.BACKGROUND:
                self._active_background[lease.provider_id] = max(
                    0, self._active_background[lease.provider_id] - 1
                )
            self._condition.notify_all()
            return True

    async def record_success(self, provider_id: str) -> None:
        async with self._condition:
            circuit = self._circuits[self._provider_key(provider_id)]
            circuit.consecutive_failures = 0
            circuit.open_until = 0.0
            self._condition.notify_all()

    async def record_failure(self, provider_id: str) -> None:
        async with self._condition:
            circuit = self._circuits[self._provider_key(provider_id)]
            circuit.consecutive_failures += 1
            threshold = self.settings.circuit_failure_threshold
            if threshold > 0 and circuit.consecutive_failures >= threshold:
                circuit.open_until = time.monotonic() + self.settings.circuit_open_seconds
            self._condition.notify_all()

    def record_background_budget_exhausted(self) -> None:
        """Record a local hard-budget rejection (no Provider request was sent)."""

        self._background_budget_exhausted += 1

    def record_singleflight_join(self) -> None:
        self._singleflight_join_count += 1

    def record_retry(self) -> None:
        self._retry_count += 1

    def get_metrics(self) -> dict:
        """返回无阻塞快照，供管理页和测试读取。"""

        now = time.monotonic()
        calls: dict[str, dict[str, int]] = defaultdict(dict)
        for (provider_id, module), samples in self._calls_per_minute.items():
            cutoff = now - 60.0
            while samples and samples[0] <= cutoff:
                samples.popleft()
            calls[provider_id][module] = len(samples)
        waits = sorted(self._queue_wait_samples)
        p95 = 0
        if waits:
            p95 = waits[min(len(waits) - 1, math.ceil(len(waits) * 0.95) - 1)]
        return {
            "in_flight": self._active_global,
            "in_flight_by_provider": dict(self._active_provider),
            "queue_depth": len(self._pending),
            "queue_limit": self.settings.queue_limit,
            "queue_depth_by_priority": {
                priority.name: sum(1 for w in self._pending if w.priority == priority)
                for priority in CallPriority
            },
            "queue_wait_p95_ms": p95,
            "calls_per_minute": {key: dict(value) for key, value in calls.items()},
            "rejected_by_backpressure": self._rejected_by_backpressure,
            "watchdog_releases": self._watchdog_releases,
            "background_budget_exhausted": self._background_budget_exhausted,
            "singleflight_join_count": self._singleflight_join_count,
            "retry_count": self._retry_count,
            "circuit_breaker_state": {
                provider_id: {
                    "state": "open" if circuit.open_until > now else "closed",
                    "consecutive_failures": circuit.consecutive_failures,
                    "open_for_seconds": max(0.0, circuit.open_until - now),
                }
                for provider_id, circuit in self._circuits.items()
            },
        }

    async def shutdown(self) -> None:
        """停止接收新请求并回收所有 lease。"""

        async with self._condition:
            self._closed = True
            leases = list(self._leases.values())
        for lease in leases:
            await lease.release()
        async with self._condition:
            self._condition.notify_all()
