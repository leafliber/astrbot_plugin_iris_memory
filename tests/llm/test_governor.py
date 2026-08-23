"""LLMCallGovernor 并发、背压与 lease 生命周期测试。"""

import asyncio

import pytest

from iris_memory.llm.governor import (
    GovernorSettings,
    LLMCircuitOpenError,
    LLMCallGovernor,
    LLMQueueFullError,
)
from iris_memory.llm.policy import CallPriority


@pytest.mark.asyncio
async def test_one_hundred_calls_respect_global_and_provider_concurrency():
    governor = LLMCallGovernor(
        GovernorSettings(
            global_concurrency=4,
            provider_concurrency=2,
            provider_background_concurrency=1,
            provider_rpm=0,
            interactive_reserved_slots=1,
        )
    )
    active = 0
    max_active = 0
    provider_active = {"p1": 0, "p2": 0}
    provider_max = {"p1": 0, "p2": 0}
    lock = asyncio.Lock()

    async def run(index: int) -> None:
        nonlocal active, max_active
        provider = "p1" if index % 2 else "p2"
        async with governor.lease(
            provider_id=provider,
            module="framework_reply",
            priority=CallPriority.INTERACTIVE,
            queue_timeout=2,
        ):
            async with lock:
                active += 1
                provider_active[provider] += 1
                max_active = max(max_active, active)
                provider_max[provider] = max(
                    provider_max[provider], provider_active[provider]
                )
            await asyncio.sleep(0.002)
            async with lock:
                active -= 1
                provider_active[provider] -= 1

    await asyncio.gather(*(run(i) for i in range(100)))

    assert max_active <= 4
    assert provider_max["p1"] <= 2
    assert provider_max["p2"] <= 2
    assert governor.get_metrics()["in_flight"] == 0


@pytest.mark.asyncio
async def test_background_load_keeps_slot_for_interactive_call():
    governor = LLMCallGovernor(
        GovernorSettings(
            global_concurrency=2,
            provider_concurrency=2,
            provider_background_concurrency=1,
            provider_rpm=0,
            interactive_reserved_slots=1,
        )
    )
    first = await governor.acquire(
        provider_id="p1",
        module="profile_analysis",
        priority=CallPriority.BACKGROUND,
    )
    blocked_background = asyncio.create_task(
        governor.acquire(
            provider_id="p1",
            module="profile_analysis",
            priority=CallPriority.BACKGROUND,
            queue_timeout=1,
        )
    )
    await asyncio.sleep(0)

    interactive = await asyncio.wait_for(
        governor.acquire(
            provider_id="p1",
            module="framework_reply",
            priority=CallPriority.INTERACTIVE,
            queue_timeout=1,
        ),
        timeout=0.2,
    )

    assert not blocked_background.done()
    await interactive.release()
    await first.release()
    second = await blocked_background
    await second.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_permit_or_queue_entry():
    governor = LLMCallGovernor(
        GovernorSettings(
            global_concurrency=1,
            provider_concurrency=1,
            provider_rpm=0,
            interactive_reserved_slots=0,
        )
    )
    held = await governor.acquire(
        provider_id="p1", module="framework_reply", queue_timeout=1
    )
    waiter = asyncio.create_task(
        governor.acquire(
            provider_id="p1", module="framework_reply", queue_timeout=1
        )
    )
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert governor.get_metrics()["queue_depth"] == 0

    await held.release()
    next_lease = await governor.acquire(
        provider_id="p1", module="framework_reply", queue_timeout=0.2
    )
    await next_lease.release()
    assert governor.get_metrics()["in_flight"] == 0


@pytest.mark.asyncio
async def test_framework_watchdog_releases_missing_response_hook_lease():
    governor = LLMCallGovernor(
        GovernorSettings(
            global_concurrency=1,
            provider_concurrency=1,
            provider_rpm=0,
            interactive_reserved_slots=0,
        )
    )
    lease = await governor.acquire(
        provider_id="p1",
        module="framework_reply",
        lease_timeout=0.02,
    )
    assert not lease.released
    await asyncio.sleep(0.05)
    assert lease.released
    metrics = governor.get_metrics()
    assert metrics["in_flight"] == 0
    assert metrics["watchdog_releases"] == 1


@pytest.mark.asyncio
async def test_queue_limit_rejects_with_backpressure_metric():
    governor = LLMCallGovernor(
        GovernorSettings(
            global_concurrency=1,
            provider_concurrency=1,
            provider_rpm=0,
            queue_limit=1,
            interactive_reserved_slots=0,
        )
    )
    held = await governor.acquire(provider_id="p1", module="framework_reply")
    queued = asyncio.create_task(
        governor.acquire(
            provider_id="p1", module="framework_reply", queue_timeout=1
        )
    )
    await asyncio.sleep(0)
    with pytest.raises(LLMQueueFullError):
        await governor.acquire(
            provider_id="p1", module="framework_reply", queue_timeout=1
        )
    assert governor.get_metrics()["rejected_by_backpressure"] == 1
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await held.release()


@pytest.mark.asyncio
async def test_consecutive_failures_open_and_then_recover_circuit():
    governor = LLMCallGovernor(
        GovernorSettings(
            provider_rpm=0,
            circuit_failure_threshold=2,
            circuit_open_seconds=0.02,
        )
    )
    await governor.record_failure("p1")
    await governor.record_failure("p1")
    with pytest.raises(LLMCircuitOpenError):
        await governor.acquire(provider_id="p1", module="framework_reply")

    await asyncio.sleep(0.03)
    lease = await governor.acquire(provider_id="p1", module="framework_reply")
    await lease.release()
