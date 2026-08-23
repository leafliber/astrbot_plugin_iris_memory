import asyncio

import pytest

from iris_memory.llm.policy import CallPriority
from iris_memory.tasks.work_queue import BoundedWorkQueue


@pytest.mark.asyncio
async def test_same_key_is_coalesced_while_running():
    started = asyncio.Event()
    release = asyncio.Event()
    handled: list[str] = []

    async def handler(payload: str) -> None:
        handled.append(payload)
        started.set()
        await release.wait()

    queue = BoundedWorkQueue(
        name="test-coalesce", handler=handler, maxsize=10, workers=1
    )
    assert queue.enqueue_once("session", "first")
    await started.wait()
    for index in range(100):
        assert queue.enqueue_once("session", f"update-{index}")
    release.set()
    await asyncio.wait_for(queue.join(), 1)

    assert handled == ["first", "update-99"]
    assert queue.get_metrics()["merged"] == 100
    await queue.shutdown()


@pytest.mark.asyncio
async def test_queue_is_bounded_and_uses_fixed_worker_count():
    blocker = asyncio.Event()

    async def handler(payload: str) -> None:
        await blocker.wait()

    queue = BoundedWorkQueue(
        name="test-bounded", handler=handler, maxsize=2, workers=1
    )
    queue.enqueue_once("running", "running", priority=CallPriority.BACKGROUND)
    await asyncio.sleep(0)
    assert queue.enqueue_once("queued-1", "one")
    assert queue.enqueue_once("queued-2", "two")
    assert not queue.enqueue_once("rejected", "three")
    assert queue.get_metrics()["rejected"] == 1
    blocker.set()
    await asyncio.wait_for(queue.join(), 1)
    await queue.shutdown()
