"""有界、可合并的插件后台工作队列。

队列只创建固定数量的 worker。相同 key 在排队时更新 payload，在执行时仅标记
一次 rerun，从而把消息洪峰合并为常数数量的后台任务。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Generic, Hashable, Optional, TypeVar

from iris_memory.core import get_logger
from iris_memory.llm.policy import CallPriority

logger = get_logger("work_queue")


PayloadT = TypeVar("PayloadT")
WorkHandler = Callable[[PayloadT], Awaitable[None]]


@dataclass
class _WorkItem(Generic[PayloadT]):
    key: Hashable
    payload: PayloadT
    priority: CallPriority
    enqueued_at: float
    sequence: int


class BoundedWorkQueue(Generic[PayloadT]):
    """固定 worker 数、有界容量、按 key singleflight 的本地工作队列。"""

    def __init__(
        self,
        *,
        name: str,
        handler: WorkHandler[PayloadT],
        maxsize: int = 500,
        workers: int = 1,
        aging_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self._handler = handler
        self._maxsize = max(1, int(maxsize))
        self._worker_count = max(1, int(workers))
        self._aging_seconds = max(0.1, float(aging_seconds))
        self._queued: Dict[Hashable, _WorkItem[PayloadT]] = {}
        self._running: set[Hashable] = set()
        self._dirty: Dict[Hashable, _WorkItem[PayloadT]] = {}
        self._wake = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._sequence = 0
        self._closed = False
        self._joined = asyncio.Event()
        self._joined.set()
        self.enqueued_count = 0
        self.merged_count = 0
        self.rejected_count = 0
        self.failure_count = 0

    def start(self) -> None:
        if self._closed or self._tasks:
            return
        for index in range(self._worker_count):
            task = asyncio.create_task(
                self._worker(index), name=f"{self.name}-worker-{index}"
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def enqueue_once(
        self,
        key: Hashable,
        payload: PayloadT,
        *,
        priority: CallPriority = CallPriority.BACKGROUND,
    ) -> bool:
        """非阻塞入队；相同 key 合并并返回 True，满队列返回 False。"""

        if self._closed:
            self.rejected_count += 1
            return False
        self.start()
        self._sequence += 1
        item = _WorkItem(
            key=key,
            payload=payload,
            priority=priority,
            enqueued_at=time.monotonic(),
            sequence=self._sequence,
        )
        if key in self._running:
            self._dirty[key] = item
            self.merged_count += 1
            self._joined.clear()
            return True
        if key in self._queued:
            current = self._queued[key]
            item.priority = min(current.priority, priority)
            item.enqueued_at = current.enqueued_at
            item.sequence = current.sequence
            self._queued[key] = item
            self.merged_count += 1
            self._wake.set()
            return True
        if len(self._queued) >= self._maxsize:
            self.rejected_count += 1
            return False
        self._queued[key] = item
        self.enqueued_count += 1
        self._joined.clear()
        self._wake.set()
        return True

    def discard(self, key: Hashable) -> None:
        self._queued.pop(key, None)
        self._dirty.pop(key, None)
        self._mark_joined_if_idle()

    def _pop_next(self) -> Optional[_WorkItem[PayloadT]]:
        if not self._queued:
            return None
        now = time.monotonic()

        def order(item: _WorkItem[PayloadT]) -> tuple[int, int]:
            aged = int((now - item.enqueued_at) / self._aging_seconds)
            return max(0, int(item.priority) - aged), item.sequence

        item = min(self._queued.values(), key=order)
        self._queued.pop(item.key, None)
        return item

    async def _worker(self, index: int) -> None:
        try:
            while not self._closed:
                item = self._pop_next()
                if item is None:
                    self._wake.clear()
                    if self._queued:
                        self._wake.set()
                        continue
                    await self._wake.wait()
                    continue
                self._running.add(item.key)
                try:
                    await self._handler(item.payload)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.failure_count += 1
                    # 静默吞异常会让 L1 总结/outbox/学习复审等后台任务
                    # 无声消失，至少留下可排查的日志线索
                    logger.warning(
                        f"工作队列任务失败：key={item.key},"
                        f" handler={getattr(self._handler, '__qualname__', self._handler)}",
                        exc_info=True,
                    )
                finally:
                    self._running.discard(item.key)
                    rerun = self._dirty.pop(item.key, None)
                    if rerun is not None and not self._closed:
                        # 运行中 key 的合并尾项至多额外占用 worker_count 个槽位。
                        self._queued[item.key] = rerun
                        self._wake.set()
                    self._mark_joined_if_idle()
        except asyncio.CancelledError:
            return

    def _mark_joined_if_idle(self) -> None:
        if not self._queued and not self._running and not self._dirty:
            self._joined.set()

    async def join(self) -> None:
        await self._joined.wait()

    def get_metrics(self) -> dict:
        return {
            "name": self.name,
            "queue_depth": len(self._queued),
            "running": len(self._running),
            "dirty": len(self._dirty),
            "enqueued": self.enqueued_count,
            "merged": self.merged_count,
            "rejected": self.rejected_count,
            "failures": self.failure_count,
        }

    async def shutdown(self) -> None:
        self._closed = True
        self._wake.set()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._queued.clear()
        self._dirty.clear()
        self._running.clear()
        self._joined.set()
