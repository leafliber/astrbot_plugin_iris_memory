"""插件级图片后台任务协调器。"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from weakref import WeakValueDictionary

from iris_memory.config import get_config
from iris_memory.core import Component, get_logger
from iris_memory.llm.policy import CallPriority
from iris_memory.tasks.work_queue import BoundedWorkQueue


logger = get_logger("image.coordinator")
Runner = Callable[[], Awaitable[None]]


@dataclass
class ImageWork:
    session_id: str
    runner: Runner
    completion: asyncio.Event
    dependency: Optional[asyncio.Event | asyncio.Future] = None


class ImageParseCoordinator(Component):
    """固定 Worker、全局有界、单会话串行的图片任务入口。"""

    def __init__(self) -> None:
        super().__init__()
        self._queue: Optional[BoundedWorkQueue[ImageWork]] = None
        # 会话锁用弱值字典：最后一个使用者释放后条目自动回收，
        # 避免长生命周期进程里每个会话各留一个永不释放的 Lock。
        # 创建由 _locks_guard 串行化，保证同一 key 拿到同一把锁。
        self._session_locks: "WeakValueDictionary[str, asyncio.Lock]" = (
            WeakValueDictionary()
        )
        self._locks_guard = threading.Lock()

    @property
    def name(self) -> str:
        return "image_coordinator"

    async def initialize(self) -> None:
        config = get_config()
        queue_limit = max(1, int(config.get("image_queue_limit", 200) or 200))
        workers = max(1, int(config.get("image_max_concurrent_parse", 2) or 2))
        self._queue = BoundedWorkQueue(
            name="iris-image",
            handler=self._run,
            maxsize=queue_limit,
            workers=workers,
        )
        self._is_available = True

    async def _run(self, work: ImageWork) -> None:
        try:
            if work.dependency is not None:
                if isinstance(work.dependency, asyncio.Event):
                    await work.dependency.wait()
                else:
                    await asyncio.shield(work.dependency)
            lock = self._get_session_lock(work.session_id)
            async with lock:
                await work.runner()
        finally:
            work.completion.set()

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """取得（或创建）会话锁；调用方持有返回值的强引用直至使用结束。"""
        with self._locks_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_id] = lock
            return lock

    def submit(
        self,
        *,
        key: str,
        session_id: str,
        runner: Runner,
        dependency: Optional[asyncio.Event | asyncio.Future] = None,
    ) -> Optional[asyncio.Event]:
        if not self._is_available or self._queue is None:
            return None
        completion = asyncio.Event()
        accepted = self._queue.enqueue_once(
            key,
            ImageWork(session_id, runner, completion, dependency),
            priority=CallPriority.NEARLINE,
        )
        if not accepted:
            completion.set()
            logger.warning(f"图片后台队列已满，拒绝任务：{key}")
            return None
        return completion

    def get_metrics(self) -> dict:
        return self._queue.get_metrics() if self._queue else {}

    async def join(self) -> None:
        if self._queue:
            await self._queue.join()

    async def shutdown(self) -> None:
        if self._queue:
            await self._queue.shutdown()
        self._queue = None
        self._session_locks.clear()
        self._reset_state()
