"""
Iris Chat Memory - 任务调度器

提供定时任务的生命周期管理和并发控制。

Features:
    - 后台任务调度（asyncio）
    - 写锁保护（防止并发写入）
    - 任务队列串行调度
    - 故障隔离（单任务失败不影响其他任务）
"""

import asyncio
import random
from typing import Dict, Optional, Callable, Awaitable, TYPE_CHECKING
from datetime import datetime

from iris_memory.core import Component, get_logger

if TYPE_CHECKING:
    from iris_memory.core import ComponentManager

logger = get_logger("tasks.scheduler")


class TaskScheduler(Component):
    """任务调度器

    管理定时任务的生命周期，提供后台任务启动/停止、写锁保护、任务状态追踪。

    Attributes:
        _tasks: 后台任务字典 {task_name: asyncio.Task}
        _write_lock: 写锁（保护存储写入）
        _task_queue: 任务队列（串行调度）
        _component_manager: 组件管理器引用
        _running: 是否正在运行

    Examples:
        >>> scheduler = TaskScheduler()
        >>> await scheduler.initialize()
        >>> scheduler.register_task("forgetting", forgetting_coroutine)
        >>> await scheduler.shutdown()
    """

    def __init__(self):
        """初始化任务调度器"""
        super().__init__()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._write_lock = asyncio.Lock()
        self._task_queue: Optional[asyncio.Queue] = None
        self._component_manager: Optional["ComponentManager"] = None
        self._running = False
        self._active_tasks: set = set()

    @property
    def name(self) -> str:
        """组件名称"""
        return "scheduler"

    async def initialize(self) -> None:
        """初始化任务调度器

        创建任务队列，启动后台任务。
        """
        try:
            # 创建任务队列
            self._task_queue = asyncio.Queue()

            # 启动队列处理器
            self._running = True
            queue_task = asyncio.create_task(self._process_queue())
            self._tasks["_queue_processor"] = queue_task

            self._is_available = True
            logger.info("TaskScheduler 初始化成功")

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"TaskScheduler 初始化失败：{e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """关闭任务调度器

        取消所有后台任务，清空任务队列。
        """
        logger.info("开始关闭 TaskScheduler...")

        self._running = False

        # 取消所有后台任务
        for task_name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"已取消任务：{task_name}")

        # 等待所有任务完成
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        self._tasks.clear()
        self._active_tasks.clear()
        self._reset_state()

        logger.info("TaskScheduler 已关闭")

    def set_component_manager(self, manager: "ComponentManager") -> None:
        """设置组件管理器引用

        Args:
            manager: 组件管理器实例
        """
        self._component_manager = manager
        logger.debug("已设置 ComponentManager 引用")

    # =========================================================================
    # 任务注册接口
    # =========================================================================

    def register_periodic_task(
        self,
        task_name: str,
        coro_func: Callable[[], Awaitable[None]],
        interval_hours: float,
    ) -> None:
        """注册周期性任务

        Args:
            task_name: 任务名称
            coro_func: 任务协程函数（无参数）
            interval_hours: 执行间隔（小时）

        Examples:
            >>> scheduler.register_periodic_task(
            ...     "forgetting",
            ...     run_forgetting_task,
            ...     interval_hours=6
            ... )
        """
        # 钳制 interval_hours 下限，防止 0 或负值导致 sleep(0) 忙循环
        if interval_hours <= 0:
            logger.warning(
                f"周期性任务 {task_name} 的 interval_hours={interval_hours} "
                f"非正数，已钳制为 1 小时"
            )
            interval_hours = 1.0

        if task_name in self._tasks:
            logger.warning(f"任务 {task_name} 已存在，将被覆盖")
            old_task = self._tasks[task_name]
            if not old_task.done():
                old_task.cancel()

                # 异步等待旧任务清理完成（不阻塞当前同步方法）。
                # 此前 cancel 后不 await，旧任务清理可能未完成就启动新任务。
                async def _await_cancelled(t):
                    try:
                        await asyncio.wait_for(t, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass

                asyncio.ensure_future(_await_cancelled(old_task))

        # 创建后台任务
        task = asyncio.create_task(
            self._periodic_task_wrapper(task_name, coro_func, interval_hours)
        )
        self._tasks[task_name] = task

        logger.info(f"已注册周期性任务：{task_name}，间隔 {interval_hours} 小时")

    async def _periodic_task_wrapper(
        self,
        task_name: str,
        coro_func: Callable[[], Awaitable[None]],
        interval_hours: float,
    ) -> None:
        """周期性任务包装器

        负责循环调度、错误捕获、日志记录。
        首次执行会在 0 到 interval_hours 之间随机延迟，避免多实例同时启动时任务重叠。

        Args:
            task_name: 任务名称
            coro_func: 任务协程函数
            interval_hours: 执行间隔（小时）
        """
        logger.info(f"周期性任务 {task_name} 已启动")

        initial_delay = random.uniform(0, interval_hours)
        logger.info(f"任务 {task_name} 首次执行将在 {initial_delay:.2f} 小时后进行")
        await asyncio.sleep(initial_delay * 3600)

        while self._running:
            try:
                logger.debug(f"开始执行任务：{task_name}")
                start_time = datetime.now()

                self._active_tasks.add(task_name)
                try:
                    await coro_func()
                finally:
                    self._active_tasks.discard(task_name)

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"任务 {task_name} 执行完成，耗时 {duration:.2f}s")

            except asyncio.CancelledError:
                logger.info(f"任务 {task_name} 已取消")
                break

            except Exception as e:
                logger.error(f"任务 {task_name} 执行失败：{e}", exc_info=True)

            await asyncio.sleep(interval_hours * 3600)

        logger.info(f"周期性任务 {task_name} 已停止")

    # =========================================================================
    # 任务队列处理
    # =========================================================================

    async def schedule_task(
        self, task_name: str, coro_func: Callable[[], Awaitable[None]]
    ) -> None:
        """调度一次性任务

        将任务添加到队列，由队列处理器串行执行。

        Args:
            task_name: 任务名称
            coro_func: 任务协程函数
        """
        if not self._task_queue:
            logger.warning("任务队列未初始化，无法调度任务")
            return

        await self._task_queue.put((task_name, coro_func))
        logger.debug(f"已调度任务：{task_name}")

    async def _process_queue(self) -> None:
        """任务队列处理器

        从队列中取出任务并串行执行，持有写锁。
        """
        logger.info("任务队列处理器已启动")

        while self._running:
            try:
                # 从队列中获取任务（带超时，避免永久阻塞）
                try:
                    task_name, coro_func = await asyncio.wait_for(
                        self._task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # 获取写锁，执行任务
                async with self._write_lock:
                    logger.debug(f"开始执行队列任务：{task_name}")
                    start_time = datetime.now()

                    try:
                        self._active_tasks.add(task_name)
                        try:
                            await coro_func()
                        finally:
                            self._active_tasks.discard(task_name)

                        duration = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            f"队列任务 {task_name} 执行完成，耗时 {duration:.2f}s"
                        )

                    except Exception as e:
                        logger.error(
                            f"队列任务 {task_name} 执行失败：{e}", exc_info=True
                        )

            except asyncio.CancelledError:
                logger.info("任务队列处理器已取消")
                break

            except Exception as e:
                logger.error(f"任务队列处理器异常：{e}", exc_info=True)

        logger.info("任务队列处理器已停止")

    # =========================================================================
    # 写锁接口
    # =========================================================================

    @property
    def write_lock(self) -> asyncio.Lock:
        """获取写锁

        Returns:
            写锁实例
        """
        return self._write_lock

    def is_task_running(self, task_name: str) -> bool:
        """检查任务是否正在执行

        Args:
            task_name: 任务名称

        Returns:
            任务是否正在执行（仅在实际执行任务逻辑时为 True，
            等待下一轮调度时为 False）
        """
        return task_name in self._active_tasks
