"""
Iris Chat Memory - 人格自迭代组件

PersonaEvolutionComponent：人格自迭代子模块的生命周期入口，持有
storage / collector / service，向钩子与调度器暴露统一调用面：
- on_message：群聊消息语料采集 + 本地计数触发检查（不在消息链调 LLM）；
- count_job_samples / count_scoped_samples：Job 语料计数查询（供触发用）；
- run_sample_prune：语料保留清理（启动一次 + 周期循环）；
- run_trigger_scan：每小时兜底扫描 + 消息计数触发的一次性任务入口；
- run_job：单 Job 迭代执行（手动/自动共用，供后续指令层调用）。

persona_evolution.db 的全部写操作共用组件级 asyncio.Lock 保证单写者；
LLM 调用一律在该锁外 await（learning 模式）。
初始化失败置 _init_error 降级，不影响 Iris Memory 其他功能。
"""

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from iris_memory.config import get_config
from iris_memory.core import (
    Component,
    InitMode,
    get_component_manager,
    get_logger,
)
from .collector import PersonaCollector
from .models import JobStatus
from .service import PersonaEvolutionService
from .storage import PersonaEvolutionStorage

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

logger = get_logger("persona_evolution")

# 语料保留清理周期（秒）
_PRUNE_INTERVAL_SECONDS = 6 * 3600

# 消息计数触发的一次性任务名（调度器串行队列执行）
_TRIGGER_SCAN_TASK = "persona_evolution_trigger_scan"

# 启动恢复对账的延迟（秒）：等 AstrBot PersonaManager 就绪
_RECONCILE_DELAY_SECONDS = 5.0


class PersonaEvolutionComponent(Component):
    """人格自迭代组件

    后台初始化（InitMode.BACKGROUND），不阻塞主流程。
    配置关闭时 _init_error 含"未启用"字样，
    供 check_component 识别为 disabled。

    Args:
        context: AstrBot Context（PersonaManager 来源），可为 None
            （发布与恢复路径将降级为不可用）
    """

    def __init__(self, context: Any = None):
        super().__init__()
        self._init_mode = InitMode.BACKGROUND
        self._context = context
        self._storage: Optional[PersonaEvolutionStorage] = None
        self._collector: Optional[PersonaCollector] = None
        self._service: Optional[PersonaEvolutionService] = None
        # persona_evolution.db 单写者锁：写库操作共用
        self._db_lock = asyncio.Lock()
        self._prune_task: Optional[asyncio.Task] = None
        self._reconcile_task: Optional[asyncio.Task] = None
        # 消息链本地计数：自上次触发检查以来新入库的语料条数
        self._pending_new_samples = 0

    @property
    def name(self) -> str:
        return "persona_evolution"

    @property
    def storage(self) -> Optional[PersonaEvolutionStorage]:
        """暴露存储实例（供后续阶段的服务/指令层使用）"""
        return self._storage

    @property
    def service(self) -> Optional[PersonaEvolutionService]:
        """暴露用例编排服务（供指令层 / Web 路由使用）"""
        return self._service

    @property
    def context(self) -> Any:
        """AstrBot Context（Web 路由读取 PersonaManager 用，可为 None）"""
        return self._context

    async def initialize(self) -> None:
        """初始化：建库迁移、实例化采集器与服务、启动清理与恢复对账"""
        config = get_config()

        if not config.get("persona_evolution.enable"):
            logger.info("人格自迭代未启用（persona_evolution.enable=false）")
            self._init_error = "人格自迭代未启用（persona_evolution.enable=false）"
            self._is_available = False
            return

        try:
            persist_dir = config.data_dir / "persona_evolution"
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._storage = PersonaEvolutionStorage(
                persist_dir / "persona_evolution.db"
            )
            self._storage.init_schema()

            store_max_chars = int(
                config.get("persona_evolution_sample_store_chars", 500) or 500
            )
            self._collector = PersonaCollector(self._storage, store_max_chars)
            self._service = PersonaEvolutionService(
                self._storage, self._context, self._db_lock
            )

            # 语料保留清理：启动先跑一轮，之后周期执行
            self._prune_task = asyncio.create_task(
                self._prune_loop(), name="persona_evolution_prune"
            )
            # 启动恢复对账（§12.2）：延迟等 PersonaManager 就绪
            self._reconcile_task = asyncio.create_task(
                self._deferred_reconcile(), name="persona_evolution_reconcile"
            )

            self._is_available = True
            logger.info(f"人格自迭代初始化成功：{persist_dir}")
        except Exception as e:
            logger.error(f"人格自迭代初始化失败：{e}", exc_info=True)
            self._init_error = str(e)
            self._is_available = False

    async def shutdown(self) -> None:
        """关闭：取消周期任务与待重试、关闭数据库"""
        for task in (self._prune_task, self._reconcile_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"后台任务关闭异常：{e}")
        self._prune_task = None
        self._reconcile_task = None
        if self._service:
            self._service.cancel_pending_retries()
        if self._storage:
            try:
                self._storage.close()
            except Exception as e:
                logger.warning(f"persona_evolution.db 关闭失败：{e}")
        self._storage = None
        self._collector = None
        self._service = None
        self._pending_new_samples = 0
        self._reset_state()

    # ------------------------------------------------------------------
    # 采集入口（供钩子调用，全部故障隔离不抛出）
    # ------------------------------------------------------------------

    async def on_message(self, event: "AstrMessageEvent") -> None:
        """群聊消息语料采集入口 + 本地计数触发检查"""
        if not self._is_available or not self._collector:
            return
        try:
            should_trigger = False
            async with self._db_lock:
                sample_id = await self._collector.collect(event)
                if sample_id is not None:
                    self._pending_new_samples += 1
                    should_trigger = self._check_local_trigger()
                    if should_trigger:
                        self._pending_new_samples = 0
            if should_trigger:
                # 满足计数门槛：一次性任务进调度队列，不在消息链调 LLM
                await self._schedule_trigger_scan()
        except Exception as e:
            logger.warning(f"人格自迭代语料采集失败：{e}")

    def _check_local_trigger(self) -> bool:
        """本地计数触发检查（消息钩子只更新本地计数）

        计数达到任一 active Job 的触发门槛即返回 True；
        完整条件（冷却/状态/哈希基线）由调度任务内的 run_job 复核。
        """
        if not self._storage:
            return False
        try:
            thresholds = [
                job.trigger_sample_count
                for job in self._storage.list_jobs()
                if job.status == JobStatus.ACTIVE.value
            ]
        except Exception:
            return False
        if not thresholds:
            return False
        return self._pending_new_samples >= min(thresholds)

    async def _schedule_trigger_scan(self) -> None:
        """把触发扫描一次性任务放入 TaskScheduler 串行队列（防重入）"""
        scheduler = self._get_scheduler()
        if scheduler is None:
            return
        try:
            if scheduler.is_task_running(_TRIGGER_SCAN_TASK):
                return
            await scheduler.schedule_task(_TRIGGER_SCAN_TASK, self.run_trigger_scan)
        except Exception as e:
            logger.warning(f"触发扫描调度失败：{e}")

    @staticmethod
    def _get_scheduler() -> Any:
        """从组件管理器解析 TaskScheduler（不可用返回 None）"""
        try:
            manager = get_component_manager()
            scheduler = manager.get_component("scheduler")
            if scheduler and getattr(scheduler, "is_available", False):
                return scheduler
        except Exception as e:
            logger.debug(f"解析 TaskScheduler 失败：{e}")
        return None

    # ------------------------------------------------------------------
    # 迭代执行入口（触发扫描 / 手动运行）
    # ------------------------------------------------------------------

    async def run_trigger_scan(self) -> int:
        """兜底扫描：启动恢复对账 + 满足条件的 Job 逐个自动执行

        注册为每小时周期任务（lifecycle），同时作为消息计数触发的
        一次性任务入口；per-Job 运行锁保证并发触发只跑一次。

        Returns:
            实际启动运行的 Job 数
        """
        if not self._is_available or not self._service:
            return 0
        try:
            await self._service.reconcile_publishing()
            return await self._service.run_trigger_scan()
        except Exception as e:
            logger.warning(f"人格自迭代兜底扫描失败：{e}")
            return 0

    async def run_job(self, job_id: int, trigger_type: str = "manual") -> Dict[str, Any]:
        """执行单个 Job 的一轮迭代（供指令层 / Web 路由手动触发）

        Args:
            job_id: Job id
            trigger_type: manual / auto

        Returns:
            service.run_job 的结果字典；组件不可用返回错误字典
        """
        if not self._is_available or not self._service:
            return {
                "ok": False,
                "job_id": job_id,
                "run_id": None,
                "error_code": "internal_error",
                "message": "人格自迭代组件不可用",
                "revision_id": None,
                "no_change": False,
            }
        return await self._service.run_job(job_id, trigger_type)

    async def _deferred_reconcile(self) -> None:
        """启动恢复对账（§12.2）：延迟执行等 PersonaManager 就绪"""
        try:
            await asyncio.sleep(_RECONCILE_DELAY_SECONDS)
            if self._service:
                summary = await self._service.reconcile_publishing()
                if any(summary[k] for k in ("applied", "publish_failed", "conflict")):
                    logger.info(f"启动恢复对账完成：{summary}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"启动恢复对账失败：{e}")

    # ------------------------------------------------------------------
    # 语料计数查询（供自动/手动触发判断用，后续阶段接入）
    # ------------------------------------------------------------------

    async def count_scoped_samples(
        self,
        group_ids: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
        since_id: int = 0,
    ) -> int:
        """按群/用户范围统计语料数（空数组=不限）"""
        if not self._is_available or not self._storage:
            return 0
        try:
            async with self._db_lock:
                return self._storage.count_samples(group_ids, user_ids, since_id)
        except Exception as e:
            logger.warning(f"语料计数查询失败：{e}")
            return 0

    async def count_job_samples(self, job_id: int, since_cursor: bool = True) -> int:
        """统计 Job 匹配范围内的语料数

        Args:
            job_id: Job id
            since_cursor: True 时只统计 last_sample_cursor 之后的新增语料
                （自动触发门槛用）；False 统计保留期内全部匹配语料
                （手动运行用）

        Returns:
            语料数；Job 不存在或组件不可用返回 0
        """
        if not self._is_available or not self._storage:
            return 0
        try:
            async with self._db_lock:
                job = self._storage.get_job(job_id)
                if not job:
                    return 0
                since_id = job.last_sample_cursor if since_cursor else 0
                return self._storage.count_samples(
                    job.source_group_ids, job.source_user_ids, since_id
                )
        except Exception as e:
            logger.warning(f"Job 语料计数查询失败：{e}")
            return 0

    # ------------------------------------------------------------------
    # 周期任务：语料保留清理
    # ------------------------------------------------------------------

    async def run_sample_prune(self) -> int:
        """执行一轮语料保留清理（30 天 / 全局上限，最旧先删）

        Returns:
            删除的条数
        """
        if not self._is_available or not self._storage:
            return 0
        try:
            config = get_config()
            retention_days = int(
                config.get("persona_evolution.sample_retention_days", 30) or 30
            )
            max_count = int(
                config.get("persona_evolution.sample_max_count", 20000) or 20000
            )
            async with self._db_lock:
                return self._storage.prune_samples(retention_days, max_count)
        except Exception as e:
            logger.warning(f"语料保留清理失败：{e}")
            return 0

    async def _prune_loop(self) -> None:
        """语料清理循环：启动先跑一轮，之后周期执行"""
        await self.run_sample_prune()
        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            await self.run_sample_prune()
