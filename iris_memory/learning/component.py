"""
Iris Chat Memory - 学习模块组件

LearningComponent：学习子模块的生命周期入口，持有
storage / collector / expression / jargon / reviewer / injector，
向钩子与调度器暴露统一调用面：
- on_message / on_response：消息与 LLM 响应采集；
- build_context：注入文本组装；
- run_review / run_jargon_scan / run_decay：周期任务入口；暗语扫描使用批量 LLM。

learning.db 的全部读写操作共用组件级 asyncio.Lock 保证单写者；
LLM 调用（审查/暗语推断）一律在锁外 await，避免阻塞注入与采集路径。
"""

import asyncio
import hashlib
from typing import Any, Dict, Optional, TYPE_CHECKING

from iris_memory.config import get_config
from iris_memory.core import (
    Component,
    InitMode,
    get_component_manager,
    get_logger,
)
from iris_memory.platform import get_adapter
from . import expression, injector
from .collector import LearningCollector
from .jargon import JargonLearner
from .persona_reviewer import PersonaLearningReviewer
from .reviewer import LearningReviewer
from .storage import DEFAULT_PERSONA_ID, LearningStorage

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.provider import LLMResponse

logger = get_logger("learning")


class LearningComponent(Component):
    """学习模块组件

    后台初始化（InitMode.BACKGROUND），不阻塞主流程。
    配置关闭时 _init_error 含"未启用"字样，
    供 check_component 识别为 disabled。
    """

    def __init__(self, context: Any = None):
        super().__init__()
        self._context = context
        self._init_mode = InitMode.BACKGROUND
        self._storage: Optional[LearningStorage] = None
        self._jargon: Optional[JargonLearner] = None
        self._reviewer: Optional[LearningReviewer] = None
        self._collector: Optional[LearningCollector] = None
        self._persona_reviewer: Optional[PersonaLearningReviewer] = None
        # learning.db 单写者锁：写库操作共用
        self._db_lock = asyncio.Lock()
        # 暗语扫描单飞，防止周期任务与手动触发重复领取批次。
        self._jargon_scan_lock = asyncio.Lock()
        # 通用质量审查与 Persona 一致性复审共用，避免对同一条目并发裁决。
        self._content_review_lock = asyncio.Lock()
        self._persona_review_tasks: Dict[str, asyncio.Task] = {}
        # 满批即时触发的审查任务引用，防止被 GC 回收（CPython 事件循环仅持弱引用）
        self._content_review_tasks: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return "learning"

    @property
    def storage(self) -> Optional[LearningStorage]:
        """暴露存储实例（供指令层使用）"""
        return self._storage

    async def initialize(self) -> None:
        """初始化学习模块：建库建表、加载词频计数、实例化子模块"""
        config = get_config()

        if not config.get("learning.enable"):
            logger.info("学习模块未启用（learning.enable=false）")
            self._init_error = "学习模块未启用（learning.enable=false）"
            self._is_available = False
            return

        try:
            persist_dir = config.data_dir / "learning"
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._storage = LearningStorage(persist_dir / "learning.db")
            self._storage.init_schema()

            self._jargon = JargonLearner(self._storage)

            self._reviewer = LearningReviewer(self._storage)
            self._persona_reviewer = PersonaLearningReviewer()
            self._collector = LearningCollector(
                self._storage, self._jargon, self._reviewer
            )

            self._is_available = True
            logger.info(f"学习模块初始化成功：{persist_dir}")
        except Exception as e:
            logger.error(f"学习模块初始化失败：{e}", exc_info=True)
            self._init_error = str(e)
            self._is_available = False

    async def shutdown(self) -> None:
        """关闭学习模块：词频计数刷盘、关闭数据库"""
        for task in list(self._persona_review_tasks.values()):
            if not task.done():
                task.cancel()
        if self._persona_review_tasks:
            await asyncio.gather(
                *self._persona_review_tasks.values(), return_exceptions=True
            )
        self._persona_review_tasks.clear()
        for task in list(self._content_review_tasks):
            if not task.done():
                task.cancel()
        if self._content_review_tasks:
            await asyncio.gather(
                *self._content_review_tasks, return_exceptions=True
            )
        self._content_review_tasks.clear()
        if self._storage:
            try:
                self._storage.close()
            except Exception as e:
                logger.warning(f"learning.db 关闭失败：{e}")
        self._storage = None
        self._jargon = None
        self._reviewer = None
        self._persona_reviewer = None
        self._collector = None
        self._reset_state()

    # ------------------------------------------------------------------
    # 采集入口（供钩子调用，全部故障隔离不抛出）
    # ------------------------------------------------------------------

    async def on_message(self, event: "AstrMessageEvent") -> None:
        """用户消息采集入口（词频统计）"""
        if not self._is_available or not self._collector:
            return
        try:
            config = get_config()
            if not config.get("learning.jargon_enable"):
                return
            adapter = get_adapter(event)
            session_id = adapter.get_session_id(event)
            user_id = adapter.get_user_id(event)
            text = getattr(event, "message_str", "") or ""
            async with self._db_lock:
                self._collector.on_message(event, session_id, user_id, text)
        except Exception as e:
            logger.warning(f"学习模块消息采集失败：{e}")

    async def on_response(self, event: "AstrMessageEvent", resp: "LLMResponse") -> None:
        """LLM 响应采集入口（对话对配对 + 表达模式提取）

        待审队列满 review_batch_size 时立即触发一轮审查。
        """
        if not self._is_available or not self._collector:
            return
        try:
            persona_id = DEFAULT_PERSONA_ID
            try:
                candidate = event.get_extra("iris_learning_persona_id")
                if isinstance(candidate, str) and candidate.strip():
                    persona_id = candidate.strip()
            except Exception:
                pass
            async with self._db_lock:
                self._collector.on_response(event, resp, str(persona_id))
            if self._reviewer and self._reviewer.is_batch_full():
                self._spawn_review()
        except Exception as e:
            logger.warning(f"学习模块响应采集失败：{e}")

    async def build_context(
        self,
        event: "AstrMessageEvent",
        meta: Optional[Dict[str, Any]] = None,
        persona_id: str = DEFAULT_PERSONA_ID,
        persona_prompt: str = "",
    ) -> str:
        """组装学习注入文本（供 llm_request_hook 调用）

        Returns:
            注入文本；不可用/无内容返回 ""
        """
        if not self._is_available or not self._storage or not self._jargon:
            if meta is not None:
                meta["skipped"] = "component_unavailable"
            return ""
        try:
            persona_id = str(persona_id or DEFAULT_PERSONA_ID)
            persona_prompt = await self._read_persona_prompt(
                persona_id, persona_prompt
            )
            reviewing = await self._ensure_persona_review(
                persona_id, persona_prompt, meta
            )
            if reviewing:
                if meta is not None:
                    meta["skipped"] = "persona_revalidation"
                    meta["persona_id"] = persona_id
                return ""
            async with self._db_lock:
                return await injector.build_learning_context(
                    event, self._storage, self._jargon, meta, persona_id
                )
        except Exception as e:
            logger.warning(f"学习上下文组装失败：{e}")
            if meta is not None:
                meta["error"] = str(e)
            return ""

    async def _read_persona_prompt(self, persona_id: str, fallback: str) -> str:
        """优先读取 PersonaManager 原始人格，避免动态 system 指令误触发。"""
        manager = getattr(self._context, "persona_manager", None)
        get_persona = getattr(manager, "get_persona", None)
        if get_persona is None:
            return fallback or ""
        try:
            persona = await get_persona(persona_id)
            prompt = getattr(persona, "system_prompt", None)
            if isinstance(prompt, str):
                return prompt
        except Exception as exc:
            logger.debug(f"读取 Persona {persona_id} 原始提示失败，使用请求值：{exc}")
        return fallback or ""

    # ------------------------------------------------------------------
    # 周期任务入口（供调度器调用）
    # ------------------------------------------------------------------

    async def run_review(self) -> None:
        """执行一轮攒批审查（满批即时触发 + 周期兜底共用）

        锁粒度：fetch/回写持 _db_lock，LLM await 在锁外，
        避免审查期间（最长 2×60s）阻塞注入与采集路径。
        """
        if not self._is_available or not self._reviewer or not self._storage:
            return
        try:
            async with self._content_review_lock:
                await self._run_review_once()
        except Exception as e:
            logger.warning(f"学习审查执行失败：{e}")

    async def _run_review_once(self) -> None:
        """单轮通用质量审查；调用方持有内容审查锁。"""
        try:
            llm_manager = self._get_llm_manager()
            if not llm_manager:
                return
            async with self._db_lock:
                pairs, patterns = self._reviewer.fetch_pending()
            if not pairs and not patterns:
                return
            verdicts = await self._reviewer.request_verdicts(
                llm_manager, pairs, patterns
            )
            if verdicts is None:
                return
            async with self._db_lock:
                self._reviewer.apply_verdicts(verdicts, pairs, patterns)
        except Exception as e:
            logger.warning(f"学习审查执行失败：{e}")

    def _spawn_review(self) -> None:
        """满批即时触发一轮审查（持有 task 引用避免被 GC 回收）

        run_review 经 _content_review_lock 串行，并发触发只会排队领取下一批，
        不会重复审查同一批 pending。
        """
        task = asyncio.create_task(self.run_review(), name="iris-learning-review")
        self._content_review_tasks.add(task)
        task.add_done_callback(self._content_review_tasks.discard)

    async def _ensure_persona_review(
        self,
        persona_id: str,
        persona_prompt: str,
        meta: Optional[Dict[str, Any]],
    ) -> bool:
        """检测 Persona 指纹变化并按需启动后台复审。"""
        if not self._storage:
            return False
        prompt = persona_prompt or ""
        if not prompt.strip():
            async with self._db_lock:
                return self._storage.is_persona_reviewing(persona_id)

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        async with self._db_lock:
            action = self._storage.observe_persona_prompt(persona_id, prompt_hash)
        if meta is not None:
            meta["persona_hash"] = prompt_hash
            meta["persona_check"] = action

        if action == "changed":
            self._schedule_persona_review(persona_id, prompt, prompt_hash)
            logger.info(f"检测到 Persona {persona_id} 已修改，已调度学习内容复审")
        elif action == "reviewing" and persona_id not in self._persona_review_tasks:
            # 重启后 reviewing 状态仍在库中，但内存任务已丢失。
            self._schedule_persona_review(persona_id, prompt, prompt_hash)
        return action in {"changed", "reviewing"}

    def _schedule_persona_review(
        self, persona_id: str, persona_prompt: str, prompt_hash: str
    ) -> None:
        current = self._persona_review_tasks.get(persona_id)
        if current and not current.done():
            return
        task = asyncio.create_task(
            self._run_persona_revalidation(persona_id, persona_prompt, prompt_hash),
            name=f"iris-learning-persona-review-{persona_id}",
        )
        self._persona_review_tasks[persona_id] = task

        def cleanup(done: asyncio.Task) -> None:
            if self._persona_review_tasks.get(persona_id) is done:
                self._persona_review_tasks.pop(persona_id, None)

        task.add_done_callback(cleanup)

    async def _run_persona_revalidation(
        self, persona_id: str, persona_prompt: str, prompt_hash: str
    ) -> None:
        """分批复审指定 Persona 的学习内容，不兼容条目直接删除。"""
        if not self._storage or not self._persona_reviewer:
            return
        try:
            llm_manager = self._get_llm_manager()
            if not llm_manager:
                raise RuntimeError("LLMManager 不可用")

            async with self._content_review_lock:
                async with self._db_lock:
                    pairs, patterns = self._storage.get_persona_review_items(persona_id)

                tagged = [("pair", row) for row in pairs] + [
                    ("pattern", row) for row in patterns
                ]
                batch_size = max(
                    1, get_config().get_int("learning.review_batch_size", 10) or 10
                )
                deleted = 0
                for start in range(0, len(tagged), batch_size):
                    batch = tagged[start : start + batch_size]
                    batch_pairs = [row for kind, row in batch if kind == "pair"]
                    batch_patterns = [row for kind, row in batch if kind == "pattern"]
                    verdicts = await self._persona_reviewer.request_verdicts(
                        llm_manager, persona_prompt, batch_pairs, batch_patterns
                    )
                    if verdicts is None:
                        raise RuntimeError("LLM 复审结果无效或不完整")

                    accepted_pairs = [
                        int(row["id"])
                        for row in batch_pairs
                        if verdicts[("pair", int(row["id"]))]
                    ]
                    rejected_pairs = [
                        int(row["id"])
                        for row in batch_pairs
                        if not verdicts[("pair", int(row["id"]))]
                    ]
                    accepted_patterns = [
                        int(row["id"])
                        for row in batch_patterns
                        if verdicts[("pattern", int(row["id"]))]
                    ]
                    rejected_patterns = [
                        int(row["id"])
                        for row in batch_patterns
                        if not verdicts[("pattern", int(row["id"]))]
                    ]

                    async with self._db_lock:
                        if not self._storage.is_persona_review_target(
                            persona_id, prompt_hash
                        ):
                            logger.info(
                                f"Persona {persona_id} 在复审期间再次修改，放弃旧批次结果"
                            )
                            return
                        result = self._storage.apply_persona_review_batch(
                            persona_id,
                            accepted_pairs,
                            rejected_pairs,
                            accepted_patterns,
                            rejected_patterns,
                        )
                    deleted += result["total"]

                async with self._db_lock:
                    applied = self._storage.finish_persona_review(
                        persona_id, prompt_hash
                    )
                if applied:
                    logger.info(
                        f"Persona {persona_id} 学习内容复审完成："
                        f"检查 {len(tagged)} 条，删除 {deleted} 条"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Persona {persona_id} 学习内容复审失败：{exc}")
            if self._storage:
                async with self._db_lock:
                    self._storage.fail_persona_review(
                        persona_id, prompt_hash, str(exc)
                    )

    async def run_jargon_scan(self) -> None:
        """执行一轮暗语扫描：本地漏斗、批量 LLM 鉴别与生命周期维护。"""
        if self._jargon_scan_lock.locked():
            logger.debug("暗语扫描已有任务执行，本轮跳过")
            return
        async with self._jargon_scan_lock:
            await self._run_jargon_scan_once()

    async def _run_jargon_scan_once(self) -> None:
        """单次暗语扫描实现；由 run_jargon_scan 保证单飞。"""
        if not self._is_available or not self._jargon:
            return
        try:
            config = get_config()
            if not config.get("learning.jargon_enable"):
                return
            async with self._db_lock:
                self._jargon.maintain()
                clusters = self._jargon.prepare_review()
            if not clusters:
                return

            llm_manager = self._get_llm_manager()
            if not llm_manager:
                return
            # 单次批量调用，LLM await 必须在数据库锁外。
            verdicts = await self._jargon.request_verdicts(clusters, llm_manager)
            async with self._db_lock:
                self._jargon.apply_verdicts(clusters, verdicts)
        except Exception as e:
            logger.warning(f"暗语扫描执行失败：{e}")

    async def run_decay(self) -> None:
        """执行一轮表达模式衰减淘汰"""
        if not self._is_available or not self._storage:
            return
        try:
            config = get_config()
            decay_days = config.get_int("learning.pattern_decay_days", 15) or 15
            max_count = config.get_int("learning.pattern_max_count", 300) or 300
            async with self._db_lock:
                expression.decay(self._storage, decay_days, max_count)
        except Exception as e:
            logger.warning(f"表达模式衰减执行失败：{e}")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _get_llm_manager() -> Any:
        """取 LLMManager 组件（不可用返回 None）"""
        try:
            manager = get_component_manager()
            if not manager:
                return None
            llm_manager = manager.get_component("llm_manager")
            if llm_manager and getattr(llm_manager, "is_available", False):
                return llm_manager
        except Exception as e:
            logger.debug(f"获取 LLMManager 失败：{e}")
        return None
