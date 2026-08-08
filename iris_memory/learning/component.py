"""
Iris Chat Memory - 学习模块组件

LearningComponent：学习子模块的生命周期入口，持有
storage / collector / expression / jargon / reviewer / injector，
向钩子与调度器暴露统一调用面：
- on_message / on_response：消息与 LLM 响应采集；
- build_context：注入文本组装；
- run_review / run_jargon_scan / run_decay：周期任务入口。

learning.db 的全部读写操作共用组件级 asyncio.Lock 保证单写者；
LLM 调用（审查/暗语推断）一律在锁外 await，避免阻塞注入与采集路径。
"""

import asyncio
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
from .reviewer import LearningReviewer
from .storage import LearningStorage

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

    def __init__(self):
        super().__init__()
        self._init_mode = InitMode.BACKGROUND
        self._storage: Optional[LearningStorage] = None
        self._jargon: Optional[JargonLearner] = None
        self._reviewer: Optional[LearningReviewer] = None
        self._collector: Optional[LearningCollector] = None
        # learning.db 单写者锁：写库操作共用
        self._db_lock = asyncio.Lock()

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
            self._jargon.load_counts()

            self._reviewer = LearningReviewer(self._storage)
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
        if self._jargon:
            try:
                self._jargon.flush()
            except Exception as e:
                logger.warning(f"暗语计数 shutdown 刷盘失败：{e}")
        if self._storage:
            try:
                self._storage.close()
            except Exception as e:
                logger.warning(f"learning.db 关闭失败：{e}")
        self._storage = None
        self._jargon = None
        self._reviewer = None
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
            async with self._db_lock:
                self._collector.on_response(event, resp)
            if self._reviewer and self._reviewer.is_batch_full():
                asyncio.create_task(self.run_review())
        except Exception as e:
            logger.warning(f"学习模块响应采集失败：{e}")

    async def build_context(
        self, event: "AstrMessageEvent", meta: Optional[Dict[str, Any]] = None
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
            async with self._db_lock:
                return await injector.build_learning_context(
                    event, self._storage, self._jargon, meta
                )
        except Exception as e:
            logger.warning(f"学习上下文组装失败：{e}")
            if meta is not None:
                meta["error"] = str(e)
            return ""

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

    async def run_jargon_scan(self) -> None:
        """执行一轮暗语扫描：刷盘词频 + 推断跨档词条含义"""
        if not self._is_available or not self._jargon:
            return
        try:
            config = get_config()
            if not config.get("learning.jargon_enable"):
                return
            async with self._db_lock:
                self._jargon.flush()
                terms = self._jargon.get_terms_for_inference()
            if not terms:
                return

            llm_manager = self._get_llm_manager()
            if not llm_manager:
                return
            l1_buffer = self._get_l1_buffer()

            for term_info in terms:
                try:
                    # 推断含 LLM await（最长 60s/词），必须在锁外执行；
                    # 仅落库写操作持锁
                    result = await self._jargon.infer(
                        term_info,
                        llm_manager,
                        l1_buffer=l1_buffer,
                        session_id=term_info.get("group_id", ""),
                    )
                    if result and self._storage:
                        meaning, confidence = result
                        async with self._db_lock:
                            self._storage.mark_jargon_inferred(
                                int(term_info["id"]), meaning, confidence
                            )
                        logger.info(
                            f"暗语含义推断成功 "
                            f"[{term_info.get('group_id')}:{term_info.get('term')}]"
                            f" = {meaning} ({confidence:.2f})"
                        )
                except Exception as e:
                    logger.warning(
                        f"暗语推断失败 [{term_info.get('term')}]：{e}"
                    )
        except Exception as e:
            logger.warning(f"暗语扫描执行失败：{e}")

    async def run_decay(self) -> None:
        """执行一轮表达模式衰减淘汰"""
        if not self._is_available or not self._storage:
            return
        try:
            config = get_config()
            decay_days = int(config.get("learning.pattern_decay_days", 15) or 15)
            max_count = int(config.get("learning.pattern_max_count", 300) or 300)
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

    @staticmethod
    def _get_l1_buffer() -> Any:
        """取 L1 缓冲组件（不可用返回 None）"""
        try:
            manager = get_component_manager()
            if not manager:
                return None
            l1 = manager.get_component("l1_buffer")
            if l1 and getattr(l1, "is_available", False):
                return l1
        except Exception as e:
            logger.debug(f"获取 L1 缓冲失败：{e}")
        return None
