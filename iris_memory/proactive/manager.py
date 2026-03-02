"""
主动回复管理器（Facade）

统一入口，编排所有子组件的初始化和消息处理流程。
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.cold_start import ColdStartStrategy
from iris_memory.proactive.core.context_engine import ContextEngine
from iris_memory.proactive.core.decision_engine import DecisionEngine
from iris_memory.proactive.core.feedback_tracker import FeedbackTracker
from iris_memory.proactive.core.llm_quota_manager import LLMQuotaManager
from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
)
from iris_memory.proactive.data.scene_initializer import SceneInitializer
from iris_memory.proactive.detectors.llm_detector import LLMDetector
from iris_memory.proactive.detectors.rule_detector import RuleDetector
from iris_memory.proactive.detectors.vector_detector import VectorDetector
from iris_memory.proactive.storage.feedback_store import FeedbackStore
from iris_memory.proactive.storage.scene_store import SceneStore
from iris_memory.proactive.strategies.router import StrategyRouter
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.manager")


class ProactiveManager:
    """主动回复管理器（Facade）

    编排流程：
    1. ContextEngine 聚合上下文
    2. DecisionEngine 决策（三级检测器 + 跟进检测）
    3. StrategyRouter 选择策略
    4. FeedbackTracker 记录回复
    """

    def __init__(
        self,
        plugin_data_path: Path,
        chroma_manager: Optional[Any] = None,
        embedding_manager: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        personality: str = "balanced",
        quiet_hours: Optional[List[int]] = None,
        max_history: int = 10,
        max_text_tokens: int = 150,
        enabled: bool = True,
        group_whitelist_mode: bool = False,
    ) -> None:
        self._plugin_data_path = plugin_data_path
        self._chroma_manager = chroma_manager
        self._embedding_manager = embedding_manager
        self._shared_state = shared_state
        self._llm_provider = llm_provider
        self._personality = personality
        self._quiet_hours = quiet_hours or [23, 7]
        self._max_history = max_history
        self._max_text_tokens = max_text_tokens
        self._enabled = enabled

        # 白名单状态（持久化到 KV 存储）
        self._group_whitelist: List[str] = []
        self._group_whitelist_mode: bool = group_whitelist_mode

        # 子组件（延迟初始化）
        self._feedback_store: Optional[FeedbackStore] = None
        self._scene_store: Optional[SceneStore] = None
        self._context_engine: Optional[ContextEngine] = None
        self._decision_engine: Optional[DecisionEngine] = None
        self._strategy_router: Optional[StrategyRouter] = None
        self._feedback_tracker: Optional[FeedbackTracker] = None
        self._quota_manager: Optional[LLMQuotaManager] = None
        self._cold_start: Optional[ColdStartStrategy] = None
        self._scene_initializer: Optional[SceneInitializer] = None

        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """初始化所有子组件"""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing ProactiveManager...")

            # 1. 存储层
            db_path = self._plugin_data_path / "proactive_feedback.db"
            self._feedback_store = FeedbackStore(db_path)
            await self._feedback_store.initialize()

            if self._chroma_manager:
                self._scene_store = SceneStore(self._chroma_manager)
                await self._scene_store.initialize()

            # 2. 冷启动策略
            self._cold_start = ColdStartStrategy()

            # 3. LLM 配额管理
            self._quota_manager = LLMQuotaManager(self._feedback_store)
            await self._quota_manager.initialize()

            # 4. 检测器
            rule_detector = RuleDetector(personality=self._personality)
            await rule_detector.initialize()

            vector_detector = VectorDetector(
                scene_store=self._scene_store,
                feedback_store=self._feedback_store,
                cold_start=self._cold_start,
                personality=self._personality,
            )
            await vector_detector.initialize()

            llm_detector = LLMDetector(
                llm_provider=self._llm_provider,
                quota_manager=self._quota_manager,
            )
            await llm_detector.initialize()

            # 5. 决策引擎
            self._decision_engine = DecisionEngine(
                rule_detector=rule_detector,
                vector_detector=vector_detector,
                llm_detector=llm_detector,
                feedback_store=self._feedback_store,
                personality=self._personality,
            )
            await self._decision_engine.initialize()

            # 6. 上下文引擎
            self._context_engine = ContextEngine(
                embedding_manager=self._embedding_manager,
                shared_state=self._shared_state,
                feedback_store=self._feedback_store,
                max_history=self._max_history,
                max_text_tokens=self._max_text_tokens,
                quiet_hours=self._quiet_hours,
            )

            # 7. 策略路由
            self._strategy_router = StrategyRouter()

            # 8. 反馈追踪
            self._feedback_tracker = FeedbackTracker(
                feedback_store=self._feedback_store,
                scene_store=self._scene_store,
            )

            # 9. 场景初始化
            self._scene_initializer = SceneInitializer(
                scene_store=self._scene_store,
                feedback_store=self._feedback_store,
                embedding_manager=self._embedding_manager,
            )
            if self._scene_store:
                try:
                    await self._scene_initializer.initialize_scenes()
                except Exception as e:
                    logger.warning(f"Scene initialization failed: {e}")

            self._initialized = True
            logger.info("ProactiveManager initialized successfully")

    async def process_message(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        session_key: str,
        session_type: str = "group",
        group_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """处理消息，决定是否主动回复

        Args:
            messages: 最近消息列表
            user_id: 用户 ID
            session_key: 会话键
            session_type: 会话类型
            group_id: 群组 ID
            extra: 额外上下文

        Returns:
            回复信息字典（含 trigger_prompt, reply_params 等），
            或 None（不回复）
        """
        if not self._enabled or not self._initialized:
            return None

        # 白名单/黑名单过滤
        if not self.is_group_allowed(group_id):
            logger.debug(f"Group {group_id} not in whitelist, skipping proactive reply")
            return None

        try:
            # 1. 构建上下文
            context = await self._context_engine.build_context(
                messages=messages,
                user_id=user_id,
                session_key=session_key,
                session_type=session_type,
                group_id=group_id,
                extra=extra,
            )

            # 2. 决策
            decision = await self._decision_engine.decide(context)

            if not decision.should_reply:
                logger.debug(
                    f"Decision: no reply for {session_key} "
                    f"(reason={decision.reason})"
                )
                return None

            # 3. 策略路由
            strategy_result = self._strategy_router.route(context, decision)

            # 4. 记录回复
            record_id = None
            if self._feedback_tracker:
                record_id = await self._feedback_tracker.record_reply(
                    context, decision,
                    content_summary=strategy_result.get("trigger_prompt", "")[:100],
                )

            # 5. 返回结果
            return {
                "trigger_prompt": strategy_result["trigger_prompt"],
                "reply_params": strategy_result["reply_params"],
                "strategy_name": strategy_result["strategy_name"],
                "decision": decision,
                "context": context,
                "record_id": record_id,
            }

        except Exception as e:
            logger.error(f"ProactiveManager.process_message failed: {e}")
            return None

    async def process_user_response(
        self,
        session_key: str,
        user_replied_directly: bool = False,
    ) -> None:
        """处理用户对主动回复的回应"""
        if self._feedback_tracker:
            await self._feedback_tracker.process_user_response(
                session_key, user_replied_directly
            )

    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取统计数据"""
        if not self._feedback_store:
            return {}
        try:
            daily = await self._feedback_store.get_daily_stats(days)
            scene_count = 0
            if self._scene_store:
                scene_count = await self._scene_store.count()
            return {
                "daily_stats": daily,
                "scene_count": scene_count,
                "enabled": self._enabled,
                "personality": self._personality,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    # ── 白名单管理 ──────────────────────────────────────────────

    @property
    def group_whitelist(self) -> List[str]:
        """已加入白名单/黑名单的群组 ID 列表"""
        return self._group_whitelist

    @group_whitelist.setter
    def group_whitelist(self, value: List[str]) -> None:
        self._group_whitelist = list(value) if value else []

    @property
    def group_whitelist_mode(self) -> bool:
        """True = 白名单模式（仅列表中的群允许）；False = 黑名单模式"""
        return self._group_whitelist_mode

    @group_whitelist_mode.setter
    def group_whitelist_mode(self, value: bool) -> None:
        self._group_whitelist_mode = bool(value)

    def is_group_allowed(self, group_id: Optional[str]) -> bool:
        """判断群组是否允许主动回复"""
        if not group_id:
            return True  # 私聊始终允许
        if not self._group_whitelist_mode:
            return True  # 未启用白名单模式，全部允许
        return group_id in self._group_whitelist

    def serialize_whitelist(self) -> Dict[str, Any]:
        """序列化白名单状态，用于 KV 存储持久化"""
        return {
            "group_whitelist": self._group_whitelist,
            "group_whitelist_mode": self._group_whitelist_mode,
        }

    def deserialize_whitelist(self, data: Any) -> None:
        """从 KV 存储反序列化白名单状态"""
        if not isinstance(data, dict):
            return
        self._group_whitelist = list(data.get("group_whitelist", []))
        self._group_whitelist_mode = bool(data.get("group_whitelist_mode", False))
        logger.debug(
            f"Whitelist loaded: mode={self._group_whitelist_mode}, "
            f"groups={self._group_whitelist}"
        )

    async def close(self) -> None:
        """关闭并释放所有资源"""
        if self._feedback_store:
            await self._feedback_store.close()
            self._feedback_store = None

        if self._scene_store:
            await self._scene_store.close()
            self._scene_store = None

        if self._decision_engine:
            await self._decision_engine.close()

        self._initialized = False
        logger.info("ProactiveManager closed")
