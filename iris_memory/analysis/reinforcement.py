"""
记忆回顾强化引擎

基于间隔重复效应 (SM-2 变体)，通过定期主动提及重要记忆强化 RIF 评分，
防止高价值陪伴记忆因长期不被访问而衰减。
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Callable, ClassVar, Coroutine, Dict, List, Optional

from iris_memory.core.types import MemoryType, StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

logger = get_logger("reinforcement")


class ReviewPromptGenerator:
    """回顾对话生成器"""

    TEMPLATES: ClassVar[Dict[str, Dict[str, List[str]]]] = {
        "fact": {
            "caring": [
                "对了，你之前说{content}，现在还是这样吗？",
                "我一直记得你提过{content}，最近有什么变化吗？",
            ],
            "casual": [
                "说起来，{content}，后来怎样了？",
                "忽然想起来，你之前跟我说过{content}~",
            ],
            "curious": [
                "我想了想，你当时说{content}，一直想问问后续~",
            ],
        },
        "emotion": {
            "caring": [
                "还记得之前你说{content}吗？现在心情好些了吗？",
                "你之前提到{content}，最近感觉怎么样？",
            ],
            "casual": [
                "想起之前聊到{content}，那时候挺有感触的~",
            ],
            "curious": [
                "说到{content}，后来有什么新的感悟吗？",
            ],
        },
        "relationship": {
            "caring": [
                "你之前说{content}，最近和ta还好吗？",
                "记得你提到{content}，现在关系怎样了？",
            ],
            "casual": [
                "忽然想起你说过{content}，后来还有联系吗？",
            ],
            "curious": [
                "关于{content}，后来有什么进展吗？",
            ],
        },
    }

    @classmethod
    def generate(cls, memory: Memory, style: str = "caring") -> str:
        """生成回顾对话文本"""
        type_key = memory.type.value if isinstance(memory.type, MemoryType) else str(memory.type)
        if type_key not in cls.TEMPLATES:
            type_key = "fact"

        style_templates = cls.TEMPLATES[type_key].get(
            style, cls.TEMPLATES[type_key]["caring"]
        )
        template = random.choice(style_templates)

        content_preview = (
            memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
        )
        return template.format(content=content_preview)


# 通知回调类型：async (user_id, group_id, text, metadata) -> None
NotifyCallback = Callable[[str, Optional[str], str, Dict[str, Any]], Coroutine[Any, Any, None]]


class MemoryReinforcementEngine:
    """记忆回顾强化引擎

    独立后台服务，拥有自己的 asyncio.Task (_review_loop)，
    与 LifecycleManager 完全解耦。通过 start()/stop() 管理生命周期。
    """

    DEFAULT_MAX_DAILY_REVIEWS = 3
    DEFAULT_REVIEW_INTERVAL_HOURS = 6
    MIN_REVIEW_INTERVAL_HOURS = 4

    def __init__(
        self,
        chroma_manager: Any = None,
        notify_callback: Optional[NotifyCallback] = None,
        review_interval_hours: Optional[int] = None,
        max_daily_reviews: Optional[int] = None,
    ):
        self._chroma = chroma_manager
        self._notify = notify_callback
        # 从配置系统读取，构造参数可覆盖
        self._review_interval_hours = (
            review_interval_hours
            if review_interval_hours is not None
            else self._load_config_interval()
        )
        self._max_daily_reviews = (
            max_daily_reviews
            if max_daily_reviews is not None
            else self._load_config_max_daily()
        )
        # 简易内存记录：{user_id: [(memory_id, review_time), ...]}
        self._review_history: Dict[str, List[tuple]] = {}
        # 后台任务
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    # ── 生命周期管理 ──

    async def start(self) -> None:
        """启动独立后台回顾循环"""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._review_loop())
        logger.info(
            f"ReinforcementEngine started: interval={self._review_interval_hours}h, "
            f"max_daily={self._max_daily_reviews}"
        )

    async def stop(self) -> None:
        """停止后台回顾循环（热更新友好）"""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"ReinforcementEngine stop error: {e}")
            self._task = None
        logger.debug("ReinforcementEngine stopped")

    async def _review_loop(self) -> None:
        """独立回顾调度循环

        每 review_interval_hours 触发一次扫描，与 LifecycleManager 互不干扰。
        """
        interval_seconds = self._review_interval_hours * 3600
        while self._is_running:
            try:
                await asyncio.sleep(interval_seconds)
                if not self._is_running:
                    break
                await self._run_review_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Review loop error: {e}")

    async def _run_review_cycle(self) -> None:
        """执行一次回顾扫描：收集所有活跃用户并推送回顾"""
        if not self._chroma or not self._notify:
            return

        try:
            active_user_ids = await self._chroma.get_active_user_ids()
        except Exception as e:
            logger.warning(f"Failed to get active users for review: {e}")
            return

        for user_id in active_user_ids:
            try:
                candidates = await self.get_review_candidates(user_id)
                for memory in candidates:
                    prompt = ReviewPromptGenerator.generate(memory)
                    await self._notify(
                        user_id,
                        getattr(memory, "group_id", None),
                        prompt,
                        {"memory_id": memory.id, "type": "memory_review"},
                    )
                    self.record_review(memory.id, user_id)
            except Exception as e:
                logger.warning(f"Review cycle failed for user {user_id}: {e}")

    # ── 配置读取 ──

    @staticmethod
    def _load_config_interval() -> int:
        """从 ConfigStore 读取回顾间隔小时数"""
        try:
            from iris_memory.config import get_store
            return get_store().get(
                "memory.reinforcement.interval_hours",
                MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS,
            )
        except Exception:
            return MemoryReinforcementEngine.DEFAULT_REVIEW_INTERVAL_HOURS

    @staticmethod
    def _load_config_max_daily() -> int:
        """从 ConfigStore 读取每日回顾上限"""
        try:
            from iris_memory.config import get_store
            return get_store().get(
                "memory.reinforcement.max_daily",
                MemoryReinforcementEngine.DEFAULT_MAX_DAILY_REVIEWS,
            )
        except Exception:
            return MemoryReinforcementEngine.DEFAULT_MAX_DAILY_REVIEWS

    # ── 回顾候选与反馈 ──

    async def get_review_candidates(
        self,
        user_id: str,
        max_count: Optional[int] = None,
    ) -> List[Memory]:
        """获取今日应回顾的记忆

        选择策略:
        1. 优先 RIF 接近阈值的记忆（即将被遗忘的）
        2. 优先有保护标记的记忆
        3. 每日最多 max_count 条
        """
        if max_count is None:
            max_count = self._max_daily_reviews
        today_count = self._get_today_review_count(user_id)
        if today_count >= max_count:
            return []

        remaining = max_count - today_count

        if not self._chroma:
            return []

        try:
            # 获取候选池
            episodic = await self._chroma.get_memories_by_storage_layer(StorageLayer.EPISODIC)
            semantic = await self._chroma.get_memories_by_storage_layer(StorageLayer.SEMANTIC)
            all_memories = (episodic or []) + (semantic or [])

            # 过滤当前用户 + 最低重要性
            candidates = [
                m for m in all_memories
                if m.user_id == user_id
                and m.importance_score >= 0.4
                and m.review_status != "pending_review"
            ]

            # 排除近期已回顾的
            filtered = []
            for mem in candidates:
                last_review = self._get_last_review_time(mem.id, user_id)
                if last_review:
                    next_review = self._calculate_next_review(mem, last_review)
                    if datetime.now() < next_review:
                        continue
                filtered.append(mem)

            # 按优先级排序
            scored = [(self._review_priority(m), m) for m in filtered]
            scored.sort(key=lambda x: x[0], reverse=True)

            return [m for _, m in scored[:remaining]]

        except Exception as e:
            logger.warning(f"Failed to get review candidates: {e}")
            return []

    def _review_priority(self, memory: Memory) -> float:
        """计算回顾优先级"""
        priority = 0.0

        # RIF 接近阈值 → 优先回顾
        if memory.rif_score < 0.5:
            priority += (0.5 - memory.rif_score) * 2.0

        # 有保护标记 → 优先
        if hasattr(memory, "is_protected") and memory.is_protected:
            priority += 0.3

        # 高重要性 → 优先
        priority += memory.importance_score * 0.2

        # 高情感权重 → 优先
        priority += memory.emotional_weight * 0.1

        return priority

    def _calculate_next_review(
        self, memory: Memory, last_review: datetime
    ) -> datetime:
        """SM-2 变体计算下次回顾时间"""
        review_count = memory.access_count

        # 易度因子：高重要性 → 更短间隔
        ef = 2.5 - memory.importance_score * 0.8
        ef = max(1.3, ef)

        if review_count <= 1:
            interval_days = 1
        elif review_count == 2:
            interval_days = 6
        else:
            interval_days = int(6 * (ef ** (review_count - 2)))

        interval_days = min(interval_days, 90)
        return last_review + timedelta(days=interval_days)

    async def process_review_response(
        self,
        memory_id: str,
        user_id: str,
        user_response: str,
    ) -> None:
        """处理用户对回顾的反馈

        - 更新 last_access_time（强化）
        - 如果用户表示"不用记了"，标记为可归档
        """
        if not self._chroma:
            return

        try:
            memory = await self._chroma.get_memory(memory_id)
            if not memory:
                return

            # 否定反馈检测
            negative_patterns = ["不用记", "忘了吧", "不用了", "算了", "不重要"]
            is_negative = any(p in user_response for p in negative_patterns)

            if is_negative:
                memory.review_status = "rejected"
                memory.importance_score = max(0.1, memory.importance_score - 0.2)
            else:
                memory.update_access()
                memory.importance_score = min(1.0, memory.importance_score + 0.05)

            await self._chroma.update_memory(memory)
        except Exception as e:
            logger.warning(f"Failed to process review response: {e}")

    def record_review(self, memory_id: str, user_id: str) -> None:
        """记录一次回顾"""
        if user_id not in self._review_history:
            self._review_history[user_id] = []
        self._review_history[user_id].append((memory_id, datetime.now()))

    # ── 内部辅助 ──

    def _get_today_review_count(self, user_id: str) -> int:
        history = self._review_history.get(user_id, [])
        today = datetime.now().date()
        return sum(1 for _, t in history if t.date() == today)

    def _get_last_review_time(
        self, memory_id: str, user_id: str
    ) -> Optional[datetime]:
        history = self._review_history.get(user_id, [])
        for mid, t in reversed(history):
            if mid == memory_id:
                return t
        return None
