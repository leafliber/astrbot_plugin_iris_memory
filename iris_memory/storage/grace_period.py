"""
宽限期管理器

在记忆被最终删除前给予用户知情权和干预机会。
"""

from datetime import datetime, timedelta
from typing import List, Literal, Optional

from iris_memory.core.types import QualityLevel, StorageLayer
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

logger = get_logger("grace_period")


class GracePeriodManager:
    """宽限期管理器

    评估即将清除的记忆，决定是直接清除、进入宽限期还是跳过（受保护）。
    """

    DEFAULT_GRACE_DAYS = 7
    SILENT_DELETE_CONFIDENCE_THRESHOLD = 0.3
    SILENT_DELETE_ACCESS_THRESHOLD = 0

    def __init__(
        self,
        chroma_manager=None,
        proactive_manager=None,
        grace_days: int = DEFAULT_GRACE_DAYS,
    ):
        self._chroma = chroma_manager
        self._proactive = proactive_manager
        self._grace_days = grace_days

    # ── 核心评估 ──

    async def evaluate_and_apply(self, memory: Memory) -> str:
        """评估记忆是否应进入宽限期或直接清除。

        Returns:
            "protected"       - 受保护，跳过
            "grace_period"    - 进入宽限期
            "silent_delete"   - 直接清除（极低价值）
            "already_pending" - 已在宽限期中
            "expired"         - 宽限期已到
        """
        # 保护检查
        if hasattr(memory, "is_protected") and memory.is_protected:
            return "protected"
        if memory.is_user_requested:
            return "protected"
        if memory.quality_level == QualityLevel.CONFIRMED:
            return "protected"

        # 已在宽限期
        if memory.grace_period_expires_at is not None:
            if datetime.now() >= memory.grace_period_expires_at:
                return "expired"
            return "already_pending"

        # 极低价值：静默清除
        if (
            memory.confidence < self.SILENT_DELETE_CONFIDENCE_THRESHOLD
            and memory.access_count <= self.SILENT_DELETE_ACCESS_THRESHOLD
            and memory.emotional_weight < 0.3
        ):
            return "silent_delete"

        # 有一定价值：进入宽限期
        await self._initiate_grace_period(memory)
        return "grace_period"

    async def _initiate_grace_period(self, memory: Memory) -> None:
        """设置宽限期并通知用户"""
        memory.grace_period_expires_at = datetime.now() + timedelta(days=self._grace_days)
        memory.review_status = "pending_review"

        # 持久化
        if self._chroma:
            try:
                await self._chroma.update_memory(memory)
            except Exception as e:
                logger.warning(f"Failed to persist grace period for {memory.id}: {e}")

        # 通知用户
        if self._proactive and not memory.grace_period_notified:
            await self._notify_user(memory)
            memory.grace_period_notified = True
            if self._chroma:
                try:
                    await self._chroma.update_memory(memory)
                except Exception:
                    pass

    async def _notify_user(self, memory: Memory) -> None:
        """通过主动回复模块通知用户"""
        content_preview = (
            memory.content[:40] + "..."
            if len(memory.content) > 40
            else memory.content
        )
        prompt = (
            f"说起来，你之前提到过「{content_preview}」，"
            f"这件事还需要我帮你记着吗？\n"
            f"（回复'保留'我就继续记住，不回复的话 {self._grace_days} 天后我会慢慢忘掉~）"
        )
        try:
            if hasattr(self._proactive, "send_review_prompt"):
                await self._proactive.send_review_prompt(
                    user_id=memory.user_id,
                    group_id=memory.group_id,
                    prompt=prompt,
                    metadata={"memory_id": memory.id, "type": "grace_period_review"},
                )
        except Exception as e:
            logger.warning(f"Failed to notify user about grace period: {e}")

    # ── 用户响应处理 ──

    async def resolve_grace_period(
        self,
        memory: Memory,
        action: Literal["keep", "archive", "upgrade"],
    ) -> Memory:
        """处理用户对宽限期记忆的决定。

        Args:
            memory: 宽限期中的记忆
            action:
                "keep"    - 保留并刷新
                "archive" - 立即归档
                "upgrade" - 升级到上层
        """
        if action == "keep":
            memory.grace_period_expires_at = None
            memory.grace_period_notified = False
            memory.review_status = None
            memory.last_access_time = datetime.now()
            memory.access_count += 1
        elif action == "archive":
            memory.review_status = "rejected"
        elif action == "upgrade":
            memory.grace_period_expires_at = None
            memory.grace_period_notified = False
            memory.review_status = "approved"
            if memory.storage_layer == StorageLayer.WORKING:
                memory.storage_layer = StorageLayer.EPISODIC
            elif memory.storage_layer == StorageLayer.EPISODIC:
                memory.storage_layer = StorageLayer.SEMANTIC

        if self._chroma:
            try:
                await self._chroma.update_memory(memory)
            except Exception as e:
                logger.warning(f"Failed to persist grace resolution for {memory.id}: {e}")

        return memory

    # ── 查询 ──

    async def get_pending_review_memories(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """获取当前处于宽限期的记忆列表
        
        Args:
            user_id: 用户ID（None 表示所有用户）
            group_id: 群组ID
            limit: 最大返回数量
        """
        if not self._chroma:
            return []
        try:
            # 从 EPISODIC 层查询所有候选
            episodic = await self._chroma.get_memories_by_storage_layer(StorageLayer.EPISODIC)
            if not episodic:
                return []
            pending = [
                m for m in episodic
                if m.review_status == "pending_review"
                and m.grace_period_expires_at is not None
                and (user_id is None or m.user_id == user_id)
                and (group_id is None or m.group_id == group_id)
            ]
            pending.sort(key=lambda m: m.grace_period_expires_at)
            return pending[:limit]
        except Exception as e:
            logger.warning(f"Failed to get pending review memories: {e}")
            return []
