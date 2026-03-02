"""
反馈追踪器

记录回复效果，动态优化场景权重。
简化反馈信号，关注可观测指标。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.models import (
    ProactiveContext,
    ProactiveDecision,
    ReplyFeedback,
    ReplyRecord,
    calculate_engagement,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.feedback_tracker")


class FeedbackTracker:
    """反馈追踪器

    职责：
    1. 记录每次主动回复
    2. 追踪回复效果（用户是否回应）
    3. 基于反馈更新场景权重
    """

    def __init__(
        self,
        feedback_store: Optional[Any] = None,
        scene_store: Optional[Any] = None,
        tracking_window: int = 300,
        ema_alpha: float = 0.2,
    ) -> None:
        """
        Args:
            feedback_store: FeedbackStore 实例
            scene_store: SceneStore 实例
            tracking_window: 反馈追踪窗口（秒）
            ema_alpha: 指数移动平均系数
        """
        self._feedback_store = feedback_store
        self._scene_store = scene_store
        self._tracking_window = tracking_window
        self._ema_alpha = ema_alpha
        # session_key → record_id 映射（等待反馈）
        self._pending_feedback: Dict[str, str] = {}

    async def record_reply(
        self,
        context: ProactiveContext,
        decision: ProactiveDecision,
        content_summary: str = "",
    ) -> Optional[str]:
        """记录一次主动回复

        Args:
            context: 上下文
            decision: 决策
            content_summary: 回复内容摘要

        Returns:
            record_id 或 None（失败时）
        """
        if not self._feedback_store:
            return None

        record_id = f"reply_{uuid.uuid4().hex[:12]}"
        scene_ids = [m.scene_id for m in decision.matched_scenes]
        if not scene_ids and decision.matched_rules:
            # L1 规则直接回复，使用虚拟 scene_id
            scene_ids = [f"rule:{r}" for r in decision.matched_rules]

        record = ReplyRecord(
            record_id=record_id,
            session_key=context.session_key,
            session_type=context.session_type,
            scene_ids=scene_ids,
            decision_type=decision.decision_type.value,
            urgency=decision.urgency.value,
            reply_type=decision.reply_type.value,
            confidence=decision.confidence,
            sent_at=datetime.now(),
            content_summary=content_summary[:200],
            topic_vector=context.conversation.current_topic_vector,
        )

        try:
            await self._feedback_store.record_reply(record)
            self._pending_feedback[context.session_key] = record_id

            # 更新每日统计
            asyncio.create_task(
                self._feedback_store.increment_daily_stats(
                    detection_type=decision.decision_type.value,
                    reply_sent=True,
                )
            )

            logger.debug(
                f"Recorded reply {record_id} for {context.session_key} "
                f"(type={decision.reply_type.value})"
            )
            return record_id
        except Exception as e:
            logger.error(f"Failed to record reply: {e}")
            return None

    async def process_user_response(
        self,
        session_key: str,
        user_replied_directly: bool = False,
    ) -> None:
        """处理用户回应（用于更新反馈）

        Args:
            session_key: 会话键
            user_replied_directly: 用户是否直接回复 Bot
        """
        record_id = self._pending_feedback.pop(session_key, None)
        if not record_id or not self._feedback_store:
            return

        feedback = ReplyFeedback(
            feedback_id=f"fb_{uuid.uuid4().hex[:12]}",
            record_id=record_id,
            user_replied=user_replied_directly,
            reply_within_window=True,
            recorded_at=datetime.now(),
        )
        engagement = calculate_engagement(feedback)
        feedback.engagement_score = engagement

        try:
            await self._feedback_store.record_feedback(feedback)
            # 更新场景权重
            await self._update_scene_weights_from_feedback(
                record_id, engagement
            )
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")

    async def process_no_response(self, session_key: str) -> None:
        """处理无回应（追踪窗口超时）"""
        record_id = self._pending_feedback.pop(session_key, None)
        if not record_id or not self._feedback_store:
            return

        feedback = ReplyFeedback(
            feedback_id=f"fb_{uuid.uuid4().hex[:12]}",
            record_id=record_id,
            user_replied=False,
            reply_within_window=False,
            engagement_score=0.0,
            recorded_at=datetime.now(),
        )

        try:
            await self._feedback_store.record_feedback(feedback)
            await self._update_scene_weights_from_feedback(
                record_id, 0.0
            )
        except Exception as e:
            logger.error(f"Failed to process no-response feedback: {e}")

    async def _update_scene_weights_from_feedback(
        self,
        record_id: str,
        engagement: float,
    ) -> None:
        """根据反馈更新场景权重"""
        if not self._feedback_store:
            return

        # 查找回复记录获取 scene_ids
        try:
            recent = await self._feedback_store.get_recent_replies(
                "", limit=100  # 使用空 session_key 表示全部搜索
            )
            record = None
            for r in recent:
                if r.record_id == record_id:
                    record = r
                    break

            if not record or not record.scene_ids:
                return

            for scene_id in record.scene_ids:
                await self._update_single_scene_weight(
                    scene_id, engagement
                )
        except Exception as e:
            logger.warning(f"Failed to update scene weights: {e}")

    async def _update_single_scene_weight(
        self,
        scene_id: str,
        engagement: float,
    ) -> None:
        """更新单个场景权重"""
        current = await self._feedback_store.get_scene_weight(scene_id)
        old_rate = current.success_rate if current else 0.5
        usage_count = current.usage_count if current else 0

        # 指数移动平均
        new_rate = (1 - self._ema_alpha) * old_rate + self._ema_alpha * engagement
        new_rate = max(0.1, min(0.95, new_rate))

        await self._feedback_store.upsert_scene_weight(
            scene_id=scene_id,
            success_rate=new_rate,
            usage_count=usage_count + 1,
            last_used=datetime.now(),
        )

        # 记录权重变更历史
        if abs(new_rate - old_rate) > 0.01:
            await self._feedback_store.record_weight_change(
                scene_id=scene_id,
                old_rate=old_rate,
                new_rate=new_rate,
                reason=f"feedback_engagement={engagement:.2f}",
            )

        # 自动停用低质量场景
        if new_rate < 0.2 and usage_count > 5 and self._scene_store:
            await self._scene_store.deactivate(scene_id)
            logger.info(
                f"Deactivated scene {scene_id}: "
                f"success_rate={new_rate:.2f}, usage={usage_count}"
            )

    def get_pending_sessions(self) -> List[str]:
        """获取等待反馈的会话列表"""
        return list(self._pending_feedback.keys())
