"""
上下文聚合引擎

统一聚合所有决策所需的上下文信息，包括对话上下文、用户画像、
群聊状态、时间维度等，输出 ProactiveContext。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.models import (
    ConversationContext,
    GroupContext,
    ProactiveContext,
    ReplyRecord,
    TemporalContext,
    UserContext,
    count_tokens,
    is_quiet_hours,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.context_engine")


class ContextEngine:
    """上下文聚合引擎

    职责：将分散的消息、用户画像、群组状态等信息统一聚合为
    ProactiveContext，供 DecisionEngine 使用。
    """

    def __init__(
        self,
        embedding_manager: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        feedback_store: Optional[Any] = None,
        max_history: int = 10,
        silence_threshold: int = 300,
        max_text_tokens: int = 150,
        quiet_hours: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            embedding_manager: 嵌入管理器
            shared_state: SharedState 共享状态（存储用户画像/情感状态）
            feedback_store: FeedbackStore 反馈存储
            max_history: 最大历史消息数
            silence_threshold: 沉默判定时长（秒）
            max_text_tokens: 向量化文本最大 token 数
            quiet_hours: 静音时段 [start, end]
        """
        self._embedding_manager = embedding_manager
        self._shared_state = shared_state
        self._feedback_store = feedback_store
        self._max_history = max_history
        self._silence_threshold = silence_threshold
        self._max_text_tokens = max_text_tokens
        self._quiet_hours = quiet_hours or [23, 7]

    async def build_context(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        session_key: str,
        session_type: str = "group",
        group_id: Optional[str] = None,
        trigger_user_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ProactiveContext:
        """构建完整的主动回复上下文

        Args:
            messages: 最近的消息列表 (dict with text, sender_id, sender_name, timestamp)
            user_id: 触发用户 ID
            session_key: 会话键
            session_type: 会话类型 ("group" / "private")
            group_id: 群组 ID（群聊时提供）
            trigger_user_id: 触发用户（默认同 user_id）
            extra: 额外上下文数据

        Returns:
            ProactiveContext 实例
        """
        extra = extra or {}
        trigger_uid = trigger_user_id or user_id
        recent = messages[-self._max_history:]

        # 构建各维度上下文（可并行的部分）
        conversation = await self._build_conversation_context(
            recent, trigger_uid
        )
        user_ctx = await self._build_user_context(user_id, session_key)
        temporal = self._build_temporal_context()
        group_ctx = self._build_group_context(
            group_id, extra
        ) if session_type == "group" and group_id else None

        # 检测状态
        has_new_user_message = extra.get("has_new_user_message", len(recent) > 0)
        new_participant_count = extra.get("new_participant_count", 0)

        return ProactiveContext(
            session_type=session_type,
            session_key=session_key,
            conversation=conversation,
            user=user_ctx,
            group=group_ctx,
            temporal=temporal,
            has_new_user_message=has_new_user_message,
            new_participant_count=new_participant_count,
        )

    # ========== 内部方法 ==========

    async def _build_conversation_context(
        self,
        messages: List[Dict[str, Any]],
        trigger_user_id: str,
    ) -> ConversationContext:
        """构建对话上下文"""
        recent_text = self._build_recent_text(
            messages, trigger_user_id, self._max_text_tokens
        )

        # 时间跨度和沉默判定
        time_span = 0.0
        silence_duration = 0.0
        if messages:
            timestamps = []
            for m in messages:
                ts = m.get("timestamp")
                if isinstance(ts, (int, float)):
                    timestamps.append(ts)
                elif isinstance(ts, datetime):
                    timestamps.append(ts.timestamp())
            if timestamps:
                time_span = max(timestamps) - min(timestamps)
                silence_duration = datetime.now().timestamp() - max(timestamps)

        # 话题连续性（简单估算：基于消息间隔）
        topic_continuity = max(0.0, 1.0 - silence_duration / self._silence_threshold) if self._silence_threshold > 0 else 0.0

        # 向量化
        base_query_vector: Optional[List[float]] = None
        current_topic_vector: Optional[List[float]] = None
        if self._embedding_manager and recent_text:
            try:
                base_query_vector = await self._embedding_manager.embed(recent_text)
                current_topic_vector = base_query_vector  # 同一文本
            except Exception as e:
                logger.warning(f"Embedding failed in context engine: {e}")

        return ConversationContext(
            recent_messages=messages,
            recent_text=recent_text,
            time_span=time_span,
            silence_duration=silence_duration,
            topic_continuity=topic_continuity,
            current_topic_vector=current_topic_vector,
            base_query_vector=base_query_vector,
        )

    async def _build_user_context(
        self, user_id: str, session_key: str
    ) -> UserContext:
        """构建用户维度上下文"""
        persona = None
        emotional_state = None
        proactive_preference = 0.5
        reply_history: List[ReplyRecord] = []

        # 从 SharedState 获取画像和情感
        if self._shared_state:
            try:
                persona = self._shared_state.get_user_persona(user_id)
            except Exception:
                pass
            try:
                emotional_state = self._shared_state.get_emotional_state(user_id)
            except Exception:
                pass

        # 从 FeedbackStore 获取回复历史
        if self._feedback_store:
            try:
                reply_history = await self._feedback_store.get_recent_replies(
                    session_key, limit=5
                )
            except Exception:
                pass

        return UserContext(
            user_id=user_id,
            persona=persona,
            emotional_state=emotional_state,
            proactive_preference=proactive_preference,
            reply_history=reply_history,
        )

    def _build_temporal_context(self) -> TemporalContext:
        """构建时间维度上下文"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        return TemporalContext(
            hour=hour,
            is_weekend=weekday >= 5,
            is_holiday=False,  # 简单实现：不做假日判断
            is_quiet_hours=is_quiet_hours(hour, self._quiet_hours),
        )

    def _build_group_context(
        self,
        group_id: Optional[str],
        extra: Dict[str, Any],
    ) -> Optional[GroupContext]:
        """构建群聊维度上下文"""
        if not group_id:
            return None
        return GroupContext(
            group_id=group_id,
            activity_level=extra.get("activity_level", 0.5),
            topic_heat=extra.get("topic_heat", {}),
            participant_count=extra.get("participant_count", 0),
            last_bot_reply_ago=extra.get("last_bot_reply_ago", 0.0),
        )

    @staticmethod
    def _build_recent_text(
        messages: List[Dict[str, Any]],
        trigger_user_id: str,
        max_tokens: int = 150,
        include_speaker: bool = True,
    ) -> str:
        """构建用于向量化的文本

        策略：
        1. 保留所有人的消息（保留上下文语义）
        2. 对触发用户的消息添加 [Focus] 标记
        3. 从最新消息开始，按 token 倒序截断
        """
        text_parts: List[str] = []
        total_tokens = 0

        for msg in reversed(messages):
            sender_id = msg.get("sender_id", "")
            sender_name = msg.get("sender_name", "")
            text = msg.get("text", "")
            if not text:
                continue

            prefix = "[Focus] " if sender_id == trigger_user_id else ""
            if include_speaker and sender_name:
                part = f"{prefix}{sender_name}: {text}"
            else:
                part = f"{prefix}{text}"

            part_tokens = count_tokens(part)
            if total_tokens + part_tokens > max_tokens:
                break

            text_parts.insert(0, part)  # 保持时间顺序
            total_tokens += part_tokens

        return "\n".join(text_parts)
