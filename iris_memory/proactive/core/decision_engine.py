"""
决策引擎

整合三级检测器结果，做出最终决策，管理多轮跟进状态。
内含 FollowUpDetector 作为内部组件。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.models import (
    DecisionType,
    FollowUpDecision,
    FollowUpState,
    LLMResult,
    PersonalityConfig,
    ProactiveContext,
    ProactiveDecision,
    ReplyRecord,
    ReplyType,
    RuleResult,
    UrgencyLevel,
    VectorResult,
    get_personality_config,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.decision_engine")


class FollowUpDetector:
    """多轮跟进检测器

    作为 DecisionEngine 的内部组件，检测是否需要多轮跟进回复。

    状态机：
    - IDLE → [首次回复] → ACTIVE
    - ACTIVE → [跟进成功] → ACTIVE
    - ACTIVE → [话题不相关/超时/他人介入/达到上限] → IDLE
    """

    def __init__(
        self,
        feedback_store: Optional[Any] = None,
        max_count: int = 2,
        time_window: int = 120,
        similarity_threshold: float = 0.6,
    ) -> None:
        self._feedback_store = feedback_store
        self._followup_states: Dict[str, FollowUpState] = {}
        self._max_count = max_count
        self._time_window = time_window
        self._similarity_threshold = similarity_threshold
        self._initialized = False

    async def initialize(self) -> None:
        """从 SQLite 恢复跟进状态"""
        if self._initialized:
            return

        if self._feedback_store:
            try:
                recent = await self._feedback_store.get_recent_followup_counts(
                    window_seconds=self._time_window
                )
                for record in recent:
                    sk = record["session_key"]
                    self._followup_states[sk] = FollowUpState(
                        count=record["followup_count"],
                    )
                logger.info(
                    f"FollowUpDetector initialized with "
                    f"{len(self._followup_states)} states"
                )
            except Exception as e:
                logger.warning(f"Failed to restore followup states: {e}")

        self._initialized = True

    def check_followup(
        self,
        context: ProactiveContext,
        last_proactive: Optional[ReplyRecord],
    ) -> Optional[FollowUpDecision]:
        """检测是否需要跟进回复

        Args:
            context: 主动回复上下文
            last_proactive: 最近一次主动回复记录

        Returns:
            FollowUpDecision 或 None
        """
        if not last_proactive:
            return None

        # 条件1：时间窗口
        time_since = (datetime.now() - last_proactive.sent_at).total_seconds()
        if time_since > self._time_window:
            self._reset_state(context.session_key)
            return None

        # 条件2：用户有新消息
        if not context.has_new_user_message:
            return None

        # 条件3：语义相关性
        if (
            last_proactive.topic_vector
            and context.conversation.current_topic_vector
        ):
            similarity = _cosine_similarity(
                last_proactive.topic_vector,
                context.conversation.current_topic_vector,
            )
            if similarity < self._similarity_threshold:
                self._reset_state(context.session_key)
                return None

        # 条件4：跟进次数上限
        current_count = self._get_followup_count(context.session_key)
        if current_count >= self._max_count:
            self._reset_state(context.session_key)
            return None

        # 条件5：群聊中无其他人介入
        if context.session_type == "group":
            if context.new_participant_count > 0:
                self._reset_state(context.session_key)
                return None

        # 通过所有条件，生成跟进决策
        self._increment_followup_count(context.session_key)
        return FollowUpDecision(
            urgency=UrgencyLevel.MEDIUM,
            reply_type="followup",
            followup_count=current_count + 1,
            original_decision_id=last_proactive.record_id,
        )

    def _reset_state(self, session_key: str) -> None:
        self._followup_states.pop(session_key, None)

    def _get_followup_count(self, session_key: str) -> int:
        state = self._followup_states.get(session_key)
        return state.count if state else 0

    def _increment_followup_count(self, session_key: str) -> None:
        if session_key not in self._followup_states:
            self._followup_states[session_key] = FollowUpState(count=0)
        self._followup_states[session_key].count += 1


class DecisionEngine:
    """决策引擎

    整合三级检测结果，按决策矩阵做出最终决定。

    决策矩阵：
    | L1结果         | L2结果         | L3结果  | 最终决策         |
    |---------------|---------------|---------|-----------------|
    | direct_reply  | -             | -       | 直接回复 (HIGH)  |
    | normal        | high_conf     | -       | 直接回复 (MEDIUM)|
    | normal        | mid_conf      | confirm | 回复 (MEDIUM)   |
    | normal        | mid_conf      | reject  | 不回复          |
    | normal        | low_conf      | -       | 不回复          |
    | fast_reject   | -             | -       | 不回复          |
    """

    def __init__(
        self,
        rule_detector: Optional[Any] = None,
        vector_detector: Optional[Any] = None,
        llm_detector: Optional[Any] = None,
        feedback_store: Optional[Any] = None,
        personality: str = "balanced",
        followup_max_count: int = 2,
        followup_time_window: int = 120,
        followup_similarity_threshold: float = 0.6,
    ) -> None:
        self._rule_detector = rule_detector
        self._vector_detector = vector_detector
        self._llm_detector = llm_detector
        self._feedback_store = feedback_store
        self._personality = personality

        # 内部组件：跟进检测器
        self._followup_detector = FollowUpDetector(
            feedback_store=feedback_store,
            max_count=followup_max_count,
            time_window=followup_time_window,
            similarity_threshold=followup_similarity_threshold,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """初始化决策引擎及子组件"""
        if self._initialized:
            return
        await self._followup_detector.initialize()
        self._initialized = True

    async def decide(
        self, context: ProactiveContext
    ) -> ProactiveDecision:
        """执行决策流程

        Args:
            context: 主动回复上下文

        Returns:
            ProactiveDecision 最终决策
        """
        start_time = time.monotonic()
        p_config = get_personality_config(
            self._personality, context.session_type
        )

        # 静音时段检查
        if context.temporal.is_quiet_hours:
            return ProactiveDecision(
                should_reply=False,
                reason="quiet_hours",
                confidence=1.0,
            )

        # === 跟进检测（优先） ===
        followup = await self._check_followup(context)
        if followup:
            elapsed = (time.monotonic() - start_time) * 1000
            return ProactiveDecision(
                should_reply=True,
                urgency=followup.urgency,
                reply_type=ReplyType.FOLLOWUP,
                decision_type=DecisionType.RULE,
                confidence=0.7,
                reason=f"followup #{followup.followup_count}",
                followup_count=followup.followup_count,
                original_decision_id=followup.original_decision_id,
                detection_latency_ms=elapsed,
            )

        # === L1: 规则检测 ===
        rule_result = RuleResult()
        if self._rule_detector:
            try:
                rule_result = await self._rule_detector.detect(context)
            except Exception as e:
                logger.error(f"Rule detection failed: {e}")

        # L1 直接回复
        if rule_result.should_reply and rule_result.score >= p_config.rule_direct_reply:
            elapsed = (time.monotonic() - start_time) * 1000
            return ProactiveDecision(
                should_reply=True,
                urgency=UrgencyLevel.HIGH,
                reply_type=rule_result.reply_type,
                decision_type=DecisionType.RULE,
                confidence=rule_result.confidence,
                reason="L1_direct_reply",
                matched_rules=rule_result.matched_rules,
                detection_latency_ms=elapsed,
            )

        # L1 快速拒绝
        fast_reject = 0.15 if context.session_type == "private" else 0.2
        if rule_result.score <= fast_reject:
            elapsed = (time.monotonic() - start_time) * 1000
            return ProactiveDecision(
                should_reply=False,
                reason="L1_fast_reject",
                confidence=1.0 - rule_result.score,
                detection_latency_ms=elapsed,
            )

        # === L2: 向量检测 ===
        vector_result = VectorResult()
        if self._vector_detector:
            try:
                vector_result = await self._vector_detector.detect(context)
            except Exception as e:
                logger.error(f"Vector detection failed: {e}")

        # L2 高置信直接回复
        threshold_high = p_config.vector_threshold_high
        threshold_mid = p_config.vector_threshold_mid

        if vector_result.should_reply and vector_result.final_score >= threshold_high:
            elapsed = (time.monotonic() - start_time) * 1000
            return ProactiveDecision(
                should_reply=True,
                urgency=UrgencyLevel.MEDIUM,
                reply_type=vector_result.reply_type,
                decision_type=DecisionType.VECTOR,
                confidence=vector_result.confidence,
                reason="L2_high_confidence",
                matched_scenes=vector_result.matches[:3],
                detection_latency_ms=elapsed,
            )

        # L2 中置信 → 进入 L3
        needs_llm = threshold_mid <= vector_result.final_score < threshold_high

        if not needs_llm:
            # 低置信，不回复
            elapsed = (time.monotonic() - start_time) * 1000
            return ProactiveDecision(
                should_reply=False,
                reason="L2_low_confidence",
                confidence=1.0 - vector_result.final_score,
                detection_latency_ms=elapsed,
            )

        # === L3: LLM 确认 ===
        llm_result = LLMResult()
        llm_used = False
        if self._llm_detector:
            try:
                llm_result = await self._llm_detector.detect(
                    context, vector_result=vector_result
                )
                llm_used = True
            except Exception as e:
                logger.error(f"LLM detection failed: {e}")

        elapsed = (time.monotonic() - start_time) * 1000

        if llm_result.should_reply:
            return ProactiveDecision(
                should_reply=True,
                urgency=llm_result.urgency,
                reply_type=llm_result.reply_type,
                decision_type=DecisionType.LLM,
                confidence=llm_result.confidence,
                reason=f"L3_confirmed: {llm_result.reason}",
                matched_scenes=vector_result.matches[:3],
                detection_latency_ms=elapsed,
                llm_used=llm_used,
            )
        else:
            return ProactiveDecision(
                should_reply=False,
                reason=f"L3_rejected: {llm_result.reason}",
                confidence=llm_result.confidence,
                detection_latency_ms=elapsed,
                llm_used=llm_used,
            )

    async def _check_followup(
        self, context: ProactiveContext
    ) -> Optional[FollowUpDecision]:
        """检查跟进回复"""
        if not self._feedback_store:
            return None
        try:
            last = await self._feedback_store.get_last_reply(context.session_key)
            return self._followup_detector.check_followup(context, last)
        except Exception as e:
            logger.warning(f"Followup check failed: {e}")
            return None

    async def close(self) -> None:
        """释放资源"""
        pass


def _cosine_similarity(
    vec_a: List[float], vec_b: List[float]
) -> float:
    """计算余弦相似度"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
