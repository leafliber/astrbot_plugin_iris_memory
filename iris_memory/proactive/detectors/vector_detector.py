"""
L2 向量检测器

语义级理解，通过向量相似度检索历史优质回复场景。
使用 ChromaDB 场景库 + SQLite 权重，结合冷启动策略。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from iris_memory.proactive.core.models import (
    PersonalityConfig,
    ProactiveContext,
    ProactiveScene,
    ReplyType,
    SceneMatch,
    SceneType,
    UrgencyLevel,
    VectorResult,
    get_personality_config,
)
from iris_memory.proactive.detectors.base import BaseDetector
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.detector.vector")


class VectorDetector(BaseDetector):
    """L2 向量检测器

    检测流程：
    1. 使用 context.conversation.base_query_vector 查询 ChromaDB
    2. 从 SQLite 获取候选场景的 success_rate
    3. 应用冷启动策略
    4. 计算加权相似度并返回结果
    """

    def __init__(
        self,
        scene_store: Optional[Any] = None,
        feedback_store: Optional[Any] = None,
        cold_start: Optional[Any] = None,
        personality: str = "balanced",
        top_k: int = 5,
    ) -> None:
        super().__init__(name="vector")
        self._scene_store = scene_store
        self._feedback_store = feedback_store
        self._cold_start = cold_start
        self._personality = personality
        self._top_k = top_k

    async def detect(self, context: ProactiveContext) -> VectorResult:
        """执行向量检测

        Args:
            context: 主动回复上下文

        Returns:
            VectorResult 检测结果
        """
        # 获取查询向量
        query_vector = context.conversation.base_query_vector
        if not query_vector:
            return VectorResult(confidence=0.0)

        if not self._scene_store:
            return VectorResult(confidence=0.0)

        p_config = get_personality_config(
            self._personality, context.session_type
        )

        # 1. ChromaDB 向量检索
        candidates = await self._scene_store.query_similar(
            query_vector=query_vector,
            top_k=self._top_k,
            where_filter={"is_active": True},
        )

        if not candidates:
            return VectorResult(confidence=0.0)

        # 2. 从 SQLite 获取权重
        scene_ids = [s.scene_id for s in candidates]
        weights = {}
        if self._feedback_store:
            try:
                weights = await self._feedback_store.get_scene_weights_batch(scene_ids)
            except Exception as e:
                logger.warning(f"Failed to get scene weights: {e}")

        # 合并权重到候选场景
        for scene in candidates:
            w = weights.get(scene.scene_id)
            if w:
                # 原始相似度已经临时存储在 success_rate
                raw_similarity = scene.success_rate
                scene.success_rate = w.success_rate
                scene.usage_count = w.usage_count
                # 恢复原始相似度到临时字段
                scene.exploration_mode = False
                # 我们需要在 SceneMatch 中使用原始相似度
                setattr(scene, "_raw_similarity", raw_similarity)
            else:
                raw_similarity = scene.success_rate
                scene.success_rate = 0.5  # 默认
                scene.usage_count = 0
                setattr(scene, "_raw_similarity", raw_similarity)

        # 3. 冷启动策略
        if self._cold_start:
            try:
                candidates = await self._cold_start.prepare_scenes_for_detection(
                    candidates, self._feedback_store
                )
            except Exception as e:
                logger.warning(f"Cold start preparation failed: {e}")

        # 4. 计算加权相似度并构建 SceneMatch 列表
        matches: List[SceneMatch] = []
        for scene in candidates:
            raw_sim = getattr(scene, "_raw_similarity", scene.success_rate)
            weighted_sim = self._calculate_weighted_similarity(
                raw_sim, context, scene
            )

            # 计算最终分数（融入 success_rate）
            final_score = self._calculate_final_score(
                weighted_sim, scene.success_rate, scene.usage_count,
                scene.exploration_mode,
            )

            # 冷启动探索扰动
            if self._cold_start and scene.exploration_mode:
                final_score = self._cold_start.apply_exploration_perturbation(
                    final_score, scene,
                    threshold_high=p_config.vector_threshold_high,
                    threshold_mid=p_config.vector_threshold_mid,
                )

            try:
                scene_type = SceneType(scene.scene_type)
            except ValueError:
                scene_type = SceneType.CHAT

            matches.append(SceneMatch(
                scene_id=scene.scene_id,
                scene_type=scene_type,
                similarity=raw_sim,
                weighted_similarity=weighted_sim,
                final_score=final_score,
                success_rate=scene.success_rate,
                usage_count=scene.usage_count,
                exploration_mode=scene.exploration_mode,
                trigger_pattern=scene.description,
            ))

        # 按最终分数排序
        matches.sort(key=lambda m: m.final_score, reverse=True)
        best = matches[0] if matches else None

        # 5. 决策
        if best:
            final = best.final_score
            threshold_high = p_config.vector_threshold_high
            threshold_mid = p_config.vector_threshold_mid

            if final >= threshold_high:
                should_reply = True
                confidence = final
            elif final >= threshold_mid:
                should_reply = False  # 需要 L3 确认
                confidence = final
            else:
                should_reply = False
                confidence = final

            # 推断回复类型
            reply_type = self._infer_reply_type(best.scene_type)

            return VectorResult(
                matches=matches,
                best_match=best,
                final_score=final,
                confidence=confidence,
                should_reply=should_reply,
                reply_type=reply_type,
            )

        return VectorResult(confidence=0.0)

    # ========== 计算方法 ==========

    @staticmethod
    def _calculate_weighted_similarity(
        base_similarity: float,
        context: ProactiveContext,
        scene: ProactiveScene,
    ) -> float:
        """多维度加权相似度计算"""
        score = base_similarity

        # 1. 情绪匹配加权（±10%）
        if scene.target_emotion and context.user.emotional_state:
            emo_state = context.user.emotional_state
            primary = getattr(emo_state, "primary", None) or getattr(emo_state, "emotion", None)
            if primary:
                if scene.target_emotion == str(primary):
                    score += 0.1
                elif scene.target_emotion == "any":
                    pass
                else:
                    score -= 0.05

        # 2. 时间模式匹配（+5%）
        if scene.time_pattern and scene.time_pattern != "any":
            if _match_time_pattern(scene.time_pattern, context.temporal.hour):
                score += 0.05

        # 3. 群活跃度调整
        if context.session_type == "group" and context.group:
            if context.group.activity_level > 0.8:
                score -= 0.05
            elif context.group.activity_level < 0.3:
                score += 0.05

        # 4. 用户偏好调整
        pref = context.user.proactive_preference
        if pref < 0.3:
            score -= 0.1
        elif pref > 0.7:
            score += 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _calculate_final_score(
        weighted_similarity: float,
        success_rate: float,
        usage_count: int,
        exploration_mode: bool,
    ) -> float:
        """计算最终匹配分数

        公式: final_score = weighted_similarity * (0.5 + 0.5 * success_rate)
        """
        base_weight = 0.5
        success_weight = 0.5 * success_rate
        return weighted_similarity * (base_weight + success_weight)

    @staticmethod
    def _infer_reply_type(scene_type: SceneType) -> ReplyType:
        """从场景类型推断回复类型"""
        mapping = {
            SceneType.QUESTION: ReplyType.QUESTION,
            SceneType.EMOTION: ReplyType.EMOTION,
            SceneType.CHAT: ReplyType.CHAT,
            SceneType.FOLLOWUP: ReplyType.FOLLOWUP,
        }
        return mapping.get(scene_type, ReplyType.CHAT)


def _match_time_pattern(pattern: str, hour: int) -> bool:
    """匹配时间模式"""
    patterns = {
        "morning": (6, 12),
        "afternoon": (12, 18),
        "evening": (18, 24),
        "night": (22, 6),
    }
    if pattern not in patterns:
        return False
    start, end = patterns[pattern]
    if start <= end:
        return start <= hour < end
    else:
        return hour >= start or hour < end
