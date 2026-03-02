"""
冷启动策略

解决新场景的探索-利用平衡问题。
三阶段策略：探索期 → 校准期 → 稳定期。
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, List, Optional, Tuple

from iris_memory.proactive.core.models import ProactiveScene
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.cold_start")


class ColdStartStrategy:
    """冷启动三阶段策略

    关键设计：
    1. usage_count 在"尝试"时就 +1，异步持久化到 SQLite
    2. 探索期不降分，在阈值附近引入随机扰动
    3. 让新场景有机会被触发，同时不抑制明显优质匹配
    """

    # 阶段1：探索期（前50次尝试）
    EXPLORATION_THRESHOLD = 50

    # 阶段2：校准期（50-200次尝试）
    CALIBRATION_THRESHOLD = 200

    @staticmethod
    def get_initial_success_rate(scene_type: str) -> float:
        """根据场景类型获取初始成功率

        Args:
            scene_type: 场景类型

        Returns:
            初始成功率
        """
        base_rates = {
            "question": 0.6,
            "emotion": 0.5,
            "chat": 0.4,
            "followup": 0.55,
        }
        return base_rates.get(scene_type, 0.5)

    @staticmethod
    async def prepare_scenes_for_detection(
        candidate_scenes: List[ProactiveScene],
        feedback_store: Optional[Any] = None,
        max_scenes: int = 5,
    ) -> List[ProactiveScene]:
        """从候选场景中准备用于检测的场景

        对所有候选场景：
        1. usage_count += 1
        2. 异步持久化到 SQLite
        3. 标记探索模式

        Args:
            candidate_scenes: 候选场景列表
            feedback_store: FeedbackStore 实例
            max_scenes: 最大场景数

        Returns:
            处理后的场景列表
        """
        selected: List[ProactiveScene] = []
        usage_updates: List[Tuple[str, int]] = []

        for scene in candidate_scenes[:max_scenes]:
            old_usage = scene.usage_count
            scene.usage_count += 1
            usage_updates.append((scene.scene_id, scene.usage_count))

            # 标记探索模式
            if old_usage < ColdStartStrategy.EXPLORATION_THRESHOLD:
                scene.exploration_mode = True
            else:
                scene.exploration_mode = False

            selected.append(scene)

        # 异步批量持久化
        if feedback_store and usage_updates:
            try:
                asyncio.create_task(
                    feedback_store.batch_update_usage_counts(usage_updates)
                )
            except Exception as e:
                logger.warning(f"Failed to persist usage counts: {e}")

        return selected

    @staticmethod
    def apply_exploration_perturbation(
        final_score: float,
        scene: ProactiveScene,
        threshold_high: float = 0.85,
        threshold_mid: float = 0.6,
    ) -> float:
        """应用探索期随机扰动

        只在阈值附近的窄区间（±0.05）引入随机扰动，
        让新场景有机会被触发。

        Args:
            final_score: 原始最终分数
            scene: 场景对象
            threshold_high: 高置信阈值
            threshold_mid: 中置信阈值

        Returns:
            扰动后的分数
        """
        if not scene.exploration_mode:
            return final_score

        # 定义窄边界区域
        high_low = threshold_high - 0.05
        high_high = threshold_high + 0.05
        mid_low = threshold_mid - 0.05
        mid_high = threshold_mid + 0.05

        in_high = high_low <= final_score <= high_high
        in_mid = mid_low <= final_score <= mid_high

        if in_high or in_mid:
            perturbation = random.uniform(-0.05, 0.08)  # 略偏正向
            final_score += perturbation

        return max(0.0, min(1.0, final_score))
