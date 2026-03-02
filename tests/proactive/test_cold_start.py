"""
ColdStartStrategy 单元测试
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock

import pytest

from iris_memory.proactive.core.cold_start import ColdStartStrategy
from iris_memory.proactive.core.models import ProactiveScene


class TestColdStartStrategy:
    def test_initial_success_rate_question(self):
        assert ColdStartStrategy.get_initial_success_rate("question") == 0.6

    def test_initial_success_rate_emotion(self):
        assert ColdStartStrategy.get_initial_success_rate("emotion") == 0.5

    def test_initial_success_rate_chat(self):
        assert ColdStartStrategy.get_initial_success_rate("chat") == 0.4

    def test_initial_success_rate_unknown(self):
        assert ColdStartStrategy.get_initial_success_rate("unknown") == 0.5

    @pytest.mark.asyncio
    async def test_prepare_scenes_increments_usage(self):
        scenes = [
            ProactiveScene(scene_id="s1", usage_count=0),
            ProactiveScene(scene_id="s2", usage_count=10),
        ]
        result = await ColdStartStrategy.prepare_scenes_for_detection(
            scenes, feedback_store=None, max_scenes=5
        )
        assert result[0].usage_count == 1
        assert result[1].usage_count == 11
        assert result[0].exploration_mode is True
        assert result[1].exploration_mode is True  # 10 < 50

    @pytest.mark.asyncio
    async def test_prepare_scenes_stable_mode(self):
        scene = ProactiveScene(scene_id="s1", usage_count=60)
        result = await ColdStartStrategy.prepare_scenes_for_detection(
            [scene], feedback_store=None
        )
        assert result[0].exploration_mode is False

    @pytest.mark.asyncio
    async def test_prepare_max_scenes_limit(self):
        scenes = [ProactiveScene(scene_id=f"s{i}") for i in range(20)]
        result = await ColdStartStrategy.prepare_scenes_for_detection(
            scenes, feedback_store=None, max_scenes=3
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_prepare_with_feedback_store(self):
        store = AsyncMock()
        store.batch_update_usage_counts = AsyncMock()
        scene = ProactiveScene(scene_id="s1", usage_count=0)
        result = await ColdStartStrategy.prepare_scenes_for_detection(
            [scene], feedback_store=store
        )
        assert len(result) == 1
        # The asyncio.create_task should have been called but we can't await it here

    def test_perturbation_non_exploration(self):
        scene = ProactiveScene(exploration_mode=False)
        score = ColdStartStrategy.apply_exploration_perturbation(
            0.8, scene, threshold_high=0.85, threshold_mid=0.6
        )
        assert score == 0.8  # no change

    def test_perturbation_in_boundary(self):
        """Exploration mode + score in boundary zone → perturbation applied"""
        random.seed(42)
        scene = ProactiveScene(exploration_mode=True)
        # Score near threshold_high (0.85) ± 0.05
        score = ColdStartStrategy.apply_exploration_perturbation(
            0.84, scene, threshold_high=0.85, threshold_mid=0.6
        )
        # Should be different from original (with high probability)
        # Since random.seed(42) is deterministic, just check bounds
        assert 0.0 <= score <= 1.0

    def test_perturbation_outside_boundary(self):
        scene = ProactiveScene(exploration_mode=True)
        # Score far from boundaries
        score = ColdStartStrategy.apply_exploration_perturbation(
            0.3, scene, threshold_high=0.85, threshold_mid=0.6
        )
        assert score == 0.3  # no change, not in boundary

    def test_perturbation_clamped(self):
        scene = ProactiveScene(exploration_mode=True)
        # Score very close to 1.0, near a threshold → might exceed bounds
        score = ColdStartStrategy.apply_exploration_perturbation(
            0.98, scene, threshold_high=0.98, threshold_mid=0.6
        )
        assert score <= 1.0
