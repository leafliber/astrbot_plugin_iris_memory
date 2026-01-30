"""
EmotionState测试
测试情感状态数据模型的核心功能
"""

import pytest
from datetime import datetime, timedelta
from iris_memory.models.emotion_state import (
    EmotionalState,
    CurrentEmotionState,
    EmotionalTrajectory,
    EmotionContext,
    EmotionConfig,
    TrendType
)
from iris_memory.core.types import EmotionType


class TestCurrentEmotionState:
    """测试CurrentEmotionState类"""

    def test_init_default(self):
        """测试默认初始化"""
        state = CurrentEmotionState()

        assert state.primary == EmotionType.NEUTRAL
        assert state.secondary == []
        assert state.intensity == 0.5
        assert state.confidence == 0.5
        assert state.contextual_correction is False
        assert isinstance(state.detected_at, datetime)

    def test_init_with_values(self):
        """测试带值的初始化"""
        detected_time = datetime(2024, 1, 1, 10, 0, 0)

        state = CurrentEmotionState(
            primary=EmotionType.JOY,
            secondary=[EmotionType.EXCITEMENT],
            intensity=0.8,
            confidence=0.9,
            detected_at=detected_time,
            contextual_correction=True
        )

        assert state.primary == EmotionType.JOY
        assert EmotionType.EXCITEMENT in state.secondary
        assert state.intensity == 0.8
        assert state.confidence == 0.9
        assert state.detected_at == detected_time
        assert state.contextual_correction is True


class TestEmotionContext:
    """测试EmotionContext类"""

    def test_init_default(self):
        """测试默认初始化"""
        context = EmotionContext()

        assert context.recent_topics == []
        assert context.active_session is None
        assert context.user_situation is None

    def test_init_with_values(self):
        """测试带值的初始化"""
        context = EmotionContext(
            recent_topics=["工作", "生活"],
            active_session="session_123",
            user_situation="在家工作"
        )

        assert len(context.recent_topics) == 2
        assert "工作" in context.recent_topics
        assert context.active_session == "session_123"
        assert context.user_situation == "在家工作"


class TestEmotionConfig:
    """测试EmotionConfig类"""

    def test_init_default(self):
        """测试默认初始化"""
        config = EmotionConfig()

        assert config.history_size == 100
        assert config.window_size == 7
        assert config.min_confidence == 0.3

    def test_init_custom(self):
        """测试自定义配置"""
        config = EmotionConfig(
            history_size=200,
            window_size=10,
            min_confidence=0.5
        )

        assert config.history_size == 200
        assert config.window_size == 10
        assert config.min_confidence == 0.5


class TestEmotionalTrajectory:
    """测试EmotionalTrajectory类"""

    def test_init_default(self):
        """测试默认初始化"""
        trajectory = EmotionalTrajectory()

        assert trajectory.trend == TrendType.STABLE
        assert trajectory.volatility == 0.5
        assert trajectory.anomaly_detected is False
        assert trajectory.needs_intervention is False
        assert trajectory.last_intervention is None

    def test_init_with_values(self):
        """测试带值的初始化"""
        intervention_time = datetime(2024, 1, 1, 10, 0, 0)

        trajectory = EmotionalTrajectory(
            trend=TrendType.IMPROVING,
            volatility=0.3,
            anomaly_detected=True,
            needs_intervention=True,
            last_intervention=intervention_time
        )

        assert trajectory.trend == TrendType.IMPROVING
        assert trajectory.volatility == 0.3
        assert trajectory.anomaly_detected is True
        assert trajectory.needs_intervention is True
        assert trajectory.last_intervention == intervention_time


class TestEmotionalState:
    """测试EmotionalState类"""

    def test_init_default(self):
        """测试默认初始化"""
        state = EmotionalState()

        assert state.current.primary == EmotionType.NEUTRAL
        assert len(state.history) == 0
        assert state.trajectory.trend == TrendType.STABLE
        assert len(state.patterns) == 0
        assert len(state.triggers) == 0
        assert len(state.soothers) == 0
        assert isinstance(state.context, EmotionContext)
        assert isinstance(state.config, EmotionConfig)

    def test_init_with_custom_config(self):
        """测试自定义配置初始化"""
        custom_config = EmotionConfig(history_size=50, window_size=5)

        state = EmotionalState(config=custom_config)

        assert state.config.history_size == 50
        assert state.config.window_size == 5

    def test_update_current_emotion_basic(self):
        """测试更新当前情感 - 基本功能"""
        state = EmotionalState()

        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )

        # 当前状态应该更新
        assert state.current.primary == EmotionType.JOY
        assert state.current.intensity == 0.8
        assert state.current.confidence == 0.9

        # 历史应该增加
        assert len(state.history) == 1

        # 模式应该更新
        assert state.patterns.get("joy", 0) == 1

    def test_update_current_emotion_with_secondary(self):
        """测试更新当前情感 - 带次要情感"""
        state = EmotionalState()

        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.7,
            confidence=0.8,
            secondary=[EmotionType.EXCITEMENT, EmotionType.CALM]
        )

        assert state.current.primary == EmotionType.JOY
        assert EmotionType.EXCITEMENT in state.current.secondary
        assert EmotionType.CALM in state.current.secondary
        assert len(state.current.secondary) == 2

    def test_update_current_emotion_multiple_times(self):
        """测试多次更新当前情感"""
        state = EmotionalState()

        # 第一次更新
        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )

        assert state.current.primary == EmotionType.JOY
        # 第一次更新后，history会有一个元素（初始的NEUTRAL）
        assert len(state.history) == 1
        # history[0]应该是初始的NEUTRAL
        assert state.history[0].primary == EmotionType.NEUTRAL

        # 第二次更新
        state.update_current_emotion(
            primary=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.8
        )

        assert state.current.primary == EmotionType.SADNESS
        assert len(state.history) == 2

        # 检查历史记录
        assert state.history[0].primary == EmotionType.NEUTRAL
        assert state.history[1].primary == EmotionType.JOY

    def test_update_current_emotion_history_limit(self):
        """测试历史记录限制"""
        state = EmotionalState()
        state.config.history_size = 3  # 设置较小的历史大小

        # 添加超过限制的历史
        for i in range(5):
            state.update_current_emotion(
                primary=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9
            )

        # 历史应该限制在3条
        assert len(state.history) == 3
        assert len(state.history) <= state.config.history_size

    def test_update_current_emotion_patterns(self):
        """测试情感模式统计"""
        state = EmotionalState()

        # 更新多次相同情感
        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )

        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.9,
            confidence=0.95
        )

        state.update_current_emotion(
            primary=EmotionType.SADNESS,
            intensity=0.7,
            confidence=0.8
        )

        # 检查模式统计
        assert state.patterns.get("joy", 0) == 2
        assert state.patterns.get("sadness", 0) == 1

    def test_analyze_trajectory_insufficient_history(self):
        """测试情感轨迹分析 - 历史不足"""
        state = EmotionalState()

        # 只添加少量历史
        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )

        # 应该直接返回，不会崩溃
        state._analyze_trajectory()

        assert state.trajectory.trend == TrendType.STABLE

    def test_analyze_trajectory_positive_trend(self):
        """测试情感轨迹分析 - 正向趋势"""
        state = EmotionalState()
        state.config.window_size = 5

        # 添加正面情感历史
        for _ in range(7):
            state.update_current_emotion(
                primary=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9
            )

        state._analyze_trajectory()

        # 应该识别到稳定或改善的趋势
        assert state.trajectory.trend in [TrendType.STABLE, TrendType.IMPROVING]

    def test_analyze_trajectory_negative_trend(self):
        """测试情感轨迹分析 - 负向趋势"""
        state = EmotionalState()
        state.config.window_size = 5

        # 先添加正面历史
        for _ in range(5):
            state.update_current_emotion(
                primary=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9
            )

        # 再添加负面历史
        for _ in range(5):
            state.update_current_emotion(
                primary=EmotionType.SADNESS,
                intensity=0.8,
                confidence=0.9
            )

        state._analyze_trajectory()

        # 应该识别到恶化或波动的趋势
        assert state.trajectory.trend in [TrendType.DETERIORATING, TrendType.VOLATILE]

    def test_analyze_trajectory_volatility(self):
        """测试情感波动性分析"""
        state = EmotionalState()
        state.config.window_size = 5

        # 添加波动的情感历史
        emotions = [
            EmotionType.JOY, EmotionType.SADNESS, EmotionType.JOY,
            EmotionType.ANGER, EmotionType.CALM
        ]

        for emotion in emotions:
            state.update_current_emotion(
                primary=emotion,
                intensity=0.8,
                confidence=0.9
            )

        state._analyze_trajectory()

        # 波动性应该被计算
        assert 0.0 <= state.trajectory.volatility <= 1.0

    def test_get_negative_ratio(self):
        """测试获取负面情感比例"""
        state = EmotionalState()

        # 添加混合情感历史
        state.update_current_emotion(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )
        state.update_current_emotion(
            primary=EmotionType.SADNESS,
            intensity=0.8,
            confidence=0.9
        )
        state.update_current_emotion(
            primary=EmotionType.ANGER,
            intensity=0.7,
            confidence=0.8
        )

        ratio = state.get_negative_ratio()

        # 应该计算负面比例
        assert 0.0 <= ratio <= 1.0
        # 3个中有2个负面,应该在0.6左右
        assert 0.5 <= ratio <= 0.7

    def test_get_negative_ratio_no_history(self):
        """测试无历史时的负面比例"""
        state = EmotionalState()

        ratio = state.get_negative_ratio()

        # 无历史应该返回0
        assert ratio == 0.0

    def test_should_filter_positive_negative(self):
        """测试负面情感时应该过滤正面记忆"""
        state = EmotionalState()
        state.current.primary = EmotionType.SADNESS
        state.current.intensity = 0.8

        result = state.should_filter_positive()

        assert result is True

    def test_should_filter_positive_neutral(self):
        """测试中性情感时不应该过滤"""
        state = EmotionalState()
        state.current.primary = EmotionType.NEUTRAL
        state.current.intensity = 0.5

        result = state.should_filter_positive()

        assert result is False

    def test_should_filter_positive_positive_low(self):
        """测试低强度正面情感时不应该过滤"""
        state = EmotionalState()
        state.current.primary = EmotionType.JOY
        state.current.intensity = 0.5

        result = state.should_filter_positive()

        assert result is False

    def test_should_filter_positive_positive_high(self):
        """测试高强度正面情感时不应该过滤"""
        state = EmotionalState()
        state.current.primary = EmotionType.JOY
        state.current.intensity = 0.9

        result = state.should_filter_positive()

        assert result is False

    def test_add_trigger(self):
        """测试添加情感触发器"""
        state = EmotionalState()

        state.add_trigger(
            trigger_type="topic",
            description="工作压力",
            emotion=EmotionType.ANXIETY
        )

        assert len(state.triggers) == 1
        assert state.triggers[0]["type"] == "topic"
        assert state.triggers[0]["description"] == "工作压力"

    def test_add_soothe(self):
        """测试添加缓解因素"""
        state = EmotionalState()

        state.add_soothe(
            soothe_type="music",
            description="古典音乐",
            emotion=EmotionType.CALM
        )

        assert len(state.soothers) == 1
        assert state.soothers[0]["type"] == "music"
        assert state.soothers[0]["description"] == "古典音乐"

    def test_update_context(self):
        """测试更新上下文"""
        state = EmotionalState()

        # 直接设置上下文属性
        state.context.recent_topics = ["工作", "生活"]
        state.context.active_session = "session_123"
        state.context.user_situation = "在家工作"

        assert len(state.context.recent_topics) == 2
        assert state.context.active_session == "session_123"
        assert state.context.user_situation == "在家工作"


class TestEmotionStateIntegration:
    """测试集成场景"""

    def test_full_emotion_tracking_workflow(self):
        """测试完整情感跟踪工作流"""
        state = EmotionalState()

        # 模拟一天的情感变化
        emotions_sequence = [
            (EmotionType.JOY, 0.8, 0.9),      # 早上
            (EmotionType.ANXIETY, 0.7, 0.8),  # 上班时
            (EmotionType.ANGER, 0.6, 0.7),    # 遇到问题
            (EmotionType.CALM, 0.7, 0.8),     # 解决后
            (EmotionType.JOY, 0.9, 0.95)      # 下班
        ]

        for primary, intensity, confidence in emotions_sequence:
            state.update_current_emotion(
                primary=primary,
                intensity=intensity,
                confidence=confidence
            )

        # 验证历史记录
        assert len(state.history) == 5

        # 验证模式统计
        assert state.patterns.get("joy", 0) == 2
        assert state.patterns.get("anxiety", 0) == 1
        assert state.patterns.get("anger", 0) == 1
        assert state.patterns.get("calm", 0) == 1

        # 验证当前状态
        assert state.current.primary == EmotionType.JOY
        assert state.current.intensity == 0.9

        # 验证轨迹分析
        state._analyze_trajectory()
        assert 0.0 <= state.trajectory.volatility <= 1.0

    def test_emotion_state_with_context(self):
        """测试带上下文的情感状态"""
        state = EmotionalState()

        # 直接设置上下文
        state.context.recent_topics = ["工作项目", "截止日期"]
        state.context.active_session = "work_session"
        state.context.user_situation = "紧张工作"

        # 更新情感
        state.update_current_emotion(
            primary=EmotionType.ANXIETY,
            intensity=0.8,
            confidence=0.9
        )

        # 验证状态
        assert state.current.primary == EmotionType.ANXIETY
        assert "工作项目" in state.context.recent_topics
        assert state.context.user_situation == "紧张工作"

        # 添加相关触发器
        state.add_trigger(
            trigger_type="topic",
            description="截止日期",
            emotion=EmotionType.ANXIETY
        )

        assert len(state.triggers) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
