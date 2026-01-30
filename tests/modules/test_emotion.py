"""
Emotion模型单元测试
测试EmotionalState数据模型
"""

import pytest
from datetime import datetime
from iris_memory.models.emotion_state import EmotionalState, CurrentEmotionState
from iris_memory.core.types import EmotionType


class TestCurrentEmotionStateInit:
    """测试CurrentEmotionState初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        current_emotion = CurrentEmotionState(
            primary=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9
        )

        assert current_emotion.primary == EmotionType.JOY
        assert current_emotion.intensity == 0.8
        assert current_emotion.confidence == 0.9

    def test_init_default_values(self):
        """测试默认值"""
        current_emotion = CurrentEmotionState(primary=EmotionType.SADNESS)

        assert current_emotion.primary == EmotionType.SADNESS
        assert current_emotion.intensity == 0.5
        assert current_emotion.confidence == 0.5


class TestEmotionalStateInit:
    """测试EmotionalState初始化"""

    def test_init_basic(self):
        """测试基本初始化"""
        state = EmotionalState()

        assert state is not None
        assert hasattr(state, 'current')
        assert hasattr(state, 'history')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
