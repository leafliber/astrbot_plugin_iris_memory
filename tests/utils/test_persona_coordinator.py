"""
测试人格协调器
测试 iris_memory.analysis.persona.coordinator 中的 PersonaConflictDetector 和 PersonaCoordinator 类
"""

import pytest

from iris_memory.analysis.persona.coordinator import (
    ConflictType,
    CoordinationStrategy,
    PersonaConflictDetector,
    PersonaCoordinator
)


class TestConflictType:
    """测试ConflictType枚举"""

    def test_conflict_type_values(self):
        """测试冲突类型的值"""
        assert ConflictType.STYLE_CONFLICT.value == "style_conflict"
        assert ConflictType.FREQUENCY_CONFLICT.value == "frequency_conflict"
        assert ConflictType.EMOTION_CONFLICT.value == "emotion_conflict"
        assert ConflictType.CONTENT_CONFLICT.value == "content_conflict"

    def test_conflict_type_count(self):
        """测试冲突类型的数量"""
        assert len(ConflictType) == 4


class TestCoordinationStrategy:
    """测试CoordinationStrategy枚举"""

    def test_coordination_strategy_values(self):
        """测试协调策略的值"""
        assert CoordinationStrategy.BOT_PRIORITY.value == "bot_priority"
        assert CoordinationStrategy.USER_PRIORITY.value == "user_priority"
        assert CoordinationStrategy.HYBRID.value == "hybrid"
        assert CoordinationStrategy.DYNAMIC.value == "dynamic"

    def test_coordination_strategy_count(self):
        """测试协调策略的数量"""
        assert len(CoordinationStrategy) == 4


class TestPersonaConflictDetectorInit:
    """测试PersonaConflictDetector初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        detector = PersonaConflictDetector()

        assert "friendly" in detector.bot_persona_keywords
        assert "professional" in detector.bot_persona_keywords
        assert "less_talkative" in detector.user_preference_keywords


class TestDetectConflicts:
    """测试detect_conflicts方法"""

    def test_detect_no_conflicts(self):
        """测试无冲突情况"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "friendly"
            }
        }

        conflicts = detector.detect_conflicts(user_persona, "friendly")

        assert conflicts == []

    def test_detect_style_conflict(self):
        """测试检测风格冲突"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }

        conflicts = detector.detect_conflicts(user_persona, "friendly")

        # 应该检测到风格冲突
        assert len(conflicts) > 0
        assert any(c["type"] == ConflictType.STYLE_CONFLICT for c in conflicts)

    def test_detect_emotion_conflict(self):
        """测试检测情感冲突"""
        detector = PersonaConflictDetector()

        user_persona = {
            "emotional": {
                "trajectory": "deteriorating"
            }
        }

        conflicts = detector.detect_conflicts(user_persona, "friendly")

        # 应该检测到情感冲突
        assert len(conflicts) > 0
        assert any(c["type"] == ConflictType.EMOTION_CONFLICT for c in conflicts)

    def test_detect_multiple_conflicts(self):
        """测试检测多个冲突"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            },
            "emotional": {
                "trajectory": "deteriorating"
            }
        }

        conflicts = detector.detect_conflicts(user_persona, "friendly")

        # 应该检测到多个冲突
        assert len(conflicts) >= 2

        conflict_types = [c["type"] for c in conflicts]
        assert ConflictType.STYLE_CONFLICT in conflict_types
        assert ConflictType.EMOTION_CONFLICT in conflict_types


class TestDetectStyleConflicts:
    """测试_detect_style_conflicts方法"""

    def test_detect_negative_preference(self):
        """测试检测负面偏好"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }

        conflicts = detector._detect_style_conflicts(user_persona, "friendly")

        assert len(conflicts) > 0
        assert conflicts[0]["type"] == ConflictType.STYLE_CONFLICT
        assert conflicts[0]["severity"] == 2

    def test_detect_no_negative_preference(self):
        """测试无负面偏好"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "friendly"
            }
        }

        conflicts = detector._detect_style_conflicts(user_persona, "friendly")

        assert len(conflicts) == 0

    def test_detect_multiple_preferences(self):
        """测试检测多个偏好"""
        detector = PersonaConflictDetector()

        user_persona = {
            "preferences": {
                "style": "less_talkative",
                "enthusiasm": "not_enthusiastic"
            }
        }

        conflicts = detector._detect_style_conflicts(user_persona, "friendly")

        # 应该检测到多个冲突
        assert len(conflicts) >= 1


class TestDetectEmotionConflicts:
    """测试_detect_emotion_conflicts方法"""

    def test_detect_deteriorating_trajectory(self):
        """测试检测恶化情感轨迹"""
        detector = PersonaConflictDetector()

        user_persona = {
            "emotional": {
                "trajectory": "deteriorating"
            }
        }

        conflicts = detector._detect_emotion_conflicts(user_persona)

        assert len(conflicts) > 0
        assert conflicts[0]["type"] == ConflictType.EMOTION_CONFLICT
        assert conflicts[0]["trajectory"] == "deteriorating"

    def test_detect_volatile_trajectory(self):
        """测试检测波动情感轨迹"""
        detector = PersonaConflictDetector()

        user_persona = {
            "emotional": {
                "trajectory": "volatile"
            }
        }

        conflicts = detector._detect_emotion_conflicts(user_persona)

        assert len(conflicts) > 0
        assert conflicts[0]["type"] == ConflictType.EMOTION_CONFLICT
        assert conflicts[0]["trajectory"] == "volatile"

    def test_detect_stable_trajectory(self):
        """测试稳定情感轨迹（无冲突）"""
        detector = PersonaConflictDetector()

        user_persona = {
            "emotional": {
                "trajectory": "stable"
            }
        }

        conflicts = detector._detect_emotion_conflicts(user_persona)

        assert len(conflicts) == 0

    def test_detect_no_emotional_data(self):
        """测试无情感数据"""
        detector = PersonaConflictDetector()

        user_persona = {}

        conflicts = detector._detect_emotion_conflicts(user_persona)

        assert len(conflicts) == 0


class TestGetResolutionSuggestions:
    """测试get_resolution_suggestions方法"""

    def test_get_suggestions_empty(self):
        """测试无冲突时的建议"""
        detector = PersonaConflictDetector()

        suggestions = detector.get_resolution_suggestions([])

        assert suggestions == []

    def test_get_suggestions_style_conflict(self):
        """测试风格冲突建议"""
        detector = PersonaConflictDetector()

        conflicts = [
            {
                "type": ConflictType.STYLE_CONFLICT,
                "description": "用户偏好'less_talkative'可能与Bot风格冲突",
                "bot_style": "friendly"
            }
        ]

        suggestions = detector.get_resolution_suggestions(conflicts)

        assert len(suggestions) > 0
        assert any("专业" in s for s in suggestions)

    def test_get_suggestions_emotion_conflict(self):
        """测试情感冲突建议"""
        detector = PersonaConflictDetector()

        conflicts = [
            {
                "type": ConflictType.EMOTION_CONFLICT,
                "description": "用户情感持续负面，建议调整回复风格"
            }
        ]

        suggestions = detector.get_resolution_suggestions(conflicts)

        assert len(suggestions) > 0
        assert any("倾听" in s or "支持" in s for s in suggestions)

    def test_get_suggestions_deduplicated(self):
        """测试建议去重"""
        detector = PersonaConflictDetector()

        conflicts = [
            {
                "type": ConflictType.EMOTION_CONFLICT,
                "description": "用户情感持续负面，建议调整回复风格"
            },
            {
                "type": ConflictType.EMOTION_CONFLICT,
                "description": "用户情感持续负面，建议调整回复风格"
            }
        ]

        suggestions = detector.get_resolution_suggestions(conflicts)

        # 建议应该去重
        assert len(set(suggestions)) == len(suggestions)


class TestPersonaCoordinatorInit:
    """测试PersonaCoordinator初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        coordinator = PersonaCoordinator()

        assert coordinator.strategy == CoordinationStrategy.HYBRID
        assert coordinator.conflict_detector is not None

    def test_init_custom_strategy(self):
        """测试自定义策略"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.BOT_PRIORITY)

        assert coordinator.strategy == CoordinationStrategy.BOT_PRIORITY


class TestCoordinatePersona:
    """测试coordinate_persona方法"""

    def test_coordinate_bot_priority(self):
        """测试Bot优先策略"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.BOT_PRIORITY)

        user_persona = {}
        prompt = coordinator.coordinate_persona(user_persona, "friendly", "")

        assert "友好" in prompt or "热情" in prompt

    def test_coordinate_user_priority(self):
        """测试用户优先策略"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.USER_PRIORITY)

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }
        prompt = coordinator.coordinate_persona(user_persona, "friendly", "")

        assert "简洁" in prompt

    def test_coordinate_hybrid(self):
        """测试混合策略"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.HYBRID)

        user_persona = {}
        prompt = coordinator.coordinate_persona(user_persona, "friendly", "")

        assert isinstance(prompt, str)

    def test_coordinate_dynamic(self):
        """测试动态策略"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.DYNAMIC)

        user_persona = {}
        prompt = coordinator.coordinate_persona(user_persona, "friendly", "")

        assert isinstance(prompt, str)

    def test_coordinate_with_memory_context(self):
        """测试带记忆上下文的协调"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.HYBRID)

        user_persona = {}
        memory_context = "【相关记忆】用户喜欢简洁"
        prompt = coordinator.coordinate_persona(user_persona, "friendly", memory_context)

        assert isinstance(prompt, str)


class TestBotPriorityPrompt:
    """测试_bot_priority_prompt方法"""

    def test_friendly_bot(self):
        """测试友好型Bot"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._bot_priority_prompt("friendly", "")

        assert "友好" in prompt or "热情" in prompt

    def test_professional_bot(self):
        """测试专业型Bot"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._bot_priority_prompt("professional", "")

        assert "专业" in prompt or "客观" in prompt

    def test_humorous_bot(self):
        """测试幽默型Bot"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._bot_priority_prompt("humorous", "")

        assert "幽默" in prompt or "风趣" in prompt

    def test_calm_bot(self):
        """测试冷静型Bot"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._bot_priority_prompt("calm", "")

        assert "平和" in prompt or "冷静" in prompt or "理性" in prompt

    def test_with_memory_context(self):
        """测试带记忆上下文"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._bot_priority_prompt("friendly", "记忆上下文")

        assert "用户画像" in prompt


class TestUserPriorityPrompt:
    """测试_user_priority_prompt方法"""

    def test_less_talkative_preference(self):
        """测试简洁偏好"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }
        prompt = coordinator._user_priority_prompt(user_persona, "", [])

        assert "简洁" in prompt

    def test_deteriorating_emotion(self):
        """测试恶化情感"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "emotional": {
                "trajectory": "deteriorating"
            }
        }
        prompt = coordinator._user_priority_prompt(user_persona, "", [])

        assert "支持" in prompt or "理解" in prompt

    def test_volatile_emotion(self):
        """测试波动情感"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "emotional": {
                "trajectory": "volatile"
            }
        }
        prompt = coordinator._user_priority_prompt(user_persona, "", [])

        assert "稳定" in prompt

    def test_with_conflicts(self):
        """测试带冲突的建议"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }
        conflicts = [
            {
                "type": ConflictType.STYLE_CONFLICT,
                "description": "测试冲突",
                "bot_style": "friendly"
            }
        ]
        prompt = coordinator._user_priority_prompt(user_persona, "", conflicts)

        assert "注意事项" in prompt


class TestHybridPrompt:
    """测试_hybrid_prompt方法"""

    def test_hybrid_basic(self):
        """测试基本混合策略"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._hybrid_prompt("friendly", {}, "", [])

        assert "友好" in prompt or "热情" in prompt
        assert "用户画像" in prompt

    def test_hybrid_with_preferences(self):
        """测试带用户偏好的混合策略"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "preferences": {
                "style": "formal"
            }
        }
        prompt = coordinator._hybrid_prompt("friendly", user_persona, "", [])

        assert "formal" in prompt or "正式" in prompt

    def test_hybrid_with_emotional_state(self):
        """测试带情感状态的混合策略"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "emotional": {
                "trajectory": "deteriorating"
            }
        }
        prompt = coordinator._hybrid_prompt("friendly", user_persona, "", [])

        assert "温和" in prompt or "支持" in prompt


class TestDynamicPrompt:
    """测试_dynamic_prompt方法"""

    def test_dynamic_no_high_severity(self):
        """测试无高严重性冲突的动态策略"""
        coordinator = PersonaCoordinator()

        prompt = coordinator._dynamic_prompt("friendly", {}, "", [])

        assert isinstance(prompt, str)

    def test_dynamic_with_high_severity(self):
        """测试高严重性冲突的动态策略"""
        coordinator = PersonaCoordinator()

        conflicts = [
            {
                "type": ConflictType.STYLE_CONFLICT,
                "severity": 2,
                "description": "测试冲突描述",
                "bot_style": "friendly"
            }
        ]
        prompt = coordinator._dynamic_prompt("friendly", {}, "", conflicts)

        # 高严重性应该用户优先
        assert isinstance(prompt, str)

    def test_dynamic_with_deteriorating_emotion(self):
        """测试恶化情感的动态策略"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "emotional": {
                "trajectory": "deteriorating"
            }
        }
        prompt = coordinator._dynamic_prompt("friendly", user_persona, "", [])

        # 恶化情感应该用户优先
        assert isinstance(prompt, str)


class TestFormatContextWithPersona:
    """测试format_context_with_persona方法"""

    def test_format_basic(self):
        """测试基本格式化"""
        coordinator = PersonaCoordinator()

        memory_context = "测试记忆上下文"
        user_persona = {}
        result = coordinator.format_context_with_persona(memory_context, user_persona, "friendly")

        assert "测试记忆上下文" in result

    def test_format_with_conflicts(self):
        """测试带冲突的格式化"""
        coordinator = PersonaCoordinator()

        user_persona = {
            "preferences": {
                "style": "less_talkative"
            }
        }
        result = coordinator.format_context_with_persona("测试上下文", user_persona, "friendly")

        # 应该包含人格协调提示
        assert "【人格协调提示】" in result


class TestSetStrategy:
    """测试set_strategy方法"""

    def test_set_strategy(self):
        """测试设置策略"""
        coordinator = PersonaCoordinator()

        coordinator.set_strategy(CoordinationStrategy.BOT_PRIORITY)

        assert coordinator.strategy == CoordinationStrategy.BOT_PRIORITY

    def test_set_strategy_multiple_times(self):
        """测试多次设置策略"""
        coordinator = PersonaCoordinator()

        coordinator.set_strategy(CoordinationStrategy.BOT_PRIORITY)
        assert coordinator.strategy == CoordinationStrategy.BOT_PRIORITY

        coordinator.set_strategy(CoordinationStrategy.USER_PRIORITY)
        assert coordinator.strategy == CoordinationStrategy.USER_PRIORITY

        coordinator.set_strategy(CoordinationStrategy.HYBRID)
        assert coordinator.strategy == CoordinationStrategy.HYBRID


class TestIntegration:
    """测试集成场景"""

    def test_full_coordination_workflow(self):
        """测试完整协调工作流"""
        coordinator = PersonaCoordinator(strategy=CoordinationStrategy.HYBRID)

        # 用户画像：偏好简洁，情感状态不佳
        user_persona = {
            "preferences": {
                "style": "less_talkative"
            },
            "emotional": {
                "trajectory": "deteriorating"
            }
        }

        memory_context = "【相关记忆】用户喜欢简洁"

        # 协调人格
        prompt = coordinator.format_context_with_persona(
            memory_context,
            user_persona,
            "friendly"
        )

        # 验证结果
        assert "测试记忆上下文" in prompt or "【相关记忆】" in prompt
        assert "【人格协调提示】" in prompt

    def test_conflict_detection_and_resolution(self):
        """测试冲突检测和解决"""
        detector = PersonaConflictDetector()

        # 用户画像：与Bot风格冲突
        user_persona = {
            "preferences": {
                "style": "less_talkative"
            },
            "emotional": {
                "trajectory": "deteriorating"
            }
        }

        # 检测冲突
        conflicts = detector.detect_conflicts(user_persona, "friendly")

        # 获取建议
        suggestions = detector.get_resolution_suggestions(conflicts)

        # 验证
        assert len(conflicts) >= 1
        assert len(suggestions) >= 1

    def test_multiple_bot_personas(self):
        """测试多个Bot人格"""
        coordinator = PersonaCoordinator()

        bot_personas = ["friendly", "professional", "humorous", "calm"]
        user_persona = {}

        for bot_persona in bot_personas:
            prompt = coordinator._bot_priority_prompt(bot_persona, "")
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_all_strategies(self):
        """测试所有策略"""
        user_persona = {}
        memory_context = "记忆上下文"

        strategies = [
            CoordinationStrategy.BOT_PRIORITY,
            CoordinationStrategy.USER_PRIORITY,
            CoordinationStrategy.HYBRID,
            CoordinationStrategy.DYNAMIC
        ]

        for strategy in strategies:
            coordinator = PersonaCoordinator(strategy=strategy)
            prompt = coordinator.coordinate_persona(
                user_persona,
                "friendly",
                memory_context
            )

            assert isinstance(prompt, str)
            assert len(prompt) > 0
