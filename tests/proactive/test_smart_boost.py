"""
主动回复智能增强测试

测试内容：
1. 配置依赖：proactive_mode = rule 时 smart_boost 不生效
2. 窗口测试：发言后窗口内增强生效，窗口外失效
3. 线性衰减：乘数随时间线性衰减
4. 规则检测增强：分数乘数正确应用，翻转决策
5. 已回复决策的延迟缩短
6. 边界条件：空 reply_score、confidence 回退
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch, PropertyMock

import pytest
import pytest_asyncio

from iris_memory.proactive.proactive_manager import ProactiveReplyManager
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDecision,
    ReplyUrgency,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def smart_boost_config():
    """智能增强启用的配置"""
    return {
        "enable_proactive_reply": True,
        "reply_cooldown": 60,
        "max_daily_replies": 20,
        "smart_boost_enabled": True,
        "smart_boost_window_seconds": 300,
        "smart_boost_score_multiplier": 1.5,
        "smart_boost_reply_threshold": 0.25,
    }


@pytest.fixture
def smart_boost_disabled_config():
    """智能增强关闭的配置"""
    return {
        "enable_proactive_reply": True,
        "reply_cooldown": 60,
        "max_daily_replies": 20,
        "smart_boost_enabled": False,
    }


@pytest.fixture
def decision_no_reply_with_score():
    """不回复但有分数的决策（接近阈值）"""
    return ProactiveReplyDecision(
        should_reply=False,
        urgency=ReplyUrgency.LOW,
        reason="low_score",
        suggested_delay=0,
        reply_context={"reply_score": 0.2, "signals": {}},
    )


@pytest.fixture
def decision_no_reply_very_low():
    """不回复且分数极低的决策"""
    return ProactiveReplyDecision(
        should_reply=False,
        urgency=ReplyUrgency.IGNORE,
        reason="low_score",
        suggested_delay=0,
        reply_context={"reply_score": 0.05, "signals": {}},
    )


@pytest.fixture
def decision_should_reply():
    """应该回复的决策"""
    return ProactiveReplyDecision(
        should_reply=True,
        urgency=ReplyUrgency.MEDIUM,
        reason="question(0.80)",
        suggested_delay=30,
        reply_context={"reply_score": 0.5, "signals": {}},
    )


@pytest.fixture
def decision_no_reply_no_score():
    """不回复且无分数的决策"""
    return ProactiveReplyDecision(
        should_reply=False,
        urgency=ReplyUrgency.IGNORE,
        reason="empty",
        suggested_delay=0,
        reply_context={},
    )


@pytest.fixture
def mock_detector_factory():
    """检测器工厂，返回指定的决策"""
    def _factory(decision):
        detector = Mock()
        detector.analyze = AsyncMock(return_value=decision)
        return detector
    return _factory


@pytest.fixture
def mock_context_with_queue():
    context = Mock()
    context._event_queue = asyncio.Queue()
    return context


@pytest.fixture
def mock_config_manager_llm():
    """proactive_mode = llm 的 ConfigManager mock"""
    mgr = Mock()
    type(mgr).proactive_mode = PropertyMock(return_value="llm")
    type(mgr).smart_boost_enabled = PropertyMock(return_value=True)
    type(mgr).smart_boost_window_seconds = PropertyMock(return_value=300)
    type(mgr).smart_boost_score_multiplier = PropertyMock(return_value=1.5)
    type(mgr).smart_boost_reply_threshold = PropertyMock(return_value=0.25)
    mgr.get_max_daily_replies = Mock(return_value=20)
    mgr.get_cooldown_seconds = Mock(return_value=60)
    return mgr


@pytest.fixture
def mock_config_manager_hybrid():
    """proactive_mode = hybrid 的 ConfigManager mock"""
    mgr = Mock()
    type(mgr).proactive_mode = PropertyMock(return_value="hybrid")
    type(mgr).smart_boost_enabled = PropertyMock(return_value=True)
    type(mgr).smart_boost_window_seconds = PropertyMock(return_value=300)
    type(mgr).smart_boost_score_multiplier = PropertyMock(return_value=1.5)
    type(mgr).smart_boost_reply_threshold = PropertyMock(return_value=0.25)
    mgr.get_max_daily_replies = Mock(return_value=20)
    mgr.get_cooldown_seconds = Mock(return_value=60)
    return mgr


@pytest.fixture
def mock_config_manager_rule():
    """proactive_mode = rule 的 ConfigManager mock"""
    mgr = Mock()
    type(mgr).proactive_mode = PropertyMock(return_value="rule")
    type(mgr).smart_boost_enabled = PropertyMock(return_value=True)
    type(mgr).smart_boost_window_seconds = PropertyMock(return_value=300)
    type(mgr).smart_boost_score_multiplier = PropertyMock(return_value=1.5)
    type(mgr).smart_boost_reply_threshold = PropertyMock(return_value=0.25)
    mgr.get_max_daily_replies = Mock(return_value=20)
    mgr.get_cooldown_seconds = Mock(return_value=60)
    return mgr


@pytest.fixture
def mock_config_manager_disabled():
    """smart_boost_enabled = False 的 ConfigManager mock"""
    mgr = Mock()
    type(mgr).proactive_mode = PropertyMock(return_value="llm")
    type(mgr).smart_boost_enabled = PropertyMock(return_value=False)
    type(mgr).smart_boost_window_seconds = PropertyMock(return_value=300)
    type(mgr).smart_boost_score_multiplier = PropertyMock(return_value=1.5)
    type(mgr).smart_boost_reply_threshold = PropertyMock(return_value=0.25)
    mgr.get_max_daily_replies = Mock(return_value=20)
    mgr.get_cooldown_seconds = Mock(return_value=60)
    return mgr


def _make_manager(
    config,
    detector_decision,
    context_with_queue,
    config_manager=None,
):
    """创建 ProactiveReplyManager 实例的辅助函数"""
    detector = Mock()
    detector.analyze = AsyncMock(return_value=detector_decision)
    return ProactiveReplyManager(
        astrbot_context=context_with_queue,
        reply_detector=detector,
        config=config,
        config_manager=config_manager,
    )


# =============================================================================
# 1. 配置依赖测试
# =============================================================================

class TestConfigDependency:
    """验证 smart_boost 在不同 proactive_mode 下的行为"""

    def test_smart_boost_disabled_when_config_off(
        self, smart_boost_disabled_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_disabled,
    ):
        """smart_boost_enabled=False 时 is_in_boost_window 返回 False"""
        manager = _make_manager(
            smart_boost_disabled_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_disabled,
        )
        manager._record_user_message("u1")
        assert manager.is_in_boost_window("u1") is False

    def test_smart_boost_disabled_when_mode_rule(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_rule,
    ):
        """proactive_mode=rule 时 is_in_boost_window 返回 False"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_rule,
        )
        manager._record_user_message("u1")
        assert manager.is_in_boost_window("u1") is False

    def test_smart_boost_enabled_when_mode_llm(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """proactive_mode=llm 时 smart_boost 正常工作"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")
        assert manager.is_in_boost_window("u1") is True

    def test_smart_boost_enabled_when_mode_hybrid(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_hybrid,
    ):
        """proactive_mode=hybrid 时 smart_boost 正常工作"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_hybrid,
        )
        manager._record_user_message("u1")
        assert manager.is_in_boost_window("u1") is True


# =============================================================================
# 2. 窗口测试
# =============================================================================

class TestBoostWindow:
    """验证时间窗口行为"""

    def test_in_window_immediately(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """刚发言时应在窗口内"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")
        assert manager.is_in_boost_window("u1") is True

    def test_not_in_window_after_expiry(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """窗口过期后应不在窗口内"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        # 模拟发言时间为 301 秒前（窗口 300 秒）
        key = "u1:private"
        manager._last_user_message_time[key] = time.time() - 301
        assert manager.is_in_boost_window("u1") is False

    def test_not_in_window_no_message(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """未发言过的用户不在窗口内"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        assert manager.is_in_boost_window("unknown_user") is False

    def test_group_session_isolation(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """私聊和群聊的窗口相互隔离"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1", group_id=None)
        manager._record_user_message("u1", group_id="g1")

        assert manager.is_in_boost_window("u1") is True
        assert manager.is_in_boost_window("u1", group_id="g1") is True
        # 不同群聊未记录
        assert manager.is_in_boost_window("u1", group_id="g2") is False


# =============================================================================
# 3. 线性衰减测试
# =============================================================================

class TestLinearDecay:
    """验证乘数随时间线性衰减"""

    def test_multiplier_at_start(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """刚发言时乘数应接近最大值"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")
        m = manager.get_boost_multiplier("u1")
        # 乘数应接近 1.5（允许小误差因为 time.time() 有少量经过）
        assert 1.4 <= m <= 1.5

    def test_multiplier_at_half_window(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """窗口过半时乘数应约为 1.25"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        key = "u1:private"
        manager._last_user_message_time[key] = time.time() - 150  # 300秒窗口的一半
        m = manager.get_boost_multiplier("u1")
        # 应约为 1 + (1.5 - 1) * 0.5 = 1.25
        assert 1.2 <= m <= 1.3

    def test_multiplier_near_end(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """窗口即将结束时乘数应接近 1.0"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        key = "u1:private"
        manager._last_user_message_time[key] = time.time() - 295  # 距窗口结束5秒
        m = manager.get_boost_multiplier("u1")
        assert 1.0 <= m <= 1.05

    def test_multiplier_outside_window(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """窗口外乘数为 1.0"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        key = "u1:private"
        manager._last_user_message_time[key] = time.time() - 301
        m = manager.get_boost_multiplier("u1")
        assert m == 1.0

    def test_multiplier_no_record(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """未记录的用户乘数为 1.0"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        m = manager.get_boost_multiplier("unknown")
        assert m == 1.0


# =============================================================================
# 4. 决策增强测试
# =============================================================================

class TestApplySmartBoost:
    """验证 _apply_smart_boost 对决策的影响"""

    def test_boost_flips_no_reply_to_reply(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """分数 0.2 × 1.5 = 0.3 >= 0.25 应翻转为回复"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")

        result = manager._apply_smart_boost(
            decision_no_reply_with_score, "u1"
        )
        assert result.should_reply is True
        assert "smart_boost" in result.reason
        assert result.reply_context["smart_boost_applied"] is True
        assert result.reply_context["reply_score"] > 0.2
        assert manager.stats["smart_boost_activations"] == 1

    def test_boost_does_not_flip_very_low_score(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_very_low, mock_config_manager_llm,
    ):
        """分数 0.05 × 1.5 = 0.075 < 0.25 不应翻转"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_very_low,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")

        result = manager._apply_smart_boost(
            decision_no_reply_very_low, "u1"
        )
        assert result.should_reply is False
        assert manager.stats["smart_boost_activations"] == 0

    def test_boost_reduces_delay_for_existing_reply(
        self, smart_boost_config, mock_context_with_queue,
        decision_should_reply, mock_config_manager_llm,
    ):
        """已回复的决策应缩短延迟"""
        manager = _make_manager(
            smart_boost_config,
            decision_should_reply,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")

        original_delay = decision_should_reply.suggested_delay
        result = manager._apply_smart_boost(
            decision_should_reply, "u1"
        )
        assert result.should_reply is True
        assert result.suggested_delay < original_delay
        assert manager.stats["smart_boost_delay_reductions"] == 1

    def test_boost_no_effect_outside_window(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """窗口外不应有增强效果"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        # 不记录发言 → 不在窗口内
        result = manager._apply_smart_boost(
            decision_no_reply_with_score, "u1"
        )
        assert result.should_reply is False

    def test_boost_no_effect_when_no_score(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_no_score, mock_config_manager_llm,
    ):
        """无分数时不应翻转"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_no_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")

        result = manager._apply_smart_boost(
            decision_no_reply_no_score, "u1"
        )
        assert result.should_reply is False

    def test_boost_uses_confidence_fallback(
        self, smart_boost_config, mock_context_with_queue,
        mock_config_manager_llm,
    ):
        """reply_score 缺失时回退到 confidence 字段"""
        # 模拟 LLMReplyDecision（有 confidence，无 reply_score）
        decision = Mock()
        decision.should_reply = False
        decision.urgency = ReplyUrgency.IGNORE
        decision.reason = "LLM判断"
        decision.suggested_delay = 0
        decision.reply_context = {}
        decision.confidence = 0.6  # 0.6 * 0.5 = 0.3; 0.3 * 1.5 = 0.45 >= 0.25

        manager = _make_manager(
            smart_boost_config,
            decision,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")

        result = manager._apply_smart_boost(decision, "u1")
        assert result.should_reply is True
        assert "smart_boost" in result.reason

    def test_boost_urgency_levels(
        self, smart_boost_config, mock_context_with_queue,
        mock_config_manager_llm,
    ):
        """增强后分数不同应对应不同紧急度"""
        # 分数 0.4 × 1.5 = 0.6 >= 0.5 → MEDIUM
        decision_high = ProactiveReplyDecision(
            should_reply=False,
            urgency=ReplyUrgency.LOW,
            reason="low_score",
            suggested_delay=0,
            reply_context={"reply_score": 0.4},
        )
        manager = _make_manager(
            smart_boost_config,
            decision_high,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        manager._record_user_message("u1")
        result = manager._apply_smart_boost(decision_high, "u1")
        assert result.urgency == ReplyUrgency.MEDIUM

        # 分数 0.2 × 1.5 = 0.3 < 0.5 → LOW
        decision_low = ProactiveReplyDecision(
            should_reply=False,
            urgency=ReplyUrgency.IGNORE,
            reason="low_score",
            suggested_delay=0,
            reply_context={"reply_score": 0.2},
        )
        result2 = manager._apply_smart_boost(decision_low, "u1")
        assert result2.urgency == ReplyUrgency.LOW


# =============================================================================
# 5. handle_batch 集成测试
# =============================================================================

class TestHandleBatchWithSmartBoost:
    """验证 handle_batch 中智能增强的完整流程"""

    @pytest.mark.asyncio
    async def test_handle_batch_does_not_record_message_time(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """handle_batch 不应记录用户发言时间
        
        智能增强窗口只在 Bot 发送主动回复后开始，
        避免持续聊天导致窗口无限延长。
        """
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        await manager.initialize()
        try:
            await manager.handle_batch(
                messages=["你好"], user_id="u1"
            )
            assert "u1:private" not in manager._last_user_message_time
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_handle_batch_boost_creates_task(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_llm,
    ):
        """增强后翻转为回复应创建任务"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_llm,
        )
        await manager.initialize()
        try:
            # 先记录一次发言（模拟之前的消息）
            manager._record_user_message("u1")
            await manager.handle_batch(
                messages=["test"], user_id="u1"
            )
            # 0.2 * ~1.5 = ~0.3 >= 0.25 应创建任务
            assert manager.pending_tasks.qsize() == 1
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_handle_batch_no_boost_when_mode_rule(
        self, smart_boost_config, mock_context_with_queue,
        decision_no_reply_with_score, mock_config_manager_rule,
    ):
        """proactive_mode=rule 时不应增强"""
        manager = _make_manager(
            smart_boost_config,
            decision_no_reply_with_score,
            mock_context_with_queue,
            mock_config_manager_rule,
        )
        await manager.initialize()
        try:
            manager._record_user_message("u1")
            await manager.handle_batch(
                messages=["test"], user_id="u1"
            )
            # 不应翻转 → 无任务
            assert manager.pending_tasks.qsize() == 0
            assert manager.stats["replies_skipped"] == 1
        finally:
            await manager.stop()


# =============================================================================
# 6. ConfigManager 集成测试
# =============================================================================

class TestConfigManagerSmartBoost:
    """验证 ConfigManager 的智能增强属性"""

    def test_smart_boost_enabled_respects_mode(self):
        """smart_boost_enabled 属性应检查 proactive_mode"""
        from iris_memory.core.config_manager import ConfigManager
        from iris_memory.core.test_utils import setup_test_config

        # mode = llm + smart_boost = True → enabled
        setup_test_config({
            "proactive_reply": {"smart_boost": True},
            "llm_enhanced": {"proactive_mode": "llm"},
        })
        mgr = ConfigManager(Mock(
            proactive_reply=Mock(smart_boost=True),
            llm_enhanced=Mock(proactive_mode="llm"),
        ))
        assert mgr.smart_boost_enabled is True

    def test_smart_boost_disabled_when_mode_rule(self):
        """proactive_mode=rule 时 smart_boost_enabled 应为 False"""
        from iris_memory.core.config_manager import ConfigManager

        mgr = ConfigManager(Mock(
            proactive_reply=Mock(smart_boost=True),
            llm_enhanced=Mock(proactive_mode="rule"),
        ))
        assert mgr.smart_boost_enabled is False

    def test_smart_boost_disabled_when_config_off(self):
        """smart_boost=False 时应为 False"""
        from iris_memory.core.config_manager import ConfigManager

        mgr = ConfigManager(Mock(
            proactive_reply=Mock(smart_boost=False),
            llm_enhanced=Mock(proactive_mode="hybrid"),
        ))
        assert mgr.smart_boost_enabled is False

    def test_smart_boost_window_defaults(self):
        """默认窗口参数"""
        from iris_memory.core.config_manager import ConfigManager

        mgr = ConfigManager()
        assert mgr.smart_boost_window_seconds == 120
        assert mgr.smart_boost_score_multiplier == 1.2
        assert mgr.smart_boost_reply_threshold == 0.35
