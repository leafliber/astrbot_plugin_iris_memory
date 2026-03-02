"""主动回复管理器测试（事件队列版本）"""

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from iris_memory.proactive.proactive_manager import ProactiveReplyManager
from iris_memory.proactive.proactive_reply_detector import (
    ProactiveReplyDecision,
    ReplyUrgency,
)


@pytest.fixture
def decision_should_reply() -> ProactiveReplyDecision:
    return ProactiveReplyDecision(
        should_reply=True,
        urgency=ReplyUrgency.HIGH,
        reason="test_reason",
        suggested_delay=0,
        reply_context={"emotion": {"primary": "joy", "intensity": 0.6}},
    )


@pytest.fixture
def decision_no_reply() -> ProactiveReplyDecision:
    return ProactiveReplyDecision(
        should_reply=False,
        urgency=ReplyUrgency.IGNORE,
        reason="ignore",
        suggested_delay=0,
        reply_context={},
    )


@pytest.fixture
def mock_reply_detector(decision_should_reply):
    detector = Mock()
    detector.analyze = AsyncMock(return_value=decision_should_reply)
    return detector


@pytest.fixture
def mock_context_with_queue():
    context = Mock()
    context._event_queue = asyncio.Queue()
    return context


@pytest_asyncio.fixture
async def manager(mock_reply_detector, mock_context_with_queue):
    proactive_manager = ProactiveReplyManager(
        astrbot_context=mock_context_with_queue,
        reply_detector=mock_reply_detector,
        config={
            "enable_proactive_reply": True,
            "reply_cooldown": 60,
            "max_daily_replies": 20,
        },
    )
    await proactive_manager.initialize()
    yield proactive_manager
    await proactive_manager.stop()


class TestInitialization:
    @pytest.mark.asyncio
    async def test_initialize_enabled(self, mock_reply_detector, mock_context_with_queue):
        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()
        assert proactive_manager.is_running is True
        assert proactive_manager.event_queue is not None
        await proactive_manager.stop()

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, mock_reply_detector):
        proactive_manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": False},
        )
        await proactive_manager.initialize()
        assert proactive_manager.is_running is False

    @pytest.mark.asyncio
    async def test_initialize_without_queue_disables_manager(self, mock_reply_detector):
        proactive_manager = ProactiveReplyManager(
            astrbot_context=Mock(spec_set=[]),
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()
        assert proactive_manager.enabled is False
        assert proactive_manager.is_running is False


class TestHandleBatch:
    @pytest.mark.asyncio
    async def test_handle_batch_creates_task(self, manager):
        await manager.handle_batch(messages=["你好", "在吗？"], user_id="u1", umo="test:FriendMessage:u1")
        assert manager.pending_tasks.qsize() == 1

    @pytest.mark.asyncio
    async def test_handle_batch_skipped_when_detector_returns_false(
        self,
        mock_context_with_queue,
        decision_no_reply,
    ):
        detector = Mock()
        detector.analyze = AsyncMock(return_value=decision_no_reply)
        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()
        await proactive_manager.handle_batch(messages=["嗯"], user_id="u1")
        assert proactive_manager.pending_tasks.qsize() == 0
        assert proactive_manager.stats["replies_skipped"] == 1
        await proactive_manager.stop()

    @pytest.mark.asyncio
    async def test_queued_sessions_prevents_duplicate(self, manager):
        """同一会话重复入队应被 _queued_sessions 阻止"""
        await manager.handle_batch(messages=["你在吗？"], user_id="u1")
        await manager.handle_batch(messages=["你在吗？"], user_id="u1")
        assert manager.pending_tasks.qsize() == 1

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self, manager):
        manager._default_max_daily = 1
        manager.daily_reply_count["u1:private"] = 1
        await manager.handle_batch(messages=["为什么"], user_id="u1")
        assert manager.pending_tasks.qsize() == 0


class TestTaskDispatch:
    @pytest.mark.asyncio
    async def test_process_task_dispatches_event(self, manager):
        with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
            event_obj = Mock()
            event_cls.return_value = event_obj

            await manager.handle_batch(
                messages=["我有点难过"],
                user_id="u1",
                group_id="g1",
                context={"sender_name": "Alice"},
                umo="test:GroupMessage:g1",
            )

            await asyncio.sleep(0.1)
            dispatched = await asyncio.wait_for(manager.event_queue.get(), timeout=1)
            assert dispatched is event_obj
            assert manager.stats["replies_sent"] == 1
            assert manager.daily_reply_count["u1:g1"] == 1

    @pytest.mark.asyncio
    async def test_stop_processes_pending_without_delay(
        self,
        mock_reply_detector,
        mock_context_with_queue,
    ):
        delayed_decision = ProactiveReplyDecision(
            should_reply=True,
            urgency=ReplyUrgency.LOW,
            reason="delay",
            suggested_delay=120,
            reply_context={},
        )
        mock_reply_detector.analyze = AsyncMock(return_value=delayed_decision)

        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()

        with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
            event_cls.return_value = Mock()
            await proactive_manager.handle_batch(
                messages=["test"],
                user_id="u2",
                umo="test:FriendMessage:u2",
            )
            await proactive_manager.stop()

        assert proactive_manager.stats["replies_sent"] >= 1


class TestWhitelist:
    def test_whitelist_normalization(self, mock_reply_detector):
        proactive_manager = ProactiveReplyManager(
            reply_detector=mock_reply_detector,
            config={
                "group_whitelist": [123, "456"],
                "dynamic_whitelist": "789",
            },
        )
        assert proactive_manager.group_whitelist == ["123", "456"]
        assert proactive_manager.get_whitelist() == ["789"]


class TestDailyReset:
    """每日计数重置测试"""

    @pytest.mark.asyncio
    async def test_daily_counts_reset_on_new_day(self, manager):
        """跨日后每日计数应自动重置"""
        manager.daily_reply_count["u1:private"] = 10
        # 模拟上次重置是昨天
        manager._last_reset_date = date.today() - timedelta(days=1)
        # _is_daily_limit_reached 内部调用 _check_daily_reset
        assert manager._is_daily_limit_reached("u1") is False
        assert manager.daily_reply_count.get("u1:private", 0) == 0

    @pytest.mark.asyncio
    async def test_daily_counts_not_reset_same_day(self, manager):
        """同一天内不应重置"""
        manager.daily_reply_count["u1:private"] = 5
        manager._last_reset_date = date.today()
        manager._check_daily_reset()
        assert manager.daily_reply_count.get("u1:private", 0) == 5

    @pytest.mark.asyncio
    async def test_daily_limit_reached_after_reset(self, manager):
        """重置后不应立即触发每日限制"""
        manager._default_max_daily = 1
        manager.daily_reply_count["u1:private"] = 1
        # 同一天：应达到限制
        assert manager._is_daily_limit_reached("u1") is True
        # 模拟跨日
        manager._last_reset_date = date.today() - timedelta(days=1)
        # 跨日后：计数被清零，不再达到限制
        assert manager._is_daily_limit_reached("u1") is False


class TestQueuedSessionsDedup:
    """队列会话去重测试"""

    @pytest.mark.asyncio
    async def test_duplicate_prevented_by_queued_sessions(self, manager):
        """同一会话的重复消息不应重复入队"""
        await manager.handle_batch(messages=["在吗？"], user_id="u1")
        await manager.handle_batch(messages=["在吗？"], user_id="u1")
        assert manager.pending_tasks.qsize() == 1

    @pytest.mark.asyncio
    async def test_different_sessions_can_queue(self, manager):
        """不同会话的消息应可以同时入队"""
        await manager.handle_batch(messages=["在吗？"], user_id="u1")
        await manager.handle_batch(messages=["你好"], user_id="u2")
        assert manager.pending_tasks.qsize() == 2

    @pytest.mark.asyncio
    async def test_queued_session_cleared_after_processing(
        self, mock_reply_detector, mock_context_with_queue
    ):
        """任务处理完成后应从队列会话集合中移除"""
        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()

        with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
            event_cls.return_value = Mock()
            await proactive_manager.handle_batch(
                messages=["test"], user_id="u1", umo="test:FriendMessage:u1"
            )
            assert "u1:private" in proactive_manager._queued_sessions
            # 等待任务处理完成
            await asyncio.sleep(0.2)
            assert "u1:private" not in proactive_manager._queued_sessions

        await proactive_manager.stop()


class TestCooldownTiming:
    """冷却时间记录时机测试"""

    @pytest.mark.asyncio
    async def test_cooldown_not_set_at_queue_time(self, manager):
        """入队时不应设置冷却时间"""
        await manager.handle_batch(messages=["在吗？"], user_id="u1")
        # last_reply_time 不应在入队时被设置
        assert "u1:private" not in manager.last_reply_time

    @pytest.mark.asyncio
    async def test_cooldown_set_after_send(
        self, mock_reply_detector, mock_context_with_queue
    ):
        """实际发送后应设置冷却时间"""
        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()

        with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
            event_cls.return_value = Mock()
            await proactive_manager.handle_batch(
                messages=["test"], user_id="u1", umo="test:FriendMessage:u1"
            )
            # 入队后还未发送，不应有 last_reply_time
            assert "u1:private" not in proactive_manager.last_reply_time
            # 等待处理完成
            await asyncio.sleep(0.2)
            # 发送后应设置 last_reply_time
            assert "u1:private" in proactive_manager.last_reply_time

        await proactive_manager.stop()


class TestUserMessageTimeRecording:
    """用户发言时间记录测试"""

    @pytest.mark.asyncio
    async def test_handle_batch_records_user_message_time(self, manager):
        """handle_batch 应记录用户发言时间"""
        await manager.handle_batch(messages=["你好"], user_id="u1")
        assert "u1:private" in manager._last_user_message_time

    @pytest.mark.asyncio
    async def test_process_task_does_not_record_user_message_time(
        self, mock_reply_detector, mock_context_with_queue
    ):
        """_process_task 不应记录用户发言时间（防止滚雪球效应）"""
        proactive_manager = ProactiveReplyManager(
            astrbot_context=mock_context_with_queue,
            reply_detector=mock_reply_detector,
            config={"enable_proactive_reply": True},
        )
        await proactive_manager.initialize()

        with patch("iris_memory.proactive.proactive_manager.ProactiveMessageEvent") as event_cls:
            event_cls.return_value = Mock()
            await proactive_manager.handle_batch(
                messages=["test"], user_id="u1", umo="test:FriendMessage:u1"
            )
            # 记录初始的用户发言时间
            initial_time = proactive_manager._last_user_message_time.get("u1:private")
            assert initial_time is not None
            # 等待处理完成
            await asyncio.sleep(0.2)
            # 处理完成后用户发言时间不应被更新（不再在 _process_task 中记录）
            assert proactive_manager._last_user_message_time["u1:private"] == initial_time

        await proactive_manager.stop()
