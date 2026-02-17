"""主动回复管理器测试（事件队列重构版）"""

import asyncio
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
    async def test_cooldown_prevents_duplicate(self, manager):
        await manager.handle_batch(messages=["你在吗？"], user_id="u1")
        await manager.handle_batch(messages=["你在吗？"], user_id="u1")
        assert manager.pending_tasks.qsize() == 1

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self, manager):
        manager._default_max_daily = 1
        manager.daily_reply_count["u1"] = 1
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
            assert manager.daily_reply_count["u1"] == 1

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
