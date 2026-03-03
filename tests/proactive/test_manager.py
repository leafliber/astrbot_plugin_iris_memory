"""test_manager.py - ProactiveManager v3 Facade 集成测试"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_memory.proactive.config import (
    FollowUpConfig,
    ProactiveConfig,
    SignalQueueConfig,
)
from iris_memory.proactive.manager import ProactiveManager
from iris_memory.proactive.models import (
    AggregatedDecision,
    ProactiveReplyResult,
    Signal,
    SignalType,
)


@pytest.fixture
def tmp_path_fixture(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def config() -> ProactiveConfig:
    return ProactiveConfig(
        enabled=True,
        group_whitelist_mode=False,
        proactive_mode="rule",
        quiet_hours=[],  # 禁用静音
        max_daily_replies=20,
        max_daily_per_user=5,
        signal_queue=SignalQueueConfig(
            check_interval_seconds=1,
            min_silence_seconds=0,
        ),
        followup=FollowUpConfig(short_window_seconds=2),
    )


@pytest.fixture
def manager(tmp_path_fixture: Path, config: ProactiveConfig) -> ProactiveManager:
    return ProactiveManager(
        plugin_data_path=tmp_path_fixture,
        config=config,
    )


class TestInitialization:
    """初始化测试"""

    @pytest.mark.asyncio
    async def test_initialize(self, manager: ProactiveManager) -> None:
        assert not manager.is_initialized
        await manager.initialize()
        assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_double_initialize_idempotent(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        await manager.initialize()
        assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_default_construction(self, tmp_path_fixture: Path) -> None:
        """不传 config 时使用默认值"""
        m = ProactiveManager(
            plugin_data_path=tmp_path_fixture,
            enabled=True,
            proactive_mode="hybrid",
        )
        assert m.enabled is True
        assert m._config.proactive_mode == "hybrid"


class TestProcessMessage:
    """消息处理测试"""

    @pytest.mark.asyncio
    async def test_process_group_message(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        result = await manager.process_message(
            messages=[{"text": "帮我看看这个问题怎么解决？", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )
        # v3 不同步返回回复结果
        assert result is None
        # 但信号应该入队了
        assert manager._signal_queue.total_signals > 0

    @pytest.mark.asyncio
    async def test_private_message_excluded(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        result = await manager.process_message(
            messages=[{"text": "帮我看看", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:private",
            session_type="private",
        )
        assert result is None
        assert manager._signal_queue.total_signals == 0

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, manager: ProactiveManager) -> None:
        manager.enabled = False
        await manager.initialize()
        result = await manager.process_message(
            messages=[{"text": "帮我看看", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )
        assert result is None
        assert manager._signal_queue.total_signals == 0

    @pytest.mark.asyncio
    async def test_whitelist_filters(
        self, tmp_path_fixture: Path, config: ProactiveConfig
    ) -> None:
        config.group_whitelist_mode = True
        config.group_whitelist = ["allowed_group"]
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=config)
        await m.initialize()

        # 不在白名单中的群
        result = await m.process_message(
            messages=[{"text": "帮我看看", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:blocked",
            session_type="group",
            group_id="blocked",
        )
        assert result is None
        assert m._signal_queue.total_signals == 0

        # 白名单内群
        await m.process_message(
            messages=[{"text": "帮我看看这个问题", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:allowed_group",
            session_type="group",
            group_id="allowed_group",
        )
        assert m._signal_queue.total_signals > 0

        await m.close()

    @pytest.mark.asyncio
    async def test_not_initialized_returns_none(self, manager: ProactiveManager) -> None:
        # 不调用 initialize
        result = await manager.process_message(
            messages=[{"text": "帮我看看"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_messages(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        result = await manager.process_message(
            messages=[],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_emotion_intensity_pass_through(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        await manager.process_message(
            messages=[{"text": "我好难过，感觉很崩溃", "sender_name": "张三"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
            extra={"emotion_intensity": 0.9},
        )
        # 应产生 emotion_high 信号
        signals = manager._signal_queue.get_signals("g1")
        types = {s.signal_type for s in signals}
        assert SignalType.EMOTION_HIGH in types

    @pytest.mark.asyncio
    async def test_followup_notified(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        # 手动创建期待
        manager._followup_planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        # 发送消息
        await manager.process_message(
            messages=[{"text": "谢谢你的回复", "sender_name": "User1"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )


class TestClearPendingTasks:
    """清除待处理任务测试"""

    @pytest.mark.asyncio
    async def test_clear_session(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        # 添加信号
        await manager.process_message(
            messages=[{"text": "帮我看看这个问题"}],
            user_id="u1",
            session_key="u1:g1",
            session_type="group",
            group_id="g1",
        )
        manager.clear_pending_tasks_for_session("u1", "g1")
        assert manager._signal_queue.get_signals("g1") == [] or \
            all(s.session_key != "u1:g1" for s in manager._signal_queue.get_signals("g1"))

    @pytest.mark.asyncio
    async def test_clear_preserves_followup_when_after_all_replies(
        self, tmp_path_fixture: Path,
    ) -> None:
        """followup_after_all_replies 启用时，clear 不清除 FollowUp 期待"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        # 手动创建期待
        m._followup_planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        assert m._followup_planner.has_active_expectation("g1")

        m.clear_pending_tasks_for_session("u1", "g1")

        # 期待应保留
        assert m._followup_planner.has_active_expectation("g1")
        await m.close()

    @pytest.mark.asyncio
    async def test_clear_removes_followup_when_after_all_replies_disabled(
        self, manager: ProactiveManager,
    ) -> None:
        """followup_after_all_replies 禁用时（默认），clear 会清除 FollowUp 期待"""
        await manager.initialize()
        assert manager._config.followup_after_all_replies is False

        manager._followup_planner.create_expectation(
            session_key="u1:g1",
            group_id="g1",
            trigger_user_id="u1",
            trigger_message="hi",
            bot_reply_summary="hello",
        )
        assert manager._followup_planner.has_active_expectation("g1")

        manager.clear_pending_tasks_for_session("u1", "g1")

        # 期待应被清除
        assert not manager._followup_planner.has_active_expectation("g1")


class TestNotifyBotReply:
    """Bot 回复后 FollowUp 通知测试"""

    @pytest.mark.asyncio
    async def test_creates_expectation_when_enabled(
        self, tmp_path_fixture: Path,
    ) -> None:
        """followup_after_all_replies 启用时创建 FollowUp 期待"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
            followup_enabled=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        m.notify_bot_reply(
            user_id="u1",
            group_id="g1",
            user_message="你好",
            bot_reply="你好呀！",
        )

        assert m._followup_planner.has_active_expectation("g1")
        await m.close()

    @pytest.mark.asyncio
    async def test_no_expectation_when_disabled(
        self, manager: ProactiveManager,
    ) -> None:
        """followup_after_all_replies 禁用时不创建 FollowUp 期待"""
        await manager.initialize()
        assert manager._config.followup_after_all_replies is False

        manager.notify_bot_reply(
            user_id="u1",
            group_id="g1",
            user_message="你好",
            bot_reply="你好呀！",
        )

        assert not manager._followup_planner.has_active_expectation("g1")

    @pytest.mark.asyncio
    async def test_no_expectation_for_private_chat(
        self, tmp_path_fixture: Path,
    ) -> None:
        """私聊不创建 FollowUp 期待"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        m.notify_bot_reply(
            user_id="u1",
            group_id=None,
            user_message="你好",
            bot_reply="你好呀！",
        )

        # 没有群 ID，不应创建期待
        assert m._followup_planner.active_expectation_count == 0
        await m.close()

    @pytest.mark.asyncio
    async def test_whitelist_filters_notify(
        self, tmp_path_fixture: Path,
    ) -> None:
        """白名单模式过滤不在白名单中的群"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
            group_whitelist_mode=True,
            group_whitelist=["allowed_g"],
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        m.notify_bot_reply(
            user_id="u1",
            group_id="blocked_g",
            user_message="你好",
            bot_reply="你好呀！",
        )
        assert not m._followup_planner.has_active_expectation("blocked_g")

        m.notify_bot_reply(
            user_id="u1",
            group_id="allowed_g",
            user_message="你好",
            bot_reply="你好呀！",
        )
        assert m._followup_planner.has_active_expectation("allowed_g")
        await m.close()

    @pytest.mark.asyncio
    async def test_not_initialized_noop(
        self, tmp_path_fixture: Path,
    ) -> None:
        """未初始化时 notify_bot_reply 不做任何事"""
        cfg = ProactiveConfig(
            enabled=True,
            followup_after_all_replies=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        # 不调用 initialize
        m.notify_bot_reply(
            user_id="u1",
            group_id="g1",
            user_message="你好",
            bot_reply="你好呀！",
        )
        # 不应崩溃

    @pytest.mark.asyncio
    async def test_followup_disabled_noop(
        self, tmp_path_fixture: Path,
    ) -> None:
        """followup_enabled=False 时不创建期待"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
            followup_enabled=False,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        m.notify_bot_reply(
            user_id="u1",
            group_id="g1",
            user_message="你好",
            bot_reply="你好呀！",
        )

        assert m._followup_planner.active_expectation_count == 0
        await m.close()

    @pytest.mark.asyncio
    async def test_replaces_old_expectation(
        self, tmp_path_fixture: Path,
    ) -> None:
        """新的 notify_bot_reply 调用会替换旧期待"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        m.notify_bot_reply(
            user_id="u1", group_id="g1",
            user_message="消息1", bot_reply="回复1",
        )
        m.notify_bot_reply(
            user_id="u2", group_id="g1",
            user_message="消息2", bot_reply="回复2",
        )

        # 同群只有一个活跃期待
        assert m._followup_planner.active_expectation_count == 1
        exp = m._followup_planner._store.get("g1")
        assert exp is not None
        assert exp.trigger_user_id == "u2"
        await m.close()

    @pytest.mark.asyncio
    async def test_long_reply_truncated(
        self, tmp_path_fixture: Path,
    ) -> None:
        """长回复在存储时截断"""
        cfg = ProactiveConfig(
            enabled=True,
            quiet_hours=[],
            followup_after_all_replies=True,
        )
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=cfg)
        await m.initialize()

        long_reply = "x" * 500
        m.notify_bot_reply(
            user_id="u1", group_id="g1",
            user_message="msg", bot_reply=long_reply,
        )

        exp = m._followup_planner._store.get("g1")
        assert exp is not None
        assert len(exp.bot_reply_summary) <= 200
        await m.close()


class TestQuietHours:
    """静音时段测试"""

    def test_quiet_hours_normal(self, tmp_path_fixture: Path) -> None:
        config = ProactiveConfig(quiet_hours=[1, 5])
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=config)
        hour = datetime.now().hour
        expected = 1 <= hour < 5
        assert m._is_quiet_hours() == expected

    def test_quiet_hours_cross_midnight(self, tmp_path_fixture: Path) -> None:
        config = ProactiveConfig(quiet_hours=[23, 7])
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=config)
        hour = datetime.now().hour
        expected = hour >= 23 or hour < 7
        assert m._is_quiet_hours() == expected

    def test_quiet_hours_empty(self, tmp_path_fixture: Path) -> None:
        config = ProactiveConfig(quiet_hours=[])
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=config)
        assert m._is_quiet_hours() is False


class TestDailyLimit:
    """每日限额测试"""

    def test_within_limit(self, manager: ProactiveManager) -> None:
        assert manager._check_daily_limit("u1") is True

    def test_global_limit_reached(self, manager: ProactiveManager) -> None:
        manager._daily_reply_date = datetime.now().strftime("%Y-%m-%d")
        manager._daily_reply_count = 20
        assert manager._check_daily_limit("u1") is False

    def test_per_user_limit_reached(self, manager: ProactiveManager) -> None:
        manager._daily_reply_date = datetime.now().strftime("%Y-%m-%d")
        manager._per_user_daily["u1"] = 5
        assert manager._check_daily_limit("u1") is False

    def test_counter_resets_next_day(self, manager: ProactiveManager) -> None:
        manager._daily_reply_date = "2020-01-01"
        manager._daily_reply_count = 999
        manager._per_user_daily["u1"] = 999
        # _check_daily_limit 会刷新计数器
        assert manager._check_daily_limit("u1") is True

    def test_increment(self, manager: ProactiveManager) -> None:
        manager._increment_daily_count("u1")
        assert manager._daily_reply_count == 1
        assert manager._per_user_daily["u1"] == 1
        manager._increment_daily_count("u1")
        assert manager._daily_reply_count == 2
        assert manager._per_user_daily["u1"] == 2


class TestWhitelistManagement:
    """白名单管理测试"""

    def test_add_group(self, manager: ProactiveManager) -> None:
        assert manager.add_group_to_whitelist("g1") is True
        assert manager.is_group_in_whitelist("g1") is True

    def test_add_duplicate(self, manager: ProactiveManager) -> None:
        manager.add_group_to_whitelist("g1")
        assert manager.add_group_to_whitelist("g1") is False

    def test_remove_group(self, manager: ProactiveManager) -> None:
        manager.add_group_to_whitelist("g1")
        assert manager.remove_group_from_whitelist("g1") is True
        assert manager.is_group_in_whitelist("g1") is False

    def test_remove_nonexistent(self, manager: ProactiveManager) -> None:
        assert manager.remove_group_from_whitelist("g1") is False

    def test_get_whitelist(self, manager: ProactiveManager) -> None:
        manager.add_group_to_whitelist("g1")
        manager.add_group_to_whitelist("g2")
        wl = manager.get_whitelist()
        assert set(wl) == {"g1", "g2"}

    def test_is_group_allowed_no_whitelist_mode(self, manager: ProactiveManager) -> None:
        # 非白名单模式，所有群都允许
        assert manager.is_group_allowed("any_group") is True

    def test_is_group_allowed_whitelist_mode(
        self, tmp_path_fixture: Path, config: ProactiveConfig
    ) -> None:
        config.group_whitelist_mode = True
        config.group_whitelist = ["g1"]
        m = ProactiveManager(plugin_data_path=tmp_path_fixture, config=config)
        assert m.is_group_allowed("g1") is True
        assert m.is_group_allowed("g2") is False

    def test_is_group_allowed_none(self, manager: ProactiveManager) -> None:
        assert manager.is_group_allowed(None) is True

    def test_serialize_deserialize(self, manager: ProactiveManager) -> None:
        manager.add_group_to_whitelist("g1")
        manager.add_group_to_whitelist("g2")
        manager.group_whitelist_mode = True
        data = manager.serialize_whitelist()

        # 新实例
        new_manager = ProactiveManager(
            plugin_data_path=manager._plugin_data_path,
        )
        new_manager.deserialize_whitelist(data)
        assert new_manager.group_whitelist_mode is True
        assert set(new_manager.get_whitelist()) == {"g1", "g2"}

    def test_deserialize_invalid(self, manager: ProactiveManager) -> None:
        # 不应崩溃
        manager.deserialize_whitelist("invalid")
        manager.deserialize_whitelist(None)
        manager.deserialize_whitelist(42)


class TestGetStats:
    """统计数据测试"""

    @pytest.mark.asyncio
    async def test_basic_stats(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        stats = await manager.get_stats()
        assert "enabled" in stats
        assert "mode" in stats
        assert "signal_queue_enabled" in stats
        assert "followup_enabled" in stats
        assert "daily_reply_count" in stats
        assert "total_signals" in stats
        assert "active_groups" in stats
        assert "active_expectations" in stats

    @pytest.mark.asyncio
    async def test_stats_before_init(self, manager: ProactiveManager) -> None:
        stats = await manager.get_stats()
        assert stats["total_signals"] == 0
        assert stats["active_groups"] == 0


class TestClose:
    """关闭测试"""

    @pytest.mark.asyncio
    async def test_close(self, manager: ProactiveManager) -> None:
        await manager.initialize()
        await manager.close()
        assert not manager.is_initialized
        assert manager._signal_queue is None
        assert manager._group_scheduler is None
        assert manager._followup_planner is None

    @pytest.mark.asyncio
    async def test_close_without_init(self, manager: ProactiveManager) -> None:
        # 不应崩溃
        await manager.close()


class TestSessionKeyBuild:
    """会话键构建测试"""

    def test_group_session(self, manager: ProactiveManager) -> None:
        key = manager._build_session_key("u1", "g1")
        assert key == "u1:g1"

    def test_private_session(self, manager: ProactiveManager) -> None:
        key = manager._build_session_key("u1")
        assert key == "u1:private"

    def test_none_group(self, manager: ProactiveManager) -> None:
        key = manager._build_session_key("u1", None)
        assert key == "u1:private"


class TestBuildSignalReply:
    """信号回复构建测试"""

    def test_build_basic(self) -> None:
        decision = AggregatedDecision(
            should_reply=True,
            session_key="u1:g1",
            group_id="g1",
            target_user_id="u1",
            aggregated_weight=0.9,
            signals=[
                Signal(
                    signal_type=SignalType.RULE_MATCH,
                    session_key="u1:g1",
                    group_id="g1",
                    user_id="u1",
                    weight=0.9,
                )
            ],
            reason="测试原因",
            recent_messages=[
                {"sender_id": "u1", "content": "你好"},
            ],
        )
        result = ProactiveManager._build_signal_reply(decision)
        assert isinstance(result, ProactiveReplyResult)
        assert result.source == "signal_queue"
        assert "主动回复场景" in result.trigger_prompt
        assert "测试原因" in result.trigger_prompt
        assert result.target_user == "u1"


class TestV2CompatParams:
    """v2 兼容参数测试"""

    def test_accepts_v2_params(self, tmp_path_fixture: Path) -> None:
        """确保旧参数不会导致崩溃"""
        m = ProactiveManager(
            plugin_data_path=tmp_path_fixture,
            chroma_manager=MagicMock(),
            embedding_manager=MagicMock(),
            shared_state=MagicMock(),
            personality="balanced",
            quiet_hours=[23, 7],
            max_history=10,
            max_text_tokens=200,
        )
        assert m._config.max_reply_tokens == 200
