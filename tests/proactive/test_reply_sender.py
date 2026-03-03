"""ProactiveReplySender 单元测试"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from iris_memory.proactive.models import ProactiveReplyResult
from iris_memory.proactive.reply_sender import ProactiveReplySender
from iris_memory.utils.llm_helper import LLMCallResult


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_context():
    """Mock AstrBot Context"""
    ctx = AsyncMock()
    ctx.send_message = AsyncMock()
    ctx.llm_generate = AsyncMock()
    return ctx


@pytest.fixture
def mock_prepare_llm_context():
    """Mock MemoryService.prepare_llm_context"""
    callback = AsyncMock(return_value="[记忆上下文] 用户喜欢编程")
    return callback


@pytest.fixture
def mock_record_chat_message():
    """Mock MemoryService.record_chat_message"""
    return AsyncMock()


@pytest.fixture
def mock_get_group_umo():
    """Mock group_id -> UMO mapping"""
    return MagicMock(return_value="platform:group123")


@pytest.fixture
def sender(
    mock_context,
    mock_prepare_llm_context,
    mock_record_chat_message,
    mock_get_group_umo,
):
    """创建 ProactiveReplySender 实例"""
    return ProactiveReplySender(
        astrbot_context=mock_context,
        prepare_llm_context=mock_prepare_llm_context,
        record_chat_message=mock_record_chat_message,
        get_group_umo=mock_get_group_umo,
        llm_provider=MagicMock(),
        llm_provider_id="test-provider",
    )


@pytest.fixture
def basic_result():
    """基本的 ProactiveReplyResult"""
    return ProactiveReplyResult(
        trigger_prompt="【主动回复场景】请自然地发起对话。",
        group_id="group123",
        session_key="user1:group123",
        target_user="user1",
        source="signal_queue",
    )


@pytest.fixture
def followup_result():
    """FollowUp 的 ProactiveReplyResult"""
    return ProactiveReplyResult(
        trigger_prompt="【跟进回复场景】用户有回应，继续对话。",
        group_id="group123",
        session_key="user1:group123",
        target_user="user1",
        recent_messages=[
            {"sender_name": "用户", "content": "是的，我也这么觉得"},
        ],
        source="followup",
    )


# ── TestSendReply ──────────────────────────────────────────


class TestSendReply:
    """测试 send_reply 方法"""

    @pytest.mark.asyncio
    async def test_successful_send(self, sender, basic_result):
        """正常发送流程"""
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(
                success=True, content="嘿，最近在忙什么项目呀？"
            ),
        ):
            result = await sender.send_reply(basic_result)

        assert result == "嘿，最近在忙什么项目呀？"
        sender._prepare_llm_context.assert_awaited_once_with(
            query="", user_id="user1", group_id="group123"
        )
        sender._record_chat_message.assert_awaited_once()
        sender._context.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_group_id_returns_none(self, sender):
        """缺少 group_id 返回 None"""
        result_no_group = ProactiveReplyResult(
            trigger_prompt="test", group_id="", target_user="user1"
        )
        result = await sender.send_reply(result_no_group)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_umo_returns_none(self, sender, basic_result):
        """找不到 UMO 返回 None"""
        sender._get_group_umo = MagicMock(return_value=None)
        result = await sender.send_reply(basic_result)
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, sender, basic_result):
        """LLM 调用失败返回 None"""
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(success=False, error="timeout"),
        ):
            result = await sender.send_reply(basic_result)

        assert result is None
        sender._context.send_message.assert_not_awaited()
        sender._record_chat_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_none(self, sender, basic_result):
        """LLM 返回空内容"""
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(success=True, content=""),
        ):
            result = await sender.send_reply(basic_result)

        assert result is None

    @pytest.mark.asyncio
    async def test_followup_send(self, sender, followup_result):
        """FollowUp 回复发送"""
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(
                success=True, content="对呀，确实值得深入研究一下。"
            ),
        ):
            result = await sender.send_reply(followup_result)

        assert result == "对呀，确实值得深入研究一下。"

    @pytest.mark.asyncio
    async def test_send_message_exception_returns_none(
        self, sender, basic_result
    ):
        """平台发送失败返回 None"""
        sender._context.send_message = AsyncMock(
            side_effect=Exception("platform error")
        )
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(
                success=True, content="test reply"
            ),
        ):
            result = await sender.send_reply(basic_result)

        assert result is None

    @pytest.mark.asyncio
    async def test_records_bot_reply(self, sender, basic_result):
        """验证 Bot 回复被记录到聊天缓冲区"""
        with patch(
            "iris_memory.proactive.reply_sender.call_llm",
            new_callable=AsyncMock,
            return_value=LLMCallResult(
                success=True, content="你好呀！"
            ),
        ):
            await sender.send_reply(basic_result)

        sender._record_chat_message.assert_awaited_once_with(
            sender_id="bot",
            sender_name=None,
            content="你好呀！",
            group_id="group123",
            is_bot=True,
            session_user_id="user1",
        )


# ── TestBuildFullPrompt ────────────────────────────────────


class TestBuildFullPrompt:
    """测试 _build_full_prompt 静态方法"""

    def test_both_contexts(self):
        """有记忆上下文和触发指令"""
        prompt = ProactiveReplySender._build_full_prompt(
            memory_context="[用户画像] 喜欢编程",
            trigger_prompt="【主动回复场景】发起对话",
        )
        assert "[用户画像] 喜欢编程" in prompt
        assert "【主动回复场景】发起对话" in prompt
        assert "自然地生成一条回复" in prompt

    def test_empty_memory_context(self):
        """无记忆上下文"""
        prompt = ProactiveReplySender._build_full_prompt(
            memory_context="",
            trigger_prompt="【主动回复场景】发起对话",
        )
        assert "【主动回复场景】发起对话" in prompt
        assert "自然地生成一条回复" in prompt

    def test_trigger_prompt_always_present(self):
        """触发指令始终存在"""
        prompt = ProactiveReplySender._build_full_prompt(
            memory_context="context",
            trigger_prompt="trigger",
        )
        assert "trigger" in prompt


# ── TestManagerIntegration ─────────────────────────────────


class TestManagerUMOTracking:
    """测试 ProactiveManager UMO 追踪"""

    def test_get_umo_returns_none_for_unknown_group(self):
        """未知群组返回 None"""
        from iris_memory.proactive.config import ProactiveConfig
        from iris_memory.proactive.manager import ProactiveManager
        from pathlib import Path

        manager = ProactiveManager(
            plugin_data_path=Path("/tmp"),
            config=ProactiveConfig(enabled=True),
        )
        assert manager.get_group_umo("unknown_group") is None

    @pytest.mark.asyncio
    async def test_umo_stored_from_process_message(self):
        """process_message 中存储 UMO"""
        from iris_memory.proactive.config import ProactiveConfig
        from iris_memory.proactive.manager import ProactiveManager
        from pathlib import Path

        manager = ProactiveManager(
            plugin_data_path=Path("/tmp"),
            config=ProactiveConfig(enabled=True),
        )
        await manager.initialize()

        await manager.process_message(
            messages=[{"text": "你好呀", "sender_name": "测试"}],
            user_id="user1",
            session_key="user1:group1",
            session_type="group",
            group_id="group1",
            extra={"umo": "platform:group1"},
        )

        assert manager.get_group_umo("group1") == "platform:group1"
        await manager.close()

    def test_set_reply_sender(self):
        """测试 set_reply_sender"""
        from iris_memory.proactive.config import ProactiveConfig
        from iris_memory.proactive.manager import ProactiveManager
        from pathlib import Path

        manager = ProactiveManager(
            plugin_data_path=Path("/tmp"),
            config=ProactiveConfig(enabled=True),
        )
        mock_sender = MagicMock()
        manager.set_reply_sender(mock_sender)
        assert manager._reply_sender is mock_sender

    def test_set_context(self):
        """测试 set_context"""
        from iris_memory.proactive.config import ProactiveConfig
        from iris_memory.proactive.manager import ProactiveManager
        from pathlib import Path

        manager = ProactiveManager(
            plugin_data_path=Path("/tmp"),
            config=ProactiveConfig(enabled=True),
        )
        mock_ctx = MagicMock()
        manager.set_context(mock_ctx, llm_provider_id="test-id")
        assert manager._astrbot_context is mock_ctx
        assert manager._llm_provider_id == "test-id"
