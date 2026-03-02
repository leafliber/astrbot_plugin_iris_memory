"""
ContextEngine 单元测试
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from iris_memory.proactive.core.context_engine import ContextEngine
from iris_memory.proactive.core.models import ProactiveContext


@pytest.fixture
def mock_embedding():
    em = AsyncMock()
    em.embed = AsyncMock(return_value=[0.1] * 384)
    return em


@pytest.fixture
def engine(mock_embedding):
    return ContextEngine(
        embedding_manager=mock_embedding,
        max_history=5,
        max_text_tokens=100,
        quiet_hours=[23, 7],
    )


def _make_messages(texts, user_id="user1"):
    now = time.time()
    msgs = []
    for i, t in enumerate(texts):
        msgs.append({
            "text": t,
            "sender_id": user_id,
            "sender_name": "Alice",
            "timestamp": now - (len(texts) - i) * 10,
        })
    return msgs


class TestContextEngine:
    @pytest.mark.asyncio
    async def test_build_context_basic(self, engine):
        msgs = _make_messages(["你好", "怎么了"])
        ctx = await engine.build_context(
            messages=msgs,
            user_id="user1",
            session_key="user1:group1",
            session_type="group",
            group_id="group1",
        )
        assert isinstance(ctx, ProactiveContext)
        assert ctx.session_type == "group"
        assert ctx.session_key == "user1:group1"
        assert ctx.conversation.recent_text != ""
        assert ctx.conversation.base_query_vector is not None

    @pytest.mark.asyncio
    async def test_build_context_private(self, engine):
        msgs = _make_messages(["好累啊"])
        ctx = await engine.build_context(
            messages=msgs,
            user_id="user1",
            session_key="user1:private",
            session_type="private",
        )
        assert ctx.session_type == "private"
        assert ctx.group is None

    @pytest.mark.asyncio
    async def test_recent_text_truncation(self, engine):
        # With max_text_tokens=100, many messages should be truncated
        # Use short messages that individually fit but collectively exceed
        msgs = _make_messages([f"消息{i}，这是一条测试话" for i in range(20)])
        ctx = await engine.build_context(
            messages=msgs,
            user_id="user1",
            session_key="user1:private",
            session_type="private",
        )
        # Recent text should contain some but not all messages
        assert len(ctx.conversation.recent_text) > 0
        # Should not contain all 20 messages
        assert ctx.conversation.recent_text.count("\n") < 19

    @pytest.mark.asyncio
    async def test_focus_marker_in_text(self, engine):
        msgs = [
            {"text": "其他人说的话", "sender_id": "other", "sender_name": "Bob", "timestamp": time.time() - 20},
            {"text": "触发用户的话", "sender_id": "user1", "sender_name": "Alice", "timestamp": time.time() - 10},
        ]
        ctx = await engine.build_context(
            messages=msgs,
            user_id="user1",
            session_key="user1:group1",
            session_type="group",
            group_id="group1",
        )
        assert "[Focus]" in ctx.conversation.recent_text

    @pytest.mark.asyncio
    async def test_embedding_failure_handled(self):
        mock_em = AsyncMock()
        mock_em.embed = AsyncMock(side_effect=Exception("embed error"))
        engine = ContextEngine(embedding_manager=mock_em)
        msgs = _make_messages(["测试"])
        ctx = await engine.build_context(
            messages=msgs, user_id="u1", session_key="u1:p", session_type="private"
        )
        assert ctx.conversation.base_query_vector is None

    @pytest.mark.asyncio
    async def test_empty_messages(self, engine):
        ctx = await engine.build_context(
            messages=[], user_id="u1", session_key="u1:p", session_type="private"
        )
        assert ctx.conversation.recent_text == ""

    @pytest.mark.asyncio
    async def test_max_history_limit(self, engine):
        # engine has max_history=5
        msgs = _make_messages([f"msg_{i}" for i in range(20)])
        ctx = await engine.build_context(
            messages=msgs, user_id="u1", session_key="u1:p", session_type="private"
        )
        # Only last 5 messages should be in recent_messages
        assert len(ctx.conversation.recent_messages) <= 5
