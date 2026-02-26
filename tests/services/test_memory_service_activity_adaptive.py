"""BusinessService 群聊自适应动态行为测试

重构后，业务逻辑（含 _build_chat_history_context / analyze_images）
已迁移到 BusinessService，此处直接构造最小 BusinessService 进行测试。
"""

from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.services.business_service import BusinessService


@pytest.fixture
def business_service_stub():
    """构建一个最小可测的 BusinessService 实例（跳过重型初始化）"""
    svc = object.__new__(BusinessService)
    svc._cfg = Mock()
    svc._storage = Mock()
    svc._analysis = Mock()
    svc._llm_enhanced = Mock()
    svc._capture = Mock()
    svc._retrieval = Mock()
    svc._kg = Mock()
    svc._shared_state = Mock()
    svc._image_analyzer = None
    svc._member_identity = None
    svc._activity_tracker = None
    return svc


class TestDynamicChatContextCount:
    """聊天上下文条数动态配置测试"""

    @pytest.mark.asyncio
    async def test_build_chat_context_uses_dynamic_limit(self, business_service_stub):
        buffer = Mock()
        buffer.max_messages = 3
        buffer.set_max_messages = Mock()
        buffer.get_recent_messages = AsyncMock(return_value=[Mock(), Mock()])
        buffer.format_for_llm = Mock(return_value="formatted chat context")

        business_service_stub._storage.chat_history_buffer = buffer
        business_service_stub._cfg.get_chat_context_count = Mock(return_value=8)

        context = await business_service_stub._build_chat_history_context("u1", "g1")

        assert context == "formatted chat context"
        business_service_stub._cfg.get_chat_context_count.assert_called_once_with("g1")
        buffer.set_max_messages.assert_called_once_with(8)
        buffer.get_recent_messages.assert_awaited_once_with(
            user_id="u1",
            group_id="g1",
            limit=8,
        )

    @pytest.mark.asyncio
    async def test_build_chat_context_disabled_by_dynamic_limit(self, business_service_stub):
        buffer = Mock()
        buffer.max_messages = 20
        buffer.set_max_messages = Mock()
        buffer.get_recent_messages = AsyncMock(return_value=[Mock()])
        buffer.format_for_llm = Mock(return_value="unused")

        business_service_stub._storage.chat_history_buffer = buffer
        business_service_stub._cfg.get_chat_context_count = Mock(return_value=0)

        context = await business_service_stub._build_chat_history_context("u1", "g1")

        assert context == ""
        buffer.get_recent_messages.assert_not_awaited()
        buffer.set_max_messages.assert_not_called()


class TestDynamicImageDailyBudget:
    """图片分析每日预算动态配置测试"""

    @pytest.mark.asyncio
    async def test_analyze_images_passes_group_dynamic_budget(self, business_service_stub):
        analyzer = Mock()
        analyzer.analyze_message_images = AsyncMock(return_value=[])
        analyzer.format_for_llm_context = Mock(return_value="")
        analyzer.format_for_memory = Mock(return_value="")

        business_service_stub._image_analyzer = analyzer
        business_service_stub._cfg.get_daily_analysis_budget = Mock(return_value=5)

        llm_ctx, mem_ctx = await business_service_stub.analyze_images(
            message_chain=[],
            user_id="u1",
            group_id="g1",
            context_text="hello",
            umo="umo",
            session_id="session-1",
        )

        assert llm_ctx == ""
        assert mem_ctx == ""
        business_service_stub._cfg.get_daily_analysis_budget.assert_called_once_with("g1")
        analyzer.analyze_message_images.assert_awaited_once_with(
            message_chain=[],
            user_id="u1",
            context_text="hello",
            umo="umo",
            session_id="session-1",
            daily_analysis_budget=5,
        )

    @pytest.mark.asyncio
    async def test_analyze_images_non_positive_budget_becomes_unlimited(self, business_service_stub):
        analyzer = Mock()
        analyzer.analyze_message_images = AsyncMock(return_value=[])
        analyzer.format_for_llm_context = Mock(return_value="")
        analyzer.format_for_memory = Mock(return_value="")

        business_service_stub._image_analyzer = analyzer
        business_service_stub._cfg.get_daily_analysis_budget = Mock(return_value=0)

        await business_service_stub.analyze_images(
            message_chain=[],
            user_id="u1",
            group_id="g1",
            context_text="hello",
            umo="umo",
            session_id="session-1",
        )

        analyzer.analyze_message_images.assert_awaited_once_with(
            message_chain=[],
            user_id="u1",
            context_text="hello",
            umo="umo",
            session_id="session-1",
            daily_analysis_budget=999999,
        )
