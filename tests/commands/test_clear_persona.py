"""命令 handler 层测试:clear 路径的 persona 透传与删除调用参数"""

from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.commands.all_handler import AllCommandHandler
from iris_memory.commands.l2_handler import L2CommandHandler
from iris_memory.commands.l3_handler import L3CommandHandler
from iris_memory.commands.base import ParsedArgs


def _make_manager(persona_id: str = "default"):
    """构建 Mock 组件管理器:返回 AsyncMock 的 L2/L3 适配器与 persona 解析"""
    from iris_memory.core.persona import PersonaResolver

    l2 = Mock()
    l2.is_available = True
    l2.delete_by_user = AsyncMock(return_value=0)
    l2.delete_by_group = AsyncMock(return_value=0)

    l3 = Mock()
    l3.is_available = True
    l3.delete_by_user = AsyncMock(return_value=0)
    l3.delete_by_group = AsyncMock(return_value=0)

    l1 = Mock()
    l1.is_available = True
    l1.clear_by_user = Mock(return_value=0)

    profile = Mock()
    profile.is_available = True
    profile.delete_user_profile = AsyncMock(return_value=True)

    manager = Mock()
    manager.get_component = Mock(
        side_effect=lambda name, cls=None: {"l2_memory": l2, "l3_kg": l3}.get(name, l1 if name == "l1_buffer" else profile)
    )

    # resolve_persona 有 isinstance 检查,必须用真实 PersonaResolver 实例
    persona_resolver = PersonaResolver(Mock())
    persona_resolver.resolve = AsyncMock(return_value=persona_id)
    manager.get_available_component = Mock(return_value=persona_resolver)
    return manager, l2, l3


def _make_event(group_id: str = "g1", user_id: str = "u1"):
    event = Mock()
    adapter = Mock()
    adapter.get_group_id = Mock(return_value=group_id)
    adapter.get_user_id = Mock(return_value=user_id)
    adapter.get_session_id = Mock(return_value=group_id)
    event._adapter = adapter
    return event, adapter


class TestL2HandlerClearPersona:
    @pytest.mark.asyncio
    async def test_clear_passes_resolved_persona(self, monkeypatch):
        manager, l2, _ = _make_manager(persona_id="persona_b")
        event, adapter = _make_event()
        monkeypatch.setattr(
            "iris_memory.commands.l2_handler.get_component_manager",
            lambda: manager,
        )
        monkeypatch.setattr(
            "iris_memory.commands.l2_handler.get_adapter",
            lambda e: adapter,
        )

        args = ParsedArgs()
        result = await L2CommandHandler()._handle_clear(event, args)

        assert result.success
        l2.delete_by_user.assert_awaited_once_with(
            "u1", "g1", persona_id="persona_b"
        )

    @pytest.mark.asyncio
    async def test_clear_group_scope_passes_persona(self, monkeypatch):
        manager, l2, _ = _make_manager(persona_id="persona_b")
        event, adapter = _make_event()
        monkeypatch.setattr(
            "iris_memory.commands.l2_handler.get_component_manager",
            lambda: manager,
        )
        monkeypatch.setattr(
            "iris_memory.commands.l2_handler.get_adapter",
            lambda e: adapter,
        )

        args = ParsedArgs()
        args.is_group_scope = True
        result = await L2CommandHandler()._handle_clear(event, args)

        assert result.success
        l2.delete_by_group.assert_awaited_once_with("g1", persona_id="persona_b")


class TestL3HandlerClearPersona:
    @pytest.mark.asyncio
    async def test_clear_passes_resolved_persona(self, monkeypatch):
        manager, _, l3 = _make_manager(persona_id="persona_b")
        event, adapter = _make_event()
        monkeypatch.setattr(
            "iris_memory.commands.l3_handler.get_component_manager",
            lambda: manager,
        )
        monkeypatch.setattr(
            "iris_memory.commands.l3_handler.get_adapter",
            lambda e: adapter,
        )

        args = ParsedArgs()
        result = await L3CommandHandler()._handle_clear(event, args)

        assert result.success
        l3.delete_by_user.assert_awaited_once_with(
            "u1", "g1", persona_id="persona_b"
        )


class TestAllHandlerClearPersona:
    @pytest.mark.asyncio
    async def test_user_scope_passes_persona_to_l2_l3(self, monkeypatch):
        manager, l2, l3 = _make_manager(persona_id="persona_b")
        event, adapter = _make_event()
        monkeypatch.setattr(
            "iris_memory.commands.all_handler.get_component_manager",
            lambda: manager,
        )
        monkeypatch.setattr(
            "iris_memory.commands.all_handler.get_adapter",
            lambda e: adapter,
        )

        args = ParsedArgs()
        result = await AllCommandHandler()._handle_clear(event, args)

        assert result.success
        l2.delete_by_user.assert_awaited_once_with(
            "u1", "g1", persona_id="persona_b"
        )
        l3.delete_by_user.assert_awaited_once_with(
            "u1", "g1", persona_id="persona_b"
        )
