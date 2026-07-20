"""PersonaResolver 测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from iris_memory.core.persona import PersonaResolver, resolve_persona, _normalize


def _make_event(umo="umo_1", cached=None):
    event = MagicMock()
    event.unified_msg_origin = umo
    extras = {}
    if cached is not None:
        extras["iris_persona_id"] = cached
    event.get_extra = Mock(
        side_effect=lambda key, default=None: extras.get(key, default)
    )
    event.set_extra = Mock(
        side_effect=lambda key, value: extras.__setitem__(key, value)
    )
    return event


def _make_context(conv_persona_id=None):
    """构造 AstrBot Context：conversation_manager 返回带 persona_id 的会话"""
    context = MagicMock()
    conversation = MagicMock()
    conversation.persona_id = conv_persona_id
    conv_mgr = MagicMock()
    conv_mgr.get_curr_conversation_id = AsyncMock(return_value="cid_1")
    conv_mgr.get_conversation = AsyncMock(return_value=conversation)
    context.conversation_manager = conv_mgr
    return context


class TestNormalize:
    def test_none_returns_empty(self):
        assert _normalize(None) == ""

    def test_default_values_return_empty(self):
        assert _normalize("default") == ""
        assert _normalize("[%None]") == ""
        assert _normalize("") == ""

    def test_real_persona_kept(self):
        assert _normalize("yuki") == "yuki"


class TestPersonaResolverDisabled:
    @pytest.mark.asyncio
    async def test_isolation_disabled_returns_default(self):
        context = _make_context(conv_persona_id="yuki")
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=False)),
            )
            assert resolver.is_enabled() is False
            result = await resolver.resolve(_make_event())
        assert result == "default"


class TestPersonaResolverFromReq:
    @pytest.mark.asyncio
    async def test_resolve_from_req_conversation(self):
        context = _make_context()
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=True)),
            )
            req = Mock()
            req.conversation = Mock(persona_id="yuki")
            result = await resolver.resolve(_make_event(), req)
        assert result == "yuki"

    @pytest.mark.asyncio
    async def test_req_none_persona_falls_back_to_default(self):
        context = _make_context(conv_persona_id=None)
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=True)),
            )
            req = Mock()
            req.conversation = Mock(persona_id=None)
            result = await resolver.resolve(_make_event(), req)
        assert result == "default"


class TestPersonaResolverFromConversation:
    @pytest.mark.asyncio
    async def test_resolve_via_conversation_manager(self):
        context = _make_context(conv_persona_id="aria")
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=True)),
            )
            result = await resolver.resolve(_make_event())
        assert result == "aria"

    @pytest.mark.asyncio
    async def test_conversation_persona_none_returns_default(self):
        context = _make_context(conv_persona_id=None)
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=True)),
            )
            result = await resolver.resolve(_make_event())
        assert result == "default"


class TestPersonaResolverCache:
    @pytest.mark.asyncio
    async def test_cached_value_reused(self):
        """已缓存的 persona 不再查询会话"""
        context = _make_context(conv_persona_id="aria")
        resolver = PersonaResolver(context)
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "iris_memory.config.get_config",
                lambda: Mock(get=Mock(return_value=True)),
            )
            event = _make_event(cached="cached_persona")
            result = await resolver.resolve(event)
        assert result == "cached_persona"
        # 不应触达 conversation_manager
        context.conversation_manager.get_curr_conversation_id.assert_not_called()


class TestResolvePersonaHelper:
    @pytest.mark.asyncio
    async def test_resolver_unavailable_returns_default(self):
        manager = MagicMock()
        manager.get_available_component.return_value = None
        result = await resolve_persona(manager, _make_event())
        assert result == "default"
