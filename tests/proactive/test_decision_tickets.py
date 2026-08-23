from unittest.mock import AsyncMock, Mock

import pytest

from iris_memory.proactive.tickets import DecisionTicketRegistry


def test_ticket_registry_dedupes_event_and_checks_owner():
    registry = DecisionTicketRegistry(timeout_seconds=120)
    ticket = registry.claim("g1", "event-1")
    assert ticket is not None
    assert registry.claim("g1", "event-2") is None
    assert not registry.release("g1", "wrong-ticket")
    assert registry.owns("g1", ticket.ticket_id, "event-1")
    assert registry.release("g1", ticket.ticket_id)
    assert registry.claim("g1", "event-1") is None
    assert registry.claim("g1", "event-2") is not None


def test_ticket_registry_recovers_stale_active_ticket():
    registry = DecisionTicketRegistry(timeout_seconds=10)
    assert registry.claim("g1", "old", now=100.0) is not None
    replacement = registry.claim("g1", "new", now=111.0)
    assert replacement is not None
    assert replacement.event_id == "new"


@pytest.mark.asyncio
async def test_stale_event_cannot_consume_newer_group_ticket():
    from main import IrisMemoryPlugin

    plugin = object.__new__(IrisMemoryPlugin)
    plugin._triggering = {"g1": 1.0}
    plugin._decision_tickets = DecisionTicketRegistry(120)
    current = plugin._decision_tickets.claim("g1", "event-new")
    assert current is not None
    plugin._decision_core = Mock()
    plugin._decision_core.decide = AsyncMock()

    event = Mock()
    event.get_group_id.return_value = "g1"
    event.get_extra.return_value = {
        "motive": "chime_in",
        "provider_id": "provider",
        "event_id": "event-old",
        "ticket_id": "stale-ticket",
    }

    assert await plugin._handle_reply_decision(event) is True
    event.stop_event.assert_called_once()
    plugin._decision_core.decide.assert_not_awaited()
    assert plugin._decision_tickets.owns(
        "g1", current.ticket_id, current.event_id
    )

