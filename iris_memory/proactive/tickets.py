"""主动回复 decision ticket 去重与所有权校验。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionTicket:
    ticket_id: str
    event_id: str
    group_id: str
    created_at: float


class DecisionTicketRegistry:
    """保证同群仅一个活动决策、同一事件在 TTL 内至多创建一次。"""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._timeout = max(1.0, float(timeout_seconds))
        self._active: dict[str, DecisionTicket] = {}
        self._seen_events: dict[str, float] = {}

    def cleanup(self, now: float | None = None) -> list[DecisionTicket]:
        current = time.time() if now is None else float(now)
        expired = [
            ticket
            for ticket in self._active.values()
            if current - ticket.created_at > self._timeout
        ]
        for ticket in expired:
            self._active.pop(ticket.group_id, None)
            self._seen_events[ticket.event_id] = current
        cutoff = current - self._timeout
        self._seen_events = {
            event_id: seen_at
            for event_id, seen_at in self._seen_events.items()
            if seen_at >= cutoff
        }
        return expired

    def claim(
        self, group_id: str, event_id: str, now: float | None = None
    ) -> DecisionTicket | None:
        current = time.time() if now is None else float(now)
        self.cleanup(current)
        if not group_id or not event_id:
            return None
        if group_id in self._active or event_id in self._seen_events:
            return None
        ticket = DecisionTicket(
            ticket_id=uuid.uuid4().hex,
            event_id=event_id,
            group_id=group_id,
            created_at=current,
        )
        self._active[group_id] = ticket
        self._seen_events[event_id] = current
        return ticket

    def get(self, group_id: str) -> DecisionTicket | None:
        self.cleanup()
        return self._active.get(group_id)

    def owns(self, group_id: str, ticket_id: str, event_id: str = "") -> bool:
        ticket = self.get(group_id)
        return bool(
            ticket
            and ticket.ticket_id == ticket_id
            and (not event_id or ticket.event_id == event_id)
        )

    def release(self, group_id: str, ticket_id: str = "") -> bool:
        ticket = self._active.get(group_id)
        if ticket is None or (ticket_id and ticket.ticket_id != ticket_id):
            return False
        self._active.pop(group_id, None)
        self._seen_events[ticket.event_id] = time.time()
        return True

    def clear(self) -> None:
        self._active.clear()
        self._seen_events.clear()

