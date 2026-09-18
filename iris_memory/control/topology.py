"""Future source/connection control shape, without registration or WS consumption."""

from dataclasses import dataclass

from ..errors import ControlError
from ..validation import identifier


@dataclass(frozen=True)
class SourceBinding:
    platform_instance: str
    bot_self: str
    kind: str
    conversation_id: str
    entry_id: str
    route_id: str | None = None

    def __post_init__(self):
        if self.kind not in ("group", "private"):
            raise ControlError("INVALID_SOURCE_KIND")
        for value in (
            self.platform_instance,
            self.bot_self,
            self.conversation_id,
            self.entry_id,
        ):
            identifier(value)
        if self.route_id:
            identifier(self.route_id)


def plan_subscriptions(sources: tuple[SourceBinding, ...]):
    if len(sources) > 10 or len({s.entry_id for s in sources}) != len(sources):
        raise ControlError("INVALID_SOURCE_TOPOLOGY")
    routes = [s.route_id for s in sources if s.route_id]
    if len(set(routes)) != len(routes):
        raise ControlError("DUPLICATE_ROUTE")
    return [routes[i : i + 8] for i in range(0, len(routes), 8)]
