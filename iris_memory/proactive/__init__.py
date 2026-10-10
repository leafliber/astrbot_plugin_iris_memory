"""Proactive policy boundary; construction is lazy to keep configuration imports pure."""

from typing import Any, Protocol


class ProactiveHooks(Protocol):
    async def start(self) -> None: ...
    async def on_message(self, event: Any) -> None: ...
    async def before_llm(self, event: Any, request: Any) -> None: ...
    async def after_sent(self, event: Any) -> None: ...
    async def close(self) -> None: ...


def create_proactive(plugin: Any) -> ProactiveHooks | None:
    if not plugin.bridge or not plugin.settings:
        return None
    from .engine import Proactive

    return Proactive(plugin)
