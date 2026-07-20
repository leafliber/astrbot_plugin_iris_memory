from __future__ import annotations

import contextvars


class ToolContext:
    _ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "iris_tool_group_id", default=None
    )

    def set_context(self, group_id: str) -> None:
        self._ctx_var.set(group_id)

    def clear_context(self) -> None:
        self._ctx_var.set(None)

    @property
    def current_group_id(self) -> str | None:
        return self._ctx_var.get()
