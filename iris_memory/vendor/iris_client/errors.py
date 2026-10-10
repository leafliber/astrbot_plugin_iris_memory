"""Exceptions deliberately avoid including HTTP requests or response bodies in repr."""

from typing import cast

from .types import ErrorField, JSONObject


class IrisError(Exception):
    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        details: JSONObject | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        self.retry_after = retry_after
        self.message = self.details.get("message")
        self.fields = cast(list[ErrorField], self.details.get("fields", []))
        self.retry_after_seconds = self.details.get("retry_after_seconds")
        self.retry_at = self.details.get("retry_at")
        self.canonical_id = self.details.get("canonical_id")
        super().__init__(f"Iris request failed ({status_code or 'transport'}, {code})")


class TransportError(IrisError):
    def __init__(self) -> None:
        super().__init__("network_error")


class ProtocolError(IrisError):
    def __init__(self) -> None:
        super().__init__("invalid_response")
