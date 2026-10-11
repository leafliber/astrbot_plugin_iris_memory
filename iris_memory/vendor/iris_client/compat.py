"""Frozen host v1 boundary (iris-core 43f8221, docs/host-api.md)."""

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from .errors import IrisError, ProtocolError
from .types import JSON, JSONObject, SearchOptions

PREFIX = "/api/v1"


def entry_path(entry_id: str, action: str) -> str:
    if not entry_id or entry_id in {".", ".."} or "/" in entry_id or "\\" in entry_id:
        raise ValueError("entry_id must be a nonempty single path segment")
    return f"{PREFIX}/entries/{quote(entry_id, safe='')}/{action}"


def search_body(options: SearchOptions | None, entry_id: str | None) -> SearchOptions:
    result = (options or {}).copy()
    if entry_id is not None:
        if "entry_id" in result and result["entry_id"] != entry_id:
            raise ValueError("conflicting search entry_id")
        result["entry_id"] = entry_id
    return result


def retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            delay = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def object_response(value: object) -> JSONObject:
    if not isinstance(value, dict):
        raise ProtocolError()
    return value


def redact(value: JSON, token: str) -> JSON:
    if isinstance(value, str):
        return value.replace(token, "[redacted]") if token else value
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    if isinstance(value, dict):
        return {str(redact(k, token)): redact(v, token) for k, v in value.items()}
    return value


def error_response(status: int, body: object, header: str | None, token: str) -> IrisError:
    error = body.get("error", {}) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    details = object_response(redact(error, token))
    code = details.get("code")
    if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code):
        code = {
            400: "invalid_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "revision_conflict",
            413: "payload_too_large",
            429: "rate_limited",
            503: "unavailable",
        }.get(status, "http_error")
    delay = retry_after(header)
    fallback = details.get("retry_after_seconds")
    if delay is None and isinstance(fallback, (int, float)) and not isinstance(fallback, bool):
        if math.isfinite(fallback) and fallback >= 0:
            delay = float(fallback)
    return IrisError(code, status_code=status, details=details, retry_after=delay)
