"""Minimal source-registration, token and ingress-configuration management DTOs."""

from ..errors import ControlError
from ..validation import exact_fields, identifier, integer
from .ingress import text

INGRESS_OPERATIONS = frozenset({"accept", "confirm", "media_upload", "media_inspect"})


def validate_request(action, payload):
    if action in {"connections/hosts/register", "connections/hosts/confirm"}:
        exact_fields(
            payload, {"key", "entry_id", "host_id", "platform_id", "external_entry_id"}
        )
        for name in ("key", "entry_id", "host_id", "platform_id"):
            identifier(payload[name])
        text(payload["external_entry_id"], 512)
    elif action in {"connections/hosts/list", "tokens/list"}:
        exact_fields(payload, {"after"})
        if payload["after"] != "":
            identifier(payload["after"])
    elif action == "tokens/create":
        exact_fields(
            payload, {"key", "host_id", "entries", "operations", "expires_at_us"}
        )
        identifier(payload["key"])
        identifier(payload["host_id"])
        entries, operations = payload["entries"], payload["operations"]
        if type(entries) is not list or not 1 <= len(entries) <= 8:
            raise ControlError("TOKEN_GROUP_LIMIT")
        for entry in entries:
            identifier(entry)
        if len(set(entries)) != len(entries):
            raise ControlError("DUPLICATE_ENTRY")
        if (
            type(operations) is not list
            or not operations
            or any(type(x) is not str for x in operations)
            or not set(operations) <= INGRESS_OPERATIONS
            or len(set(operations)) != len(operations)
        ):
            raise ControlError("TOKEN_SCOPE_NOT_ALLOWED")
        integer(payload["expires_at_us"], 1, 2**63 - 1)
    elif action == "tokens/revoke":
        exact_fields(payload, {"key", "token_id", "expected_revision"})
        identifier(payload["key"])
        identifier(payload["token_id"])
        integer(payload["expected_revision"], 1)
    elif action == "configuration/read":
        exact_fields(payload, {"version_id"})
        if payload["version_id"] is not None:
            identifier(payload["version_id"])
    else:
        raise ControlError("HTTP_ROUTE_NOT_ALLOWED", 403)
