"""DTOs pinned to the reviewed HTTP envelope and managed health implementation."""

from dataclasses import dataclass, field

from ..errors import ControlError
from ..validation import boolean, exact_fields, identifier, integer

OUTCOMES = frozenset(
    {
        "COMMITTED",
        "OBSERVED",
        "ABSENT",
        "UNCONFIRMED",
        "NOT_COMMITTED",
        "REJECTED",
        "FAILED",
    }
)


@dataclass(frozen=True)
class Reply:
    http_status: int
    outcome: str
    cleanup_pending: bool
    data: dict | None = field(repr=False)
    error: dict | None = field(default=None, repr=False)

    @property
    def observed(self):
        return self.http_status == 200 and self.outcome == "OBSERVED"


def envelope(status, value):
    # managed_http.py's dispatched-work timeout is a finite implementation
    # compatibility branch, not a general extension to the OpenAPI envelope.
    timeout_key = type(value) is dict and "operation_key" in value
    exact_fields(
        value,
        {"version", "outcome", "cleanup_pending"},
        {"data", "error", "state"} | ({"operation_key"} if timeout_key else set()),
    )
    if type(value["version"]) is not int or value["version"] != 1:
        raise ControlError("PROTOCOL_VERSION", 502)
    if (
        not isinstance(value["outcome"], str)
        or value["outcome"] not in OUTCOMES
        or ("state" in value and value["state"] != value["outcome"])
    ):
        raise ControlError("PROTOCOL_OUTCOME", 502)
    boolean(value["cleanup_pending"])
    if timeout_key:
        if (
            status != 202
            or value["outcome"] != "UNCONFIRMED"
            or value["cleanup_pending"] is not True
        ):
            raise ControlError("PROTOCOL_TIMEOUT_CONFLICT", 502)
        if value["operation_key"] is not None:
            identifier(value["operation_key"])
        # Validate, then deliberately project out the untrusted echoed key.
        # Only local durable key/input/binding may identify an original operation.
    data, error = value.get("data"), value.get("error")
    if data is not None and type(data) is not dict:
        raise ControlError("PROTOCOL_DATA", 502)
    if error is not None:
        exact_fields(
            error, {"code", "operation", "field", "reason"}, {"cleanup_pending"}
        )
        for key in ("code", "operation", "field", "reason"):
            identifier(error[key])
        if "cleanup_pending" in error:
            boolean(error["cleanup_pending"])
    if value["outcome"] in ("OBSERVED", "COMMITTED") and data is None:
        raise ControlError("PROTOCOL_DATA", 502)
    if status >= 400 and value["outcome"] in ("COMMITTED", "OBSERVED", "ABSENT"):
        raise ControlError("HTTP_OUTCOME_CONFLICT", 502)
    if status == 202 and value["outcome"] != "UNCONFIRMED":
        raise ControlError("HTTP_OUTCOME_CONFLICT", 502)
    return Reply(status, value["outcome"], value["cleanup_pending"], data, error)


HEALTH_FLAGS = {
    "business_ready",
    "cleanup_pending",
    "initialized",
    "listener_alive",
    "persistent_recovery_complete",
    "ws_available",
    "notifications_paused",
}


def health_data(value):
    exact_fields(value, HEALTH_FLAGS | {"state", "model_dispatch"})
    for name in HEALTH_FLAGS:
        boolean(value[name])
    identifier(value["state"])
    if value["model_dispatch"] not in ("ENABLED", "PAUSED"):
        raise ControlError("PROTOCOL_HEALTH", 502)
    return dict(value)


def health(status, value):
    if status != 200:
        raise ControlError("HEALTH_HTTP_ERROR", 502)
    exact_fields(value, {"version", "health"})
    if type(value["version"]) is not int or value["version"] != 1:
        raise ControlError("PROTOCOL_VERSION", 502)
    return health_data(value["health"])


def capabilities(value):
    exact_fields(
        value,
        {
            "protocol_version",
            "http_envelope_version",
            "entries",
            "operations",
            "route_ids",
            "event_types",
            "media",
            "websocket",
            "confirmation",
            "unknown_retry",
        },
    )
    for key in ("protocol_version", "http_envelope_version"):
        if type(value[key]) is not int or value[key] != 1:
            raise ControlError("PROTOCOL_VERSION", 502)
    for key, maximum in [
        ("entries", 8),
        ("operations", 32),
        ("route_ids", 16),
        ("event_types", 4),
    ]:
        values = value[key]
        if type(values) is not list or len(values) > maximum:
            raise ControlError("CAPABILITY_SCHEMA_LIMIT", 502)
        for item in values:
            identifier(item)
        if len(set(values)) != len(values):
            raise ControlError("PROTOCOL_CAPABILITIES", 502)
    if (
        value["confirmation"] != "ORIGINAL_KEY_AND_INPUT"
        or value["unknown_retry"] is not False
    ):
        raise ControlError("PROTOCOL_CONFIRMATION", 502)
    expected_media = {
        "blob_max_bytes": 1048576,
        "chunk_max_bytes": 65536,
        "progress_durable": False,
        "ready_before_reference": True,
    }
    exact_fields(value["media"], expected_media)
    for key, expected in expected_media.items():
        if (
            type(value["media"][key]) is not type(expected)
            or value["media"][key] != expected
        ):
            raise ControlError("PROTOCOL_MEDIA", 502)
    if value["websocket"] != {
        "path": "/api/host/ws",
        "subprotocol": "iris.communication.v1",
    }:
        raise ControlError("PROTOCOL_WS", 502)
    return value


def management_status(value):
    extra = {
        "instance_id",
        "environment_timezone",
        "default_timezone",
        "session_id",
        "session_revision",
        "account_revision",
        "mode_epoch",
        "audit_access",
        "capabilities",
    }
    exact_fields(value, HEALTH_FLAGS | {"state", "model_dispatch"} | extra)
    result = health_data(
        {k: value[k] for k in HEALTH_FLAGS | {"state", "model_dispatch"}}
    )
    result["instance_id"] = identifier(value["instance_id"])
    identifier(value["session_id"])
    for key in ("session_revision", "account_revision", "mode_epoch"):
        if value[key] is not None:
            integer(value[key])
    for key in ("environment_timezone", "default_timezone"):
        if not isinstance(value[key], str) or len(value[key]) > 128:
            raise ControlError("PROTOCOL_STATUS", 502)
    boolean(value["audit_access"])
    exact_fields(value["capabilities"], {"audio_video_qualification", "rerank"})
    for v in value["capabilities"].values():
        boolean(v)
    # Session identities, revisions and environment details never leave the backend.
    return result
