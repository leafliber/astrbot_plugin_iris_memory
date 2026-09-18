"""Synthetic projections of the reviewed managed implementation; no Core runs."""

HEALTH = {
    "state": "READY",
    "business_ready": True,
    "model_dispatch": "PAUSED",
    "cleanup_pending": False,
    "initialized": True,
    "listener_alive": True,
    "persistent_recovery_complete": True,
    "ws_available": False,
    "notifications_paused": True,
}
CAPABILITIES = {
    "protocol_version": 1,
    "http_envelope_version": 1,
    "entries": ["entry-a"],
    "operations": ["confirm"],
    "route_ids": [],
    "event_types": [],
    "media": {
        "blob_max_bytes": 1048576,
        "chunk_max_bytes": 65536,
        "progress_durable": False,
        "ready_before_reference": True,
    },
    "websocket": {"path": "/api/host/ws", "subprotocol": "iris.communication.v1"},
    "confirmation": "ORIGINAL_KEY_AND_INPUT",
    "unknown_retry": False,
}
STATUS = {
    **HEALTH,
    "instance_id": "instance-a",
    "environment_timezone": "UTC",
    "default_timezone": "UTC",
    "session_id": "session-id",
    "session_revision": 1,
    "account_revision": 1,
    "mode_epoch": None,
    "audit_access": False,
    "capabilities": {"audio_video_qualification": False, "rerank": False},
}


def envelope(data=None, outcome="OBSERVED", cleanup=False):
    result = {"version": 1, "outcome": outcome, "cleanup_pending": cleanup}
    if data is not None:
        result["data"] = data
    return result


def dispatched_timeout(operation_key=None):
    """Exact dispatched-work deadline branch, with no data/error.

    Core 1f2cc1e52d6f3a6fff2d41e97b161804c2295ccc:
    companion_memory/management/managed_http.py:299-303, `if not done`.
    capabilities {} yields null; nested input.key / operation_key is echoed
    for keyed work. Serve with HTTP 202. This fixture never starts Core.
    """
    return {
        "version": 1,
        "outcome": "UNCONFIRMED",
        "state": "UNCONFIRMED",
        "operation_key": operation_key,
        "cleanup_pending": True,
    }
