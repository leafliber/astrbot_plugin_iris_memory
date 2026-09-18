import copy

import pytest

from iris_memory.core_client.protocol import (
    OUTCOMES,
    capabilities,
    envelope,
    health,
    management_status,
)
from iris_memory.errors import ControlError
from iris_memory.validation import decode, origin
from tests.fixtures import CAPABILITIES, HEALTH, STATUS
from tests.fixtures import envelope as wire


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_actual_outcomes_keep_cleanup_separate(outcome):
    status = 202 if outcome == "UNCONFIRMED" else 200
    reply = envelope(
        status,
        wire({} if outcome in ("COMMITTED", "OBSERVED") else None, outcome, True),
    )
    assert reply.outcome == outcome and reply.cleanup_pending
    assert reply.observed == (outcome == "OBSERVED")


def test_error_without_data_and_redacted_repr():
    value = wire(outcome="REJECTED")
    value["error"] = dict(
        code="ACCESS_DENIED", operation="http", field="credential", reason="EXPIRED"
    )
    reply = envelope(403, value)
    assert reply.data is None and not reply.observed
    assert "EXPIRED" not in repr(reply)


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"version": True},
        {"outcome": "OK"},
        {"cleanup_pending": 0},
        {"extra": "x"},
    ],
)
def test_closed_envelope_rejects_invalid(change):
    with pytest.raises(ControlError):
        envelope(200, wire({}) | change)


def test_http_status_is_not_commit():
    assert envelope(200, wire({}, "OBSERVED")).outcome == "OBSERVED"
    for status, payload in [
        (202, wire({}, "COMMITTED")),
        (500, wire({}, "OBSERVED")),
        (200, wire()),
    ]:
        with pytest.raises(ControlError):
            envelope(status, payload)


def test_health_and_management_status_are_distinct():
    assert health(200, {"version": 1, "health": HEALTH}) == HEALTH
    with pytest.raises(ControlError):
        health(200, wire(HEALTH))
    result = management_status(STATUS)
    assert result["instance_id"] == "instance-a"
    assert "session_id" not in result
    assert result["model_dispatch"] == "PAUSED"


def test_capabilities_do_not_imply_ready_and_schema_is_closed():
    assert "business_ready" not in capabilities(CAPABILITIES)
    for key, value in [
        ("protocol_version", 2),
        ("entries", ["e"] * 10),
        ("unknown_retry", True),
    ]:
        changed = copy.deepcopy(CAPABILITIES)
        changed[key] = value
        with pytest.raises(ControlError):
            capabilities(changed)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host",
        "http://user:password@host",
        "https://host/path",
        "https://host?token=x",
        "http://host#x",
        "http://host\n",
        "http://host:0",
        "http://host:99999",
        "http://host\\evil",
    ],
)
def test_invalid_origins(url):
    with pytest.raises(ControlError):
        origin(url)


def test_canonical_origin_and_json_strictness():
    assert origin("HTTPS://Example.COM:443/") == "https://example.com"
    assert origin("http://[::1]:8080") == "http://[::1]:8080"
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"):
        with pytest.raises(ControlError):
            decode(raw)
