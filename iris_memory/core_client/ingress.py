"""Closed event-v2 carrier. Limits are supplied by an authenticated configuration read."""

import json
import re
from dataclasses import dataclass
from datetime import datetime

from ..errors import ControlError
from ..validation import exact_fields, identifier, integer


@dataclass(frozen=True)
class IngressLimits:
    event_bytes: int
    occurrences: int
    interpretation_bytes: int
    blob_bytes: int
    chunk_bytes: int

    def __post_init__(self):
        integer(self.event_bytes, 256, 8192)
        integer(self.occurrences, 1, 2)
        integer(self.interpretation_bytes, 0, 512)
        integer(self.blob_bytes, 1, 1048576)
        integer(self.chunk_bytes, 1, 65536)


def canonical(value):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        controls = {"b": "0008", "f": "000c", "n": "000a", "r": "000d", "t": "0009"}
        return re.sub(
            r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})',
            lambda m: "\\u" + controls[m[0][1]] if m[0][1] in controls else m[0],
            text,
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ControlError("INVALID_EVENT_ENCODING") from None


def text(value, maximum=512, *, nullable=False, empty=False):
    if value is None and nullable:
        return
    if type(value) is not str or (not value and not empty):
        raise ControlError("INVALID_EVENT_TEXT")
    try:
        if len(value) > maximum or len(value.encode("utf-8")) > maximum:
            raise ControlError("EVENT_TEXT_LIMIT")
    except UnicodeError:
        raise ControlError("INVALID_EVENT_ENCODING") from None


def timestamp(value):
    if value is None:
        return
    text(value, 64)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except ValueError:
        raise ControlError("EVENT_TIME_OFFSET_REQUIRED") from None


def interpretation(value, limit):
    if value is None:
        return
    exact_fields(value, {"status", "text", "source_ref", "coverage"})
    state, body, source = value["status"], value["text"], value["source_ref"]
    if state not in {"COMPLETE", "PARTIAL", "EMPTY", "MISSING", "FAILED", "REFUSED"}:
        raise ControlError("INVALID_INTERPRETATION")
    if state in {"COMPLETE", "PARTIAL"}:
        text(body, limit)
        if not body.strip():
            raise ControlError("INVALID_INTERPRETATION")
    elif (
        state == "EMPTY"
        and body != ""
        or state in {"MISSING", "FAILED"}
        and body is not None
        or state == "REFUSED"
        and body != "敏感信息无法访问"
    ):
        raise ControlError("INVALID_INTERPRETATION")
    if state == "MISSING":
        if source is not None:
            raise ControlError("INVALID_INTERPRETATION")
    else:
        text(source, 128)
        if len(canonical(source)) > 130:
            raise ControlError("INTERPRETATION_SOURCE_LIMIT")
    if body is not None:
        text(body, limit, empty=True)
    coverage = (
        "COMPLETE"
        if state in {"COMPLETE", "EMPTY"}
        else "EXPLICIT_PARTIAL"
        if state == "PARTIAL"
        else "UNSPECIFIED"
    )
    if value["coverage"] != coverage:
        raise ControlError("INVALID_INTERPRETATION_COVERAGE")


def event_v2(value, limits):
    required = {
        "event_version",
        "event_kind",
        "sender",
        "body",
        "quotation",
        "media",
        "correlation",
        "extensions",
    }
    exact_fields(
        value, required, {"external_event_id", "client_event_key", "occurred_at"}
    )
    if ("external_event_id" in value) == ("client_event_key" in value):
        raise ControlError("INVALID_EVENT_IDENTITY")
    if "external_event_id" in value:
        text(value["external_event_id"])
    else:
        identifier(value["client_event_key"])
    if type(value["event_version"]) is not int or value["event_version"] != 2:
        raise ControlError("EVENT_VERSION")
    if value["event_kind"] not in {
        "MESSAGE",
        "PERCEPTION",
        "SELF_OUTPUT",
        "ACTION_RESULT",
    }:
        raise ControlError("INVALID_EVENT_KIND")
    sender = exact_fields(
        value["sender"], {"subject_id", "display_name", "role", "identity_source"}
    )
    text(sender["subject_id"])
    text(sender["display_name"], nullable=True)
    identifier(sender["role"])
    identifier(sender["identity_source"])
    timestamp(value.get("occurred_at"))
    text(value["body"], 8192, empty=True)
    if type(value["quotation"]) is not list or len(value["quotation"]) > 16:
        raise ControlError("QUOTATION_LIMIT")
    for quote in value["quotation"]:
        exact_fields(quote, {"body", "author", "event_id", "occurred_at"})
        text(quote["body"], 8192, empty=True)
        text(quote["author"], nullable=True)
        text(quote["event_id"], nullable=True)
        timestamp(quote["occurred_at"])
    if type(value["media"]) is not list or len(value["media"]) > limits.occurrences:
        raise ControlError("MEDIA_OCCURRENCE_LIMIT")
    for medium in value["media"]:
        exact_fields(
            medium, {"reference_id", "occurrence_id", "modality", "interpretation"}
        )
        identifier(medium["reference_id"])
        identifier(medium["occurrence_id"])
        modality(medium["modality"])
        interpretation(medium["interpretation"], limits.interpretation_bytes)
    if not value["body"] and not value["media"]:
        raise ControlError("EMPTY_EVENT_UNREPRESENTABLE")
    if value["correlation"] is not None:
        exact_fields(value["correlation"], {"correlation_id", "state"})
        identifier(value["correlation"]["correlation_id"])
        if value["correlation"]["state"] not in {
            "INTENDED",
            "PREPARED",
            "EMITTED",
            "RESULT",
        }:
            raise ControlError("INVALID_CORRELATION")
    if type(value["extensions"]) is not dict or value["extensions"]:
        raise ControlError("EVENT_EXTENSIONS_UNSUPPORTED")
    if len(canonical(value)) > limits.event_bytes:
        raise ControlError("EVENT_CANONICAL_LIMIT")
    return value


def modality(value):
    if value not in {"IMAGE", "AUDIO", "VIDEO"}:
        raise ControlError("INVALID_MODALITY")
    return value
