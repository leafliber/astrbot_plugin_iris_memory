"""Project only proven receipt branches; observation and cleanup are independent."""

from ..errors import ControlError
from ..validation import boolean, exact_fields, identifier, integer
from .ingress import IngressLimits, timestamp


def receipt(value):
    exact_fields(
        value,
        {
            "schema_version",
            "identity",
            "command_version",
            "fingerprint_version",
            "fingerprint",
            "commit_id",
            "recorded_at",
            "result_schema_version",
            "result",
        },
    )
    for key in (
        "schema_version",
        "command_version",
        "fingerprint_version",
        "result_schema_version",
    ):
        integer(value[key], 1, 1)
    exact_fields(
        value["identity"],
        {
            "database_id",
            "owner_namespace",
            "operation_kind",
            "scope_id",
            "operation_key",
        },
    )
    for item in value["identity"].values():
        identifier(item)
    identifier(value["commit_id"])
    identifier(value["fingerprint"])
    timestamp(value["recorded_at"])
    if type(value["result"]) is not dict:
        raise ControlError("INVALID_RECEIPT")
    return value["result"]


def committed(reply):
    if reply.outcome != "COMMITTED":
        return None
    exact_fields(reply.data, {"receipt", "source"})
    if reply.data["source"] not in {"NEW", "EXISTING"}:
        raise ControlError("INVALID_RECEIPT_SOURCE")
    return receipt(reply.data["receipt"])


def acceptance(reply, entry):
    result = committed(reply)
    if result is None:
        return None
    exact_fields(
        result,
        {
            "operation_id",
            "entry_id",
            "batch_id",
            "candidate_id",
            "source_id",
            "terminal",
            "object_refs",
            "history",
            "retired_source_ids",
            "targets",
            "storage_execution",
            "model_adapter",
            "candidate_origin",
            "facts",
        },
    )
    if (
        result["entry_id"] != entry
        or result["terminal"] != "ACCEPTED"
        or result["storage_execution"] != "ACTUAL"
    ):
        raise ControlError("ACCEPTANCE_RECEIPT_MISMATCH")
    identifier(result["operation_id"])
    return result["operation_id"]


def media_fact(reply):
    if reply.outcome == "COMMITTED":
        result = committed(reply)
    elif reply.observed and set(reply.data) == {"value"}:
        result = receipt(reply.data["value"])
    else:
        return None
    exact_fields(
        result,
        {
            "upload_id",
            "blob_id",
            "generation",
            "state",
            "byte_count",
            "targets",
            "change",
        },
    )
    if result["state"] not in {"READY", "UPLOADING", "REUPLOAD_REQUIRED"}:
        raise ControlError("INVALID_MEDIA_RECEIPT")
    identifier(result["upload_id"])
    integer(result["byte_count"], 0, 1048576)
    integer(result["generation"], 0, 2**63 - 1)
    if result["state"] == "READY":
        identifier(result["blob_id"])
    return {k: result[k] for k in ("upload_id", "state", "byte_count")}


def inspection(reply):
    if not reply.observed:
        return None
    value = exact_fields(
        reply.data,
        {
            "upload_id",
            "state",
            "volatile_offset",
            "reupload_required",
            "completion",
            "observed_at_us",
            "progress_durable",
        },
    )
    identifier(value["upload_id"])
    if value["state"] not in {
        "UPLOADING",
        "REUPLOAD_REQUIRED",
        "SEALED",
        "PUBLISHING",
        "READY",
        "ABANDONED",
        "FAULTED",
    }:
        raise ControlError("INVALID_UPLOAD_STATE")
    if value["volatile_offset"] is not None:
        integer(value["volatile_offset"], 0, 1048576)
    boolean(value["reupload_required"])
    if value["progress_durable"] is not False or value["reupload_required"] != (
        value["state"] == "REUPLOAD_REQUIRED"
    ):
        raise ControlError("INVALID_UPLOAD_PROGRESS")
    integer(value["observed_at_us"], 0, 2**63 - 1)
    if value["state"] == "READY":
        completion = value["completion"]
        if (
            type(completion) is not dict
            or completion.get("state") != "READY"
            or completion.get("upload_id") != value["upload_id"]
        ):
            raise ControlError("INVALID_UPLOAD_COMPLETION")
    return value


def configuration_limits(reply):
    if not reply.observed:
        raise ControlError("INGRESS_LIMITS_UNVERIFIED", 409)
    data = exact_fields(reply.data, {"status", "version_id", "values", "birth_version"})
    status = data["status"]
    if (
        type(status) is not dict
        or status.get("state") != "APPLIED"
        or status.get("admission_closed") is not False
        or status.get("published_version") != data["version_id"]
        or status.get("authoritative_version") != data["version_id"]
    ):
        raise ControlError("CONFIGURATION_NOT_APPLIED", 409)
    try:
        content, runtime = data["values"]["content"], data["values"]["runtime"]
        limits = IngressLimits(
            runtime["ingress.event_max_bytes"],
            content["media.event_occurrence_limit"],
            content["media.interpretation_text_max_bytes"],
            content["media.blob_max_bytes"],
            content["media.upload_chunk_bytes"],
        )
    except (KeyError, TypeError):
        raise ControlError("INGRESS_LIMITS_UNVERIFIED", 409) from None
    return data["version_id"], limits
