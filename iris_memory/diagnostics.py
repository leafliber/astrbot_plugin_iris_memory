"""Bounded, memory-only diagnostics. No text payload can enter this projection."""

import copy
import hashlib
import math
from collections import OrderedDict, deque
from datetime import datetime, timezone

from .config import Config

CODES = frozenset(
    """unknown_event disabled_or_missing_token invalid_configuration
proactive_initialization_failed decorate_failed container_host_enabled astrbot_context_conflict
pure_at_waiting_conflict host_settings_unavailable pure_at_degraded queue_full
background_request_failed receive_failed output_capture_failed prepare_without_ingest
prepare_degraded send_unconfirmed reminder_inbox_full goal_dedup_result
reminder_send_unconfirmed reminder_poll_failed image_upload_failed image_upload_degraded
proactive_goals_unavailable proactive_task_failed proactive_decision_failed proactive_scheduler_failed
message_received prepare_received prepare_failed prepare_cancelled proactive_call proactive_sent
forward_degraded forward_expanded image_filtered image_cache_hit image_quota_exhausted
image_storage_unavailable image_cache_stale image_processing_unavailable
command_request_failed tool_request_failed""".split()
)
# Routine activity is useful as counters, but must not evict fault evidence.
ROUTINE_CODES = frozenset(
    """disabled_or_missing_token goal_dedup_result message_received prepare_received
proactive_call proactive_sent forward_expanded image_filtered image_cache_hit
image_quota_exhausted""".split()
)
MOTIVES = frozenset(("chime_in", "follow_up", "initiate", "watch", "reminder"))
REASONS = (
    frozenset(
        """other disabled awaiting_message normal_reply quiet_hours cooldown daily_limit
 decision_interval private_care_disabled rolling_limit initiate_interval unanswered_limit
 explicit_reply new_activity no_signal accepted busy quiet_resume automatic_disabled not_silent
 due scheduled waiting reserved probability cancelled failed input_unavailable provider_unavailable
 invalid_decision goal_unavailable watch empty_content send_unconfirmed sent""".split()
    )
    | MOTIVES
)
STAGES = frozenset(
    ("message", "scheduler", "decision_gate", "calls", "sends", "outcome", "reminder")
)
PARTITIONS = ("persona", "memories", "recent_messages", "state", "goals", "hints", "entry_profile")


def reference(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Diagnostics:
    def __init__(self, config: Config):
        self.config = config
        self.started_at = stamp()
        self.counts: dict[str, int] = {}
        self.logs = deque(maxlen=config.diagnostics_log_limit)
        self.prepares = deque(maxlen=config.diagnostics_prepare_limit)
        self.sessions = OrderedDict()

    def record(self, code: str) -> str:
        code = code if isinstance(code, str) and code in CODES else "unknown_event"
        if self.config.diagnostics_enabled:
            self.counts[code] = self.counts.get(code, 0) + 1
            if code not in ROUTINE_CODES:
                self.logs.append({"at": stamp(), "code": code})
        return code

    def prepared(
        self, entry_id: str, data, duration: float, *, purpose: str, outcome: str, judge: bool
    ):
        if not self.config.diagnostics_enabled:
            return
        self.record(
            {
                "ok": "prepare_received",
                "failed": "prepare_failed",
                "cancelled": "prepare_cancelled",
            }.get(outcome, "unknown_event")
        )
        raw = data if isinstance(data, dict) else {}
        counts = {}
        for key in PARTITIONS:
            part = raw.get(key)
            counts[key] = (
                len(part) if isinstance(part, list) else int(isinstance(part, dict) and bool(part))
            )
        profile = raw.get("entry_profile")
        profile = profile if isinstance(profile, dict) else {}
        version = profile.get("version")
        judgment = raw.get("judgment")
        status = judgment.get("status") if isinstance(judgment, dict) else None
        status = (
            status
            if isinstance(status, str) and status in {"applied", "disabled", "degraded"}
            else "unknown"
        )
        hints = raw.get("hints")
        if isinstance(hints, list) and any(
            isinstance(h, dict)
            and h.get("code") == "recall_judgment"
            and h.get("status") == "degraded"
            for h in hints
        ):
            status = "degraded"
        if not judge and status == "unknown":
            status = "disabled"
        self.prepares.append(
            {
                "at": stamp(),
                "entry_ref": reference(entry_id),
                "purpose": purpose if purpose in {"reply", "proactive"} else "unknown",
                "outcome": outcome if outcome in {"ok", "failed", "cancelled"} else "unknown",
                "duration_ms": round(max(0, duration) * 1000, 2)
                if type(duration) in (int, float) and math.isfinite(duration)
                else 0,
                "partitions": counts,
                "judgment": status,
                "entry_profile_version": version
                if type(version) is int and 0 <= version <= 2**63 - 1
                else None,
                "entry_profile_stale": profile.get("possibly_stale")
                if type(profile.get("possibly_stale")) is bool
                else None,
            }
        )

    def proactive(self, origin: str, entry_id: str, stage: str, code: str):
        if not self.config.diagnostics_enabled or stage not in STAGES:
            return
        allowed = MOTIVES if stage in {"calls", "sends"} else REASONS
        code = code if isinstance(code, str) and code in allowed else "other"
        if stage in {"calls", "sends"}:
            self.record("proactive_call" if stage == "calls" else "proactive_sent")
        key = reference(origin)
        row = self.sessions.setdefault(
            key,
            {
                "session_ref": key,
                "entry_ref": reference(entry_id),
                "since": stamp(),
                "counts": {key: {} for key in sorted(STAGES)},
            },
        )
        row["entry_ref"] = reference(entry_id)
        counts = row["counts"][stage]
        counts[code] = counts.get(code, 0) + 1
        self.sessions.move_to_end(key)
        while len(self.sessions) > self.config.diagnostics_session_limit:
            self.sessions.popitem(last=False)

    def reset_session(self, origin: str):
        self.sessions.pop(reference(origin), None)

    def snapshot(self) -> dict:
        return copy.deepcopy(
            {
                "enabled": self.config.diagnostics_enabled,
                "started_at": self.started_at,
                "limits": {
                    "logs": self.logs.maxlen,
                    "prepares": self.prepares.maxlen,
                    "sessions": self.config.diagnostics_session_limit,
                },
                "counts": self.counts,
                "logs": list(self.logs),
                "prepares": list(self.prepares),
                "sessions": list(self.sessions.values()),
            }
        )
