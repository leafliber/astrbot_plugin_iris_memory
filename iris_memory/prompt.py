"""Only this projection interprets Iris response partitions / runtime hints."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Any

from .config import SECTIONS, Config
from .input_cleanup import clean_material


def safe_hints(hints: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    fixed = {
        "memory_gaps": "本入口有尚未记住的消息区间。",
        "model_service": "部分模型用途暂不可用；仅根据现有材料回复。",
        "model_paused": "部分模型用途已暂停，接收消息不等于完成学习。",
        "embedding_unconfigured": "未配置向量模型，本次使用全文检索。",
        "embedding_uncalibrated": "向量检索尚未标定，本次使用全文检索。",
        "embedding_fallback": "本次向量检索失败，使用全文检索。",
    }
    for hint in hints:
        code = hint.get("code")
        if code in fixed:
            output.append(fixed[code])
        elif code == "recall_judgment" and hint.get("status") == "degraded":
            output.append("本次召回判断降级，相关记忆未经过完整判断。")
        elif code == "other_entries_pending":
            count = hint.get("entry_count")
            if type(count) is not int or count <= 0 or hint.get("time_basis") != "received_at":
                continue
            try:
                start, end = (datetime.fromisoformat(hint[k]) for k in ("time_from", "time_to"))
                if start.tzinfo is None or end.tzinfo is None or start > end:
                    continue
                start, end = (v.astimezone(timezone.utc).isoformat() for v in (start, end))
            except (TypeError, ValueError, KeyError):
                continue
            output.append(f"另有 {count} 个入口有待学习消息；接收时间为 {start} 至 {end}。")
        # Never pass arbitrary `message`, errors, call diagnostics, or other entries.
    return list(dict.fromkeys(output))


def project(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: value[k] for k in keys if k in value}


def memory_material(memory: dict[str, Any]) -> dict[str, Any]:
    result = project(
        memory,
        (
            "id",
            "content",
            "reason",
            "kind",
            "stance",
            "belief",
            "event_time",
            "speaker_name",
            "speaker_subject_id",
            "about",
            "world",
            "lifecycle",
            "subject_annotations",
        ),
    )
    if "sources" in memory:
        result["sources"] = [
            project(source, ("message_id", "entry_id", "entry_name", "occurred_at"))
            for source in memory["sources"]
            if isinstance(source, dict)
        ]
    return result


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


@dataclass(frozen=True)
class Prompt:
    text: str
    memory_ids: tuple[int, ...]
    reminder_ids: tuple[int, ...]


def render(prepared: dict[str, Any], reminders: list[dict[str, Any]], config: Config) -> Prompt:
    template = Template(config.prompt_template)
    used = set(template.get_identifiers())
    data = {
        "persona": project(
            prepared.get("persona") or {}, ("content", "version", "generated_at", "needs_update")
        ),
        "entry_profile": project(
            prepared["entry_profile"], ("text", "version", "generated_at", "possibly_stale")
        )
        if isinstance(prepared.get("entry_profile"), dict)
        else None,
        "memories": [memory_material(m) for m in prepared.get("memories", [])],
        "recent_messages": [
            project(
                m,
                (
                    "kind",
                    "sender_name",
                    "content",
                    "occurred_at",
                    "unlearned",
                    "quote_author_name",
                    "quote_content",
                ),
            )
            for m in prepared.get("recent_messages", [])
        ],
        "state": project(
            prepared.get("state") or {},
            ("activity", "details", "mood", "started_at", "possibly_stale", "duration_seconds"),
        ),
        "goals": [
            project(
                g,
                (
                    "id",
                    "content",
                    "kind",
                    "state",
                    "deadline",
                    "deadline_at",
                    "overdue",
                    "due_soon",
                ),
            )
            for g in prepared.get("goals", [])
        ],
        "hints": safe_hints(prepared.get("hints", [])),
        "reminders": [
            project(n, ("id", "content", "deadline_at", "reminder_kind")) for n in reminders
        ],
    }
    data = clean_material(data, config)
    # Fair section quotas prevent recent messages from crowding out state/goals.
    # Entire memory/reminder records are either present or absent; never report
    # clipped/omitted memories as used or clipped reminders as offered.
    base = len(template.substitute(dict.fromkeys(SECTIONS, "")))
    occurrences = sum(
        1
        for match in template.pattern.finditer(template.template)
        if match.group("named") or match.group("braced")
    )
    quota = max(0, (config.max_prompt_chars - base) // max(1, occurrences))
    values = dict.fromkeys(SECTIONS, "")
    included: dict[str, Any] = {}
    for key in used:
        item = data[key]
        if key == "entry_profile" and item is None:
            # Optional host extension: old cores / ungenerated profiles render empty.
            included[key] = None
            continue
        if isinstance(item, list):
            kept: list[Any] = []
            candidates = list(reversed(item)) if key == "recent_messages" else item
            for candidate in candidates:
                trial = [candidate, *kept] if key == "recent_messages" else [*kept, candidate]
                if len(encode(trial)) <= quota:
                    kept = trial
            item = kept
        serialized = encode(item)
        # Oversized persona/state is omitted as a whole instead of invalid JSON.
        if len(serialized) > quota:
            item = None
            serialized = "null" if quota >= 4 else ""
        values[key] = serialized
        included[key] = item
    text = template.substitute(values)
    return Prompt(
        text,
        tuple(m["id"] for m in included.get("memories") or [] if m.get("reason") == "relevant"),
        tuple(n["id"] for n in included.get("reminders") or []),
    )
