"""Platform mapping; image transfer is isolated in media.py."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from .config import Config
from .vendor.iris_client.compat import validate_entry_id
from .vendor.iris_client.types import Message


def stable_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


@dataclass(frozen=True)
class Entry:
    id: str
    platform: str
    kind: str
    origin: str


def map_entry(event: Any, config: Config) -> Entry:
    platform = str(event.get_platform_name())
    platform_id = str(event.get_platform_id())
    group = str(event.get_group_id() or "")
    raw = raw_message(event)
    if platform in {"qq_official", "qq_official_webhook"}:
        group = str(raw.get("group_openid") or raw.get("channel_id") or group)
    kind = "group" if group else "private"
    session_id = group or str(event.get_sender_id() or event.get_session_id())
    origin = str(event.unified_msg_origin)
    values = {
        "prefix": config.entry_prefix,
        "platform": quote(platform, safe=""),
        "platform_id": quote(platform_id, safe=""),
        "kind": kind,
        "session_id": quote(session_id, safe=""),
    }
    entry_id = config.entry_overrides.get(origin) or Template(config.entry_template).substitute(
        values
    )
    if len(entry_id) > 200:
        entry_id = config.entry_prefix + "sha256:" + stable_key(entry_id)
    validate_entry_id(entry_id)
    namespace = f"astrbot:{quote(platform, safe='')}:{quote(platform_id, safe='')}"
    if platform in {"qq_official", "qq_official_webhook"}:
        # QQ group / C2C / guild IDs are separate identity spaces.
        scene = "guild" if raw.get("channel_id") or raw.get("guild_id") else kind
        namespace += ":" + scene
    if len(namespace) > 100:
        namespace = "astrbot:" + stable_key(namespace)
    return Entry(entry_id, namespace, kind, origin)


def component_kind(component: Any) -> str:
    value = getattr(component, "type", "")
    return str(getattr(value, "value", value)).lower() or type(component).__name__.lower()


def content_text(chain: list[Any], descriptions: list[str] | None = None) -> str:
    parts: list[str] = []
    images = iter(descriptions or [])
    for component in chain:
        kind = component_kind(component)
        if kind == "plain":
            parts.append(str(component.text))
        elif kind == "image":
            # Only trusted host-provided text, never image URLs/files/base64.
            description = next(images, "")
            parts.append(f"[图片，宿主理解：{description}]" if description else "[图片，未理解]")
        elif kind in {"record", "video", "file"}:
            parts.append(
                {"record": "[音频，未理解]", "video": "[视频，未理解]", "file": "[附件]"}[kind]
            )
        elif kind in {"face", "at", "atall"}:
            parts.append({"face": "[表情]", "at": "[提及]", "atall": "[提及全体]"}[kind])
    return "".join(parts)


def input_message(event: Any, entry: Entry) -> Message:
    message_id = str(getattr(event.message_obj, "message_id", "") or "")
    # No stable platform ID means no promise of dedupe across process restarts.
    if not message_id:
        message_id = event.get_extra("iris_fallback_id") or uuid4().hex
        event.set_extra("iris_fallback_id", message_id)
    stamp = getattr(event.message_obj, "timestamp", None)
    try:
        occurred = datetime.fromtimestamp(float(stamp), timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        occurred = datetime.now(timezone.utc).isoformat()
    descriptions = event.get_extra("iris_image_descriptions", [])
    if not isinstance(descriptions, list) or not all(isinstance(v, str) for v in descriptions):
        descriptions = []
    message: Message = {
        "sender": sender_name(event),
        "account_id": str(event.get_sender_id()) or None,
        "content": content_text(event.get_messages(), descriptions),
        "occurred_at": occurred,
        "dedupe_key": "in:" + stable_key(entry.id, entry.platform, message_id),
        "platform": entry.platform,
        "entry_kind": entry.kind,
        "kind": "message",
    }
    for component in event.get_messages():
        if component_kind(component) == "reply":
            message["quote_author"] = str(getattr(component, "sender_nickname", "") or "未知引用者")
            message["quote_author_account_id"] = (
                str(getattr(component, "sender_id", "") or "") or None
            )
            message["quote_content"] = str(
                getattr(component, "message_str", "") or getattr(component, "text", "") or ""
            )
            break
    return message


def output_message(entry: Entry, text: str, key: str) -> Message:
    return {
        "sender": "我",
        "content": text,
        "kind": "self_output",
        "dedupe_key": "out:" + stable_key(entry.id, key),
        "platform": entry.platform,
        "entry_kind": entry.kind,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def raw_message(event: Any) -> dict:
    raw = getattr(event.message_obj, "raw_message", {})
    data = raw if isinstance(raw, dict) else getattr(raw, "raw_data", {})
    return data if isinstance(data, dict) else {}


def sender_name(event: Any) -> str:
    name = str(event.get_sender_name() or "")
    account = str(event.get_sender_id() or "")
    if event.get_platform_name() in {"qq_official", "qq_official_webhook"} and (
        not name or name == account
    ):
        return ("成员_" if event.get_group_id() else "用户_") + stable_key(account)[:8]
    return name or account or "未知发送者"
