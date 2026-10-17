"""Platform mapping; image transfer is isolated in media.py."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from .config import Config, validate_entry_id
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


class SenderNameCache:
    """Latest observed nicknames, bounded and isolated like core account identities."""

    def __init__(self, limit: int = 1024) -> None:
        self.limit = limit
        self.names: OrderedDict[str, str] = OrderedDict()

    def resolve(self, event: Any, entry: Entry) -> str:
        account = str(event.get_sender_id() or "")
        key = stable_key(entry.platform, account)
        name = usable_sender_name(event)
        if account and name:
            self.names[key] = name
            self.names.move_to_end(key)
            if len(self.names) > self.limit:
                self.names.popitem(last=False)
        elif account and is_notice(event) and key in self.names:
            self.names.move_to_end(key)
            return self.names[key]
        return sender_name(event)


def is_notice(event: Any) -> bool:
    raw = raw_message(event)
    return bool(
        raw.get("post_type") in {"notice", "request", "meta_event"}
        or raw.get("notice_type")
        or raw.get("request_type")
        or any(component_kind(c) == "poke" for c in event.get_messages())
    )


def notice_content(event: Any) -> str:
    # Adapter-generated text/quotes can contain bare account IDs. Describe only
    # the known event type; never turn those placeholders into names or speech.
    raw = raw_message(event)
    if raw.get("sub_type") == "poke" or any(
        component_kind(c) == "poke" for c in event.get_messages()
    ):
        return "有成员戳了戳某人"
    if raw.get("post_type") == "request" or raw.get("request_type"):
        return {
            "group": "收到入群申请或邀请",
            "friend": "收到好友申请",
        }.get(raw.get("request_type"), "收到一条平台申请")
    return {
        "group_increase": "有成员加入了群聊",
        "group_decrease": "有成员离开了群聊",
        "group_recall": "群内有消息被撤回",
        "friend_recall": "有消息被撤回",
        "group_admin": "群管理员状态发生变化",
        "group_ban": "群内发言权限发生变化",
        "group_upload": "有成员上传了文件",
        "friend_add": "新增了好友",
    }.get(raw.get("notice_type"), "发生了一条平台通知")


def input_message(event: Any, entry: Entry, names: SenderNameCache | None = None) -> Message:
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
        "sender": names.resolve(event, entry) if names is not None else sender_name(event),
        "account_id": str(event.get_sender_id() or "") or None,
        "content": content_text(event.get_messages(), descriptions),
        "occurred_at": occurred,
        "dedupe_key": "in:" + stable_key(entry.id, entry.platform, message_id),
        "platform": entry.platform,
        "entry_kind": entry.kind,
        "kind": "message",
    }
    if is_notice(event):
        message["content"] = notice_content(event)
        if not message["sender"]:
            # Host v1 requires sender; core ignores this label for kind=event
            # and binds the message to scene without creating a person/account.
            message.update(kind="event", sender="场景")
            message.pop("account_id", None)
        return message
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


def other_output_message(entry: Entry, text: str, key: str) -> Message:
    message = output_message(entry, text, key)
    message.update(sender="场景", kind="event", content="［机器人账号发送，非角色发言］" + text)
    return message


def raw_message(event: Any) -> dict:
    raw = getattr(event.message_obj, "raw_message", {})
    data = raw if isinstance(raw, dict) else getattr(raw, "raw_data", {})
    return data if isinstance(data, dict) else {}


def usable_sender_name(event: Any) -> str:
    name = str(event.get_sender_name() or "").strip()
    account = str(event.get_sender_id() or "")
    if (
        not name
        or name == account
        or name.isdecimal()
        or name in {"N/A", "未知发送者"}
        or len(name) > 200
        or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in name)
    ):
        return ""
    return name


def sender_name(event: Any) -> str:
    name = str(event.get_sender_name() or "")
    account = str(event.get_sender_id() or "")
    if event.get_platform_name() in {"qq_official", "qq_official_webhook"} and (
        not name or name == account
    ):
        return ("成员_" if event.get_group_id() else "用户_") + stable_key(account)[:8]
    if is_notice(event):
        return usable_sender_name(event)
    return name or account or "未知发送者"
