"""Stable platform-instance identities; display names are never identifiers."""

import hashlib
from dataclasses import asdict, dataclass


def digest(*parts):
    return hashlib.sha256("\0".join(str(p) for p in parts).encode()).hexdigest()


@dataclass(frozen=True)
class Identity:
    key: str
    session_key: str
    platform: str
    realm: str
    user: str
    display_name: str
    umo: str
    message_id: str
    is_group: bool

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_event(cls, event):
        platform = event.get_platform_name()
        meta = event.platform_meta
        instance = str(getattr(meta, "id", "") or getattr(meta, "name", platform))
        self_id = str(getattr(event.message_obj, "self_id", ""))
        user = str(event.get_sender_id())
        group = str(event.get_group_id() or "")
        realm = digest(platform, instance, self_id)[:32]
        key = (
            "chat:" + digest(realm, "group" if group else "private", group or user)[:32]
        )
        session = "session:" + digest(key, str(getattr(event, "session_id", "")))[:32]
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        return cls(
            key,
            session,
            platform,
            realm,
            user,
            str(event.get_sender_name() or user),
            str(event.unified_msg_origin),
            message_id,
            bool(group),
        )
