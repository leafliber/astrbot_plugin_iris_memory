"""Bounded snapshots of public OneBot message material, without changing the event."""

import uuid
from datetime import datetime, timezone

from ..control.sources import source_identity
from ..errors import ControlError
from ..validation import encode

COVERAGE = [
    {
        "shape": "文本／中文／控制字符",
        "state": "partial",
        "reason": "保留 raw_message 文本段；超出 Core 规范字节预算时保存受阻材料",
    },
    {
        "shape": "图片／音频／视频",
        "state": "partial",
        "reason": "仅事件自带 http(s) 原件；每事件最多两项且须媒体 READY",
    },
    {
        "shape": "引用",
        "state": "partial",
        "reason": "保留事件自带引用 ID；宿主提供的纯文本引用链可映射，复杂引用保留缺口",
    },
    {
        "shape": "合并转发／纯 @／混合 @",
        "state": "unsupported",
        "reason": "事件 v2 无原生平台结构字段；不伪造正文",
    },
    {
        "shape": "编辑／撤回／通知",
        "state": "unsupported",
        "reason": "无合法原始事件修订协议；宿主不一定交付公开 handler",
    },
    {
        "shape": "超长消息／超限原件",
        "state": "unsupported",
        "reason": "不截断、不压缩、不拆分冒充完整交付",
    },
    {
        "shape": "机器人生成／发送后回调",
        "state": "diagnostic_only",
        "reason": "仅证明相应回调阶段，不能证明平台送达",
    },
    {
        "shape": "带平台 ID 的 message_sent 回声",
        "state": "partial",
        "reason": "DTO 可表达；本基线 aiocqhttp 不处理 message_sent，真实宿主路径未可达；不按正文关联",
    },
    {
        "shape": "过滤前／停机期间／平台未提供历史",
        "state": "unavailable",
        "reason": "公开 handler 无法补足，完整捕获保证未实现",
    },
]


def freeze(value, maximum=262144):
    remaining, nodes = maximum, 2048

    def own(item, depth=0):
        nonlocal remaining, nodes
        nodes -= 1
        if nodes < 0 or depth > 8:
            raise ControlError("RAW_STRUCTURE_LIMIT")
        if item is None or type(item) in (bool, int, float):
            remaining -= 24
            result = item
        elif type(item) is str:
            if len(item) > remaining:
                raise ControlError("RAW_RECORD_LIMIT")
            remaining -= len(item.encode("utf-8")) + 8
            result = item
        elif isinstance(item, dict) and len(item) <= 64:
            result = {}
            for key, val in item.items():
                if type(key) is not str:
                    raise ControlError("RAW_STRUCTURE_UNSUPPORTED")
                result[own(key, depth + 1)] = own(val, depth + 1)
        elif type(item) in (list, tuple) and len(item) <= 128:
            result = [own(val, depth + 1) for val in item]
        else:
            raise ControlError("RAW_STRUCTURE_UNSUPPORTED")
        if remaining < 0:
            raise ControlError("RAW_RECORD_LIMIT")
        return result

    result = own(value)
    encode(result, maximum)
    return result


def identity(event):
    if event.get_platform_name() != "aiocqhttp":
        raise ControlError("PLATFORM_UNSUPPORTED")
    platform = event.platform_meta.id
    bot = str(event.get_self_id())
    kind = "private" if event.is_private_chat() else "group"
    conversation = (
        str(event.get_sender_id()) if kind == "private" else str(event.get_group_id())
    )
    return source_identity(platform, bot, kind, conversation)


def snapshot(event, maximum):
    raw = freeze(event.message_obj.raw_message, maximum)
    if type(raw) is not dict:
        raise ControlError("RAW_STRUCTURE_UNSUPPORTED")
    material = {"raw": raw, "event": None, "media": [], "stage": "public_handler"}
    platform_id = raw.get("message_id")
    external = (
        str(platform_id)
        if type(platform_id) in (str, int) and str(platform_id)
        else None
    )
    sender = raw.get("user_id")
    if type(sender) not in (str, int):
        return material, external, "SENDER_ID_UNAVAILABLE"
    if raw.get("post_type") not in {"message", "message_sent"}:
        return material, external, "PLATFORM_MUTATION_UNSUPPORTED"
    own = str(sender) == str(event.get_self_id())
    if own and (raw.get("post_type") != "message_sent" or external is None):
        return material, external, "SELF_OUTPUT_STAGE_UNPROVEN"
    chain = raw.get("message")
    if type(chain) is not list:
        # A CQ string may encode structured elements; do not reinterpret it as plain text.
        return material, external, "RAW_CHAIN_UNAVAILABLE"
    body, media, quotes, reason = [], [], [], None
    quote_budget = maximum
    for component in chain:
        if (
            type(component) is not dict
            or set(component) != {"type", "data"}
            or type(component["data"]) is not dict
        ):
            reason = "RAW_COMPONENT_UNSUPPORTED"
            break
        kind, data = component["type"], component["data"]
        if kind == "text" and set(data) == {"text"} and type(data["text"]) is str:
            body.append(data["text"])
        elif kind == "reply" and set(data) == {"id"} and type(data["id"]) in (str, int):
            quote = {
                "body": "",
                "author": None,
                "event_id": str(data["id"]),
                "occurred_at": None,
            }
            host_chain = event.get_messages()
            if type(host_chain) is not list or len(host_chain) > 128:
                reason = "QUOTATION_MATERIAL_LIMIT"
                break
            for item in host_chain:
                if type(item).__name__ == "Reply" and str(item.id) == str(data["id"]):
                    chain = getattr(item, "chain", None)
                    if chain:
                        if len(chain) > 128 or any(
                            type(part).__name__ != "Plain" for part in chain
                        ):
                            reason = "COMPLEX_QUOTATION_UNREPRESENTABLE"
                            break
                        for part in chain:
                            if (
                                type(part.text) is not str
                                or len(part.text) > quote_budget
                            ):
                                raise ControlError("QUOTATION_MATERIAL_LIMIT")
                            quote_budget -= len(part.text.encode("utf-8"))
                            if quote_budget < 0:
                                raise ControlError("QUOTATION_MATERIAL_LIMIT")
                        quote["body"] = "".join(part.text for part in chain)
                        quote["author"] = (
                            str(item.sender_id) if item.sender_id else None
                        )
                    # Adapter's reply.time is local conversion time, not a platform timestamp.
            quotes.append(quote)
        elif kind in {"image", "record", "video"}:
            url = data.get("url")
            if (
                type(url) is not str
                or len(url) > 4096
                or not url.startswith(("https://", "http://"))
            ):
                reason = "MEDIA_RESOURCE_UNAVAILABLE"
                break
            media.append(
                {
                    "id": str(uuid.uuid4()),
                    "url": url,
                    "modality": {"image": "IMAGE", "record": "AUDIO", "video": "VIDEO"}[
                        kind
                    ],
                }
            )
        else:
            reason = "PLATFORM_STRUCTURE_UNREPRESENTABLE"
            break
    if len(media) > 2:
        reason = "MEDIA_OCCURRENCE_LIMIT"
    display = (
        raw.get("sender", {}).get("nickname")
        if type(raw.get("sender")) is dict
        else None
    )
    event_value = {
        "event_version": 2,
        "event_kind": "SELF_OUTPUT" if own else "MESSAGE",
        "sender": {
            "subject_id": str(sender),
            "display_name": display,
            "role": "SELF" if own else "USER",
            "identity_source": "ONEBOT_V11",
        },
        "body": "".join(body),
        "quotation": quotes,
        "media": [],
        "correlation": None,
        "extensions": {},
    }
    event_value["external_event_id" if external else "client_event_key"] = (
        external or str(uuid.uuid4())
    )
    when = raw.get("time")
    if type(when) is int:
        try:
            event_value["occurred_at"] = datetime.fromtimestamp(
                when, timezone.utc
            ).isoformat()
        except (ValueError, OverflowError, OSError):
            reason = "PLATFORM_TIME_UNREPRESENTABLE"
    material.update(event=event_value, media=media)
    # IDs depend on occurrence identity, not byte equality. Stable across platform redelivery.
    if external:
        import hashlib

        for index, medium in enumerate(media):
            medium["id"] = hashlib.sha256(
                encode([identity(event), external, index]).encode()
            ).hexdigest()
    return material, external, reason
