"""Bounded merged-forward expansion; quoted media stays a count, never an upload."""

import asyncio
import re
from typing import Any

from .config import Config
from .mapping import component_kind, raw_message

MAX_BYTES = 32768
TRUNCATED = "\n[已截断]"
_FORWARD_KINDS = {"forward", "node", "nodes"}
_CQ = re.compile(r"\[CQ:([a-zA-Z_]+)(?:,[^\]]*)?\]")


def _kind(value: Any) -> str:
    return str(value.get("type", "")).lower() if isinstance(value, dict) else component_kind(value)


def _field(value: Any, name: str, default=None):
    if isinstance(value, dict):
        data = value.get("data")
        return (data if isinstance(data, dict) else value).get(name, default)
    return getattr(value, name, default)


def _label(value: Any, default="未知作者") -> str:
    return (
        str(value).replace("\r", " ").replace("\n", " ")[:200]
        if isinstance(value, (str, int)) and value
        else default
    )


def _raw_text(text: str) -> str:
    placeholders = {
        "image": "[图片×1]",
        "record": "[音频]",
        "video": "[视频]",
        "file": "[附件]",
        "face": "[表情]",
        "at": "[提及]",
    }
    return _CQ.sub(lambda m: placeholders.get(m[1].lower(), "[平台组件]"), text)


class _Expansion:
    def __init__(self, event, config, record):
        self.event, self.config, self.record = event, config, record
        self.text = ""
        self.full = False
        self.nodes = self.parts = 0
        self.title = "合并转发"
        self.cache: dict[str, dict] = {}

    def line(self, text: str) -> None:
        if self.full:
            return
        combined = self.text + ("\n" if self.text else "") + text[: MAX_BYTES + 1]
        raw = combined.encode("utf-8")
        if len(raw) > MAX_BYTES:
            self.text = (
                raw[: MAX_BYTES - len(TRUNCATED.encode())].decode("utf-8", errors="ignore")
                + TRUNCATED
            )
            self.full = True
        else:
            self.text = combined

    def step(self) -> bool:
        self.parts += 1
        if self.parts > self.config.forward_max_nodes * 32:
            self.line("[已截断：组件数量上限]")
            self.full = True
        return not self.full

    async def walk(self, item, depth=0, ancestry=()):
        if not self.step():
            return
        kind = _kind(item)
        if kind in _FORWARD_KINDS and depth >= self.config.forward_max_depth:
            self.line("[已截断：嵌套深度上限]")
            return
        if kind in {"nodes", "forward"} and self.title == "合并转发":
            self.title = _label(_field(item, "title"), "合并转发")
        if kind == "nodes":
            nodes = _field(item, "nodes", [])
            if isinstance(nodes, list):
                for child in nodes:
                    if self.full:
                        break
                    await self.walk(child, depth, ancestry)
        elif kind == "node" or (isinstance(item, dict) and "sender" in item):
            await self.node(item, depth + 1, ancestry)
        elif kind == "forward":
            identity = _field(item, "id")
            if not isinstance(identity, (str, int)) or not identity or len(str(identity)) > 300:
                self.line("[合并转发未展开]")
                self.record("forward_degraded")
                return
            identity = str(identity)
            if identity in ancestry:
                self.line("[已截断：循环转发]")
                return
            payload = await self.fetch(identity)
            if self.title == "合并转发":
                self.title = _label(payload.get("title"), "合并转发")
            nodes = payload.get("messages")
            if not isinstance(nodes, list) or not nodes:
                self.line("[合并转发未展开]")
                self.record("forward_degraded")
                return
            for child in nodes:
                if self.full:
                    break
                await self.walk(child, depth, (*ancestry, identity))

    async def fetch(self, identity):
        if identity in self.cache:
            return self.cache[identity]
        payload = {}
        bot = getattr(self.event, "bot", None)
        if self.event.get_platform_name() in {"aiocqhttp", "onebot", "onebot_v11"} and callable(
            getattr(bot, "call_action", None)
        ):
            route = {}
            self_id = getattr(self.event.message_obj, "self_id", None)
            if isinstance(self_id, (str, int)) and self_id:
                route["self_id"] = self_id
            try:
                result = await bot.call_action("get_forward_msg", message_id=identity, **route)
                if isinstance(result, dict):
                    payload = result.get("data", result)
                    if not isinstance(payload, dict):
                        payload = {}
            except Exception:
                self.record("forward_degraded")
        self.cache[identity] = payload
        return payload

    async def node(self, item, depth, ancestry):
        if self.nodes >= self.config.forward_max_nodes:
            self.line("[已截断：消息数量上限]")
            self.full = True
            return
        self.nodes += 1
        sender = _field(item, "sender", {})
        if not isinstance(sender, dict):
            sender = {}
        name = _label(
            sender.get("card")
            or sender.get("nickname")
            or _field(item, "name")
            or _field(item, "nickname")
            or sender.get("user_id")
            or _field(item, "uin")
            or _field(item, "user_id")
        )
        content = _field(item, "content", _field(item, "message", []))
        if isinstance(content, str):
            self.line(name + "：" + _raw_text(content)[: MAX_BYTES + 1])
            return
        if not isinstance(content, list):
            self.line(name + "：[内容未展开]")
            return
        parts, images, chars = [], 0, 0

        def flush():
            nonlocal images, chars
            if images:
                parts.append(f"[图片×{images}]")
            if parts:
                self.line(name + "：" + "".join(parts))
            parts.clear()
            images = chars = 0

        for part in content:
            if not self.step():
                break
            kind = _kind(part)
            if kind in _FORWARD_KINDS:
                flush()
                await self.walk(part, depth, ancestry)
            elif kind == "image":
                images += 1
            else:
                if images:
                    parts.append(f"[图片×{images}]")
                    images = 0
                value = (
                    _field(part, "text", "")
                    if kind in {"plain", "text"}
                    else {
                        "record": "[音频]",
                        "video": "[视频]",
                        "file": "[附件]",
                        "face": "[表情]",
                        "at": "[提及]",
                        "atall": "[提及全体]",
                    }.get(kind, "")
                )
                if isinstance(value, str):
                    value = value[: MAX_BYTES + 1]
                    parts.append(value)
                    chars += len(value)
                    if chars > MAX_BYTES:
                        flush()
        flush()


async def attach_forwarded(event, message: dict, config: Config, record) -> None:
    if not config.forward_enabled:
        return
    roots = [part for part in event.get_messages() if _kind(part) in _FORWARD_KINDS]
    if not roots:
        raw = raw_message(event).get("message", [])
        if isinstance(raw, list):
            roots = [part for part in raw if _kind(part) in _FORWARD_KINDS]
    if not roots:
        return
    expansion = _Expansion(event, config, record)
    previous = message.get("quote_content")
    if previous:
        expansion.line(
            "【原引用】" + _label(message.get("quote_author")) + "：" + previous[: MAX_BYTES + 1]
        )
    try:
        async with asyncio.timeout(config.forward_timeout):
            for part in roots:
                if expansion.full:
                    break
                await expansion.walk(part)
    except Exception:
        expansion.line("[合并转发读取失败或超时]")
        record("forward_degraded")
    message["quote_author"] = expansion.title
    message["quote_author_account_id"] = None
    message["quote_content"] = expansion.text or "[合并转发未展开]"
    if not message.get("content", "").strip():
        message["content"] = "[合并转发]"
    record("forward_expanded")
