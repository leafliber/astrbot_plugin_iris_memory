"""Conservative text normalization; never classify or remove alleged instructions."""

import re
from functools import lru_cache
from typing import Any

from .config import Config

# Preserve line breaks, tabs, Unicode composition, ZWJ/ZWNJ and variation selectors.
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff]")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CQ = re.compile(r"\[CQ:([a-z_]+)(?:,([^\]\r\n]{0,4096}))?\]")
_CODE = re.compile(r"(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)")
_TEXT_KEYS = {
    "content",
    "quote_content",
    "text",
    "activity",
    "details",
    "mood",
    "sender_name",
    "quote_author_name",
    "speaker_name",
    "entry_name",
}


def _platform(match: re.Match) -> str:
    kind, params = match.groups()
    fields = dict(part.split("=", 1) for part in (params or "").split(",") if "=" in part)
    if kind in {"image", "record", "video", "file", "forward"}:
        return {
            "image": "[图片]",
            "record": "[音频]",
            "video": "[视频]",
            "file": "[附件]",
            "forward": "[转发消息]",
        }[kind]
    if kind == "at":
        return "[提及全体]" if fields.get("qq") == "all" else "[提及]"
    if kind in {"face", "reply"}:
        return "[表情]" if kind == "face" else "[引用消息]"
    return "[平台组件]"


@lru_cache(maxsize=32)
def _repeat_pattern(limit: int) -> re.Pattern:
    # Deliberately exclude identifiers, numbers, whitespace and formatting syntax.
    return re.compile(r"([!?！？。…~～哈呵啊呜])\1{" + str(limit - 1) + r",}")


def clean_text(text: str, config: Config, *, raw_cq: bool = False) -> str:
    """raw_cq requires known OneBot wire data, never Plain/quote/Iris text."""
    if not config.input_cleanup_enabled or not text:
        return text
    if raw_cq:
        # Convert wire segments BEFORE unescaping user literals. Escaped brackets
        # must not turn an actual text example into an image/mention placeholder.
        text = _CQ.sub(_platform, text)
        text = text.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "&")
    text = _ANSI.sub("", text)
    text = _CONTROLS.sub(
        "", text.replace("\r\n", "\n").replace("\r", "\n").replace("\x85", "\n")
    ).lstrip("\ufeff")
    # Code examples still lose actual controls, but retain literal platform syntax.
    parts = _CODE.split(text)
    repeat = _repeat_pattern(config.input_cleanup_repeat_limit)
    for index in range(0, len(parts), 2):
        parts[index] = repeat.sub(lambda m: m[1] * 8 + f"[同字符共{len(m[0])}个]", parts[index])
    return "".join(parts)


def _message_limit(text: str) -> str:
    # Host v1 limits both content fields to UTF-8 bytes, including our marker.
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= 32768:
        return text
    marker = "[已截断]"
    return encoded[: 32768 - len(marker.encode("utf-8"))].decode("utf-8", errors="ignore") + marker


def clean_message(message: dict, config: Config) -> dict:
    result = dict(message)
    for key in ("content", "quote_content"):
        if isinstance(result.get(key), str):
            result[key] = _message_limit(clean_text(result[key], config))
    if message.get("content") and not result.get("content"):
        result["content"] = "[空消息]"
    return result


def clean_material(value: Any, config: Config) -> Any:
    """Copy only projected fields; preserve metadata IDs, dates, enums and ordering."""
    if isinstance(value, list):
        return [clean_material(item, config) for item in value]
    if isinstance(value, dict):
        return {
            key: clean_text(item, config)
            if key in _TEXT_KEYS and isinstance(item, str)
            else clean_material(item, config)
            for key, item in value.items()
        }
    return value
