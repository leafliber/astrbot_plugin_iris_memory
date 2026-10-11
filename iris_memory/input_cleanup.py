"""Conservative text normalization; never classify or remove alleged instructions."""

import re
from functools import lru_cache
from typing import Any

from .config import Config

# Preserve line breaks, tabs, Unicode composition, ZWJ/ZWNJ and variation selectors.
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff]")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CQ = re.compile(r"\[CQ:([a-z_]+)(?:,([^\]\r\n]{0,4096}))?\]")
_MENTION = re.compile(r"<@!?([0-9]{1,32})>")
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
            "forward": "[合并转发]",
        }[kind]
    key = "qq" if kind == "at" else "id"
    identifier = fields.get(key, "")
    if kind in {"at", "face", "reply"} and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", identifier):
        if kind == "at":
            return "@全体" if identifier == "all" else "@" + identifier
        return f"[{'表情' if kind == 'face' else '引用消息'}:{identifier}]"
    # Future/structured payloads are left intact rather than guessing their meaning.
    return match.group()


@lru_cache(maxsize=32)
def _repeat_pattern(limit: int) -> re.Pattern:
    # Deliberately exclude identifiers, numbers, whitespace and formatting syntax.
    return re.compile(r"([!?！？。…~～哈呵啊呜])\1{" + str(limit - 1) + r",}")


def clean_text(text: str, config: Config) -> str:
    if not config.input_cleanup_enabled or not text:
        return text
    text = _ANSI.sub("", text)
    text = _CONTROLS.sub(
        "", text.replace("\r\n", "\n").replace("\r", "\n").replace("\x85", "\n")
    ).lstrip("\ufeff")
    # Code examples still lose actual controls, but retain literal platform syntax.
    parts = _CODE.split(text)
    repeat = _repeat_pattern(config.input_cleanup_repeat_limit)
    for index in range(0, len(parts), 2):
        part = _CQ.sub(_platform, parts[index])
        part = _MENTION.sub(r"@\1", part)
        parts[index] = repeat.sub(lambda m: m[1] * 8 + f"[同字符共{len(m[0])}个]", part)
    return "".join(parts)


def clean_message(message: dict, config: Config) -> dict:
    result = dict(message)
    for key in ("content", "quote_content"):
        if isinstance(result.get(key), str):
            result[key] = clean_text(result[key], config)
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
