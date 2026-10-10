"""Small AstrBot conveniences, rewritten from v3 behavior without its subsystems."""

import re

from .mapping import component_kind


def pure_mention(event) -> bool:
    own = str(event.get_self_id() or "")
    official = event.get_platform_name() in {"qq_official", "qq_official_webhook"}
    mentioned = False
    for item in event.get_messages():
        kind = component_kind(item)
        if kind == "at" and (
            str(getattr(item, "qq", "")) == own
            or official
            and str(getattr(item, "qq", "")) == "qq_official"
        ):
            mentioned = True
        elif kind == "plain" and not str(item.text).strip():
            continue
        else:
            return False
    return mentioned


async def prepare_pure_mention(context, event, settings) -> None:
    if not settings.pure_at_reply or not pure_mention(event):
        return
    cfg = context.get_config(umo=event.unified_msg_origin)
    if cfg.get("platform_settings", {}).get("empty_mention_waiting", True):
        return
    if event.get_extra("provider_request") is not None:
        return
    manager = context.conversation_manager
    origin = event.unified_msg_origin
    conversation_id = await manager.get_curr_conversation_id(origin)
    if not conversation_id:
        conversation_id = await manager.new_conversation(
            origin, platform_id=event.get_platform_id()
        )
    conversation = await manager.get_conversation(origin, conversation_id)
    if conversation is None:
        return
    # AstrBot's empty-request guard requires a nonempty transport prompt;
    # assemble_context drops whitespace before sending to the model.
    request = event.request_llm(
        prompt=" ",
        session_id=conversation_id,
        contexts=[],
        system_prompt="",
        conversation=conversation,
    )
    event.set_extra("provider_request", request)
    event.set_extra("iris_pure_at", True)


def strip_markdown(text: str) -> str:
    # Protect code so multiplication, underscores, headings in code stay literal.
    protected = []

    def code(match):
        protected.append(match.group(1))
        return f"\x00{len(protected) - 1}\x00"

    text = re.sub(r"```[^\n]*\n(.*?)```", code, text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", code, text)
    text = re.sub(r"!?\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s?)", "", text)
    text = re.sub(r"(?m)^\s{0,3}[-*+]\s+", "• ", text)
    for marker in (r"\*\*", "__", "~~", r"\*", "_"):
        text = re.sub(r"(?<!\w)" + marker + r"(\S(?:.*?\S)?)" + marker + r"(?!\w)", r"\1", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)


def friendly_error(text: str) -> str | None:
    # Don't rewrite ordinary model answers about APIs or software errors.
    signatures = ("AstrBot 请求失败", "错误类型:", "错误信息:", "请在平台日志查看")
    if sum(s in text for s in signatures) < 2:
        return None
    lower = text.lower()
    if any(s in lower for s in ("429", "rate limit", "限流")):
        return "请求有些频繁，请稍后再试。"
    if any(s in lower for s in ("timeout", "network", "连接", "网络")):
        return "网络暂时不太稳定，请稍后再试。"
    return "刚才的请求遇到了一点问题，请稍后再试。"


def decorate_result(event, settings, host_config) -> None:
    result = event.get_result()
    if result is None:
        return
    plain = [item for item in result.chain if component_kind(item) == "plain"]
    text = "".join(str(item.text) for item in plain)
    if settings.error_friendly and not result.is_llm_result():
        friendly = friendly_error(text)
        if friendly:
            result.chain.clear()
            result.message(friendly)
            return
    if not settings.strip_markdown:
        return
    use_t2i = getattr(result, "use_t2i_", None)
    if use_t2i is True:
        return
    try:
        threshold = max(50, int(host_config.get("t2i_word_threshold", 150)))
    except (TypeError, ValueError):
        threshold = 150
    if use_t2i is None and host_config.get("t2i", False) and len(text) > threshold:
        return
    for item in plain:
        item.text = strip_markdown(str(item.text))
