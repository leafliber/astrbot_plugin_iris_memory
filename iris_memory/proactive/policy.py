"""Deterministic local gates. No HTTP or Provider calls in this module."""

import re
from datetime import datetime, time, timedelta

LEVELS = {
    "低": "low",
    "中": "medium",
    "高": "high",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
DISPLAY = {"low": "低", "medium": "中", "high": "高"}
MOTIVES = ("chime_in", "follow_up", "initiate", "watch", "reminder")


def minute_of_day(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("invalid quiet time")
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def muted(state, now: float) -> bool:
    start, end = minute_of_day(state.quiet_start), minute_of_day(state.quiet_end)
    local = datetime.fromtimestamp(now)
    minute = local.hour * 60 + local.minute
    if start == end:
        return False
    return start <= minute < end if start < end else minute >= start or minute < end


def crossed_quiet(state, since: float, now: float) -> bool:
    """Also detect a quiet window crossed while the computer/event loop was asleep."""
    start, end = minute_of_day(state.quiet_start), minute_of_day(state.quiet_end)
    if not since or now <= since or start == end:
        return False
    first = datetime.fromtimestamp(since).date() - timedelta(days=1)
    last = datetime.fromtimestamp(now).date()
    if (last - first).days > 3:
        return True
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        begin = datetime.combine(day, time(start // 60, start % 60)).timestamp()
        finish = datetime.combine(
            day + timedelta(days=int(start > end)), time(end // 60, end % 60)
        ).timestamp()
        if since < finish and now >= begin:
            return True
    return False


def topic_terms(text: str) -> set[str]:
    terms = set()
    for word in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text.casefold()[:1000]):
        if word.isascii() or len(word) <= 2:
            terms.add(word)
        else:
            terms.update(word[i : i + 2] for i in range(len(word) - 1))
    return terms - {
        "关注",
        "跟进",
        "提醒",
        "需要",
        "可以",
        "一下",
        "今天",
        "明天",
        "事情",
        "一下",
        "the",
        "and",
        "this",
        "that",
    }


def related_goals(
    goals: list[dict], text: str, sender: str, name: str, anchors: list[dict], now: float
) -> list[int]:
    terms = topic_terms(text)
    matches = []
    for goal in goals:
        content = str(goal.get("content", ""))
        people = goal.get("people") or []
        if (
            any(person and (person in people or person in content) for person in (sender, name))
            or topic_terms(content) & terms
            or any(
                a["goal_id"] == goal["id"] and a["expires_at"] > now and sender in a["users"]
                for a in anchors
            )
        ):
            matches.append(goal["id"])
    return matches


def message_motive(
    config, state, *, text: str, named: bool, matches: list[int], now: float
) -> str | None:
    if matches:
        return "follow_up"
    question = bool(re.search(r"[?？]|怎么|如何|为什么|吗[呀啊呢]?|是否|求助", text))
    # Low: only explicit attention; medium: also questions; high: lighter question cues.
    if named or (state.willingness != "low" and question):
        return "chime_in"
    if state.willingness == "high" and any(
        word in text for word in ("大家觉得", "有人知道", "求推荐", "怎么办")
    ):
        return "chime_in"
    if (
        config.proactive_watch_enabled
        and len(text.strip()) >= 10
        and now - state.last_watch >= config.proactive_watch_interval_seconds
    ):
        return "watch"
    return None
