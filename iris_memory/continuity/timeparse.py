"""相对时间解析：把口语时间表达解析为时间窗口。

所有相对表达都以 **原始消息时间**（``base_ts``）在指定时区（默认
Asia/Shanghai）下的本地日期为基准，而不是以解析发生的时刻为基准。

设计原则：绝不编造精确时间。返回的永远是一个窗口 ``[start_ts, end_ts)``，
``precision`` 描述其模糊程度：

- ``day``          ：某一整天（今天/明天/周五/15号…）
- ``part_of_day``  ：某天中的一段（下午、晚上；"3点" 也落在此档，窗口为 ±1h）
- ``range``        ：确定边界的多日区间（周末、下周、下个月、月底、月初）
- ``fuzzy``        ：边界本身是估计（过几天、这几天、最近）

无法识别 → ``None``。

过去时（``prefer_past``）：X月X号 / X号 / 裸“周X”默认取“今天或之后”的下一次出现；
``prefer_past=True``（或文本自带过去标记：了/的/过/考完/面完/已经/昨天/刚…，
且无“要/会/准备”等将来标记）时改取最近一次已过去（含今天）的出现。
即使没有过去标记，X月X号若要向后滚动 >300 天也视为过去日期；滚动距离
超过阈值而改取过去日期时 precision 记为 ``fuzzy``（X月X号：无过去语境、仅因
滚动 >300 天；X号：过去语境下滚动 >25 天）。
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from astrbot.api import logger

__all__ = ["ParsedTime", "parse", "has_past_marker", "has_future_marker"]

DEFAULT_TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class ParsedTime:
    start_ts: float
    end_ts: float
    precision: str  # 'day' | 'part_of_day' | 'range' | 'fuzzy'
    basis: str  # 原文中命中的子串（多个片段时为覆盖它们的最短原文子串）


# ---------------------------------------------------------------------------
# 词表 / 正则（模块级编译）
# ---------------------------------------------------------------------------

_CN_NUM_CHARS = "零〇一二两三四五六七八九十"
_N = rf"[0-9０-９]{{1,2}}|[{_CN_NUM_CHARS}]{{1,3}}"
_NUM_BEFORE = rf"0-9０-９{_CN_NUM_CHARS}"

# X月X日 / X月X号
_MONTH_DAY_RE = re.compile(rf"(?<![{_NUM_BEFORE}])(?P<m>{_N})月(?P<d>{_N})[日号]")
# 下个月X号
_NEXT_MONTH_DAY_RE = re.compile(rf"下个?月(?P<d>{_N})[日号]")
# X号 / X日（前面不能是 月/周/期/拜 或数字，避免与上面冲突、避免“星期日”）
_DAY_OF_MONTH_RE = re.compile(rf"(?<![月周期拜{_NUM_BEFORE}])(?P<d>{_N})[日号]")
# X月底 / X月初
_MONTH_PART_RE = re.compile(rf"(?<![{_NUM_BEFORE}个])(?P<m>{_N})月(?P<part>底|末|初)")
# 下个月(底/初) / 这个月 / 本月 / 月底 / 月初
_MONTH_SPAN_RE = re.compile(
    r"(?P<which>下个?月|这个?月|本月|(?<![0-9０-９零〇一二两三四五六七八九十个上])月(?=[底末初]))"
    r"(?P<part>底|末|初)?"
)

# 日词：长词在前（大后天先于后天）
_DAYWORD_RE = re.compile(
    r"大后天|大前天|今天|今日|今儿|今早|今晚|明天|明日|明儿|明早|明晚|后天|昨天|昨日|昨晚|前天"
)
_DAYWORD_OFFSET = {
    "大后天": 3, "大前天": -3, "今天": 0, "今日": 0, "今儿": 0, "今早": 0, "今晚": 0,
    "明天": 1, "明日": 1, "明儿": 1, "明早": 1, "明晚": 1, "后天": 2,
    "昨天": -1, "昨日": -1, "昨晚": -1, "前天": -2,
}
_MINGHOU_RE = re.compile(r"明后天|明后两天")
# 日词自带的时段
_DAYWORD_PART = {"今早": "早上", "今晚": "晚上", "明早": "早上", "明晚": "晚上", "昨晚": "晚上"}

# 周X / 星期X / 礼拜X（带可选前缀）
_WEEKDAY_RE = re.compile(
    r"(?<![0-9０-９一二两三四五六七八九十几每])"
    r"(?P<pre>下下个?|下个?|这个?|本|上个?)?(?:周|星期|礼拜)(?P<w>[一二三四五六日天1-7])"
)
_WEEKDAY_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
                  "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
# 周末
_WEEKEND_RE = re.compile(r"(?<![每])(?P<pre>下个?|这个?|本|上个?)?(?:周末|礼拜六日)")
# 这周 / 下周（不跟具体星期）
_WEEK_SPAN_RE = re.compile(r"(?P<pre>下个?|这个?|本)(?:周|星期|礼拜)(?![一二三四五六日天末1-7])")

# 模糊区间：(起始偏移天, 结束偏移天) —— 窗口为 [base+a 天 00:00, base+b+1 天 00:00)
_FUZZY_RE = re.compile(r"过两三天|过两天|过几天|过些天|过段时间|过阵子|这几天|这两天|最近|近期")
_FUZZY_SPAN = {
    "过两天": (2, 3), "过两三天": (2, 3), "过几天": (2, 5), "过些天": (3, 7),
    "过段时间": (7, 30), "过阵子": (7, 30),
    "这几天": (0, 3), "这两天": (0, 2), "最近": (0, 7), "近期": (0, 7),
}

# 时段（小时，可超过 24 表示跨到次日）
_PART_RE = re.compile(r"凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|半夜")
_PART_WINDOW = {
    "凌晨": (0, 6), "清晨": (5, 8), "早上": (6, 9), "早晨": (6, 9), "上午": (8, 12),
    "中午": (11, 14), "下午": (13, 18), "傍晚": (17, 19), "晚上": (18, 24),
    "夜里": (21, 24), "半夜": (23, 27),
}
_PM_PARTS = frozenset({"下午", "傍晚", "晚上", "夜里"})

# X点 / X点半 / X点钟。排除“有一点/这一点/一点点”等非时间用法；
# “早一点/晚一点”在解析时对中文数字另行排除（“早8点”仍可用）。
_HOUR_RE = re.compile(
    rf"(?<![有快慢多少好差这那])(?P<h>[0-9０-９]{{1,2}}|[{_CN_NUM_CHARS}]{{1,3}})点(?!点)(?P<half>半)?(?P<zhong>钟)?"
)

_CN_HOUR_BAD_PREV = frozenset("早晚再稍得")

# 过去 / 将来标记（只用于决定 X月X号、X号、裸周X 的滚动方向）
_PAST_MARKER_RE = re.compile(
    r"已经|昨天|昨晚|前天|刚[刚才]?|考完|面完|[考面交做办去见完比聊谈测][了过的](?![解])|试[了过]"
)
_FUTURE_MARKER_RE = re.compile(r"要|会|将|准备|打算|计划|快[要到]|还没|等到|下[个周次]|明天|后天|出(?:成绩|结果|分)|公布|放榜|通知")
_MD_PAST_ROLL_DAYS = 300
_D_PAST_ROLL_DAYS = 25


def has_past_marker(text: str | None) -> bool:
    """文本含过去标记且不含将来标记（“周五要考的”不算过去）。"""
    if not text or not isinstance(text, str):
        return False
    return bool(_PAST_MARKER_RE.search(text)) and not _FUTURE_MARKER_RE.search(text)


def has_future_marker(text: str | None) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(_FUTURE_MARKER_RE.search(text))


_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _num(s: str) -> int | None:
    """解析 0–99 的阿拉伯/中文数字；无法解析返回 None。"""
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    if "十" in s:
        a, _, b = s.partition("十")
        if "十" in b or len(a) > 1 or len(b) > 1:
            return None
        tens = _CN_DIGITS.get(a) if a else 1
        ones = _CN_DIGITS.get(b) if b else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    return None


# ---------------------------------------------------------------------------
# 日期工具
# ---------------------------------------------------------------------------


def _add_months(y: int, m: int, n: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + n
    return idx // 12, idx % 12 + 1


def _first_of_next_month(d: date) -> date:
    y, m = _add_months(d.year, d.month, 1)
    return date(y, m, 1)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


class _Span:
    __slots__ = ("start", "end")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


def _basis(text: str, spans: list[_Span]) -> str:
    lo = min(s.start for s in spans)
    hi = max(s.end for s in spans)
    return text[lo:hi]


# ---------------------------------------------------------------------------
# 锚点解析：返回 (kind, payload, span)
#   kind = 'day'   → payload = date
#   kind = 'range' → payload = (start_date, end_date_exclusive, precision)
# ---------------------------------------------------------------------------


def _find_anchor(text: str, today: date, prefer_past: bool = False):
    # 1. X月X日
    m = _MONTH_DAY_RE.search(text)
    if m:
        mon, day = _num(m.group("m")), _num(m.group("d"))
        if mon is None or day is None or not (1 <= mon <= 12) or not (1 <= day <= 31):
            return None
        fwd = next(
            (d for y in range(today.year, today.year + 9)
             if (d := _safe_date(y, mon, day)) is not None and d >= today),
            None,
        )
        past = next(
            (d for y in range(today.year, today.year - 9, -1)
             if (d := _safe_date(y, mon, day)) is not None and d <= today),
            None,
        )
        far = fwd is not None and (fwd - today).days > _MD_PAST_ROLL_DAYS
        if past is not None and (prefer_past or far):
            # 只凭“滚动太远”推断为过去时，精度降为 fuzzy
            return "day", past, _Span(*m.span()), ("fuzzy" if far and not prefer_past else "day")
        return ("day", fwd, _Span(*m.span())) if fwd is not None else None

    # 2. 下个月X号
    m = _NEXT_MONTH_DAY_RE.search(text)
    if m:
        day = _num(m.group("d"))
        y, mon = _add_months(today.year, today.month, 1)
        d = _safe_date(y, mon, day) if day else None
        return ("day", d, _Span(*m.span())) if d else None

    # 3. X月底 / X月初
    m = _MONTH_PART_RE.search(text)
    if m:
        mon = _num(m.group("m"))
        if mon is None or not (1 <= mon <= 12):
            return None
        for y in (today.year, today.year + 1):
            s, e = _month_part_window(y, mon, m.group("part"))
            if e > today:
                return "range", (s, e, "range"), _Span(*m.span())
        return None

    # 4. 下个月 / 这个月 / 月底 / 月初
    m = _MONTH_SPAN_RE.search(text)
    if m:
        which, part = m.group("which"), m.group("part")
        span = _Span(*m.span())
        if which.startswith("下"):
            y, mon = _add_months(today.year, today.month, 1)
            if part:
                s, e = _month_part_window(y, mon, part)
            else:
                s = date(y, mon, 1)
                e = _first_of_next_month(s)
            return "range", (s, e, "range"), span
        if which == "月":  # 月底/月初（当前月，已过则下月）
            s, e = _month_part_window(today.year, today.month, part)
            if e <= today:
                y, mon = _add_months(today.year, today.month, 1)
                s, e = _month_part_window(y, mon, part)
            return "range", (max(s, today) if s <= today < e else s, e, "range"), span
        # 这个月 / 本月 (+可选 底/初)
        if part:
            s, e = _month_part_window(today.year, today.month, part)
            return "range", (s, e, "range"), span
        return "range", (today, _first_of_next_month(today), "range"), span

    # 5. X号 / X日
    m = _DAY_OF_MONTH_RE.search(text)
    if m:
        day = _num(m.group("d"))
        if day is None or not (1 <= day <= 31):
            return None
        y, mon = today.year, today.month
        fwd = past = None
        for i in range(13):
            yy, mm = _add_months(y, mon, i)
            d = _safe_date(yy, mm, day)
            if d is not None and d >= today:
                fwd = d
                break
        for i in range(13):
            yy, mm = _add_months(y, mon, -i)
            d = _safe_date(yy, mm, day)
            if d is not None and d <= today:
                past = d
                break
        if prefer_past and past is not None:
            far = fwd is not None and (fwd - today).days > _D_PAST_ROLL_DAYS
            return "day", past, _Span(*m.span()), ("fuzzy" if far else "day")
        return ("day", fwd, _Span(*m.span())) if fwd is not None else None

    # 6. 明后天（两天区间） / 今天/明天/后天…
    m = _MINGHOU_RE.search(text)
    if m:
        s = today + timedelta(days=1)
        return "range", (s, s + timedelta(days=2), "range"), _Span(*m.span())
    m = _DAYWORD_RE.search(text)
    if m:
        return "day", today + timedelta(days=_DAYWORD_OFFSET[m.group(0)]), _Span(*m.span())

    # 7. 周X
    m = _WEEKDAY_RE.search(text)
    if m:
        pre = m.group("pre") or ""
        idx = _WEEKDAY_INDEX[m.group("w")]
        monday = today - timedelta(days=today.weekday())
        target = monday + timedelta(days=idx)
        if pre.startswith("下下"):
            target += timedelta(days=14)
        elif pre.startswith("下"):
            target += timedelta(days=7)
        elif pre.startswith("上"):
            target -= timedelta(days=7)
        elif not pre and prefer_past and target > today:
            # 裸“周X” + 过去语境：本周还没到 → 上一次出现
            target -= timedelta(days=7)
        elif not pre and not prefer_past and target < today:
            # 裸“周X”：本周已过 → 下一次出现
            target += timedelta(days=7)
        return "day", target, _Span(*m.span())

    # 8. 周末
    m = _WEEKEND_RE.search(text)
    if m:
        pre = m.group("pre") or ""
        monday = today - timedelta(days=today.weekday())
        sat = monday + timedelta(days=5)
        if pre.startswith("下"):
            sat += timedelta(days=7)
        elif pre.startswith("上"):
            sat -= timedelta(days=7)
        return "range", (sat, sat + timedelta(days=2), "range"), _Span(*m.span())

    # 9. 这周 / 下周
    m = _WEEK_SPAN_RE.search(text)
    if m:
        monday = today - timedelta(days=today.weekday())
        if m.group("pre").startswith("下"):
            s = monday + timedelta(days=7)
            return "range", (s, s + timedelta(days=7), "range"), _Span(*m.span())
        return "range", (today, monday + timedelta(days=7), "range"), _Span(*m.span())

    # 10. 过几天 / 最近…
    m = _FUZZY_RE.search(text)
    if m:
        a, b = _FUZZY_SPAN[m.group(0)]
        return (
            "range",
            (today + timedelta(days=a), today + timedelta(days=b + 1), "fuzzy"),
            _Span(*m.span()),
        )
    return None


def _month_part_window(y: int, mon: int, part: str) -> tuple[date, date]:
    last = calendar.monthrange(y, mon)[1]
    if part == "初":
        return date(y, mon, 1), date(y, mon, 6)  # 1–5 号
    # 底 / 末：最后 7 天
    return date(y, mon, last - 6), _first_of_next_month(date(y, mon, 1))


def _resolve_hour(h: int, part: str | None) -> int | None:
    """把口语小时换算成 0–24 的钟点；无法判定返回 None。"""
    if h > 24:
        return None
    if part in _PM_PARTS and h < 12:
        return h + 12
    if part == "中午" and 1 <= h <= 3:
        return h + 12
    if part in ("晚上", "夜里") and h == 12:
        return 24
    if part in ("凌晨", "半夜") and h == 12:
        return 0 if part == "凌晨" else 24
    if part is None and 1 <= h <= 6:
        # 无时段的 1–6 点按口语习惯视作下午（窗口仍为 ±1h，不精确到分）
        return h + 12
    return h


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def parse(
    text: str, base_ts: float, tz: str = DEFAULT_TZ, prefer_past: bool = False
) -> ParsedTime | None:
    """解析 ``text`` 中的相对时间（以 ``base_ts`` 所在本地日期为基准）。

    ``prefer_past=True``（或 ``text`` 自带过去标记）时，X月X号 / X号 / 裸周X
    取最近一次已过去（含今天）的出现，而不是向后滚动。
    """
    if not text or not isinstance(text, str):
        return None
    try:
        past = bool(prefer_past) or has_past_marker(text)
        return _parse(text, float(base_ts), ZoneInfo(tz), past)
    except Exception as e:  # 解析失败只降级为“无时间”
        logger.warning(f"[continuity] timeparse failed: {type(e).__name__}")
        return None


def _parse(text: str, base_ts: float, zone: ZoneInfo, prefer_past: bool = False) -> ParsedTime | None:
    base_dt = datetime.fromtimestamp(base_ts, zone)
    today = base_dt.date()

    def at(d: date, hours: float = 0.0) -> float:
        return (datetime.combine(d, time(0), tzinfo=zone) + timedelta(hours=hours)).timestamp()

    anchor = _find_anchor(text, today, prefer_past)
    day_precision = "day"
    if anchor and len(anchor) == 4:
        day_precision = anchor[3]
        anchor = anchor[:3]

    # 时段与钟点
    part_m = _PART_RE.search(text)
    part = part_m.group(0) if part_m else None
    day_part = None
    if anchor and anchor[0] == "day":
        day_m = _DAYWORD_RE.match(text, anchor[2].start)
        if day_m and day_m.end() == anchor[2].end:
            day_part = _DAYWORD_PART.get(day_m.group(0))
    eff_part = part or day_part

    hour_m = None
    for hm in _HOUR_RE.finditer(text):
        raw = hm.group("h")
        is_digit = raw[0].isdigit()
        # 中文数字的“X点”歧义大（“一点”），需有日/时段锚点或“半/钟”佐证
        if not is_digit and not (anchor or eff_part or hm.group("half") or hm.group("zhong")):
            continue
        if not is_digit and hm.start() > 0 and text[hm.start() - 1] in _CN_HOUR_BAD_PREV:
            continue
        if _num(raw) is None:
            continue
        hour_m = hm
        break

    # 多日区间：时段/钟点不细化
    if anchor and anchor[0] == "range":
        s, e, prec = anchor[1]
        return ParsedTime(at(s), at(e), prec, text[anchor[2].start:anchor[2].end])

    spans: list[_Span] = []
    if anchor:
        day = anchor[1]
        spans.append(anchor[2])
        explicit_day = True
    else:
        day = today
        explicit_day = False

    if hour_m is not None:
        h = _resolve_hour(_num(hour_m.group("h")), eff_part)
        if h is not None:
            center = h + (0.5 if hour_m.group("half") else 0.0)
            spans.append(_Span(*hour_m.span()))
            if part_m is not None:
                spans.append(_Span(*part_m.span()))
            start, end = at(day, center - 1), at(day, center + 1)
            if not explicit_day and end <= base_ts:
                start, end = at(day + timedelta(days=1), center - 1), at(day + timedelta(days=1), center + 1)
            return ParsedTime(start, end, "part_of_day", _basis(text, spans))

    if eff_part is not None:
        a, b = _PART_WINDOW[eff_part]
        if part_m is not None:
            spans.append(_Span(*part_m.span()))
        start, end = at(day, a), at(day, b)
        if not explicit_day and end <= base_ts:
            start, end = at(day + timedelta(days=1), a), at(day + timedelta(days=1), b)
        return ParsedTime(start, end, "part_of_day", _basis(text, spans))

    if anchor:
        return ParsedTime(at(day), at(day + timedelta(days=1)), day_precision, _basis(text, spans))
    return None
