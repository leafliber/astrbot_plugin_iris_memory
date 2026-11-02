"""timeparse：相对时间解析（Asia/Shanghai，固定基准时间）。"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity.timeparse import (
    ParsedTime,
    parse,
)

SH = ZoneInfo("Asia/Shanghai")


def ts(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=SH).timestamp()


def local(t):
    return datetime.fromtimestamp(t, SH)


# 2026-11-02 是周一；2026-11-07 是周六
MON = ts(2026, 11, 2, 10, 0)
SAT = ts(2026, 11, 7, 15, 0)


def day_window(y, mo, d):
    return ts(y, mo, d), ts(y, mo, d) + 86400


@pytest.mark.parametrize(
    "text,expected",
    [
        ("今天", (2026, 11, 2)),
        ("明天交报告", (2026, 11, 3)),
        ("后天", (2026, 11, 4)),
        ("大后天去医院", (2026, 11, 5)),
        ("昨天", (2026, 11, 1)),
        ("周五交报告", (2026, 11, 6)),
        ("星期三", (2026, 11, 4)),
        ("礼拜天", (2026, 11, 8)),
        ("下周五", (2026, 11, 13)),
        ("这周一", (2026, 11, 2)),
        ("15号", (2026, 11, 15)),
        ("12月3日", (2026, 12, 3)),
        ("下个月5号", (2026, 12, 5)),
        ("二十号", (2026, 11, 20)),
    ],
)
def test_day_precision(text, expected):
    r = parse(text, MON)
    assert isinstance(r, ParsedTime)
    assert r.precision == "day"
    assert (r.start_ts, r.end_ts) == day_window(*expected)
    assert r.basis in text and r.basis


def test_dahoutian_not_confused_with_houtian():
    assert parse("大后天", MON).basis == "大后天"


def test_friday_when_today_is_saturday_rolls_to_next_week():
    r = parse("周五", SAT)
    assert (r.start_ts, r.end_ts) == day_window(2026, 11, 13)
    # 下周五 → 下周的周五，同为 11-13
    assert parse("下周五", SAT).start_ts == ts(2026, 11, 13)
    # 当天的星期 → 今天
    assert parse("周六", SAT).start_ts == ts(2026, 11, 7)


def test_day_of_month_rolls_to_next_month_and_year():
    assert parse("1号", MON).start_ts == ts(2026, 12, 1)
    assert parse("2号", MON).start_ts == ts(2026, 11, 2)  # 当天不滚动
    # 12-20 时说 5号 → 次年 1 月
    assert parse("5号", ts(2026, 12, 20, 9)).start_ts == ts(2027, 1, 5)
    # 11 月没有 31 号 → 下一个有 31 号的月份
    assert parse("31号", MON).start_ts == ts(2026, 12, 31)
    # 已过的月日 → 次年
    assert parse("3月1日", MON).start_ts == ts(2027, 3, 1)


def test_invalid_date_returns_none():
    assert parse("2月30日", MON) is None
    assert parse("45号", MON) is None


def test_afternoon_window():
    r = parse("明天下午开会", MON)
    assert r.precision == "part_of_day"
    assert local(r.start_ts) == datetime(2026, 11, 3, 13, tzinfo=SH)
    assert local(r.end_ts) == datetime(2026, 11, 3, 18, tzinfo=SH)
    assert r.basis == "明天下午"


def test_part_of_day_without_day_rolls_forward_when_passed():
    # 22:00 说“早上” → 次日早上
    r = parse("早上去跑步", ts(2026, 11, 2, 22))
    assert local(r.start_ts) == datetime(2026, 11, 3, 6, tzinfo=SH)
    # 10:00 说“晚上” → 当天晚上
    r = parse("晚上吃火锅", MON)
    assert local(r.start_ts) == datetime(2026, 11, 2, 18, tzinfo=SH)
    assert local(r.end_ts) == datetime(2026, 11, 3, 0, tzinfo=SH)


def test_evening_twelve_is_midnight():
    r = parse("明天晚上12点", MON)
    assert local(r.start_ts) == datetime(2026, 11, 3, 23, tzinfo=SH)
    assert local(r.end_ts) == datetime(2026, 11, 4, 1, tzinfo=SH)


def test_minghoutian_is_two_day_range():
    r = parse("明后天", MON)
    assert r.precision == "range"
    assert (r.start_ts, r.end_ts) == (ts(2026, 11, 3), ts(2026, 11, 5))


def test_compound_daywords():
    r = parse("明晚", MON)
    assert r.precision == "part_of_day"
    assert local(r.start_ts) == datetime(2026, 11, 3, 18, tzinfo=SH)


def test_hour_is_fuzzy_window_never_exact_minutes():
    r = parse("周五下午3点交", MON)
    assert r.precision == "part_of_day"
    assert local(r.start_ts) == datetime(2026, 11, 6, 14, tzinfo=SH)
    assert local(r.end_ts) == datetime(2026, 11, 6, 16, tzinfo=SH)
    assert r.basis == "周五下午3点"
    r = parse("明天晚上八点半", MON)
    assert local(r.start_ts) == datetime(2026, 11, 3, 19, 30, tzinfo=SH)
    assert local(r.end_ts) == datetime(2026, 11, 3, 21, 30, tzinfo=SH)


def test_chinese_yidian_is_not_an_hour():
    assert parse("有一点累", MON) is None
    assert parse("这一点很重要", MON) is None
    r = parse("明天早一点来", MON)
    assert r.precision == "day"


def test_fuzzy_ranges():
    r = parse("过几天再说", MON)
    assert r.precision == "fuzzy"
    assert r.start_ts == ts(2026, 11, 4)
    assert r.end_ts == ts(2026, 11, 8)
    assert r.basis == "过几天"
    assert parse("最近有点忙", MON).precision == "fuzzy"
    assert parse("过两天", MON).start_ts == ts(2026, 11, 4)


def test_weekend_and_month_ranges():
    r = parse("这周末去爬山", MON)
    assert r.precision == "range"
    assert (r.start_ts, r.end_ts) == (ts(2026, 11, 7), ts(2026, 11, 9))
    assert parse("下周末", MON).start_ts == ts(2026, 11, 14)
    r = parse("下个月", MON)
    assert (r.start_ts, r.end_ts) == (ts(2026, 12, 1), ts(2027, 1, 1))
    r = parse("月底交", MON)
    assert (r.start_ts, r.end_ts) == (ts(2026, 11, 24), ts(2026, 12, 1))
    # 11-02 已在月初窗口内
    r = parse("月初", MON)
    assert r.end_ts == ts(2026, 11, 6)
    # 11-20 说月初 → 下月初
    assert parse("月初", ts(2026, 11, 20)).start_ts == ts(2026, 12, 1)
    # 12 月下个月 → 次年 1 月
    assert parse("下个月", ts(2026, 12, 15)).start_ts == ts(2027, 1, 1)


def test_relative_to_original_message_time_not_now():
    # 消息发送于 23:59，明天 = 次日；与解析运行时刻无关
    r = parse("明天", ts(2026, 11, 2, 23, 59))
    assert r.start_ts == ts(2026, 11, 3)


def test_timezone_is_shanghai_not_utc():
    # 2026-11-02 17:00 UTC = 11-03 01:00 上海 → "明天" 是 11-04
    base = datetime(2026, 11, 2, 17, tzinfo=timezone.utc).timestamp()
    assert parse("明天", base).start_ts == ts(2026, 11, 4)


@pytest.mark.parametrize("text", ["", "哈哈", "好的", "我觉得还行", "一周一次", None])
def test_unknown_returns_none(text):
    assert parse(text, MON) is None


def test_bad_timezone_returns_none():
    assert parse("明天", MON, tz="Not/AZone") is None


# ---------------------------------------------------------------------------
# 过去日期（state F5）
# ---------------------------------------------------------------------------

MON_PM = ts(2026, 11, 2, 15, 30)
WED_PM = ts(2026, 11, 4, 15, 30)


@pytest.mark.parametrize(
    "text,base,prefer_past,expected,precision",
    [
        # X月X号 向后滚 >300 天 → 视为过去，fuzzy
        ("11月1号面试，等结果", MON_PM, False, (2026, 11, 1), "fuzzy"),
        # X号 + 过去标记，向后滚 >25 天 → 过去，fuzzy
        ("1号考的", MON_PM, False, (2026, 11, 1), "fuzzy"),
        # X号 + 过去标记，小滚动 → 最近一次过去，day
        ("5号考的", MON_PM, False, (2026, 10, 5), "day"),
        # 裸周X + 过去标记 → 本周已过的那天
        ("周一考的", WED_PM, False, (2026, 11, 2), "day"),
        ("周二面完了", WED_PM, False, (2026, 11, 3), "day"),
        # 显式 prefer_past（pending_result）
        ("周一", WED_PM, True, (2026, 11, 2), "day"),
        ("周五", WED_PM, True, (2026, 10, 30), "day"),
        ("11月3号", WED_PM, True, (2026, 11, 3), "day"),
    ],
)
def test_past_resolution(text, base, prefer_past, expected, precision):
    r = parse(text, base, prefer_past=prefer_past)
    assert r is not None
    assert (r.start_ts, r.end_ts) == day_window(*expected)
    assert r.precision == precision


@pytest.mark.parametrize(
    "text,base,expected",
    [
        ("周五面试", WED_PM, (2026, 11, 6)),
        ("1号交报告", MON_PM, (2026, 12, 1)),
        ("12月1号", MON_PM, (2026, 12, 1)),
        ("周一要考的", WED_PM, (2026, 11, 9)),  # 将来标记压过“的”
        ("周五去了解一下", WED_PM, (2026, 11, 6)),
    ],
)
def test_future_unchanged_without_past_context(text, base, expected):
    r = parse(text, base)
    assert (r.start_ts, r.end_ts) == day_window(*expected)
    assert r.precision == "day"
