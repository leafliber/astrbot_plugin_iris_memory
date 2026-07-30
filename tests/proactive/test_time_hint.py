"""proactive.time_hint 测试

覆盖与 AstrBot 主管线 _append_system_reminders 一致的时间提示构建、
配置解析（时区 / datetime_system_prompt 开关）以及 system_reminder 包裹。
"""

from types import SimpleNamespace

from iris_memory.proactive.time_hint import (
    build_datetime_reminder,
    resolve_datetime_reminder,
    wrap_system_reminder,
)


class TestBuildDatetimeReminder:
    def test_format_matches_astrbot(self):
        text = build_datetime_reminder("UTC")
        assert text.startswith("Current datetime: ")
        assert "(UTC)" in text
        assert ", Weekday: " in text

    def test_disabled_returns_empty(self):
        assert build_datetime_reminder("UTC", enabled=False) == ""

    def test_bad_timezone_falls_back_to_local(self):
        text = build_datetime_reminder("Not/AZone")
        assert text.startswith("Current datetime: ")
        assert ", Weekday: " in text

    def test_none_timezone_uses_local(self):
        text = build_datetime_reminder(None)
        assert text.startswith("Current datetime: ")


def _ctx(provider_settings=None, timezone="Asia/Shanghai"):
    cfg = {"timezone": timezone}
    if provider_settings is not None:
        cfg["provider_settings"] = provider_settings
    return SimpleNamespace(get_config=lambda umo=None: cfg)


class TestResolveDatetimeReminder:
    def test_reads_timezone_and_default_enabled(self):
        text = resolve_datetime_reminder(_ctx(), "umo1")
        assert text.startswith("Current datetime: ")

    def test_disabled_via_provider_settings(self):
        ctx = _ctx(provider_settings={"datetime_system_prompt": False})
        assert resolve_datetime_reminder(ctx, "umo1") == ""

    def test_provider_settings_timezone_preferred(self):
        ctx = _ctx(provider_settings={"timezone": "UTC"}, timezone="Asia/Shanghai")
        assert "(UTC)" in resolve_datetime_reminder(ctx, "umo1")

    def test_exception_returns_empty(self):
        ctx = SimpleNamespace(get_config=lambda umo=None: (_ for _ in ()).throw(RuntimeError("boom")))
        assert resolve_datetime_reminder(ctx, "umo1") == ""


class TestWrapSystemReminder:
    def test_wraps(self):
        assert wrap_system_reminder("Current datetime: x") == (
            "<system_reminder>Current datetime: x</system_reminder>"
        )

    def test_empty_returns_empty(self):
        assert wrap_system_reminder("") == ""
