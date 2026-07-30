from __future__ import annotations

import datetime
import zoneinfo
from typing import TYPE_CHECKING

from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.star import Context

WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def build_datetime_reminder(timezone: str | None, enabled: bool = True) -> str:
    """构建与 AstrBot 主管线一致的时间提示。

    格式对齐 astrbot.core.astr_main_agent._append_system_reminders：
    ``Current datetime: YYYY-MM-DD HH:MM (TZ), Weekday: X``。
    主动发起通路直连 ``Context.llm_generate``，不经过主管线，需自行注入，
    否则 LLM 无当前时间锚点，会从滑动窗口里的旧消息推断时间（如早上说晚上好）。
    """
    if not enabled:
        return ""
    now = None
    if timezone:
        try:
            now = datetime.datetime.now(zoneinfo.ZoneInfo(timezone))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Iris Reply: 时区设置错误: %s, 使用本地时区", exc)
    if now is None:
        now = datetime.datetime.now().astimezone()
    current_time = now.strftime("%Y-%m-%d %H:%M (%Z)")
    weekday = WEEKDAY_NAMES[now.weekday()]
    return f"Current datetime: {current_time}, Weekday: {weekday}"


def resolve_datetime_reminder(context: Context, umo: str | None = None) -> str:
    """从 AstrBot 配置解析时区与开关并返回时间提示，失败时返回空串。

    读取 ``provider_settings.datetime_system_prompt``（默认开启）与顶层
    ``timezone``，与主管线 ``_decorate_llm_request`` 的取值口径保持一致。
    """
    try:
        cfg = context.get_config(umo)
        provider_settings = cfg.get("provider_settings", {}) or {}
        enabled = bool(provider_settings.get("datetime_system_prompt", True))
        timezone = provider_settings.get("timezone") or cfg.get("timezone")
        return build_datetime_reminder(timezone, enabled)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Iris Reply: resolve datetime reminder failed: %s", exc)
        return ""


def wrap_system_reminder(reminder: str) -> str:
    """将时间提示包裹为 ``<system_reminder>`` 块，空串原样返回。"""
    if not reminder:
        return ""
    return f"<system_reminder>{reminder}</system_reminder>"
