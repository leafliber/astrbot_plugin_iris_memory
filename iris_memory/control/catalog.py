"""Complete action inventory and conservative server-side effective states."""

import json
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("catalog.json").read_text())
EDITABLE = frozenset({"plugin.enabled", "observation.enabled"})


def effective(
    record,
    intents,
    *,
    lifecycle="ready",
    permission=None,
    mode=None,
    source_allowed=None,
    observed_at=None,
):
    key = record["action_key"]
    desired = intents.get(key, False)
    state, reason = record["availability"], record["reason"]
    enabled = False
    if state == "implemented":
        if lifecycle != "ready":
            state, reason = "unknown", "插件生命周期未就绪"
        elif not desired and not key.startswith("foundation."):
            state, reason = "disabled", "用户意图关闭"
        elif record.get("parent") and not intents.get(record["parent"], False):
            state, reason = "parent_disabled", "父级关闭；子项意图保留"
        elif (
            key != "plugin.enabled"
            and record.get("parent")
            and not intents.get("plugin.enabled", False)
        ):
            state, reason = "global_disabled", "全局禁止"
        elif source_allowed is False:
            state, reason = "source_disabled", "来源未获准"
        elif permission is False:
            state, reason = "permission_denied", "权限不足"
        elif mode == "paused":
            state, reason = "mode_paused", "运行模式暂停"
        elif record.get("requires_observation") and observed_at is None:
            state, reason = "unknown", "缺少当前观测"
        else:
            state, reason, enabled = (
                "effective",
                "本地控制有效" if key == "plugin.enabled" else "基础服务可用",
                True,
            )
    return {
        **record,
        "desired": desired,
        "effective": enabled,
        "state": state,
        "reason": reason,
        "editable": key in EDITABLE,
        "observed_at": observed_at,
        "inflight_policy": "停止新增；保留原确认及必要清理义务",
    }


def catalog(settings, lifecycle="ready"):
    return [
        {
            **effective(row, settings["intents"], lifecycle=lifecycle),
            "revision": settings["revision"],
        }
        for row in CATALOG
    ]
