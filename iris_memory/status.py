"""Small diagnostic projections: no credentials, payloads or remote error text."""

HEALTH_STATES = {
    "normal",
    "temporarily_unavailable",
    "invalid_key",
    "configuration_error",
    "account_problem",
    "usage_limit",
    "rate_limited",
}
MODEL_KINDS = ("chat", "embedding", "recall_judge", "goal_dedup_judge", "image_understanding")


def count(value):
    return value if type(value) is int and value >= 0 else 0


def summarize(raw: dict) -> dict:
    health = raw.get("model_health") or {}
    return {
        "service": "ready" if raw.get("service") == "ready" else "unknown",
        "model_health": {
            name: {
                "state": item.get("state") if item.get("state") in HEALTH_STATES else "unknown",
                "configured": item.get("configured")
                if type(item.get("configured")) is bool
                else None,
            }
            for name in MODEL_KINDS
            if isinstance((item := health.get(name)), dict)
        },
        "pending_count": sum(
            count(e.get("pending_count")) for e in raw.get("entries", raw.get("backlog", []))
        ),
        "usage": {
            period: {
                key: count((raw.get("usage", {}).get(period) or {}).get(key))
                for key in ("calls", "tokens", "calls_without_usage", "failures")
            }
            for period in ("today", "week")
        },
    }


def status_text(summary: dict) -> str:
    states = "、".join(f"{k}={v['state']}" for k, v in summary["model_health"].items()) or "未知"
    today = summary["usage"]["today"]
    return (
        f"Iris 连接成功；服务 {summary['service']}\n模型健康：{states}\n"
        f"授权范围待学习消息：{summary['pending_count']}\n"
        f"今日用量：{today['calls']} 次调用，{today['tokens']} token；"
        f"{today['calls_without_usage']} 次未报告用量。\n接收不等于已学习。"
    )
