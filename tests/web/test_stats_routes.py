"""后台统计路由测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime

import pytest
from quart import Quart

from iris_memory.web.routes import stats as stats_routes


class _ComponentManager:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def get_component(self, *_args):
        return self.llm_manager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_days"),
    [("", 7), ("?days=1", 1), ("?days=30", 30), ("?days=14", 7)],
)
async def test_token_stats_range_defaults_and_validation(
    monkeypatch, query, expected_days
):
    llm_manager = type(
        "FakeLLMManager",
        (),
        {
            "is_available": True,
            "get_token_stats_for_days": AsyncMock(
                return_value={
                    "global": {
                        "total_input_tokens": 10,
                        "total_output_tokens": 5,
                        "total_calls": 1,
                        "successful_calls": 1,
                        "failed_calls": 0,
                        "pending_calls": 0,
                    }
                }
            ),
        },
    )()
    monkeypatch.setattr(
        stats_routes,
        "get_component_manager",
        lambda: _ComponentManager(llm_manager),
    )
    app = Quart(__name__)
    app.add_url_rule("/stats/token", view_func=stats_routes.get_token_stats)

    response = await app.test_client().get(f"/stats/token{query}")
    payload = await response.get_json()

    assert response.status_code == 200
    assert payload["days"] == expected_days
    llm_manager.get_token_stats_for_days.assert_awaited_once_with(expected_days)


@pytest.mark.asyncio
async def test_llm_governance_endpoint_exposes_metrics_and_alerts(monkeypatch):
    llm_manager = type(
        "FakeLLMManager",
        (),
        {
            "is_available": True,
            "get_governor_metrics": lambda self: {
                "queue_depth": 8,
                "queue_limit": 10,
                "queue_wait_p95_ms": 1200,
                "calls_per_minute": {"p1": {"l1_summarizer": 25}},
            },
            "get_recent_call_logs": lambda self, limit=100: [
                {"module": "profile_analysis", "success": False}
                for _ in range(5)
            ],
        },
    )()
    monkeypatch.setattr(
        stats_routes,
        "get_component_manager",
        lambda: _ComponentManager(llm_manager),
    )
    monkeypatch.setattr(
        "iris_memory.config.get_config",
        lambda: SimpleNamespace(get=lambda key, default=None: 30),
    )
    app = Quart(__name__)
    app.add_url_rule(
        "/stats/llm-governance",
        view_func=stats_routes.get_llm_governance_stats,
    )

    response = await app.test_client().get("/stats/llm-governance")
    payload = await response.get_json()

    assert response.status_code == 200
    assert payload["metrics"]["queue_depth"] == 8
    assert {alert["type"] for alert in payload["alerts"]} == {
        "provider_rpm_high",
        "queue_depth_high",
        "queue_wait_high",
        "module_failure_streak",
    }


def test_duplicate_real_image_provider_calls_raise_alert(monkeypatch):
    monkeypatch.setattr(
        "iris_memory.config.get_config",
        lambda: SimpleNamespace(get=lambda key, default=None: 30),
    )
    now = datetime.now().isoformat()
    calls = [
        {
            "call_id": f"call-{index}",
            "timestamp": now,
            "module": "image_parsing",
            "provider_id": "vision-1",
            "success": index == 0,
            "metadata": {
                "image_hash": "same-hash",
                "provider_call_started": True,
            },
        }
        for index in range(2)
    ]
    # 本地排队失败不属于真实 Provider 调用，不应被计为第三次。
    calls.append(
        {
            "call_id": "queued-only",
            "timestamp": now,
            "module": "image_parsing",
            "provider_id": "vision-1",
            "success": False,
            "metadata": {
                "image_hash": "same-hash",
                "provider_call_started": False,
            },
        }
    )

    alerts = stats_routes._governor_alerts(
        {"queue_depth": 0, "queue_limit": 10, "calls_per_minute": {}}, calls
    )
    duplicate = next(
        alert for alert in alerts if alert["type"] == "duplicate_image_provider_call"
    )
    assert duplicate["image_hash"] == "same-hash"
    assert duplicate["value"] == 2
