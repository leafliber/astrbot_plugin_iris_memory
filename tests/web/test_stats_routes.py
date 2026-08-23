"""后台统计路由测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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
