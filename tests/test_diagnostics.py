import ast
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_memory.config import Config
from iris_memory.diagnostics import CODES, Diagnostics, reference
from iris_memory.vendor.iris_client import IrisError
from tests.conftest import Context, HostEvent
from tests.test_bridge import bridge_factory as bridge_factory
from tests.test_bridge import prepared
from tests.test_image_policy import image_event, picture
from tests.test_proactive import enable, settle
from tests.test_proactive import setup as setup


def test_bounded_logs_fixed_codes_and_detached_snapshot():
    diagnostics = Diagnostics(Config(diagnostics_log_limit=2))
    diagnostics.record("receive_failed")
    diagnostics.record("iris_test_token PRIVATE_BODY")
    diagnostics.record("prepare_degraded")
    snapshot = diagnostics.snapshot()
    assert len(snapshot["logs"]) == 2
    assert [row["code"] for row in snapshot["logs"]] == ["unknown_event", "prepare_degraded"]
    assert "iris_test_token" not in json.dumps(snapshot) and "PRIVATE_BODY" not in json.dumps(
        snapshot
    )
    snapshot["logs"][0]["code"] = "changed"
    assert diagnostics.snapshot()["logs"][0]["code"] == "unknown_event"


@pytest.mark.parametrize(
    "profile",
    [
        None,
        {"text": "PRIVATE_BODY", "version": 4, "possibly_stale": True},
        {"text": "PRIVATE_BODY", "version": "iris_test_token", "possibly_stale": "PRIVATE_BODY"},
    ],
)
def test_prepare_metadata_only_and_profile_compatibility(profile):
    diagnostics = Diagnostics(Config(diagnostics_prepare_limit=2))
    data = prepared()
    data["entry_profile"] = profile
    data["judgment"] = {"status": "degraded", "error": "iris_test_token", "reason": "PRIVATE_BODY"}
    data["hints"].append(
        {"code": "recall_judgment", "status": "degraded", "message": "PRIVATE_BODY"}
    )
    for i in range(3):
        diagnostics.prepared(
            "private-entry", data, 0.025, purpose="reply", outcome="ok", judge=True
        )
    rows = diagnostics.snapshot()["prepares"]
    assert len(rows) == 2
    assert rows[-1]["entry_ref"] == reference("private-entry")
    assert rows[-1]["duration_ms"] == 25
    assert rows[-1]["partitions"]["memories"] == 2
    assert rows[-1]["partitions"]["recent_messages"] == 1
    assert rows[-1]["judgment"] == "degraded"
    assert rows[-1]["entry_profile_version"] == (4 if profile and profile["version"] == 4 else None)
    output = json.dumps(rows)
    for forbidden in ("PRIVATE_BODY", "iris_test_token", "小林", "private-entry", "recall-1"):
        assert forbidden not in output


@pytest.mark.parametrize(
    "status,expected",
    [
        ("applied", "applied"),
        ("disabled", "disabled"),
        ("degraded", "degraded"),
        ("PRIVATE_BODY", "unknown"),
    ],
)
def test_judgment_whitelist(status, expected):
    diagnostics = Diagnostics(Config())
    diagnostics.prepared(
        "entry", {"judgment": {"status": status}}, 0.1, purpose="reply", outcome="ok", judge=True
    )
    assert diagnostics.snapshot()["prepares"][0]["judgment"] == expected


def test_malformed_metadata_and_disabled_collection():
    config = Config()
    diagnostics = Diagnostics(config)
    diagnostics.prepared(
        "entry",
        {
            "persona": "PRIVATE_BODY",
            "memories": None,
            "entry_profile": False,
            "hints": [None],
            "judgment": [],
        },
        float("nan"),
        purpose="PRIVATE_BODY",
        outcome="PRIVATE_BODY",
        judge=False,
    )
    output = diagnostics.snapshot()
    assert output["prepares"][0]["duration_ms"] == 0
    assert "PRIVATE_BODY" not in json.dumps(output)
    disabled = Diagnostics(replace(config, diagnostics_enabled=False))
    disabled.record("receive_failed")
    disabled.prepared("entry", prepared(), 0.1, purpose="reply", outcome="ok", judge=True)
    disabled.proactive("origin", "entry", "message", "disabled")
    assert (
        disabled.snapshot()["logs"]
        == disabled.snapshot()["prepares"]
        == disabled.snapshot()["sessions"]
        == []
    )


def test_stats_bounded_codes_lru_reset():
    diagnostics = Diagnostics(Config(diagnostics_session_limit=2))
    for origin in ("a", "b", "a", "c"):
        diagnostics.proactive(origin, "entry", "message", "no_signal")
    snapshots = diagnostics.snapshot()["sessions"]
    assert {r["session_ref"] for r in snapshots} == {reference("a"), reference("c")}
    a = next(r for r in snapshots if r["session_ref"] == reference("a"))
    assert a["counts"]["message"]["no_signal"] == 2
    diagnostics.proactive("a", "entry", "calls", "PRIVATE_BODY")
    assert "PRIVATE_BODY" not in json.dumps(diagnostics.snapshot())
    diagnostics.reset_session("a")
    assert len(diagnostics.snapshot()["sessions"]) == 1


async def test_prepare_records_success_failure_and_cancel(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    fake_iris.prepared["entry_profile"] = {"text": "PRIVATE_BODY", "version": 9}
    await plugin.before_llm(HostEvent(), SimpleNamespace(system_prompt="", contexts=[]))
    assert plugin.diagnostics.snapshot()["prepares"][-1]["entry_profile_version"] == 9

    async def failing(*args):
        raise RuntimeError("iris_test_token PRIVATE_BODY")

    plugin.bridge.client.prepare = failing
    with pytest.raises(RuntimeError):
        await plugin.bridge.prepare_material("entry", {}, purpose="reply")

    async def hanging(*args):
        await asyncio.sleep(10)

    plugin.bridge.client.prepare = hanging
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await plugin.bridge.prepare_material("entry", {}, purpose="proactive")
    snapshot = plugin.diagnostics.snapshot()
    assert [x["outcome"] for x in snapshot["prepares"]] == ["ok", "failed", "cancelled"]
    assert "PRIVATE_BODY" not in json.dumps(snapshot) and "iris_test_token" not in json.dumps(
        snapshot
    )


async def test_authenticated_page_no_http_and_route_cleanup(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    plugin.bridge.record("iris_test_token PRIVATE_BODY")
    count = len(fake_iris.requests)
    page = await plugin.page_diagnostics()
    assert len(fake_iris.requests) == count
    assert page["logs"][-1]["code"] == "unknown_event"
    assert "iris_test_token" not in json.dumps(page)
    module = __import__(plugin.__class__.__module__, fromlist=["request"])
    module.request.username = None
    assert (await plugin.page_diagnostics())["status_code"] == 401
    module.request.username = "test-admin"
    assert any(api[0].endswith("/diagnostics") for api in plugin.context.registered_web_apis)
    await plugin.terminate()
    assert not plugin.context.registered_web_apis


async def test_disabled_plugin_has_safe_startup_log(plugin_module):
    module, logs = plugin_module
    plugin = module.IrisMemoryPlugin(Context(), {"enabled": False})
    page = await plugin.page_diagnostics()
    assert page["logs"] == []
    assert page["counts"]["disabled_or_missing_token"] == 1
    await plugin.terminate()


async def test_message_gate_decision_counts_and_reset(setup, fake_iris):
    create, event = setup
    plugin, engine, provider = await create()
    initial = event()
    state = await enable(plugin, initial)
    await plugin.receive(event("普通一句话"))
    await plugin.receive(event("Iris 有空吗？"))
    await settle(plugin)
    await plugin.receive(event("Iris 又来一句？"))
    row = plugin.diagnostics.snapshot()["sessions"][0]
    assert row["counts"]["message"]["disabled"] == 1
    assert row["counts"]["message"]["no_signal"] == 1
    assert row["counts"]["message"]["chime_in"] == 1
    assert row["counts"]["message"]["cooldown"] == 1
    assert row["counts"]["calls"]["chime_in"] == 1
    assert row["counts"]["sends"]["chime_in"] == 1
    assert row["counts"]["outcome"]["sent"] == 1
    assert plugin.diagnostics.snapshot()["prepares"][-1]["purpose"] == "proactive"
    assert len(provider.calls) == 1
    engine.update(state.origin, {"reset": True})
    assert plugin.diagnostics.snapshot()["sessions"] == []


async def test_scheduler_and_failed_provider_are_counted(setup):
    create, event = setup
    plugin, engine, provider = await create(
        proactive_initiate_enabled=True,
        proactive_silence_minutes=1,
        proactive_initiate_probability=1,
    )
    await enable(plugin, event())
    await engine.tick()
    engine.clock.advance(301)
    await engine.tick()
    provider.error = True
    engine.clock.advance(301)
    await engine.tick()
    await settle(plugin)
    row = plugin.diagnostics.snapshot()["sessions"][0]
    assert row["counts"]["scheduler"]["not_silent"] == 1
    assert row["counts"]["scheduler"]["reserved"] == 1
    assert row["counts"]["calls"]["initiate"] == 1
    assert row["counts"]["outcome"]["failed"] == 1
    assert not row["counts"]["sends"]


@pytest.mark.parametrize(
    "changes",
    [
        {"diagnostics_enabled": 1},
        {"diagnostics_log_limit": 0},
        {"diagnostics_log_limit": 5001},
        {"diagnostics_prepare_limit": 0},
        {"diagnostics_prepare_limit": 501},
        {"diagnostics_session_limit": True},
        {"diagnostics_session_limit": 1001},
    ],
)
def test_diagnostics_config(changes):
    with pytest.raises(ValueError):
        Config.from_mapping(changes)


async def test_diagnostics_response_is_no_store(plugin_factory, monkeypatch):
    plugin = await plugin_factory()
    module = __import__(plugin.__class__.__module__, fromlist=["json_response"])
    seen = {}

    def response(data, **kwargs):
        seen.update(kwargs)
        return data

    monkeypatch.setattr(module, "json_response", response)
    await plugin.page_diagnostics()
    assert seen["headers"]["Cache-Control"] == "no-store"


async def test_observation_and_failed_send_statistics(setup):
    create, event = setup
    plugin, engine, provider = await create()
    state = await enable(plugin, event())
    provider.response["speak"] = False
    await plugin.receive(event("Iris 有空吗？"))
    await settle(plugin)
    row = plugin.diagnostics.snapshot()["sessions"][0]
    assert row["counts"]["outcome"]["watch"] == 1
    assert row["counts"]["calls"]["chime_in"] == 1
    assert not row["counts"]["sends"]
    engine.update(state.origin, {"reset": True})
    provider.response["speak"] = True

    async def failed_send(*args):
        return False

    plugin._send_to_session = failed_send
    await plugin.receive(event("Iris 再试试？"))
    await settle(plugin)
    row = plugin.diagnostics.snapshot()["sessions"][0]
    assert row["counts"]["outcome"]["send_unconfirmed"] == 1
    assert not row["counts"]["sends"]


def test_every_record_literal_in_plugin_source_is_registered():
    root = Path(__file__).parents[1]
    seen = set()
    paths = [root / "main.py", *(root / "iris_memory").rglob("*.py")]
    for path in paths:
        if "vendor" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if (
                name in {"record", "_record"}
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                seen.add(node.args[0].value)
    assert {
        "command_request_failed",
        "tool_request_failed",
        "image_processing_unavailable",
        "image_cache_hit",
        "image_cache_stale",
        "image_filtered",
        "image_quota_exhausted",
    } <= seen
    assert seen <= CODES, "Unregistered fixed event codes: " + str(sorted(seen - CODES))


def test_routine_events_only_count_and_do_not_evict_warnings():
    diagnostics = Diagnostics(Config(diagnostics_log_limit=2))
    diagnostics.record("receive_failed")
    normal = [
        "message_received",
        "prepare_received",
        "image_cache_hit",
        "image_filtered",
        "image_quota_exhausted",
        "goal_dedup_result",
        "proactive_call",
        "proactive_sent",
        "forward_expanded",
    ]
    for _ in range(100):
        for code in normal:
            diagnostics.record(code)
    data = diagnostics.snapshot()
    assert [row["code"] for row in data["logs"]] == ["receive_failed"]
    assert all(data["counts"][code] == 100 for code in normal)
    assert data["counts"]["receive_failed"] == 1
    data["counts"]["image_cache_hit"] = 0
    assert diagnostics.snapshot()["counts"]["image_cache_hit"] == 100
    disabled = Diagnostics(Config(diagnostics_enabled=False))
    disabled.record("message_received")
    assert disabled.snapshot()["counts"] == {} and disabled.snapshot()["logs"] == []


async def test_bridge_warns_only_for_faults_and_preserves_command_tool_codes(
    plugin_factory, plugin_module
):
    plugin = await plugin_factory()
    _, logs = plugin_module
    logs.clear()
    for code in [
        "message_received",
        "prepare_received",
        "image_cache_hit",
        "image_filtered",
        "image_quota_exhausted",
    ]:
        plugin.bridge.record(code)
        assert plugin.bridge.counts[code] == 1
    assert logs == [] and plugin.diagnostics.snapshot()["logs"] == []
    for code in ["command_request_failed", "tool_request_failed", "image_processing_unavailable"]:
        plugin.bridge.record(code)
        assert logs[-1] == "Iris: " + code
        assert plugin.diagnostics.snapshot()["logs"][-1]["code"] == code
        assert plugin.diagnostics.snapshot()["counts"][code] == 1


@pytest.mark.parametrize("mode", ["success", "cache-404", "receive-failed", "retry-failed"])
async def test_image_ingest_counts_only_confirmed_receive_once(bridge_factory, mode):
    bridge, client, logs, _ = bridge_factory()

    async def upload(body):
        return {"id": "media-test"}

    client.upload_media = upload
    await bridge.receive(image_event(picture(), message_id="prime"))
    await bridge.queue.join()
    baseline = bridge.diagnostics.snapshot()["counts"].get("message_received", 0)
    attempts = []
    receive = client.receive_message

    async def capture(entry_id, message):
        attempts.append(dict(message))
        if mode in {"cache-404", "retry-failed"} and len(attempts) == 1:
            raise IrisError("not_found", status_code=404)
        if mode in {"receive-failed", "retry-failed"}:
            raise IrisError("unavailable", status_code=503)
        return await receive(entry_id, message)

    client.receive_message = capture
    event = image_event(picture(), message_id="second")
    await bridge.receive(event)
    await bridge.queue.join()
    snapshot = bridge.diagnostics.snapshot()
    confirmed = mode in {"success", "cache-404"}
    assert event.get_extra("iris_received").result() is confirmed
    assert snapshot["counts"].get("message_received", 0) - baseline == int(confirmed)
    assert snapshot["counts"]["image_cache_hit"] == 1
    assert "unknown_event" not in snapshot["counts"]
    expected_warnings = []
    if mode in {"cache-404", "retry-failed"}:
        expected_warnings.append("image_cache_stale")
        assert len(attempts) == 2
        assert "media_ids" not in attempts[-1]
    else:
        assert len(attempts) == 1
    if not confirmed:
        expected_warnings.append("background_request_failed")
    assert [row["code"] for row in snapshot["logs"]] == expected_warnings
    assert logs == ["Iris: " + code for code in expected_warnings]
