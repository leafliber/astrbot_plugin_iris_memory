"""Actual managed deadline envelope over loopback, SQLite and fresh processes.

Fixture provenance: Core 1f2cc1e5 management/managed_http.py:299-303 (`if not
 done` after dispatch). Synthetic operation routes below are test-only; the
production transport allowlist and operation descriptor registry stay unchanged.
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from iris_memory.application import LOCAL_BASELINE, Application
from iris_memory.control.operations import OperationDescriptor, OriginalOperations
from iris_memory.control.store import Store
from iris_memory.core_client.protocol import envelope
from iris_memory.validation import decode
from tests.fixtures import dispatched_timeout
from tests.fixtures import envelope as wire

WORKER = Path(__file__).resolve()
ROOT = WORKER.parents[1]


@asynccontextmanager
async def serve(handler):
    app = web.Application(client_max_size=32768)
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    try:
        await site.start()
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("key", [None, "original-key", "a" * 128])
async def test_dispatched_timeout_connection_observation(tmp_path, key):
    calls = []

    async def handler(request):
        calls.append((request.method, request.path))
        if request.path == "/health":
            return web.json_response({}, status=503)
        assert request.method == "POST" and request.path == "/api/host/capabilities"
        assert await request.json() == {}
        return web.json_response(dispatched_timeout(key), status=202)

    async with serve(handler) as base:
        app = Application(tmp_path)
        await app.initialize()
        try:
            config = await app.save_connection(0, base, "synthetic-host-token")
            result = await app.check_connection()
            assert (
                result["http_status"],
                result["outcome"],
                result["cleanup_pending"],
            ) == (202, "UNCONFIRMED", True)
            assert result["permission"] == result["protocol"] == "unknown"
            assert result["health"] is None and result["capabilities"] is None
            assert result["errors"] == [
                "HEALTH_HTTP_ERROR",
                "HOST_CAPABILITIES_UNCONFIRMED",
            ]
            overview = await app.overview()
            assert overview["observations"]["connection"]["value"] == result
            assert (await app.store.settings())["binding"] == config["binding"]
            assert "operation_key" not in json.dumps(overview)
            assert "synthetic-host-token" not in json.dumps(overview)
            assert not app.operations.descriptors
        finally:
            await app.terminate()
        reopened = Application(tmp_path)
        await reopened.initialize()
        try:
            observed = (await reopened.overview())["observations"]["connection"]
            assert observed["value"] == result and observed["current"] is False
            assert calls == [("GET", "/health"), ("POST", "/api/host/capabilities")]
        finally:
            await reopened.terminate()


@pytest.mark.parametrize(
    ("change", "status", "code"),
    [
        ({"operation_key": ""}, 202, "INVALID_IDENTIFIER"),
        ({"operation_key": "x" * 129}, 202, "INVALID_IDENTIFIER"),
        ({"operation_key": "bad/key"}, 202, "INVALID_IDENTIFIER"),
        ({"operation_key": 12}, 202, "INVALID_IDENTIFIER"),
        ({"operation_key": False}, 202, "INVALID_IDENTIFIER"),
        ({"operation_key": {}}, 202, "INVALID_IDENTIFIER"),
        ({"state": "COMMITTED"}, 202, "PROTOCOL_OUTCOME"),
        ({"state": None}, 202, "PROTOCOL_OUTCOME"),
        ({"unexpected": True}, 202, "INVALID_FIELDS"),
        ({"cleanup_pending": False}, 202, "PROTOCOL_TIMEOUT_CONFLICT"),
        ({"cleanup_pending": 1}, 202, "INVALID_BOOLEAN"),
        ({}, 200, "PROTOCOL_TIMEOUT_CONFLICT"),
        (
            {"outcome": "COMMITTED", "state": "COMMITTED", "data": {}},
            202,
            "PROTOCOL_TIMEOUT_CONFLICT",
        ),
        ({"operation_key": "synthetic-host-token"}, 202, "REMOTE_CREDENTIAL_ECHO"),
    ],
)
async def test_invalid_deadline_branch_is_rejected_before_observation(
    tmp_path, change, status, code
):
    async def handler(request):
        if request.path == "/health":
            return web.json_response({}, status=503)
        assert request.path == "/api/host/capabilities"
        return web.json_response(dispatched_timeout() | change, status=status)

    async with serve(handler) as base:
        app = Application(tmp_path)
        await app.initialize()
        try:
            await app.save_connection(0, base, "synthetic-host-token")
            result = await app.check_connection()
            assert code in result["errors"]
            assert "http_status" not in result and "outcome" not in result
            assert result["permission"] == result["protocol"] == "unknown"
            assert result["health"] is None and result["capabilities"] is None
            assert "operation_key" not in json.dumps(await app.overview())
            assert "synthetic-host-token" not in json.dumps(
                await app.diagnostics("alice")
            )
        finally:
            await app.terminate()


def test_timeout_key_is_validated_but_not_projected():
    value = dispatched_timeout("private-original-key")
    value.pop("state")  # Contract permits omission; if present it must match.
    reply = envelope(202, value)
    assert reply.data is None and reply.error is None and not reply.observed
    assert "private-original-key" not in repr(reply) + json.dumps(asdict(reply))
    assert (
        envelope(202, wire(outcome="UNCONFIRMED", cleanup=False)).cleanup_pending
        is False
    )


async def _worker(directory, base, phase):
    store = Store(Path(directory))
    await store.open()
    timeout = aiohttp.ClientTimeout(total=5, connect=1)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, cookie_jar=aiohttp.DummyCookieJar(), trust_env=False
        ) as session:

            async def request(action, key, payload):
                async with session.post(
                    base + "/synthetic/" + action,
                    json={"key": key, "input": payload},
                    allow_redirects=False,
                ) as response:
                    # Finite fixture transport, never available to production.
                    assert (
                        response.content_length is not None
                        and response.content_length <= 8192
                    )
                    raw = await response.content.readexactly(response.content_length)
                    return envelope(response.status, decode(raw, 8192))

            descriptor = OperationDescriptor(
                "synthetic",
                lambda _: None,
                lambda key, payload: request("submit", key, payload),
                lambda key, payload: request("confirm", key, payload),
            )
            dispatcher = OriginalOperations(store, (descriptor,))
            if phase == "submit":
                config = await store.connection(0, base, "synthetic-host-token")
                await store.observe(
                    "status", config["binding"], {}, instance_id="instance-a"
                )
            config = await store.settings()
            try:
                op = await dispatcher.begin(
                    config["binding"],
                    "instance-a",
                    "synthetic",
                    "local-original",
                    {"value": 7},
                )
            except aiohttp.ClientError:
                assert phase == "submit"
                op = await dispatcher.begin(
                    config["binding"],
                    "instance-a",
                    "synthetic",
                    "local-original",
                    {"value": 7},
                )
            if phase == "submit":
                assert (
                    op["state"] in ("UNCONFIRMED", "UNKNOWN") and op["cleanup_pending"]
                )
                os._exit(
                    17
                )  # Crash after durable observation; no graceful store close.
            assert (
                op["original_key"] == "local-original"
                and op["instance_id"] == "instance-a"
            )
            assert op["binding"] == config["binding"]
            confirmed = await dispatcher.confirm(op["id"])
            assert (
                confirmed["state"] == "COMMITTED" and not confirmed["cleanup_pending"]
            )
            assert confirmed["original_input"] == '{"value":7}'
    finally:
        await store.close()


@pytest.mark.parametrize(
    "lose_response", [False, True], ids=["received-202", "lost-response"]
)
async def test_deadline_crash_restart_only_confirms_original_input(
    tmp_path, lose_response
):
    requests = []

    async def handler(request):
        assert request.method == "POST"
        body = await request.json()
        requests.append((request.path, body))
        if request.path == "/synthetic/submit":
            if lose_response:
                request.transport.close()
            # A different returned key must never replace the durable local key.
            return web.json_response(dispatched_timeout("remote-other-key"), status=202)
        assert request.path == "/synthetic/confirm"
        return web.json_response(wire({}, "COMMITTED"))

    async with serve(handler) as base:
        env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
        for phase, expected_exit in [("submit", 17), ("confirm", 0)]:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    str(WORKER),
                    str(tmp_path),
                    base,
                    phase,
                ],
                env=env,
                capture_output=True,
                timeout=15,
            )
            assert result.returncode == expected_exit, result.stderr.decode()
        assert requests == [
            ("/synthetic/submit", {"key": "local-original", "input": {"value": 7}}),
            ("/synthetic/confirm", {"key": "local-original", "input": {"value": 7}}),
        ]


async def test_local_qualification_records_are_not_live_authority(tmp_path):
    app = Application(tmp_path)
    await app.initialize()
    try:
        info = await app.overview()
        assert info["local_baseline"] == LOCAL_BASELINE
        assert LOCAL_BASELINE["core_head"] == "1f2cc1e52d6f3a6fff2d41e97b161804c2295ccc"
        assert (
            LOCAL_BASELINE["core_final_qualification"]
            == "accepted_engineering_with_recorded_limitations"
        )
        assert (
            LOCAL_BASELINE["host_pages_qualification"]
            == "historical_local_verification"
        )
        assert LOCAL_BASELINE["actual_plugin_core_integration"] == "not_run"
        assert (
            LOCAL_BASELINE["remote_build"]
            == LOCAL_BASELINE["current_environment_qualification"]
            == "unknown"
        )
        assert info["observations"] == {} and not app.operations.descriptors
        records = {r["action_key"]: r for r in await app.features()}
        for key in ("foundation.host_qualification", "foundation.core_qualification"):
            assert (
                records[key]["state"] == "verified_record"
                and not records[key]["effective"]
            )
        assert (
            records["foundation.integration_qualification"]["state"]
            == "pending_verification"
        )
        assert all(
            not r["effective"] and (not r["implemented"] or r["state"] == "partial")
            for r in records.values()
            if r["feature_id"].startswith("F")
        )
    finally:
        await app.terminate()


if __name__ == "__main__":
    asyncio.run(_worker(*sys.argv[1:]))
