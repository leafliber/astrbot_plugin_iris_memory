import asyncio
import json
import time
from pathlib import Path

import pytest

from iris_memory.application import Application
from iris_memory.control.catalog import CATALOG, effective
from iris_memory.control.topology import SourceBinding, plan_subscriptions
from iris_memory.errors import ControlError
from tests.fixtures import CAPABILITIES, HEALTH, STATUS, envelope


class FakeTransport:
    instances = []

    def __init__(self, gate):
        self.gate, self.closed, self.calls = gate, False, []
        self.instances.append(self)

    async def request(self, base, method, path, headers, body):
        assert not self.closed
        self.calls.append((base, method, path, headers, body))
        if path == "/health":
            return 200, {"version": 1, "health": HEALTH}
        return 200, envelope(CAPABILITIES if "host" in path else STATUS)

    async def close(self):
        self.closed = True


async def test_lifecycle_init_failure_and_repeated_cleanup(tmp_path):
    def fail(_):
        raise RuntimeError("injected constructor failure")

    app = Application(tmp_path, fail)
    with pytest.raises(RuntimeError):
        await app.initialize()
    assert app.store.db is None
    await app.terminate()
    await app.terminate()
    assert app.state == "closed"
    with pytest.raises(ControlError, match="CLOSED"):
        await app.initialize()
    async with asyncio.timeout(2):
        app2 = Application(tmp_path, FakeTransport)
        await app2.initialize()
        await app2.initialize()
        await app2.terminate()
    assert app2.store.db is None and app2.transport is None


async def test_auth_control_offline_and_recovery(tmp_path):
    app = Application(tmp_path, FakeTransport)
    await app.initialize()
    try:
        with pytest.raises(ControlError, match="AUTHENTICATION"):
            async with app.request(None):
                pytest.fail("must not enter")
        overview = await app.overview()
        assert overview["model_usage"]["core"] is None
        assert overview["observations"] == {}
        config = await app.save_connection(0, "http://localhost:12345", "host-secret")
        result = await app.check_connection()
        assert result["protocol"] == "compatible"
        assert "host-secret" not in json.dumps(await app.overview())
        with pytest.raises(ControlError, match="NOT_IMPLEMENTED"):
            await app.store.intent(config["revision"], "memory.remember", True)
        assert (await app.store.settings())["revision"] == config["revision"]
        await app.store.intent(config["revision"], "plugin.enabled", True)
    finally:
        await app.terminate()
    new = Application(tmp_path, FakeTransport)
    await new.initialize()
    try:
        result = await new.overview()
        assert result["settings"]["intents"]["plugin.enabled"]
        assert not result["observations"]["connection"]["current"]
    finally:
        await new.terminate()


async def test_admin_sessions_user_isolation_expiry_and_connection_reset(tmp_path):
    app = Application(tmp_path, FakeTransport)
    await app.initialize()
    try:
        config = await app.save_connection(0, "https://example.test", "host-secret")
        await app.authorize_admin(
            "alice", "alice-session", "alice-csrf", config["revision"]
        )
        with pytest.raises(ControlError, match="REAUTHORIZE"):
            await app.admin_status("bob")
        status = await app.admin_status("alice")
        assert status["instance_id"] == "instance-a"
        assert "alice-session" not in json.dumps(await app.overview())
        assert "alice-session" not in (
            tmp_path / "control.sqlite3"
        ).read_bytes().decode("latin1")
        client, _, binding = app.admin["alice"]
        app.admin["alice"] = client, time.monotonic() - 1, binding
        with pytest.raises(ControlError, match="REAUTHORIZE"):
            await app.admin_status("alice")
        assert client.transport.closed
        config = await app.store.settings()
        await app.authorize_admin(
            "alice", "new-session", "new-csrf", config["revision"]
        )
        config = await app.store.settings()
        await app.save_connection(config["revision"], "https://other.test")
        assert not app.admin
    finally:
        await app.terminate()


def test_catalog_complete_at_independent_action_level():
    expected = set(
        json.loads((Path(__file__).with_name("catalog_expectations.json")).read_text())[
            "action_keys"
        ]
    )
    keys = [r["action_key"] for r in CATALOG]
    assert expected <= set(keys)
    assert len(keys) == len(set(keys))
    assert {r["feature_id"] for r in CATALOG if r["feature_id"] != "BASE"} == {
        f"F{i:02}" for i in range(1, 62)
    }
    assert {r["limitation_id"] for r in CATALOG if "limitation_id" in r} == {
        f"U{i:02}" for i in range(1, 13)
    }
    partial = {
        r["action_key"]
        for r in CATALOG
        if r["implemented"] and r["feature_id"].startswith("F")
    }
    assert partial == {"observation.enabled", "ingress.enabled"}
    assert all(
        r["availability"] == "partial" for r in CATALOG if r["action_key"] in partial
    )
    assert any(r["availability"] == "pending_verification" for r in CATALOG)


def test_parent_source_permission_and_mode_matrix_preserves_intents():
    row = dict(
        action_key="child",
        implemented=True,
        availability="implemented",
        reason="",
        parent="parent",
    )
    intents = {"plugin.enabled": True, "parent": True, "child": True}
    assert effective(row, intents)["effective"]
    for kwargs, state in [
        ({"permission": False}, "permission_denied"),
        ({"source_allowed": False}, "source_disabled"),
        ({"mode": "paused"}, "mode_paused"),
        ({"lifecycle": "closed"}, "unknown"),
    ]:
        result = effective(row, intents, **kwargs)
        assert (
            result["state"] == state and not result["effective"] and result["desired"]
        )
    result = effective(row, intents | {"parent": False})
    assert result["state"] == "parent_disabled" and result["desired"]
    result = effective(row, intents | {"plugin.enabled": False}, source_allowed=True)
    assert result["state"] == "global_disabled" and result["desired"]
    assert effective(row, intents)["effective"]


def test_ten_distinct_sources_and_two_future_connections():
    sources = tuple(
        SourceBinding(
            "qq",
            "self",
            "group" if i % 2 else "private",
            f"chat-{i}",
            f"entry-{i}",
            f"route-{i}",
        )
        for i in range(10)
    )
    assert list(map(len, plan_subscriptions(sources))) == [8, 2]


async def test_remote_credentials_never_enter_observations(tmp_path):
    class Reflecting(FakeTransport):
        async def request(self, base, method, path, headers, body):
            if path == "/health":
                return await super().request(base, method, path, headers, body)
            return 200, envelope({**CAPABILITIES, "entries": ["host-secret"]})

    app = Application(tmp_path, Reflecting)
    await app.initialize()
    try:
        await app.save_connection(0, "http://localhost:12345", "host-secret")
        result = await app.check_connection()
        assert result["errors"] == ["REMOTE_CREDENTIAL_ECHO"]
        assert result["capabilities"] is None
        assert "host-secret" not in json.dumps(await app.overview())
    finally:
        await app.terminate()
