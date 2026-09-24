# ruff: noqa: F811
import asyncio
import os
import sqlite3
import sys

import pytest

from iris_memory.application import Application
from tests.ingress_fixture import event, png_bytes
from tests.test_ingress_delivery import configure, core, until  # noqa: F401


@pytest.mark.parametrize("phase", ["before", "unknown", "media_ready", "media_mid"])
async def test_independent_process_crash_original_recovery(tmp_path, core, phase):
    core.tokens["secret-group-0"] = ["entry-0"]
    core.unknown = True
    if phase == "media_mid":
        core.mid_chunk = True
        core.media_data = png_bytes(256)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.s2_crash_worker",
        str(tmp_path),
        core.base,
        phase,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), 20)
    assert proc.returncode == 17, (stdout, stderr)
    initial = len(core.accepts)
    assert initial == (0 if phase in {"before", "media_mid"} else 1)
    app = Application(tmp_path)
    await app.initialize()
    try:
        before = await app.delivery.status()
        if phase == "media_ready":
            assert before["media_reserved_bytes"] == len(core.media_data)
            assert before["media"][0]["state"] == "READY"
        core.unknown = False
        if phase == "media_mid":
            core.mid_chunk = False
            for upload in core.uploads.values():
                upload.update(bytes=b"", state="REUPLOAD_REQUIRED")
        for row in before["items"]:
            if phase not in {"before", "media_mid"}:
                await app.delivery_confirm(row["id"])

        async def done():
            status = await app.delivery.status()
            return (
                status["counts"].get("confirmed") == 1
                and status["event_bytes"] == 0
                and status["media_reserved_bytes"] == 0
            )

        await until(done, 15)
        status = await app.delivery.status()
        assert len(core.accepts) == 1
        assert bool(core.confirms) == (phase not in {"before", "media_mid"})
        assert status["media_reserved_bytes"] == status["event_bytes"] == 0
        assert any(
            x["reason"] == "UNCLEAN_SESSION_COVERAGE_UNKNOWN" for x in status["gaps"]
        )
    finally:
        await app.terminate()


async def test_binding_change_does_not_resubmit_unknown(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        core.unknown = True
        await app.delivery.capture(event(1))

        async def submitted():
            return len(core.accepts) == 1

        await until(submitted)
        row = (await app.delivery.status())["items"][0]
        config = await app.store.settings()
        await app.group_save(
            config["revision"], "group-0", "host", ["entry-0"], "new-token"
        )
        core.tokens["new-token"] = ["entry-0"]
        await app.delivery.work(row["id"])
        current = await app.delivery.queue.get(row["id"])
        assert (
            current["reason"] == "ORIGINAL_BINDING_CHANGED"
            and len(core.accepts) == 1
            and not core.confirms
        )
    finally:
        await app.terminate()


async def test_disk_failure_reports_memory_only_and_closes_intake(
    tmp_path, core, monkeypatch
):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)

        async def fail(*args, **kwargs):
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(app.delivery.queue, "admit", fail)
        monkeypatch.setattr(app.delivery.queue, "_gap", fail)
        await app.delivery.capture(event(1))
        await app.delivery.capture(event(2))
        status = await app.delivery.status()
        assert (
            status["memory_only_gaps"] == 2
            and status["storage_failed"]
            and not core.accepts
        )
    finally:
        await app.terminate()


async def test_unknown_platform_does_not_poison_ingress(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        item = event()
        item.get_platform_name = lambda: "webchat"
        await app.delivery.capture(item)
        assert not app.delivery.queue.storage_failed
    finally:
        await app.terminate()


async def test_v1_migration_preserves_intents_credentials_and_unknown(tmp_path):
    import json

    from iris_memory.control.store import Store

    store = Store(tmp_path)
    await store.open()
    await store.connection(0, "http://127.0.0.1:45678", "old-secret")
    config = await store.settings()
    await store.observe("status", config["binding"], {}, instance_id="old-instance")
    config = await store.settings()
    original, _ = await store.create_operation(
        config["binding"],
        "old-instance",
        "legacy-operation",
        "original-key",
        {"old": "input"},
    )
    async with store.transaction() as db:
        _, value = await store._settings(db)
        value.pop("groups")
        value.pop("ingress_limits")
        value["intents"].pop("observation.enabled")
        value["intents"]["learning.enabled"] = True
        await db.execute("UPDATE settings SET value=?", (json.dumps(value),))
        await db.execute("PRAGMA user_version=1")
    await store.close()
    app = Application(tmp_path)
    await app.initialize()
    try:
        config = await app.store.settings()
        assert (
            config["intents"]["learning.enabled"]
            and not config["intents"]["observation.enabled"]
        )
        assert config["groups"] == [] and config["sources"] == []
        row = await app.store.operation(original["id"])
        assert row["state"] == "UNKNOWN" and row["original_key"] == "original-key"
        assert json.loads(row["original_input"]) == {"old": "input"}
        assert await app.sources.credential(config["credential_ref"]) == "old-secret"
    finally:
        await app.terminate()


async def test_reload_waits_for_bounded_local_capture_without_cancelling_host(
    tmp_path, core, monkeypatch
):
    app = Application(tmp_path)
    await app.initialize()
    await configure(app, core)
    entered, release = asyncio.Event(), asyncio.Event()
    delivery = app.delivery
    original = delivery.queue.admit

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.delivery.queue, "admit", delayed)
    capture = asyncio.create_task(app.delivery.capture(event(1)))
    await entered.wait()
    closing = asyncio.create_task(app.terminate())
    await asyncio.sleep(0)
    assert not closing.done() and not capture.cancelled()
    release.set()
    await asyncio.gather(capture, closing)
    assert (
        app.store.db is None
        and app.delivery is None
        and not delivery.capture_completions
    )


@pytest.mark.parametrize("mismatch", [None, "host", "instance"])
async def test_explicit_reauthorization_only_confirms_original(
    tmp_path, core, mismatch
):
    from iris_memory.delivery.confirmation import confirm_with_current_group
    from iris_memory.errors import ControlError

    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        core.unknown = True
        await app.delivery.capture(event(1))

        async def submitted():
            return len(core.accepts) == 1

        await until(submitted)
        row = (await app.delivery.status())["items"][0]
        original = await app.delivery.queue.get(row["id"])
        c = await app.store.settings()
        await app.group_save(
            c["revision"], "group-0", "host", ["entry-0"], "replacement-token"
        )
        core.tokens["replacement-token"] = ["entry-0"]
        c = await app.store.settings()
        await app.authorize_admin(
            "alice", "synthetic-session", "synthetic-csrf", c["revision"]
        )
        if mismatch == "host":
            core.token_hosts["replacement-token"] = "wrong-host"
        if mismatch == "instance":
            core.status_instance = "different-instance"
        c = await app.store.settings()
        core.unknown = False
        if mismatch:
            with pytest.raises(ControlError, match="ORIGINAL_AUTHORITY"):
                await confirm_with_current_group(
                    app, "alice", row["id"], "group-0", c["revision"]
                )
            assert not core.confirms
        else:
            result = await confirm_with_current_group(
                app, "alice", row["id"], "group-0", c["revision"]
            )
            assert (
                result["outcome"] == "COMMITTED"
                and not result["first_submit_authorized"]
            )
            assert len(core.confirms) == 1
            current = await app.delivery.queue.get(row["id"])
            assert (
                current["state"] == "CONFIRMED"
                and current["binding"] == original["binding"]
            )
        assert len(core.accepts) == 1
    finally:
        await app.terminate()
