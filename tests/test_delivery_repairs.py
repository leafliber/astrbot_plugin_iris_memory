# ruff: noqa: F811
"""Deterministic confirmation interleavings and faulted resource release."""

import asyncio
import json
import os
import sqlite3
import sys

import pytest

from iris_memory.application import Application
from iris_memory.delivery.confirmation import confirm_with_current_group
from iris_memory.errors import ControlError
from tests.ingress_fixture import event
from tests.test_ingress_delivery import configure, core, until  # noqa: F401


async def unknown(app, core):
    await configure(app, core)
    core.unknown = True
    await app.delivery.capture(event(1))

    async def idle_unknown():
        status = await app.delivery.status()
        return (
            status["items"]
            and status["items"][0]["submitted"] == 1
            and not app.delivery.jobs
        )

    await until(idle_unknown)
    app.delivery.task.cancel()
    await asyncio.gather(app.delivery.task, return_exceptions=True)
    return (await app.delivery.status())["items"][0]["id"]


@pytest.mark.parametrize("reauthorized", [False, True])
async def test_stale_manual_snapshot_cannot_resurrect_terminal(
    tmp_path, core, monkeypatch, reauthorized
):
    app = Application(tmp_path)
    await app.initialize()
    try:
        ident = await unknown(app, core)
        queue = app.delivery.queue
        original = await queue.get(ident)
        entered, release = asyncio.Event(), asyncio.Event()
        schedule = queue.schedule_confirmation

        async def stale_request(event_id):
            # The old implementation's read/write gap, now OUTSIDE atomic scheduling.
            row = await queue.get(event_id)
            assert row["state"] == "UNKNOWN"
            entered.set()
            await release.wait()
            return await schedule(event_id)

        monkeypatch.setattr(queue, "schedule_confirmation", stale_request)
        manual = asyncio.create_task(app.delivery_confirm(ident))
        await entered.wait()
        core.unknown = False
        if reauthorized:
            config = await app.store.settings()
            await app.group_save(
                config["revision"], "group-0", "host", ["entry-0"], "replacement-token"
            )
            core.tokens["replacement-token"] = ["entry-0"]
            config = await app.store.settings()
            await app.authorize_admin(
                "alice", "synthetic-session", "synthetic-csrf", config["revision"]
            )
            config = await app.store.settings()
            await confirm_with_current_group(
                app, "alice", ident, "group-0", config["revision"]
            )
        else:
            await app.delivery.work(ident)
        completed = await queue.get(ident)
        assert completed["state"] == "CONFIRMED" and completed["material"] is None
        release.set()
        assert await manual == {"scheduled": False, "operation": "already_terminal"}
        assert await queue.get(ident) == completed
        # A worker already selected before replacement authority completed is harmless.
        await app.delivery.work(ident)
        assert await queue.get(ident) == completed
        assert completed["original_key"] == original["original_key"]
        assert len(core.accepts) == len(core.confirms) == 1
        await app.delivery.capture(event(2))
        heads = await queue.heads()
        assert len(heads) == 1 and heads[0]["id"] != ident
        await app.delivery.work(heads[0]["id"])
        assert (await queue.get(heads[0]["id"]))["state"] == "CONFIRMED"
        assert len(core.accepts) == 2 and len(core.confirms) == 1
    finally:
        await app.terminate()


async def test_cleanup_inflight_repeat_and_pruned_confirmation(
    tmp_path, core, monkeypatch
):
    app = Application(tmp_path)
    await app.initialize()
    try:
        ident = await unknown(app, core)
        queue = app.delivery.queue
        await queue.state(
            ident, "CONFIRMED", cleanup=True, reason="CLEANUP_PENDING", delay=60
        )
        before = await queue.get(ident)
        for _ in range(3):
            assert (await app.delivery_confirm(ident))["scheduled"]
        scheduled = await queue.get(ident)
        assert {k: v for k, v in scheduled.items() if k != "next_attempt"} == {
            k: v for k, v in before.items() if k != "next_attempt"
        }
        entered, release = asyncio.Event(), asyncio.Event()
        ingress = app.host.ingress

        async def paused(*args, **kwargs):
            if args[3] == "accept/resolve":
                entered.set()
                await release.wait()
            return await ingress(*args, **kwargs)

        monkeypatch.setattr(app.host, "ingress", paused)
        core.unknown = False
        # Real scheduler owns one worker per source while repeated actions arrive.
        app.delivery.task = asyncio.create_task(app.delivery.run())
        await entered.wait()
        for _ in range(3):
            await app.delivery_confirm(ident)
        assert len(app.delivery.jobs) == 1 and len(core.accepts) == 1
        release.set()

        async def cleaned():
            return (await queue.get(ident))[
                "material"
            ] is None and not app.delivery.jobs

        await until(cleaned)
        assert len(core.confirms) == len(core.accepts) == 1
        assert not (await app.delivery_confirm(ident))["scheduled"]
        # Confirmed diagnostics may be pruned; a stale page must not recreate them.
        async with app.store.transaction() as db:
            await db.execute("DELETE FROM delivery_events WHERE id=?", (ident,))
        with pytest.raises(ControlError, match="DELIVERY_NOT_FOUND"):
            await app.delivery_confirm(ident)
        assert not await queue.heads()
    finally:
        await app.terminate()


def inject_close_failure(db, phase, monkeypatch):
    execute, commit = db.execute, db.commit

    def fault_execute(sql, *args, **kwargs):
        if (phase == "BEGIN" and sql == "BEGIN IMMEDIATE") or (
            phase == "UPDATE" and sql.startswith("UPDATE delivery_session SET clean=1")
        ):
            raise sqlite3.OperationalError("synthetic close failure")
        return execute(sql, *args, **kwargs)

    async def fault_commit():
        if phase == "COMMIT":
            raise sqlite3.OperationalError("synthetic close failure")
        await commit()

    monkeypatch.setattr(db, "execute", fault_execute)
    monkeypatch.setattr(db, "commit", fault_commit)


@pytest.mark.parametrize("phase", ["BEGIN", "UPDATE", "COMMIT"])
async def test_close_database_failure_releases_all_and_reopens_in_process(
    tmp_path, core, monkeypatch, phase
):
    app = Application(tmp_path)
    await app.initialize()
    ident = await unknown(app, core)
    config = await app.store.settings()
    await app.authorize_admin(
        "alice", "synthetic-session", "synthetic-csrf", config["revision"]
    )
    delivery, db = app.delivery, app.store.db
    sessions = [
        delivery.media.session,
        app.transport._session,
        app.admin["alice"][0].transport._session,
    ]
    entered = asyncio.Event()

    async def pending_page():
        async with app.request("alice"):
            entered.set()
            async with app.admission.slot():
                await asyncio.Event().wait()

    requests = [asyncio.create_task(pending_page()) for _ in range(6)]
    await entered.wait()
    await asyncio.sleep(0)
    assert app.admission.active == 4 and app.admission.waiting == 2
    inject_close_failure(db, phase, monkeypatch)
    with pytest.raises(ControlError, match="RESOURCE_RELEASE_FAILED"):
        await app.terminate()
    await app.terminate()
    assert app.released and app.state == "closed"
    assert all(session.closed for session in sessions)
    assert all(task.done() for task in requests)
    assert not db._thread.is_alive()
    assert not delivery.queue.clean_saved
    assert delivery.queue.storage_failed and delivery.queue.close_attempted
    assert app.release_failures == [
        {"resource": "delivery", "error_type": "ExceptionGroup"}
    ]
    core.unknown = False
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.delivery_reopen_worker",
        str(tmp_path),
        ident,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), 20)
    assert proc.returncode == 0, (stdout, stderr)
    result = json.loads(stdout)
    assert (
        result["unclean"]
        and result["state"] == "CONFIRMED"
        and result["submitted"] == 1
    )
    assert len(core.accepts) == len(core.confirms) == 1


async def test_cancelled_close_joins_capture_and_resources(tmp_path, core, monkeypatch):
    app = Application(tmp_path)
    await app.initialize()
    await configure(app, core)
    entered, release = asyncio.Event(), asyncio.Event()
    original = app.delivery.queue.admit
    db = app.store.db

    async def paused(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(app.delivery.queue, "admit", paused)
    capture = asyncio.create_task(app.delivery.capture(event(1)))
    await entered.wait()
    closing = asyncio.create_task(app.terminate())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and app.store.db is db
    release.set()
    await capture
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert app.released and not db._thread.is_alive()
    await app.terminate()


async def test_partial_initialize_failure_keeps_original_and_releases(
    tmp_path, monkeypatch
):
    from iris_memory.delivery.engine import DeliveryEngine

    start = DeliveryEngine.start
    references = {}

    async def fail_start(self):
        await start(self)
        references.update(
            db=self.app.store.db,
            download=self.media.session,
            host=self.app.transport._session,
        )
        inject_close_failure(self.app.store.db, "UPDATE", monkeypatch)
        raise RuntimeError("synthetic init failure")

    monkeypatch.setattr(DeliveryEngine, "start", fail_start)
    app = Application(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic init failure"):
        await app.initialize()
    assert app.released and app.release_failures
    assert references["download"].closed and references["host"].closed
    assert not references["db"]._thread.is_alive()
    await app.terminate()


async def test_failed_session_close_retains_owner_and_other_resources_close(
    tmp_path, core, monkeypatch
):
    app = Application(tmp_path)
    await app.initialize()
    await configure(app, core)
    config = await app.store.settings()
    for name in ("alice", "bob"):
        await app.authorize_admin(
            name, "synthetic-session", "synthetic-csrf", config["revision"]
        )
    delivery, db = app.delivery, app.store.db
    alice, bob = app.admin["alice"][0], app.admin["bob"][0]
    host_session = app.transport._session
    real_media_close, real_admin_close = delivery.media.close, alice.close

    async def failure():
        raise OSError("synthetic session close failure")

    monkeypatch.setattr(delivery.media, "close", failure)
    monkeypatch.setattr(alice, "close", failure)
    with pytest.raises(ControlError, match="RESOURCE_RELEASE_FAILED"):
        await app.terminate()
    assert app.state == "release_failed" and app.delivery is delivery
    assert set(app.admin) == {"alice"} and not delivery.media.session.closed
    assert (
        not alice.transport._session.closed
        and bob.transport._session.closed
        and host_session.closed
    )
    assert app.store.db is None and not db._thread.is_alive()
    assert not delivery.queue.clean_saved
    monkeypatch.setattr(delivery.media, "close", real_media_close)
    monkeypatch.setattr(alice, "close", real_admin_close)
    await app.terminate()
    assert app.released and app.state == "closed"
    assert not delivery.queue.clean_saved and len(app.release_failures) == 2


@pytest.mark.parametrize(
    "state,submitted,retry,code",
    [
        ("SAVED", 0, 0, "DELIVERY_NOT_DISPATCHED"),
        ("NOT_COMMITTED", 1, 0, "ORIGINAL_CONFIRMATION_ONLY"),
        ("SAVED", 1, 1, "ORIGINAL_CONFIRMATION_ONLY"),
    ],
)
async def test_confirmation_cannot_authorize_first_submission(
    tmp_path, core, state, submitted, retry, code
):
    app = Application(tmp_path)
    await app.initialize()
    try:
        ident = await unknown(app, core)
        async with app.store.transaction() as db:
            await db.execute(
                "UPDATE delivery_events SET state=?,submitted=?,retry_authorized=? WHERE id=?",
                (state, submitted, retry, ident),
            )
        before = await app.delivery.queue.get(ident)
        with pytest.raises(ControlError, match=code):
            await app.delivery_confirm(ident)
        assert await app.delivery.queue.get(ident) == before
        assert len(core.accepts) == 1 and not core.confirms
    finally:
        await app.terminate()
