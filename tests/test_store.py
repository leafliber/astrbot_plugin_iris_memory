import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from iris_memory.control.operations import OperationDescriptor, OriginalOperations
from iris_memory.control.store import Store
from iris_memory.core_client.protocol import Reply
from iris_memory.errors import ControlError

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
async def store(tmp_path):
    value = Store(tmp_path / "private")
    await value.open()
    yield value
    await value.close()


async def bound(store):
    config = await store.connection(0, "http://localhost:12345", "secret-host")
    return await store.observe(
        "status", config["binding"], {}, instance_id="instance-a"
    )


async def test_real_sqlite_cas_between_two_connections(store):
    second = Store(store.directory)
    await second.open()
    try:
        results = await asyncio.gather(
            store.intent(0, "plugin.enabled", True),
            second.intent(0, "plugin.enabled", False),
            return_exceptions=True,
        )
        assert (
            sum(
                isinstance(r, ControlError) and r.code == "REVISION_CONFLICT"
                for r in results
            )
            == 1
        )
        assert (await store.settings())["revision"] == 1
    finally:
        await second.close()


async def test_credentials_and_binding_changes(store):
    config = await bound(store)
    await store.observe("connection", config["binding"], {"connected": True})
    row, _ = await store.create_operation(
        config["binding"], "instance-a", "synthetic", "key-a", {"value": 1}
    )
    changed = await store.connection(config["revision"], "http://localhost:12346")
    assert changed["credential_ref"] is None
    assert await store.credential(changed["binding"]) is None
    assert await store.observations() == {}
    with pytest.raises(ControlError, match="BINDING_MISMATCH"):
        await store.operation(row["id"])
    assert (await store.operations())["pending"] == 1
    assert os.stat(store.directory).st_mode & 0o777 == 0o700
    assert os.stat(store.directory / "control.sqlite3").st_mode & 0o777 == 0o600
    assert "secret-host" not in json.dumps(await store.operations())


async def test_same_key_conflict_capacity_and_unknown_retention(store):
    config = await bound(store)
    args = config["binding"], "instance-a", "synthetic"
    first, created = await store.create_operation(*args, "key-0", {"value": 1})
    assert created
    repeated, created = await store.create_operation(*args, "key-0", {"value": 1})
    assert not created and first["id"] == repeated["id"]
    with pytest.raises(ControlError, match="ORIGINAL_INPUT_CONFLICT"):
        await store.create_operation(*args, "key-0", {"value": 2})
    with pytest.raises(ControlError, match="PAYLOAD_TOO_LARGE"):
        await store.create_operation(*args, "large", {"value": "a" * 32768})
    for index in range(1, 256):
        await store.create_operation(*args, f"key-{index}", {})
    with pytest.raises(ControlError, match="CAPACITY_FULL"):
        await store.create_operation(*args, "overflow", {})
    await store.db.execute(
        "UPDATE operations SET updated_at=?", (time.time() - 8 * 86400,)
    )
    await store.prune()
    assert (await store.operations())["pending"] == 256
    await store.finish_operation(first["id"], Reply(200, "COMMITTED", True, {}))
    await store.prune()
    assert (await store.operations())["pending"] == 256
    await store.finish_operation(first["id"], Reply(200, "COMMITTED", False, {}))
    await store.create_operation(*args, "after-cleanup", {})
    assert (await store.operations())["pending"] == 256


async def test_history_bounded_and_pending_never_ttl_deleted(store):
    config = await bound(store)
    first, _ = await store.create_operation(
        config["binding"], "instance-a", "synthetic", "unknown", {}
    )
    for n in range(1002):
        row, _ = await store.create_operation(
            config["binding"], "instance-a", "synthetic", f"done-{n}", {}
        )
        await store.finish_operation(row["id"], Reply(200, "COMMITTED", False, {}))
    result = await store.operations()
    assert result["total"] == 1001 and result["pending"] == 1
    await store.db.execute(
        "UPDATE operations SET updated_at=?", (time.time() - 8 * 86400,)
    )
    await store.prune()
    assert (await store.operations())["total"] == 1
    assert (await store.operation(first["id"]))["state"] == "UNKNOWN"


async def test_process_crash_recovery_original_confirm_only(tmp_path):
    directory = tmp_path / "crash"
    script = """import asyncio,os,sys
from pathlib import Path
from iris_memory.control.store import Store
async def run():
 s=Store(Path(sys.argv[1]));await s.open();c=await s.connection(0,'http://localhost:12345','host-secret');c=await s.observe('status',c['binding'],{},instance_id='instance-a');await s.create_operation(c['binding'],'instance-a','synthetic','original-key',{'value':7});os._exit(17)
asyncio.run(run())"""
    env = os.environ | {
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(directory)],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 17
    store = Store(directory)
    await store.open()
    calls = []

    async def submit(key, payload):
        pytest.fail("Unknown submission must never replay")

    async def confirm(key, payload):
        calls.append((key, payload))
        return Reply(200, "COMMITTED", False, {})

    try:
        config = await store.settings()
        dispatcher = OriginalOperations(
            store, (OperationDescriptor("synthetic", lambda _: None, submit, confirm),)
        )
        op = await dispatcher.begin(
            config["binding"], "instance-a", "synthetic", "original-key", {"value": 7}
        )
        assert op["state"] == "UNKNOWN" and not calls
        await dispatcher.confirm(op["id"])
        assert calls == [("original-key", {"value": 7})]
        assert (await store.operations())["pending"] == 0
    finally:
        await store.close()


async def test_lost_response_and_instance_switch(store):
    config = await bound(store)
    sent = []

    async def send(key, payload):
        sent.append(key)
        raise TimeoutError()

    async def confirm(key, payload):
        return Reply(202, "UNCONFIRMED", True, {})

    dispatcher = OriginalOperations(
        store, (OperationDescriptor("synthetic", lambda _: None, send, confirm),)
    )
    with pytest.raises(TimeoutError):
        await dispatcher.begin(
            config["binding"], "instance-a", "synthetic", "key", {"a": 1}
        )
    row = await dispatcher.begin(
        config["binding"], "instance-a", "synthetic", "key", {"a": 1}
    )
    assert sent == ["key"] and row["state"] == "UNKNOWN"
    await dispatcher.confirm(row["id"])
    updated = await store.observe(
        "status", config["binding"], {}, instance_id="instance-b"
    )
    assert updated["binding"] != config["binding"]
    with pytest.raises(ControlError, match="BINDING_MISMATCH"):
        await dispatcher.confirm(row["id"])


async def test_production_dispatcher_has_no_business_ports(store):
    with pytest.raises(ControlError, match="NOT_IMPLEMENTED"):
        await OriginalOperations(store).begin("x", "instance-a", "accept", "key", {})
