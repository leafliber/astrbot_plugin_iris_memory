import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from iris_memory.application import Application
from iris_memory.control.barrier import BindingBarrier
from iris_memory.core_client.ingress import (
    IngressLimits,
    canonical,
    event_v2,
    interpretation,
)
from iris_memory.delivery.store import Capacity
from iris_memory.errors import ControlError
from iris_memory.platforms.onebot import identity, snapshot
from tests.ingress_fixture import HTTPFixture, event

LIMITS = IngressLimits(8192, 2, 512, 1048576, 65536)


@pytest.fixture
async def core():
    fixture = HTTPFixture()
    await fixture.start()
    yield fixture
    await fixture.close()


async def configure(app, core, size=1):
    await app.save_connection(0, core.base)
    config = await app.store.settings()
    await app.store.observe("status", config["binding"], {}, instance_id="instance-a")
    for first in range(0, size, 8):
        config = await app.store.settings()
        entries = [f"entry-{i}" for i in range(first, min(first + 8, size))]
        token = f"secret-group-{first}"
        core.tokens[token] = entries
        await app.group_save(
            config["revision"], f"group-{first}", "host", entries, token
        )
        await app.group_check(f"group-{first}")
    config = await app.store.settings()
    await app.sources.limits(config["binding"], "v1", asdict(LIMITS))
    for i in range(size):
        source = {
            "platform_instance": "onebot-test",
            "bot_self": "100",
            "kind": "group",
            "conversation_id": f"group-{i}",
            "entry_id": f"entry-{i}",
            "group_id": f"group-{(i // 8) * 8}",
            "enabled": False,
        }
        config = await app.store.settings()
        await app.source_save(config["revision"], source)
        config = await app.store.settings()
        await app.source_save(config["revision"], {**source, "enabled": True})
    for key in ("plugin.enabled", "observation.enabled"):
        config = await app.store.settings()
        await app.save_intent(config["revision"], key, True)


async def until(predicate, seconds=10):
    async with asyncio.timeout(seconds):
        while not await predicate():  # noqa: ASYNC110 - observe an external HTTP ledger
            await asyncio.sleep(0.02)


def test_v2_canonical_and_actual_limits():
    material, _, _ = snapshot(event(), 262144)
    value = material["event"]
    assert event_v2(value, LIMITS)["body"] == "  原文\n\t中文  "
    assert b"\\u000a" in canonical(value) and b"\\u0009" in canonical(value)
    assert canonical({"x": "\\n"}) == b'{"x":"\\\\n"}'
    with pytest.raises(ControlError, match="CANONICAL_LIMIT"):
        event_v2(value, IngressLimits(256, 2, 512, 1048576, 65536))
    with pytest.raises(ControlError, match="VERSION"):
        event_v2({**value, "event_version": 1}, LIMITS)


@pytest.mark.parametrize(
    "state,body,source,coverage",
    [
        ("MISSING", None, None, "UNSPECIFIED"),
        ("FAILED", None, "external", "UNSPECIFIED"),
        ("COMPLETE", "看见中文", "external", "COMPLETE"),
        ("PARTIAL", "部分", "external", "EXPLICIT_PARTIAL"),
        ("EMPTY", "", "external", "COMPLETE"),
        ("REFUSED", "敏感信息无法访问", "external", "UNSPECIFIED"),
    ],
)
def test_interpretation_states(state, body, source, coverage):
    value = dict(status=state, text=body, source_ref=source, coverage=coverage)
    interpretation(value, 512)
    with pytest.raises(ControlError):
        interpretation({**value, "coverage": "invented"}, 512)
    if state in {"COMPLETE", "PARTIAL"}:
        with pytest.raises(ControlError):
            interpretation({**value, "text": "中" * 171}, 512)


async def test_dispatch_barrier_two_readers_and_exclusive_change():
    gate = BindingBarrier()
    active = asyncio.Event()
    entered = []

    async def reader(n):
        async with gate.read():
            entered.append(n)
            await active.wait()

    tasks = [asyncio.create_task(reader(i)) for i in range(2)]
    await asyncio.sleep(0)
    changed = asyncio.Event()

    async def writer():
        async with gate:
            changed.set()

    w = asyncio.create_task(writer())
    await asyncio.sleep(0)
    assert len(entered) == 2 and not changed.is_set()
    active.set()
    await asyncio.gather(*tasks, w)
    assert changed.is_set()


@pytest.mark.parametrize("size", [5, 8, 10])
async def test_topology_mixed_engineering_sample(tmp_path, core, size):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core, size)
        started = time.monotonic()
        core.drop_accept = True
        peaks = {
            "http_waiting": 0,
            "event_bytes": 0,
            "media_reserved_bytes": 0,
            "disk_bytes": 0,
            "unknown": 0,
        }

        async def sample():
            state = await app.delivery.status()
            peaks["http_waiting"] = max(peaks["http_waiting"], app.admission.waiting)
            for name in ("event_bytes", "media_reserved_bytes"):
                peaks[name] = max(peaks[name], state[name])
            peaks["disk_bytes"] = max(
                peaks["disk_bytes"],
                sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()),
            )
            peaks["unknown"] = max(peaks["unknown"], state["states"].get("UNKNOWN", 0))
            return state

        for n in range(20):
            for source in range(size):
                await app.delivery.capture(
                    event(
                        n,
                        f"group-{source}",
                        media=core.base + "/original.png" if n == 0 else None,
                    )
                )
                await sample()

        async def drained():
            state = await sample()
            return (
                state["counts"].get("confirmed", 0) == size * 20
                and state["event_bytes"] == 0
                and state["media_reserved_bytes"] == 0
                and state["active_jobs"] == 0
            )

        await until(drained, 25)
        status = await app.delivery.status()
        assert len(core.accepts) == size * 20 == len(set(core.accepts))
        assert status["peak_jobs"] <= 2 and status["media_reserved_bytes"] == 0
        assert status["event_bytes"] == 0 and status["counts"].get("gaps", 0) == 0
        assert [len(g["entries"]) for g in (await app.source_status())["groups"]] == (
            [8, 2] if size == 10 else [size]
        )
        for i in range(size):
            original = [
                v["input"]["event"]["external_event_id"]
                for v in core.originals.values()
                if v["entry_id"] == f"entry-{i}"
            ]
            assert original == [str(n) for n in range(20)]
        assert all(v["bytes"] == core.media_data for v in core.uploads.values())
        if size == 10:
            denied = await app.host.ingress(
                core.base,
                "secret-group-0",
                "entry-9",
                "accept",
                {"key": "denied", "event": snapshot(event(99), 262144)[0]["event"]},
                limits=LIMITS,
            )
            assert denied.http_status == 403 and len(core.accepts) == 200
        report = {
            "sources": size,
            "events": size * 20,
            "status": status,
            "real_core": False,
            "peaks_sampled": peaks,
            "elapsed_seconds": time.monotonic() - started,
            "disconnect": "one accepted response dropped; original resolution only",
            "actual_submit_count": len(core.accepts),
            "actual_confirm_count": len(core.confirms),
        }
        async with app.store.lock:
            async with app.store.db.execute(
                "SELECT updated-created FROM delivery_events"
            ) as cursor:
                latencies = sorted(r[0] for r in await cursor.fetchall())
        report["latency_seconds"] = {
            "max": max(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[int(len(latencies) * 0.95)],
        }
        (tmp_path / "sample.json").write_text(json.dumps(report))
        if destination := os.environ.get("IRIS_S2_EVIDENCE"):
            await asyncio.to_thread(
                Path(destination, f"sample-{size}.json").write_text,
                json.dumps(report, ensure_ascii=False, indent=2),
            )
    finally:
        await app.terminate()


async def test_unknown_fifo_close_binding_and_original_confirmation(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core, 2)
        core.unknown = True
        await app.delivery.capture(event(1))

        async def submitted():
            return len(core.accepts) == 1

        await until(submitted)
        await app.delivery.capture(event(2))
        await app.delivery.capture(event(1))
        await app.delivery.capture(event(3, text="same text"))
        await app.delivery.capture(event(4, text="same text"))
        config = await app.store.settings()
        await app.save_intent(config["revision"], "plugin.enabled", False)
        status = await app.delivery.status()
        assert status["counts"]["duplicate"] == 1 and len(core.accepts) == 1
        first = min(status["items"], key=lambda x: x["seq"])
        core.unknown = False
        await app.delivery_confirm(first["id"])

        async def confirmed():
            return (await app.delivery.status())["counts"].get("confirmed") == 1

        await until(confirmed)
        assert len(core.accepts) == 1 and core.confirms
        await app.terminate()
        app = Application(tmp_path)
        await app.initialize()
        config = await app.store.settings()
        await app.save_intent(config["revision"], "plugin.enabled", True)

        async def all_done():
            return (await app.delivery.status())["counts"].get("confirmed") == 4

        await until(all_done)
        assert len(core.accepts) == 4
        assert (
            len(
                {
                    v["input"]["event"]["external_event_id"]
                    for v in core.originals.values()
                }
            )
            == 4
        )
    finally:
        await app.terminate()


async def test_capacity_unsupported_and_frozen_material(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        app.delivery.queue.capacity = Capacity(events=2, source_events=2)
        core.unknown = True
        first = event(1)
        await app.delivery.capture(first)
        first.message_obj.raw_message["message"][0]["data"]["text"] = (
            "changed after callback"
        )
        await app.delivery.capture(event(2))
        await app.delivery.capture(event(3))
        status = await app.delivery.status()
        assert status["counts"]["queued"] == 2 and status["counts"]["gaps"] == 1
        rows = [await app.delivery.queue.get(row["id"]) for row in status["items"]]
        assert all(
            row["material"]["event"]["body"] != "changed after callback" for row in rows
        )
        assert len(core.accepts) <= 1
    finally:
        await app.terminate()


async def test_media_restart_inspects_before_resuming(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        app.delivery.task.cancel()
        await asyncio.gather(app.delivery.task, return_exceptions=True)
        await app.delivery.capture(event(1, media=core.base + "/original.png"))
        row = await app.delivery.queue.get(
            (await app.delivery.status())["items"][0]["id"]
        )
        medium = row["material"]["media"][0]
        record = await app.delivery.media.download(row, medium, LIMITS.blob_bytes)
        token = "secret-group-0"
        original = {"key": record["original_key"], "modality": "IMAGE"}
        await app.delivery.queue.update_media(record["id"], dispatched=1)
        reply = await app.host.ingress(
            core.base, token, "entry-0", "media/begin", original
        )
        upload = reply.data["receipt"]["result"]["upload_id"]
        core.uploads[upload]["state"] = "REUPLOAD_REQUIRED"
        before = len(core.requests)
        reference = await app.delivery.media.upload(row, medium, token, LIMITS)
        assert core.requests[before] == "inspect" and reference == upload
        assert core.uploads[upload]["bytes"] == core.media_data
    finally:
        await app.terminate()


def test_plain_mentions_long_and_no_reliable_id():
    raw = event()
    raw.message_obj.raw_message["message"] = [{"type": "at", "data": {"qq": "100"}}]
    assert snapshot(raw, 262144)[2] == "PLATFORM_STRUCTURE_UNREPRESENTABLE"
    raw = event()
    raw.message_obj.raw_message.pop("message_id")
    a = snapshot(raw, 262144)[0]["event"]["client_event_key"]
    b = snapshot(raw, 262144)[0]["event"]["client_event_key"]
    assert a != b
    assert identity(raw) != identity(event(conversation="different"))


@pytest.mark.parametrize(
    "kind",
    [
        "mention",
        "forward",
        "edit",
        "too_many_media",
        "oversize_record",
        "missing_media",
    ],
)
async def test_unrepresentable_raw_shapes_remain_explicit(tmp_path, core, kind):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await configure(app, core)
        item = event()
        raw = item.message_obj.raw_message
        if kind == "mention":
            raw["message"] = [{"type": "at", "data": {"qq": "100"}}]
        elif kind == "forward":
            raw["message"] = [{"type": "forward", "data": {"id": "forward-id"}}]
        elif kind == "edit":
            raw["post_type"] = "notice"
        elif kind == "too_many_media":
            raw["message"] = [
                {"type": "image", "data": {"url": core.base + "/original.png"}}
            ] * 3
        elif kind == "oversize_record":
            raw["message"] = [{"type": "text", "data": {"text": "中" * 100000}}]
        else:
            raw["message"] = [
                {"type": "image", "data": {"file": "untrusted-local-path"}}
            ]
        await app.delivery.capture(item)
        status = await app.delivery.status()
        assert status["states"] == {"BLOCKED": 1} and not core.accepts
        row = await app.delivery.queue.get(status["items"][0]["id"])
        assert row["reason"] and status["counts"]["gaps"] == 1
        if kind == "oversize_record":
            assert (
                row["material"]["raw"] is None
                and row["material"]["full_raw_retained"] is False
            )
        else:
            assert row["material"]["raw"] == raw
    finally:
        await app.terminate()


def test_ten_source_group_allocation_is_eight_plus_two():
    from iris_memory.control.sources import validate_group_topology

    group = {
        "group_id": "first",
        "host_id": "host",
        "entries": [f"entry-{i}" for i in range(8)],
    }
    validate_group_topology([group], "second", "host", ["entry-8", "entry-9"])
    with pytest.raises(ControlError, match="EIGHT_PLUS_TWO"):
        validate_group_topology(
            [{**group, "entries": group["entries"][:5]}],
            "second",
            "host",
            [f"entry-{i}" for i in range(5, 10)],
        )
    with pytest.raises(ControlError, match="EIGHT_PLUS_TWO"):
        validate_group_topology(
            [group], "second", "host", ["entry-8", "entry-9", "entry-10"]
        )
