import asyncio
import json
import sys

import pytest

from iris_memory.api import PagesAPI
from iris_memory.config import defaults, validate
from iris_memory.control import Control
from iris_memory.errors import IrisError
from iris_memory.outbox import Outbox


async def test_control_without_optional_dependencies(tmp_path, host, monkeypatch):
    monkeypatch.setitem(sys.modules, "iris_memory_core", None)
    monkeypatch.setitem(sys.modules, "iris_memory_sdk", None)
    c = Control(tmp_path, host)
    await c.start()
    assert not c.modules
    assert not (tmp_path / "core").exists()
    api = PagesAPI(c)
    assert (await api.dispatch("status", username="admin"))["accepting"]
    with pytest.raises(IrisError, match="登录"):
        await api.dispatch("status", username="")
    with pytest.raises(IrisError, match="未知"):
        await api.dispatch("../core/raw", username="admin")
    await c.close()
    await c.close()


async def test_sqlite_revision_persistence_and_bounded_queue(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    await c.apply({"queue_items": 1}, 0)
    q = Outbox(c)
    assert await q.enqueue("one", {"text": "hello"})
    assert not await q.enqueue("one", {"text": "hello"})
    with pytest.raises(IrisError, match="队列已满"):
        await q.enqueue("two", {})
    await q.finish("one", "accepted")
    rows = await c.store.run(
        lambda db: list(db.execute("SELECT payload FROM deliveries"))
    )
    assert rows[0][0] == "{}"
    await c.close()
    await c.start()
    assert c.revision == 1
    assert c.settings["queue_items"] == 1
    await c.close()


async def test_concurrent_config_cas_and_validation(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    results = await asyncio.gather(
        c.apply({"context_tokens": 100}, 0),
        c.apply({"context_tokens": 200}, 0),
        return_exceptions=True,
    )
    assert sum(isinstance(r, IrisError) for r in results) == 1
    with pytest.raises(IrisError):
        await c.apply({"modules": {"context": True}}, c.revision)
    with pytest.raises(IrisError):
        await c.apply({"remote_url": "https://password:secret@example.org"}, c.revision)
    assert c.accepting
    await c.close()


async def test_failed_start_rolls_back_configuration(tmp_path, host):
    class Failing:
        def __init__(self, c):
            self.closed = False

        async def start(self):
            raise IrisError("test_failure", "failure")

        async def close(self):
            self.closed = True

    c = Control(tmp_path, host, backend_factory=Failing)
    await c.start()
    with pytest.raises(IrisError):
        await c.apply({"modules": {"memory": True}}, 0)
    assert c.revision == 0 and c.settings == defaults() and c.accepting
    assert not c.modules
    assert await c.store.get("config", "active") is None
    await c.close()


async def test_fifty_hot_cycles_and_shared_ownership(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    before = {t for t in asyncio.all_tasks()}
    for _ in range(50):
        await c.apply({"modules": {"persona": True, "diagnostics": True}}, c.revision)
        c.logs.emit("cycle", token="safe")
        await c.apply({"modules": {"persona": False, "diagnostics": False}}, c.revision)
    assert not c.modules and not host.shared_closed
    assert {t for t in asyncio.all_tasks()} <= before
    await c.close()


async def test_inflight_task_cancelled_on_hot_disable(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    entered = asyncio.Event()

    class Slow:
        async def work(self):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            pass

    c.modules["persona"] = Slow()
    task = asyncio.create_task(c.run("persona", "work"))
    await entered.wait()
    await c.apply({}, 0)
    with pytest.raises(IrisError, match="取消"):
        await task
    await c.close()


async def test_budget_is_atomic_and_persistent(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    await c.apply({"daily_model_calls": 2, "daily_model_tokens": 100}, 0)
    results = await asyncio.gather(
        *(c.budget.reserve(50) for _ in range(3)), return_exceptions=True
    )
    assert sum(isinstance(r, IrisError) for r in results) == 1
    await c.close()
    await c.start()
    with pytest.raises(IrisError, match="耗尽"):
        await c.budget.reserve(1)
    await c.close()


async def test_optional_logs_redact_rotate_and_filter(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    assert not (tmp_path / "logs").exists()
    await c.apply(
        {
            "remote_token": "very-secret-token",
            "log_megabytes": 1,
            "modules": {"diagnostics": True},
        },
        0,
    )
    c.logs.emit(
        "test.secret",
        api_key="abc",
        body="body secret",
        error=ValueError("Bearer very-secret-token"),
    )
    await c.apply({"modules": {"diagnostics": False}}, c.revision)
    text = "".join(p.read_text() for p in (tmp_path / "logs").glob("*.jsonl"))
    assert (
        "very-secret-token" not in text
        and "body secret" not in text
        and '"api_key": "abc"' not in text
    )
    assert "redacted" in text and "sha256" in text
    assert len((await c.logs.query(operation="test.secret"))["records"]) == 1
    await c.close()


@pytest.mark.parametrize(
    "change",
    [
        {"queue_items": True},
        {"modules": {"imaginary": True}},
        {"log_level": "SECRET"},
        {"remote_bindings": {"x": {}}},
        {"timezone": "../bad"},
        {"model_timeout": float("nan")},
    ],
)
def test_config_rejects_invalid_values(change):
    with pytest.raises(IrisError):
        validate(defaults(), change)


def test_only_mode_in_astrbot_schema():
    from pathlib import Path

    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text())
    assert set(schema) == {"core_mode"}


async def test_concurrent_conversation_updates_do_not_drop_capture(
    tmp_path, host, identity
):
    c = Control(tmp_path, host)
    await c.start()
    await asyncio.gather(*(c.remember_conversation(identity) for _ in range(20)))
    assert (await c.store.get("conversations", identity.key))["revision"] == 20
    await c.close()
