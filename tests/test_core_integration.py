"""Opt-in real iris-core main, empty model config, offline ephemeral credential.

Only the credential creation subprocess stdout is held in memory. Never log it,
put it in a fixture, write it into a config file, or include it in assertion output.
"""

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iris_memory.bridge import Bridge
from iris_memory.config import Config
from iris_memory.vendor.iris_client import AsyncIrisClient, RetryPolicy
from tests.conftest import Result
from tests.test_bridge import Event
from tests.test_capture import notice_event


@pytest.mark.integration
async def test_real_core_receive_prepare_search_goals_notifications(tmp_path):
    source = os.environ.get("IRIS_CORE_PATH")
    if not source:
        pytest.skip("Set IRIS_CORE_PATH to the read-only iris-core checkout")
    core = Path(source).resolve()
    python = Path(os.environ.get("IRIS_CORE_PYTHON", str(core / ".venv/bin/python")))
    if not python.is_file() or not (core / "src/iris/cli.py").is_file():
        pytest.fail("A prepared iris-core environment is required", pytrace=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip()
    env = {"PATH": os.defpath, "PYTHONPATH": str(core / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    base = [
        str(python),
        "-B",
        "-c",
        # Uvicorn's logging config re-enables disabled loggers, but retains
        # logger filters. Keep setup codes out of both console and file handlers.
        "import logging; logging.getLogger('iris.auth').addFilter(lambda record: False); "
        "from iris.cli import main; raise SystemExit(main())",
        "--data-dir",
        str(tmp_path / "data"),
    ]

    def offline(*args):
        result = subprocess.run(
            base + list(args),
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode:
            pytest.fail("Offline core CLI failed (output withheld)", pytrace=False)
        return result.stdout

    offline("setup", "--name", "Iris integration test")
    credential = json.loads(
        offline("tokens", "create", "--host", "astrbot-test", "--prefix", "astrbot:")
    )["token"]
    models = tmp_path / "empty-models.toml"
    models.write_text("# Deliberately empty: no model endpoints or credentials.\n")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        base + ["serve", "--port", str(port), "--no-open", "--models-config", str(models)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        async with AsyncIrisClient(
            f"http://127.0.0.1:{port}",
            token=credential,
            timeout=2,
            retry=RetryPolicy(max_retries=0),
        ) as client:
            until = time.monotonic() + 20
            while True:
                try:
                    status = await client.status()
                    if status.get("service") == "ready":
                        break
                except Exception:
                    pass
                if process.poll() is not None or time.monotonic() > until:
                    pytest.fail(
                        "Core failed to become ready (credential/output withheld)", pytrace=False
                    )
                await asyncio.sleep(0.1)
            entry = "astrbot:integration:group:escaped%3F%23"
            message = {
                "sender": "测试用户",
                "account_id": "test-user",
                "platform": "astrbot:test",
                "content": "我喜欢桂花茶",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "dedupe_key": "integration-message-1",
                "entry_kind": "group",
            }
            receipt = await client.receive_message(entry, message)
            repeated = await client.receive_message(entry, message)
            assert receipt["message_ids"] == repeated["message_ids"]
            prepared = await client.prepare(entry, {"judge": False})
            assert prepared["recent_messages"][0]["content"] == "我喜欢桂花茶"
            assert (
                set(("persona", "memories", "recent_messages", "state", "goals", "hints"))
                <= prepared.keys()
            )
            persona = await client.prepare(
                entry, {"judge": False, "recent_limit": 0, "memory_limit": 0, "goal_limit": 0}
            )
            assert "persona" in persona
            search = await client.search({"text": "桂花茶"}, entry_id=entry)
            assert search["memories"] == []  # No models: don't claim text was learned.
            feedback = await client.feedback(prepared["recall_id"], [])
            assert feedback["accepted"] == []
            learn = await client.learn(entry)
            assert learn["accepted"] is True
            goal_body = {
                "content": "测试喝水提醒",
                "entry_id": entry,
                "host_key": "integration-goal-1",
                "deadline": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "reminder_minutes": 0,
            }
            created = await client.create_goal(goal_body)
            repeated_goal = await client.create_goal(goal_body)
            goal_id = created["goal"]["id"]
            assert goal_id == repeated_goal["goal"]["id"]
            goals = await client.list_goals({"entry_id": entry})
            goal = next(g for g in goals["items"] if g["id"] == goal_id)
            assert goal["entry_id"] == entry and goal["state"] == "open"
            until = time.monotonic() + 10
            while True:
                notifications = await client.pull_notifications(after=0)
                if any(
                    n.get("goal_id") == goal_id and n.get("kind") == "goal_reminder"
                    for n in notifications["items"]
                ):
                    break
                if time.monotonic() > until:
                    pytest.fail("No due-goal notification appeared", pytrace=False)
                await asyncio.sleep(0.2)
            assert notifications["next_cursor"] >= 1
            assert await client.pull_notifications(after=0)  # Replay is supported.
            after_poll = await client.list_goals({"entry_id": entry})
            assert next(g for g in after_poll["items"] if g["id"] == goal_id)["state"] == "open"
            completed = await client.update_goal(
                goal_id, {"state": "completed", "expected_revision": goal["revision"]}
            )
            assert completed["state"] == "completed"
            final_status = await client.status()
            assert final_status["usage"]["today"]["calls"] == 0
            await verify_output_notice_capture(f"http://127.0.0.1:{port}", credential, tmp_path)
            latency = await verify_live_sse(
                f"http://127.0.0.1:{port}",
                credential,
                entry,
                notifications["next_cursor"],
                tmp_path,
            )
            # Store only public execution metadata, never client repr or configuration.
            (tmp_path / "integration-result.json").write_text(
                json.dumps(
                    {
                        "core_head": head,
                        "result": "passed",
                        "model_calls_expected": 0,
                        "sse_enqueue_seconds": round(latency, 4),
                        "sse_platform_sends": 1,
                        "steps": [
                            "receive",
                            "dedupe",
                            "prepare",
                            "persona",
                            "search",
                            "feedback",
                            "learn",
                            "goals",
                            "notifications",
                            "complete",
                            "output_provenance_and_scene_sender",
                            "notice_cached_name_and_no_account_named_people",
                            "sse_enqueue",
                            "sse_and_poll_single_send",
                            "sse_unload",
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=5)
        if credential.encode() in output or "首次设置码".encode() in output:
            pytest.fail("Core emitted a temporary credential; log withheld", pytrace=False)
        (tmp_path / "core-service.log").write_bytes(output)
        credential = None


async def verify_live_sse(base_url, credential, entry, pull_cursor, tmp_path):
    # The fresh goal is created only after initial repair/connection. A 60s pull
    # interval makes delivery within 2s evidence of push, not a periodic poll.
    client = AsyncIrisClient(
        base_url, token=credential, timeout=2, retry=RetryPolicy(max_retries=0)
    )
    config = Config(
        base_url=base_url,
        token=credential,
        images_enabled=False,
        reminder_interval=60,
        request_timeout=2,
        reminder_mode="active",
        reminder_target="integration:GroupMessage:events",
        reminder_entry_id=entry,
    )
    enqueued, release = asyncio.Event(), asyncio.Event()
    sends, logs = [], []

    async def send(target, text):
        enqueued.set()
        await release.wait()
        sends.append(True)
        return True

    bridge = Bridge(config, client, tmp_path / "plugin-events", logs.append, send)
    # Simulate an existing plugin that already handled the previous test goal.
    bridge.inbox.advance(pull_cursor)
    bridge.start()
    try:
        async with asyncio.timeout(5):
            while bridge.diagnostics.reminder_transport["mode"] != "sse":
                await asyncio.sleep(0.01)
        started = time.monotonic()
        await client.create_goal(
            {
                "content": "SSE 集成测试提醒",
                "entry_id": entry,
                "host_key": "integration-sse-goal",
                "deadline": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                "reminder_minutes": 0,
            }
        )
        await asyncio.wait_for(enqueued.wait(), 2)
        latency = time.monotonic() - started
        assert latency < 2  # Core queries the stream every ~1s, with scheduling margin.
        assert bridge.inbox.count() == 1
        notice = bridge.inbox.pending()[0]
        assert notice["kind"] == "goal_reminder"
        release.set()
        await bridge.poll()
        await bridge.poll()
        await bridge.queue.join()
        assert len(sends) == 1 and bridge.inbox.count() == 0
        assert bridge.counts["reminder_sent"] == 1
        assert (await client.status())["usage"]["today"]["calls"] == 0
        assert not logs
        return latency
    finally:
        release.set()
        await bridge.close()
        assert all(task.done() for task in bridge.tasks + bridge.reminders.tasks)


async def verify_output_notice_capture(base_url, credential, tmp_path):
    """Exercise the bridge over real HTTP, then inspect only this fresh test DB."""
    client = AsyncIrisClient(
        base_url, token=credential, timeout=2, retry=RetryPolicy(max_retries=0)
    )
    config = Config(
        base_url=base_url,
        token=credential,
        request_timeout=2,
        images_enabled=False,
        reminders_enabled=False,
    )
    logs = []

    async def no_active_send(*args):
        raise AssertionError("capture test does not send active reminders")

    def people():
        uri = (tmp_path / "data/iris.db").as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            return dict(db.execute("SELECT id,name FROM subjects WHERE kind='person'"))

    # Core's pre-existing scene subject is itself kind=person. Compare identities
    # before/after so the assertion checks creations rather than reserved labels.
    original_people = people()
    bridge = Bridge(config, client, tmp_path / "plugin-capture", logs.append, no_active_send)
    bridge.start()
    try:
        for kind in ("poke", "group_increase", "group_request", "group_recall"):
            event = notice_event(kind, name="10001", instance="capture", message_id=kind)
            await bridge.receive(event)
            assert await event.get_extra("iris_received")
        entry = event.get_extra("iris_entry")
        scene = await client.prepare(entry.id, {"judge": False})
        assert len(scene["recent_messages"]) == 4
        assert all(
            m["kind"] == "event" and m["sender_subject_id"] == "scene"
            for m in scene["recent_messages"]
        )
        assert all(
            "10001" not in m["content"] and m["quote_author_subject_id"] is None
            for m in scene["recent_messages"]
        )
        assert people() == original_people

        chat = Event(user="10004", instance="capture", message_id="named-chat")
        chat.get_sender_name = lambda: "测试昵称"
        await bridge.receive(chat)
        chat.result = Result("其他插件的复读")
        await chat.send(chat.result)
        chat.result = None  # Unknown source is also non-role output.
        await chat.send(Result("来源不明的输出"))
        chat.result = Result("Iris 的 LLM 回复", llm=True)
        await chat.send(chat.result)
        cached = notice_event("poke", name="10004", instance="capture", message_id="cached")
        cached.user = "10004"
        await bridge.receive(cached)
        await bridge.queue.join()
        assert not logs
        recent = (await client.prepare(entry.id, {"judge": False, "recent_limit": 100}))[
            "recent_messages"
        ]
        outputs = [m for m in recent if m["content"].startswith("［机器人账号发送，非角色发言］")]
        assert len(outputs) == 2
        assert all(
            m["kind"] == "event" and m["sender_subject_id"] == "scene" and m["sender_name"] != "我"
            for m in outputs
        )
        own = [m for m in recent if m["sender_subject_id"] == "self"]
        assert len(own) == 1 and own[0]["content"] == "Iris 的 LLM 回复"
        assert own[0]["kind"] == "self_output"
        last = recent[-1]
        assert last["sender_name"] == "测试昵称" and last["kind"] == "message"
        assert last["content"] == "有成员戳了戳某人"
        current_people = people()
        assert len(current_people) == len(original_people) + 1
        assert {
            name for subject, name in current_people.items() if subject not in original_people
        } == {"测试昵称"}
        assert not any(name.isdecimal() for name in current_people.values())
    finally:
        await bridge.close()
