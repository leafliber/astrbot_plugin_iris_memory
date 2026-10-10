"""Opt-in real iris-core main, empty model config, offline ephemeral credential.

Only the credential creation subprocess stdout is held in memory. Never log it,
put it in a fixture, write it into a config file, or include it in assertion output.
"""

import asyncio
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iris_memory.vendor.iris_client import AsyncIrisClient, RetryPolicy


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
            # Store only public execution metadata, never client repr or configuration.
            (tmp_path / "integration-result.json").write_text(
                json.dumps(
                    {
                        "core_head": head,
                        "result": "passed",
                        "model_calls_expected": 0,
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
        if credential.encode() in output:
            pytest.fail("Core emitted the ephemeral credential; log withheld", pytrace=False)
        (tmp_path / "core-service.log").write_bytes(output)
        credential = None
