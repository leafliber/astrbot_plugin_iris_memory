"""Run reproducible checks and bind logs to exact plugin files in a JSON ledger."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(names))
        if name and (ROOT / name).is_file()
    }
    digest = hashlib.sha256(
        json.dumps(
            files, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {"files": files, "tree_sha256": digest}


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("aiohttp", "aiosqlite", "pytest", "pytest-asyncio", "ruff")
        },
        "host_checkout": os.environ.get("IRIS_HOST_CHECKOUT"),
        "host_root": os.environ.get("ASTRBOT_ROOT"),
        "real_core": False,
        "real_model_calls": 0,
        "real_chat_sources": 0,
    }
    commands = [
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                f"--junitxml={output / 'pytest.xml'}",
            ],
        ),
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("javascript", ["node", "--check", "pages/iris/app.js"]),
        ("diff", ["git", "diff", "--check"]),
    ]
    ledger = {"environment": environment, "runs": []}
    for name, command in commands:
        before = fingerprint()
        start = time.time()
        log = output / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=120
            )
        after = fingerprint()
        ledger["runs"].append(
            {
                "name": name,
                "command": command,
                "cwd": str(ROOT),
                "started_at": start,
                "finished_at": time.time(),
                "exit_code": result.returncode,
                "log": str(log),
                "tested_fingerprint": before,
                "unchanged_during_run": before == after,
            }
        )
        (output / "checks.json").write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2)
        )
        print(name, result.returncode, flush=True)
    return int(any(run["exit_code"] for run in ledger["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
