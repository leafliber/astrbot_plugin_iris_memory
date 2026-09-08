"""Optional bounded structured logging, independent of Core and model providers."""

import asyncio
import hashlib
import json
import logging
import re
import time
import traceback
from pathlib import Path


class Diagnostics:
    def __init__(self, directory, logger=None):
        self.directory = Path(directory) / "logs"
        self.logger = logger or logging.getLogger("iris_memory.v4")
        self.queue = asyncio.Queue(maxsize=512)
        self.bytes = 0
        self.dropped = 0
        self.sequence = 0
        self.task = None
        self.settings = {}
        self.secrets = []

    def sanitize(self, value):
        if isinstance(value, dict):
            return {
                k: "[redacted]"
                if re.search(
                    r"(?i)(password|secret|api_key|authorization|bearer|remote_token)",
                    k,
                )
                else self.sanitize(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.sanitize(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                if secret:
                    value = value.replace(secret, "[redacted]")
            return re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [redacted]", value)
        return value

    async def start(self, settings):
        self.settings = settings
        self.secrets = [settings.get("remote_token", "")]
        if self.task:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task = asyncio.create_task(self._writer(), name="iris-log-writer")

    def emit(self, operation, *, level="INFO", error=None, body=None, **fields):
        if error:
            fields.update(
                error_type=type(error).__name__,
                error=str(error),
                stack="".join(traceback.format_exception(error)),
            )
        if level in {"ERROR", "WARNING"}:
            self.logger.warning(
                "Iris %s: %s",
                operation,
                self.sanitize(str(error or fields.get("reason", ""))),
            )
        if (
            self.task is None
            or logging._nameToLevel[level]
            < logging._nameToLevel[self.settings["log_level"]]
        ):
            return
        self.sequence += 1
        record = {
            "time": time.time(),
            "sequence": self.sequence,
            "operation": operation,
            "level": level,
            **fields,
        }
        if body is not None:
            text = json.dumps(body, ensure_ascii=False, default=str)
            record["body"] = (
                body
                if self.settings.get("log_bodies")
                else {
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "bytes": len(text.encode()),
                }
            )
        text = json.dumps(self.sanitize(record), ensure_ascii=False, default=str) + "\n"
        size = len(text.encode())
        if self.queue.full() or size + self.bytes > 1048576:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                self.logger.warning(
                    "Iris log buffer overflow; dropped=%d", self.dropped
                )
            return
        self.bytes += size
        self.queue.put_nowait((text, size))

    def _write(self, text):
        maximum = self.settings["log_megabytes"] * 1048576
        path = self.directory / "current.jsonl"
        if path.exists() and path.stat().st_size + len(text.encode()) > min(
            1048576, maximum
        ):
            path.rename(self.directory / f"{time.time_ns()}.jsonl")
        with path.open("a", encoding="utf-8") as file:
            file.write(text)
        path.chmod(0o600)
        files = sorted(self.directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        cutoff = time.time() - self.settings["log_days"] * 86400
        for file in files:
            if file == path:
                continue
            size = file.stat().st_size
            if total > maximum or file.stat().st_mtime < cutoff:
                file.unlink()
                total -= size

    async def _writer(self):
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                text, size = item
                try:
                    await asyncio.to_thread(self._write, text)
                except Exception:
                    self.dropped += 1
                    self.logger.error("Iris log write failed; dropped=%d", self.dropped)
                finally:
                    self.bytes -= size
            finally:
                self.queue.task_done()

    async def close(self):
        if not self.task:
            return
        task, self.task = self.task, None
        await self.queue.put(None)
        await task

    async def query(self, *, limit=100, before=None, operation="", level=""):
        limit = max(1, min(int(limit), 200))
        if not self.directory.exists():
            return {"records": [], "dropped": self.dropped}

        def read():
            records = []
            # Rotation bounds each file; read at most 2 MiB per query.
            budget = 2097152
            for file in sorted(
                self.directory.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                with file.open("rb") as stream:
                    size = file.stat().st_size
                    offset = max(0, size - budget)
                    stream.seek(offset)
                    lines = stream.read(budget).splitlines()
                    if offset:
                        lines = lines[1:]
                budget -= min(size, budget)
                for line in reversed(lines):
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if (
                        before
                        and row["time"] >= before
                        or operation
                        and operation not in row["operation"]
                        or level
                        and row["level"] != level
                    ):
                        continue
                    records.append(self.sanitize(row))
                    if len(records) >= limit:
                        return records
                if budget <= 0:
                    break
            return records

        return {"records": await asyncio.to_thread(read), "dropped": self.dropped}
