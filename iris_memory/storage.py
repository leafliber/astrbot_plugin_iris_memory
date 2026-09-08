"""One bounded SQLite connection for plugin configuration, persona and delivery state."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .errors import Conflict, IrisError


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.executor = None
        self.connection = None
        self.pending = set()
        self.closing = False

    async def start(self):
        if self.executor:
            return
        self.closing = False
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="iris-plugin-db"
        )

        def open_database():
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / "plugin.sqlite3"
            self.connection = sqlite3.connect(path, timeout=3)
            os.chmod(path, 0o600)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    collection TEXT NOT NULL, key TEXT NOT NULL,
                    revision INTEGER NOT NULL, value TEXT NOT NULL,
                    PRIMARY KEY(collection, key)
                );
                CREATE TABLE IF NOT EXISTS history (
                    collection TEXT NOT NULL, key TEXT NOT NULL,
                    revision INTEGER NOT NULL, value TEXT NOT NULL,
                    PRIMARY KEY(collection, key, revision)
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, error TEXT
                );
                CREATE INDEX IF NOT EXISTS delivery_due ON deliveries(state, expires);
            """)

        try:
            await self.run(open_database, connection=False)
        except BaseException:
            await self.close()
            raise

    async def run(self, operation, *, connection=True):
        if self.executor is None or self.closing:
            raise IrisError("store_closed", "插件配置库已关闭")
        if len(self.pending) >= 64:
            raise IrisError("busy", "配置请求过多，请稍后重试")

        def execute():
            if not connection:
                return operation()
            with self.connection:
                return operation(self.connection)

        future = asyncio.get_running_loop().run_in_executor(self.executor, execute)
        self.pending.add(future)
        future.add_done_callback(self.pending.discard)
        return await asyncio.shield(future)

    async def get(self, collection, key):
        def read(db):
            row = db.execute(
                "SELECT revision,value FROM documents WHERE collection=? AND key=?",
                (collection, key),
            ).fetchone()
            return {"revision": row[0], "value": json.loads(row[1])} if row else None

        return await self.run(read)

    async def list(self, collection, *, limit=100, offset=0):
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        return await self.run(
            lambda db: [
                {"key": r[0], "revision": r[1], "value": json.loads(r[2])}
                for r in db.execute(
                    "SELECT key,revision,value FROM documents WHERE collection=? ORDER BY key LIMIT ? OFFSET ?",
                    (collection, limit, offset),
                )
            ]
        )

    async def put(self, collection, key, value, *, expected_revision, history=False):
        text = encode(value)
        if len(text.encode()) > 262144:
            raise IrisError("too_large", "单条配置或人格数据超过 256 KiB")

        def write(db):
            row = db.execute(
                "SELECT revision FROM documents WHERE collection=? AND key=?",
                (collection, key),
            ).fetchone()
            current = row[0] if row else 0
            if current != expected_revision:
                raise Conflict()
            revision = current + 1
            db.execute(
                "INSERT INTO documents VALUES(?,?,?,?) ON CONFLICT(collection,key) DO UPDATE SET revision=excluded.revision,value=excluded.value",
                (collection, key, revision, text),
            )
            if history:
                db.execute(
                    "INSERT INTO history VALUES(?,?,?,?)",
                    (collection, key, revision, text),
                )
            return {"revision": revision, "value": value}

        return await self.run(write)

    async def history(self, collection, key, *, limit=100):
        return await self.run(
            lambda db: [
                {"revision": r[0], "value": json.loads(r[1])}
                for r in db.execute(
                    "SELECT revision,value FROM history WHERE collection=? AND key=? ORDER BY revision DESC LIMIT ?",
                    (collection, key, max(1, min(int(limit), 100))),
                )
            ]
        )

    async def close(self):
        if self.executor is None:
            return
        self.closing = True
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        executor, self.executor = self.executor, None
        if self.connection:
            await asyncio.get_running_loop().run_in_executor(
                executor, self.connection.close
            )
            self.connection = None
        await asyncio.to_thread(executor.shutdown, wait=True)
