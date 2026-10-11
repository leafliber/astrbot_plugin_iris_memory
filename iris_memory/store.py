"""Single-process notification inbox: checkpoint and page content commit together."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def open_private_database(path: Path, *, timeout: float = 5.0) -> sqlite3.Connection:
    """Create privately before SQLite writes; tighten existing files on reopen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(path, 0o600)
    finally:
        os.close(fd)
    return sqlite3.connect(path, timeout=timeout)


class Inbox:
    def __init__(self, path: Path) -> None:
        self.db = open_private_database(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            INSERT OR IGNORE INTO cursor VALUES(1,0);
            CREATE TABLE IF NOT EXISTS inbox (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        """)

    @property
    def cursor(self) -> int:
        return self.db.execute("SELECT value FROM cursor WHERE id=1").fetchone()[0]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM inbox").fetchone()[0]

    def save_page(self, page: dict[str, Any]) -> None:
        value = page["next_cursor"]
        if type(value) is not int or value < self.cursor:
            raise ValueError("invalid notification cursor")
        for item in page["items"]:
            if type(item["id"]) is not int or not 0 < item["id"] <= value:
                raise ValueError("invalid notification ID")
        with self.db:
            for item in page["items"]:
                if item["id"] > self.cursor:
                    self.db.execute(
                        "INSERT OR IGNORE INTO inbox VALUES(?,?)",
                        (item["id"], json.dumps(item, ensure_ascii=False)),
                    )
            self.db.execute("UPDATE cursor SET value=? WHERE id=1", (value,))

    def pending(self) -> list[dict[str, Any]]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM inbox ORDER BY id")]

    def consume(self, ids: tuple[int, ...] | list[int]) -> None:
        # This is a local 'offered to reply / accepted by adapter' checkpoint,
        # never an Iris goal completion or a claim that a person read a message.
        with self.db:
            self.db.executemany("DELETE FROM inbox WHERE id=?", ((n,) for n in ids))

    def close(self) -> None:
        self.db.close()
