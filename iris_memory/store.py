"""Durable notification inbox; pull and SSE share acceptance by notification ID."""

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


class InboxFull(Exception):
    """Backpressure: do not acknowledge a notification that was not persisted."""


class Inbox:
    capacity = 1000
    receipt_capacity = 2048

    def __init__(self, path: Path) -> None:
        self.db = open_private_database(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            INSERT OR IGNORE INTO cursor VALUES(1,0);
            CREATE TABLE IF NOT EXISTS inbox (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS accepted (id INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO accepted SELECT id FROM inbox WHERE id > (SELECT value FROM cursor WHERE id=1);
        """)

    @property
    def cursor(self) -> int:
        return self.db.execute("SELECT value FROM cursor WHERE id=1").fetchone()[0]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM inbox").fetchone()[0]

    def _accept(self, item: dict[str, Any], *, covered_by_page: bool = False) -> bool:
        number = item.get("id")
        if type(number) is not int or number <= 0:
            raise ValueError("invalid notification ID")
        if (
            number <= self.cursor
            or self.db.execute("SELECT 1 FROM accepted WHERE id=?", (number,)).fetchone()
        ):
            return False
        receipts = self.db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
        if self.count() >= self.capacity or (
            not covered_by_page and receipts >= self.receipt_capacity
        ):
            raise InboxFull()
        self.db.execute(
            "INSERT INTO inbox VALUES(?,?)", (number, json.dumps(item, ensure_ascii=False))
        )
        # Retain this ID even after delivery, until pull has covered its range.
        self.db.execute("INSERT INTO accepted VALUES(?)", (number,))
        return True

    def accept(self, item: dict[str, Any]) -> bool:
        with self.db:
            return self._accept(item)

    def _advance(self, value: int) -> None:
        self.db.execute("UPDATE cursor SET value=MAX(value,?) WHERE id=1", (value,))
        self.db.execute("DELETE FROM accepted WHERE id<=?", (value,))

    def advance(self, value: int) -> None:
        # Only a fully accepted pull page can advance this watermark; an SSE
        # frame ID or the largest notification ID is never a pull cursor.
        if type(value) is not int or value < 0:
            raise ValueError("invalid notification cursor")
        with self.db:
            self._advance(value)

    def save_page(self, page: dict[str, Any]) -> None:
        value = page["next_cursor"]
        if type(value) is not int or value < self.cursor:
            raise ValueError("invalid notification cursor")
        for item in page["items"]:
            if type(item["id"]) is not int or not 0 < item["id"] <= value:
                raise ValueError("invalid notification ID")
        with self.db:
            for item in page["items"]:
                # These IDs are removed by _advance in this same transaction.
                # A full SSE receipt set must not block pull from filling gaps.
                self._accept(item, covered_by_page=True)
            self._advance(value)

    def pending(self) -> list[dict[str, Any]]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM inbox ORDER BY id")]

    def consume(self, ids: tuple[int, ...] | list[int]) -> None:
        # This is a local 'offered to reply / accepted by adapter' checkpoint,
        # never an Iris goal completion or a claim that a person read a message.
        with self.db:
            self.db.executemany("DELETE FROM inbox WHERE id=?", ((n,) for n in ids))

    def close(self) -> None:
        self.db.close()
