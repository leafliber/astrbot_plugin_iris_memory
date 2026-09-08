"""Bounded delivery journal. Acknowledged payloads are removed, never a memory replica."""

import json
import time

from .errors import IrisError
from .storage import encode


class Outbox:
    def __init__(self, control):
        self.control = control

    async def enqueue(self, key, payload, *, kind="observation"):
        text, now = encode(payload), time.time()
        settings = self.control.settings

        def write(db):
            if db.execute("SELECT 1 FROM deliveries WHERE id=?", (key,)).fetchone():
                return False
            db.execute("DELETE FROM deliveries WHERE expires < ?", (now,))
            count, size = db.execute(
                "SELECT count(*), coalesce(sum(length(cast(payload AS BLOB))),0) FROM deliveries WHERE state NOT IN ('accepted','cancelled','expired')"
            ).fetchone()
            if (
                count >= settings["queue_items"]
                or size + len(text.encode()) > settings["queue_bytes"]
            ):
                raise IrisError("queue_full", "交付队列已满；本条未采集，请查看诊断")
            db.execute(
                "INSERT INTO deliveries(id,kind,state,payload,created,expires) VALUES(?,?,'pending',?,?,?)",
                (key, kind, text, now, now + settings["queue_ttl_hours"] * 3600),
            )
            return True

        return await self.control.store.run(write)

    async def pending(self, kind="observation"):
        return await self.control.store.run(
            lambda db: [
                dict(r) | {"payload": json.loads(r["payload"])}
                for r in db.execute(
                    "SELECT * FROM deliveries WHERE kind=? AND state IN ('pending','retry') AND expires>? ORDER BY created LIMIT 8",
                    (kind, time.time()),
                )
            ]
        )

    async def finish(self, key, state, error=None):
        # Keep only short-lived receipts for deduplication after acknowledgment.
        await self.control.store.run(
            lambda db: db.execute(
                "UPDATE deliveries SET state=?,attempts=attempts+1,error=?,payload=CASE WHEN ? IN ('accepted','cancelled','expired') THEN '{}' ELSE payload END WHERE id=?",
                (state, str(error)[:128] if error else None, state, key),
            )
        )
