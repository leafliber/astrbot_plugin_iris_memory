"""Finite transport obligations; confirmed bodies are erased, never a chat archive."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass

from ..errors import ControlError
from ..validation import encode, integer


@dataclass(frozen=True)
class Capacity:
    events: int = 2048
    source_events: int = 256
    event_bytes: int = 32 * 1024 * 1024
    record_bytes: int = 256 * 1024
    media_bytes: int = 256 * 1024 * 1024
    source_media_bytes: int = 64 * 1024 * 1024
    blob_bytes: int = 1024 * 1024
    workers: int = 2

    def __post_init__(self):
        upper = (2048, 256, 33554432, 262144, 268435456, 67108864, 1048576, 2)
        for value, maximum in zip(asdict(self).values(), upper):
            integer(value, 1, maximum)
        if (
            self.source_events > self.events
            or self.record_bytes > self.event_bytes
            or self.blob_bytes > self.source_media_bytes
            or self.source_media_bytes > self.media_bytes
        ):
            raise ControlError("INVALID_DELIVERY_CAPACITY")


class DeliveryStore:
    def __init__(self, control, capacity=None):
        self.control = control
        self.capacity = capacity or Capacity()
        self.memory_gaps = 0
        self.storage_failed = False
        self.close_attempted = False
        self.clean_saved = False

    async def open(self):
        async with self.control.lock:
            await self.control._live().executescript("""
                CREATE TABLE IF NOT EXISTS delivery_events (
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                  source_id TEXT NOT NULL, external_id TEXT, digest TEXT NOT NULL,
                  binding TEXT NOT NULL, original_key TEXT NOT NULL,
                  material TEXT, bytes INTEGER NOT NULL, state TEXT NOT NULL,
                  cleanup_pending INTEGER NOT NULL DEFAULT 0, reason TEXT,
                  submitted INTEGER NOT NULL DEFAULT 0, confirms INTEGER NOT NULL DEFAULT 0,
                  confirmed_once INTEGER NOT NULL DEFAULT 0, retry_authorized INTEGER NOT NULL DEFAULT 0,
                  created REAL NOT NULL, updated REAL NOT NULL, next_attempt REAL NOT NULL DEFAULT 0,
                  UNIQUE(source_id, external_id));
                CREATE INDEX IF NOT EXISTS delivery_source_order ON delivery_events(source_id,seq);
                CREATE TABLE IF NOT EXISTS delivery_confirmation_bindings (event_id TEXT PRIMARY KEY, binding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS delivery_media (
                  id TEXT PRIMARY KEY, event_id TEXT NOT NULL, source_id TEXT NOT NULL,
                  original_key TEXT NOT NULL, state TEXT NOT NULL, upload_id TEXT,
                  bytes INTEGER NOT NULL, digest TEXT, retained INTEGER NOT NULL DEFAULT 1,
                  dispatched INTEGER NOT NULL DEFAULT 0, cleanup_pending INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS delivery_counts (name TEXT PRIMARY KEY, count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS delivery_gaps (
                  source_id TEXT NOT NULL, reason TEXT NOT NULL, first REAL NOT NULL, last REAL NOT NULL,
                  count INTEGER NOT NULL, PRIMARY KEY(source_id,reason));
                CREATE TABLE IF NOT EXISTS delivery_session (id INTEGER PRIMARY KEY CHECK(id=1), clean INTEGER NOT NULL, last REAL NOT NULL);
            """)
            async with self.control._live().execute(
                "PRAGMA table_info(delivery_events)"
            ) as cur:
                columns = {row[1] for row in await cur.fetchall()}
            for name in ("confirmed_once", "retry_authorized"):
                if name not in columns:
                    await self.control._live().execute(
                        f"ALTER TABLE delivery_events ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT clean,last FROM delivery_session WHERE id=1"
            ) as cur:
                old = await cur.fetchone()
            if old:
                await self._gap(
                    db,
                    "all",
                    "UNOBSERVED_BETWEEN_SESSIONS"
                    if old[0]
                    else "UNCLEAN_SESSION_COVERAGE_UNKNOWN",
                    old[1],
                )
            await db.execute(
                "INSERT OR REPLACE INTO delivery_session VALUES(1,0,?)", (time.time(),)
            )
            await self._prune(db)

    async def close(self):
        # A failed close must not become a clean session on a later release attempt.
        if self.close_attempted:
            return
        self.close_attempted = True
        try:
            async with self.control.transaction() as db:
                await db.execute(
                    "UPDATE delivery_session SET clean=1,last=? WHERE id=1",
                    (time.time(),),
                )
            self.clean_saved = True
        except BaseException:
            self.storage_failed = True
            raise

    async def _count(self, db, name, number=1):
        await db.execute(
            "INSERT INTO delivery_counts VALUES(?,?) ON CONFLICT(name) DO UPDATE SET count=count+excluded.count",
            (name, number),
        )

    async def _gap(self, db, source, reason, first=None):
        # Closed caller reason vocabulary; overflow collapses to an explicit coarse interval.
        async with db.execute("SELECT COUNT(*) FROM delivery_gaps") as cur:
            count = (await cur.fetchone())[0]
        if count >= 127:
            source, reason = "all", "COARSE_OVERFLOW"
        now = time.time()
        await db.execute(
            "INSERT INTO delivery_gaps VALUES(?,?,?,?,1) ON CONFLICT(source_id,reason) DO UPDATE SET last=excluded.last,count=count+1",
            (source, reason, first or now, now),
        )
        await self._count(db, "gaps")

    async def gap(self, source, reason):
        try:
            async with self.control.transaction() as db:
                await self._gap(db, source, reason)
        except Exception:
            self.memory_gaps += 1
            self.storage_failed = True

    async def admit(self, source, binding, material, external_id, reason=None):
        cap = self.capacity
        serialized = encode(material, cap.record_bytes)
        size = len(serialized.encode())
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        event_id, original_key = str(uuid.uuid4()), str(uuid.uuid4())
        now = time.time()
        async with self.control.transaction() as db:
            await self._count(db, "input")
            if external_id is not None:
                async with db.execute(
                    "SELECT id,digest FROM delivery_events WHERE source_id=? AND external_id=?",
                    (source, external_id),
                ) as cur:
                    old = await cur.fetchone()
                if old:
                    if old[1] != digest:
                        await self._gap(db, source, "DUPLICATE_ID_CONTENT_CONFLICT")
                        return None
                    await self._count(db, "duplicate")
                    return old[0]
            async with db.execute(
                "SELECT COUNT(*),COALESCE(SUM(bytes),0),COALESCE(SUM(source_id=?),0) FROM delivery_events WHERE state!='CONFIRMED' OR cleanup_pending=1 OR material IS NOT NULL",
                (source,),
            ) as cur:
                total, occupied, local = await cur.fetchone()
            if (
                total >= cap.events
                or local >= cap.source_events
                or occupied + size > cap.event_bytes
            ):
                await self._gap(db, source, "EVENT_CAPACITY_FULL")
                return None
            await db.execute(
                "INSERT INTO delivery_events(seq,id,source_id,external_id,digest,binding,original_key,material,bytes,state,reason,created,updated) VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    source,
                    external_id,
                    digest,
                    encode(binding),
                    original_key,
                    serialized,
                    size,
                    "BLOCKED" if reason else "SAVED",
                    reason,
                    now,
                    now,
                ),
            )
            await self._count(db, "queued")
            if reason:
                await self._gap(db, source, reason)
            await self._prune(db)
        return event_id

    async def get(self, event_id):
        async with self.control.lock:
            async with self.control._live().execute(
                "SELECT * FROM delivery_events WHERE id=?", (event_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            raise ControlError("DELIVERY_NOT_FOUND", 404)
        value = dict(row)
        value["binding"] = json.loads(value["binding"])
        value["material"] = json.loads(value["material"]) if value["material"] else None
        return value

    async def heads(self):
        async with self.control.lock:
            async with self.control._live().execute(
                "SELECT id,source_id FROM delivery_events WHERE seq IN (SELECT MIN(seq) FROM delivery_events WHERE state NOT IN ('CONFIRMED','BLOCKED','REJECTED') OR cleanup_pending=1 OR (state='CONFIRMED' AND material IS NOT NULL) GROUP BY source_id) AND next_attempt<=? ORDER BY next_attempt,seq LIMIT 10",
                (time.time(),),
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]

    async def schedule_confirmation(self, event_id):
        # Inspect and schedule the CURRENT obligation in one transaction. Never copy
        # state, cleanup facts or original input from a caller's earlier snapshot.
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT submitted,state,cleanup_pending,material,retry_authorized "
                "FROM delivery_events WHERE id=?",
                (event_id,),
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise ControlError("DELIVERY_NOT_FOUND", 404)
            if not row["submitted"]:
                raise ControlError("DELIVERY_NOT_DISPATCHED", 409)
            if not row["cleanup_pending"] and (
                row["state"] in {"BLOCKED", "REJECTED"}
                or (row["state"] == "CONFIRMED" and row["material"] is None)
            ):
                return {"scheduled": False, "operation": "already_terminal"}
            if row["retry_authorized"] or row["state"] == "NOT_COMMITTED":
                raise ControlError("ORIGINAL_CONFIRMATION_ONLY", 409)
            if row["material"] is None:
                raise ControlError("ORIGINAL_INPUT_UNAVAILABLE", 409)
            await db.execute(
                "UPDATE delivery_events SET next_attempt=0 WHERE id=?", (event_id,)
            )
            return {"scheduled": True, "operation": "original_confirmation"}

    async def state(self, event_id, state, *, reason=None, cleanup=False, delay=0):
        if state not in {
            "SAVED",
            "PAUSED",
            "MEDIA_PENDING",
            "UNKNOWN",
            "CONFIRMED",
            "BLOCKED",
            "REJECTED",
            "ABSENT_UNKNOWN",
            "NOT_COMMITTED",
        }:
            raise ControlError("INVALID_DELIVERY_STATE")
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT state,confirmed_once FROM delivery_events WHERE id=?",
                (event_id,),
            ) as cur:
                old = await cur.fetchone()
            if not old:
                raise ControlError("DELIVERY_NOT_FOUND", 404)
            await db.execute(
                "UPDATE delivery_events SET state=?,reason=?,cleanup_pending=?,updated=?,next_attempt=? WHERE id=?",
                (
                    state,
                    reason,
                    int(cleanup),
                    time.time(),
                    time.time() + delay,
                    event_id,
                ),
            )
            if not old[1] and state == "CONFIRMED":
                await self._count(db, "confirmed")
                await db.execute(
                    "UPDATE delivery_events SET confirmed_once=1 WHERE id=?",
                    (event_id,),
                )

    async def dispatch(self, event_id, *, confirm=False):
        async with self.control.transaction() as db:
            # Persist before the first network byte. Cancellation keeps this obligation.
            column = "confirms" if confirm else "submitted"
            await db.execute(
                f"UPDATE delivery_events SET state='UNKNOWN',cleanup_pending=1,retry_authorized=0,{column}={column}+1,updated=? WHERE id=?",
                (time.time(), event_id),
            )
            await self._count(db, "confirms" if confirm else "submits")

    async def freeze_event(self, event_id, material):
        serialized = encode(material, self.capacity.record_bytes)
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT submitted,bytes FROM delivery_events WHERE id=?", (event_id,)
            ) as cur:
                row = await cur.fetchone()
            if row[0]:
                raise ControlError("ORIGINAL_EVENT_IMMUTABLE", 409)
            size = len(serialized.encode())
            async with db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM delivery_events"
            ) as cur:
                used = (await cur.fetchone())[0]
            if used - row[1] + size > self.capacity.event_bytes:
                raise ControlError("EVENT_CAPACITY_FULL", 429)
            await db.execute(
                "UPDATE delivery_events SET material=?,bytes=? WHERE id=?",
                (serialized, size, event_id),
            )

    async def media(self, event_id):
        async with self.control.lock:
            async with self.control._live().execute(
                "SELECT * FROM delivery_media WHERE event_id=? ORDER BY id", (event_id,)
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]

    async def reserve_media(self, event_id, source, media_id):
        cap = self.capacity
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT id FROM delivery_media WHERE id=?", (media_id,)
            ) as cur:
                if await cur.fetchone():
                    return
            async with db.execute(
                "SELECT COALESCE(SUM(bytes),0),COALESCE(SUM(CASE WHEN source_id=? THEN bytes ELSE 0 END),0) FROM delivery_media WHERE retained=1",
                (source,),
            ) as cur:
                total, local = await cur.fetchone()
            if (
                total + cap.blob_bytes > cap.media_bytes
                or local + cap.blob_bytes > cap.source_media_bytes
            ):
                raise ControlError("MEDIA_CAPACITY_FULL", 429)
            await db.execute(
                "INSERT INTO delivery_media(id,event_id,source_id,original_key,state,bytes) VALUES(?,?,?,?,?,?)",
                (
                    media_id,
                    event_id,
                    source,
                    str(uuid.uuid4()),
                    "RESERVED",
                    cap.blob_bytes,
                ),
            )

    async def update_media(self, media_id, **values):
        allowed = {
            "state",
            "upload_id",
            "bytes",
            "digest",
            "retained",
            "dispatched",
            "cleanup_pending",
        }
        if not values or values.keys() - allowed:
            raise ControlError("INVALID_MEDIA_UPDATE")
        async with self.control.transaction() as db:
            await db.execute(
                "UPDATE delivery_media SET "
                + ",".join(k + "=?" for k in values)
                + " WHERE id=?",
                (*values.values(), media_id),
            )

    async def cleaned(self, event_id):
        async with self.control.transaction() as db:
            async with db.execute(
                "SELECT COUNT(*) FROM delivery_media WHERE event_id=? AND retained=1",
                (event_id,),
            ) as cur:
                if (await cur.fetchone())[0]:
                    return
            await db.execute(
                "UPDATE delivery_events SET material=NULL,bytes=0 WHERE id=? AND state='CONFIRMED' AND cleanup_pending=0",
                (event_id,),
            )
            await self._prune(db)

    async def _prune(self, db):
        await db.execute(
            "DELETE FROM delivery_events WHERE state='CONFIRMED' AND cleanup_pending=0 AND material IS NULL AND (updated<? OR seq IN (SELECT seq FROM delivery_events WHERE state='CONFIRMED' AND cleanup_pending=0 AND material IS NULL ORDER BY seq DESC LIMIT -1 OFFSET 1000))",
            (time.time() - 7 * 86400,),
        )
        await db.execute(
            "DELETE FROM delivery_media WHERE retained=0 AND event_id NOT IN (SELECT id FROM delivery_events)"
        )

        await db.execute(
            "DELETE FROM delivery_confirmation_bindings WHERE event_id NOT IN (SELECT id FROM delivery_events)"
        )

    async def status(self, offset=0, limit=50):
        integer(offset, 0, 3048)
        integer(limit, 1, 100)
        async with self.control.lock:
            db = self.control._live()
            async with db.execute(
                "SELECT id,seq,source_id,state,cleanup_pending,reason,submitted,confirms,bytes,created,updated FROM delivery_events ORDER BY seq DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ) as cur:
                items = [dict(row) for row in await cur.fetchall()]
            async with db.execute(
                "SELECT state,COUNT(*) FROM delivery_events GROUP BY state"
            ) as cur:
                states = dict(await cur.fetchall())
            async with db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM delivery_events"
            ) as cur:
                event_bytes = (await cur.fetchone())[0]
            async with db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM delivery_media WHERE retained=1"
            ) as cur:
                media_bytes = (await cur.fetchone())[0]
            async with db.execute(
                "SELECT * FROM delivery_gaps ORDER BY last DESC LIMIT 128"
            ) as cur:
                gaps = [dict(row) for row in await cur.fetchall()]
            async with db.execute("SELECT * FROM delivery_counts") as cur:
                counts = dict(await cur.fetchall())
            async with db.execute(
                "SELECT event_id,state,bytes,retained,dispatched,cleanup_pending FROM delivery_media WHERE event_id IN (SELECT id FROM delivery_events ORDER BY seq DESC LIMIT ? OFFSET ?) LIMIT 200",
                (limit, offset),
            ) as cur:
                media = [dict(row) for row in await cur.fetchall()]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "states": states,
            "media": media,
            "counts": counts,
            "event_bytes": event_bytes,
            "media_reserved_bytes": media_bytes,
            "limits": asdict(self.capacity),
            "gaps": gaps,
            "memory_only_gaps": self.memory_gaps,
            "storage_failed": self.storage_failed,
            "gap_precision": "区间为聚合观察，不保证区间内每条丢失；内存计数不保证重启保留",
            "dedup_retention": "已确认身份最多 1000 项／7 天；更旧平台重复不能保证本地识别",
        }
