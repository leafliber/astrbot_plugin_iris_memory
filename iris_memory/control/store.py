"""Plugin-owned transactions, credential references and bounded original operations.

Credentials are plaintext in a private (0700 / 0600) local database, not encrypted.
Only safe projections are exported; the host OS account remains the trust boundary.
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from ..errors import ControlError
from ..validation import encode, identifier, integer, origin, secret

TERMINAL = {"COMMITTED", "NOT_COMMITTED", "REJECTED", "ABSENT"}
PENDING = "(state NOT IN ('COMMITTED','NOT_COMMITTED','REJECTED','ABSENT') OR cleanup_pending=1)"
DEFAULT = {
    "origin": "",
    "credential_ref": None,
    "binding": "",
    "instance_id": None,
    "intents": {"plugin.enabled": False},
    "sources": [],
    "ws_connections": [],
}


class Store:
    def __init__(self, directory: Path):
        self.directory = directory
        self.db = None
        self.lock = asyncio.Lock()

    async def open(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ControlError("UNSAFE_DATA_DIRECTORY")
        os.chmod(self.directory, 0o700)
        path = self.directory / "control.sqlite3"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.db = await aiosqlite.connect(path, isolation_level=None)
        self.db.row_factory = aiosqlite.Row
        try:
            async with self.db.execute("PRAGMA user_version") as cursor:
                version = (await cursor.fetchone())[0]
            if version not in (0, 1):
                raise ControlError("STORE_VERSION_UNSUPPORTED", 503)
            await self.db.execute("PRAGMA journal_mode=DELETE")
            await self.db.execute("PRAGMA synchronous=FULL")
            await self.db.execute("PRAGMA secure_delete=ON")
            await self.db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS credentials (ref TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (name TEXT PRIMARY KEY, binding TEXT NOT NULL, observed_at REAL NOT NULL, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS operations (id TEXT PRIMARY KEY, binding TEXT NOT NULL, instance_id TEXT NOT NULL, kind TEXT NOT NULL, original_key TEXT NOT NULL, original_input TEXT NOT NULL, digest TEXT NOT NULL, state TEXT NOT NULL, cleanup_pending INTEGER NOT NULL, http_status INTEGER, created_at REAL NOT NULL, updated_at REAL NOT NULL, UNIQUE(binding,original_key));
                PRAGMA user_version=1;
                COMMIT;
            """)
            await self.db.execute(
                "INSERT OR IGNORE INTO settings VALUES (1,0,?)", (encode(DEFAULT),)
            )
            await self.prune()
        except BaseException:
            await self.close()
            raise

    async def close(self):
        async with self.lock:
            if self.db is not None:
                await self.db.close()
                self.db = None

    def _live(self):
        if self.db is None:
            raise ControlError("STORE_CLOSED", 503)
        return self.db

    @asynccontextmanager
    async def transaction(self):
        async with self.lock:
            db = self._live()
            await db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                await db.commit()
            except BaseException:
                await asyncio.shield(db.rollback())
                raise

    async def _settings(self, db):
        async with db.execute(
            "SELECT revision,value FROM settings WHERE id=1"
        ) as cursor:
            row = await cursor.fetchone()
        return row["revision"], json.loads(row["value"])

    async def settings(self):
        async with self.lock:
            revision, value = await self._settings(self._live())
            return {"revision": revision, **value}

    async def _cas(self, db, expected):
        integer(expected)
        revision, value = await self._settings(db)
        if expected != revision:
            raise ControlError("REVISION_CONFLICT", 409, revision=revision)
        return revision, value

    async def connection(self, expected, address, token=None):
        address = origin(address)
        if token is not None and token != "":
            secret(token)
        async with self.transaction() as db:
            revision, value = await self._cas(db, expected)
            changed = value["origin"] != address or token is not None
            if token is None and value["origin"] and value["origin"] != address:
                # Never carry a saved credential to a different origin implicitly.
                token = ""
            if token is not None:
                await db.execute("DELETE FROM credentials")
                ref = str(uuid.uuid4()) if token else None
                if ref:
                    await db.execute(
                        "INSERT INTO credentials VALUES (?,?)", (ref, token)
                    )
                value["credential_ref"] = ref
            value["origin"] = address
            if changed:
                value["binding"] = str(uuid.uuid4())
                value["instance_id"] = None
                await db.execute("DELETE FROM observations")
            await db.execute(
                "UPDATE settings SET revision=?,value=? WHERE id=1",
                (revision + 1, encode(value)),
            )
        return await self.settings()

    async def intent(self, expected, key, desired):
        from .catalog import EDITABLE

        if key not in EDITABLE or type(desired) is not bool:
            raise ControlError("CONTROL_NOT_IMPLEMENTED", 409)
        async with self.transaction() as db:
            revision, value = await self._cas(db, expected)
            value["intents"][key] = desired
            await db.execute(
                "UPDATE settings SET revision=?,value=? WHERE id=1",
                (revision + 1, encode(value)),
            )
        return await self.settings()

    async def credential(self, binding):
        async with self.lock:
            db = self._live()
            _, value = await self._settings(db)
            if value["binding"] != binding:
                raise ControlError("CONNECTION_CHANGED", 409)
            async with db.execute(
                "SELECT value FROM credentials WHERE ref=?", (value["credential_ref"],)
            ) as cursor:
                row = await cursor.fetchone()
            return row["value"] if row else None

    async def observe(self, name, binding, value, *, instance_id=None):
        if name not in {"connection", "status"}:
            raise ControlError("INVALID_OBSERVATION")
        async with self.transaction() as db:
            revision, config = await self._settings(db)
            if binding != config["binding"]:
                raise ControlError("CONNECTION_CHANGED", 409)
            if instance_id is not None:
                identifier(instance_id)
                if config["instance_id"] != instance_id:
                    if config["instance_id"] is not None:
                        config["binding"] = str(uuid.uuid4())
                        await db.execute("DELETE FROM observations")
                    config["instance_id"] = instance_id
                    await db.execute(
                        "UPDATE settings SET revision=?,value=? WHERE id=1",
                        (revision + 1, encode(config)),
                    )
            await db.execute(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?)",
                (name, config["binding"], time.time(), encode(value)),
            )
        return await self.settings()

    async def observations(self):
        async with self.lock:
            async with self._live().execute("SELECT * FROM observations") as cursor:
                rows = await cursor.fetchall()
        return {
            r["name"]: {
                "binding": r["binding"],
                "observed_at": r["observed_at"],
                "value": json.loads(r["value"]),
            }
            for r in rows
        }

    async def create_operation(self, binding, instance_id, kind, key, payload):
        identifier(kind)
        identifier(key)
        identifier(instance_id)
        serialized = encode(payload)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        async with self.transaction() as db:
            _, config = await self._settings(db)
            if (
                not instance_id
                or config["binding"] != binding
                or config["instance_id"] != instance_id
            ):
                raise ControlError("OPERATION_BINDING_MISMATCH", 409)
            async with db.execute(
                "SELECT * FROM operations WHERE binding=? AND original_key=?",
                (binding, key),
            ) as cur:
                row = await cur.fetchone()
            if row:
                if (
                    row["digest"] != digest
                    or row["kind"] != kind
                    or row["instance_id"] != instance_id
                ):
                    raise ControlError("ORIGINAL_INPUT_CONFLICT", 409)
                return dict(row), False
            async with db.execute(
                f"SELECT COUNT(*) FROM operations WHERE {PENDING}"
            ) as cur:
                if (await cur.fetchone())[0] >= 256:
                    raise ControlError("OPERATION_CAPACITY_FULL", 429)
            now, op_id = time.time(), str(uuid.uuid4())
            await db.execute(
                "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    op_id,
                    binding,
                    instance_id,
                    kind,
                    key,
                    serialized,
                    digest,
                    "UNKNOWN",
                    1,
                    None,
                    now,
                    now,
                ),
            )
            async with db.execute(
                "SELECT * FROM operations WHERE id=?", (op_id,)
            ) as cur:
                row = await cur.fetchone()
            return dict(row), True

    async def operation(self, op_id, *, check_binding=True):
        async with self.lock:
            db = self._live()
            async with db.execute(
                "SELECT * FROM operations WHERE id=?", (op_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                raise ControlError("OPERATION_NOT_FOUND", 404)
            if check_binding:
                _, config = await self._settings(db)
                if (config["binding"], config["instance_id"]) != (
                    row["binding"],
                    row["instance_id"],
                ):
                    raise ControlError("OPERATION_BINDING_MISMATCH", 409)
            return dict(row)

    async def finish_operation(self, op_id, reply):
        from ..core_client.protocol import OUTCOMES

        if reply.outcome not in OUTCOMES:
            raise ControlError("PROTOCOL_OUTCOME")
        async with self.transaction() as db:
            await db.execute(
                "UPDATE operations SET state=?,cleanup_pending=?,http_status=?,updated_at=? WHERE id=?",
                (
                    reply.outcome,
                    int(reply.cleanup_pending),
                    reply.http_status,
                    time.time(),
                    op_id,
                ),
            )
            await self._prune(db)

    async def _prune(self, db):
        await db.execute(
            f"DELETE FROM operations WHERE NOT {PENDING} AND updated_at<?",
            (time.time() - 7 * 86400,),
        )
        await db.execute(
            f"DELETE FROM operations WHERE id IN (SELECT id FROM operations WHERE NOT {PENDING} ORDER BY updated_at DESC,id DESC LIMIT -1 OFFSET 1000)"
        )

    async def prune(self):
        async with self.transaction() as db:
            await self._prune(db)

    async def operations(self, offset=0, limit=50):
        await self.prune()
        integer(offset, 0, 1256)
        integer(limit, 1, 100)
        async with self.lock:
            db = self._live()
            async with db.execute(
                "SELECT id,binding,instance_id,kind,state,cleanup_pending,http_status,created_at,updated_at FROM operations ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ) as cursor:
                rows = [dict(r) for r in await cursor.fetchall()]
            async with db.execute(
                f"SELECT COUNT(*),COALESCE(SUM({PENDING}),0) FROM operations"
            ) as cursor:
                total, pending = await cursor.fetchone()
        return {
            "items": rows,
            "total": total,
            "pending": pending,
            "offset": offset,
            "limit": limit,
        }
