"""Stable source identities and credential groups, independent of participants."""

import hashlib
import time
import uuid

from ..core_client.ingress import IngressLimits
from ..errors import ControlError
from ..validation import boolean, encode, exact_fields, identifier, secret


def source_identity(platform_instance, bot_self, kind, conversation_id):
    if kind not in {"group", "private"}:
        raise ControlError("INVALID_SOURCE_KIND")
    for value in (platform_instance, bot_self, conversation_id):
        identifier(value)
    components = [platform_instance, bot_self, kind, conversation_id]
    return "source:" + hashlib.sha256(encode(components).encode()).hexdigest()


def validate_group_topology(groups, group_id, host_id, entries):
    others = [g for g in groups if g["group_id"] != group_id]
    if len(others) >= 2 or any(
        g["host_id"] != host_id or set(g["entries"]) & set(entries) for g in others
    ):
        raise ControlError("GROUP_TOPOLOGY_CONFLICT", 409)
    sizes = [len(g["entries"]) for g in others] + [len(entries)]
    if sum(sizes) > 10 or (sum(sizes) == 10 and sorted(sizes) != [2, 8]):
        raise ControlError("TEN_SOURCES_REQUIRE_EIGHT_PLUS_TWO", 409)


class Sources:
    def __init__(self, store):
        self.store = store

    async def save(self, expected, source):
        exact_fields(
            source,
            {
                "platform_instance",
                "bot_self",
                "kind",
                "conversation_id",
                "entry_id",
                "group_id",
                "enabled",
            },
        )
        identity = source_identity(
            *(
                source[k]
                for k in ("platform_instance", "bot_self", "kind", "conversation_id")
            )
        )
        identifier(source["entry_id"])
        identifier(source["group_id"])
        boolean(source["enabled"])
        async with self.store.transaction() as db:
            revision, value = await self.store._cas(db, expected)
            sources = value["sources"]
            old = next((s for s in sources if s["source_id"] == identity), None)
            if old and any(old[k] != source[k] for k in ("entry_id", "group_id")):
                raise ControlError("SOURCE_BINDING_IMMUTABLE", 409)
            if not old and (source["enabled"] or len(sources) >= 10):
                raise ControlError("NEW_SOURCE_MUST_BE_DISABLED_OR_CAPACITY", 409)
            if any(
                s["entry_id"] == source["entry_id"] and s["source_id"] != identity
                for s in sources
            ):
                raise ControlError("DUPLICATE_ENTRY", 409)
            group = next(
                (g for g in value["groups"] if g["group_id"] == source["group_id"]),
                None,
            )
            if not group or source["entry_id"] not in group["entries"]:
                raise ControlError("SOURCE_GROUP_MISMATCH", 409)
            current = {
                **source,
                "source_id": identity,
                "version": (old["version"] + 1) if old else 1,
                "route_id": None,
            }
            value["sources"] = [s for s in sources if s["source_id"] != identity] + [
                current
            ]
            await db.execute(
                "UPDATE settings SET revision=?,value=? WHERE id=1",
                (revision + 1, encode(value)),
            )
        return identity

    async def group(self, expected, group_id, host_id, entries, token):
        identifier(group_id)
        identifier(host_id)
        secret(token)
        if type(entries) is not list or not 1 <= len(entries) <= 8:
            raise ControlError("TOKEN_GROUP_LIMIT")
        for entry in entries:
            identifier(entry)
        if len(set(entries)) != len(entries):
            raise ControlError("DUPLICATE_ENTRY")
        async with self.store.transaction() as db:
            revision, value = await self.store._cas(db, expected)
            validate_group_topology(value["groups"], group_id, host_id, entries)
            old = next((g for g in value["groups"] if g["group_id"] == group_id), None)
            if not old and len(value["groups"]) >= 2:
                raise ControlError("GROUP_CAPACITY", 409)
            for other in value["groups"]:
                if other["group_id"] != group_id and (
                    set(other["entries"]) & set(entries) or other["host_id"] != host_id
                ):
                    raise ControlError("GROUP_TOPOLOGY_CONFLICT", 409)
            if any(
                s["group_id"] == group_id and s["entry_id"] not in entries
                for s in value["sources"]
            ):
                raise ControlError("GROUP_HAS_BOUND_SOURCES", 409)
            async with db.execute("SELECT COUNT(*) FROM credentials") as cur:
                if (await cur.fetchone())[0] >= 256:
                    raise ControlError("CREDENTIAL_HISTORY_CAPACITY", 429)
            ref = str(uuid.uuid4())
            await db.execute("INSERT INTO credentials VALUES (?,?)", (ref, token))
            group = {
                "group_id": group_id,
                "host_id": host_id,
                "entries": entries,
                "credential_ref": ref,
                "binding": str(uuid.uuid4()),
                "connection_binding": value["binding"],
                "instance_id": value["instance_id"],
                "version": old["version"] + 1 if old else 1,
                "observation": None,
            }
            value["groups"] = [
                g for g in value["groups"] if g["group_id"] != group_id
            ] + [group]
            await db.execute(
                "UPDATE settings SET revision=?,value=? WHERE id=1",
                (revision + 1, encode(value)),
            )

    async def credential(self, ref):
        async with self.store.lock:
            async with self.store._live().execute(
                "SELECT value FROM credentials WHERE ref=?", (ref,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            raise ControlError("GROUP_CREDENTIAL_MISSING", 409)
        return row[0]

    async def observe(self, group_id, binding, observation):
        async with self.store.transaction() as db:
            revision, value = await self.store._settings(db)
            group = next(
                (g for g in value["groups"] if g["group_id"] == group_id), None
            )
            if group is None or group["binding"] != binding:
                raise ControlError("GROUP_BINDING_CHANGED", 409)
            group["observation"] = {"at": time.time(), **observation}
            await db.execute("UPDATE settings SET value=? WHERE id=1", (encode(value),))

    async def limits(self, binding, version, limits):
        IngressLimits(**limits)
        identifier(version)
        async with self.store.transaction() as db:
            _, value = await self.store._settings(db)
            if value["binding"] != binding or value["instance_id"] is None:
                raise ControlError("CONNECTION_CHANGED", 409)
            value["ingress_limits"] = {
                "binding": binding,
                "version": version,
                "at": time.time(),
                "values": limits,
            }
            await db.execute("UPDATE settings SET value=? WHERE id=1", (encode(value),))

    @staticmethod
    def safe(config, started_at):
        groups = []
        for group in config["groups"]:
            observation = group["observation"]
            current = bool(
                observation
                and observation["at"] >= started_at
                and time.time() - observation["at"] <= 60
                and group["connection_binding"] == config["binding"]
                and group["instance_id"] == config["instance_id"]
            )
            groups.append(
                {k: v for k, v in group.items() if k != "credential_ref"}
                | {"current": current, "credential_configured": True}
            )
        return {
            "sources": config["sources"],
            "groups": groups,
            "revision": config["revision"],
            "ingress_limits": config["ingress_limits"],
        }
