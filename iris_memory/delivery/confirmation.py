"""Explicit, reauthorized original resolution; never transfer first-submit authority."""

import asyncio
import hashlib
import time

from ..core_client.ingress import IngressLimits
from ..core_client.receipts import acceptance
from ..errors import ControlError
from ..validation import encode, exact_fields


async def confirm_with_current_group(app, username, event_id, group_id, revision):
    try:
        async with asyncio.timeout(60):
            return await _confirm(app, username, event_id, group_id, revision)
    except TimeoutError:
        raise ControlError("ORIGINAL_CONFIRMATION_DEADLINE", 504) from None


async def _confirm(app, username, event_id, group_id, revision):
    async with app._connection_lock:
        async with app._admin_lock:
            await app._expire_admin()
            session = app.admin.get(username)
            if not session:
                raise ControlError("ADMIN_SESSION_REAUTHORIZE", 403)
            config = await app.store.settings()
            if config["revision"] != revision:
                raise ControlError(
                    "REVISION_CONFLICT", 409, revision=config["revision"]
                )
            row = await app.delivery.queue.get(event_id)
            if not row["submitted"] or row["material"] is None:
                raise ControlError("ORIGINAL_CONFIRMATION_ONLY", 409)
            binding = row["binding"]
            status = await session[0].status()
            group = next(
                (g for g in config["groups"] if g["group_id"] == group_id), None
            )
            if (
                session[2] != config["binding"]
                or status["instance_id"] != binding["instance_id"]
                or status["instance_id"] != config["instance_id"]
                or not group
                or group["connection_binding"] != config["binding"]
                or group["instance_id"] != config["instance_id"]
                or group["host_id"] != binding["host_id"]
                or binding["entry_id"] not in group["entries"]
            ):
                raise ControlError("ORIGINAL_AUTHORITY_MISMATCH", 409)
            token = await app.sources.credential(group["credential_ref"])
            token_id = hashlib.sha256(token.encode()).hexdigest()
            # Capabilities do not identify the token's host. Verify it using the legal
            # management metadata read as well; bounded to four 64-row pages.
            cursor, authority = "", None
            for _ in range(4):
                reply = await session[0].ingress("tokens/list", {"after": cursor})
                if not reply.observed:
                    raise ControlError("ORIGINAL_AUTHORITY_UNVERIFIED", 409)
                data = exact_fields(reply.data, {"items", "after"})
                if type(data["items"]) is not list or len(data["items"]) > 64:
                    raise ControlError("MANAGEMENT_PAGE_LIMIT", 502)
                authority = next(
                    (x for x in data["items"] if x.get("object_id") == token_id), None
                )
                if authority or data["after"] is None:
                    break
                if data["after"] == cursor:
                    raise ControlError("MANAGEMENT_CURSOR_LOOP", 502)
                cursor = data["after"]
            if (
                not authority
                or authority.get("revoked") is not False
                or authority.get("host_id") != binding["host_id"]
                or binding["entry_id"] not in authority.get("entries", [])
                or "confirm" not in authority.get("operations", [])
                or authority.get("expires_at_us", 0) <= time.time_ns() // 1000
            ):
                raise ControlError("ORIGINAL_AUTHORITY_UNVERIFIED", 403)
            capabilities = await app.host.capabilities(config["origin"], token)
            if (
                not capabilities.observed
                or "confirm" not in capabilities.data["operations"]
                or set(capabilities.data["entries"]) != set(group["entries"])
            ):
                raise ControlError("ORIGINAL_AUTHORITY_UNVERIFIED", 403)
            async with app.store.transaction() as db:
                await db.execute(
                    "INSERT OR REPLACE INTO delivery_confirmation_bindings VALUES(?,?)",
                    (
                        event_id,
                        encode(
                            {
                                "connection": config["binding"],
                                "instance_id": config["instance_id"],
                                "origin": config["origin"],
                                "group_binding": group["binding"],
                                "credential_ref": group["credential_ref"],
                            }
                        ),
                    ),
                )
            await app.delivery.queue.dispatch(event_id, confirm=True)
            reply = await app.host.ingress(
                config["origin"],
                token,
                binding["entry_id"],
                "accept/resolve",
                {"key": row["original_key"], "event": row["material"]["event"]},
                limits=IngressLimits(**binding["limits"]["values"]),
            )
            if acceptance(reply, binding["entry_id"]):
                await app.delivery.queue.state(
                    event_id, "CONFIRMED", cleanup=reply.cleanup_pending
                )
                if not reply.cleanup_pending:
                    await app.delivery.media.cleanup(event_id)
            else:
                # Absence or denial through replacement authority is not a new-submit permit.
                await app.delivery.queue.state(
                    event_id,
                    "UNKNOWN",
                    reason="REAUTHORIZED_" + reply.outcome,
                    cleanup=reply.cleanup_pending,
                    delay=10,
                )
            return {
                "outcome": reply.outcome,
                "cleanup_pending": reply.cleanup_pending,
                "first_submit_authorized": False,
            }
