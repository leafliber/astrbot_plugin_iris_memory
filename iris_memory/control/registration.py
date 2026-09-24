"""Durable, session-isolated management originals. Token possession is separate."""

import json
import uuid

from ..core_client.management import validate_request
from ..core_client.protocol import Reply
from ..core_client.receipts import committed
from ..errors import ControlError
from ..validation import exact_fields, identifier
from .sources import validate_group_topology

KINDS = {
    "source_register": ("connections/hosts/register", "connections/hosts/confirm"),
    "token_create": ("tokens/create", "tokens/create"),
    "token_revoke": ("tokens/revoke", "tokens/revoke"),
}


class Registration:
    def __init__(self, app):
        self.app = app

    async def perform(
        self,
        username,
        kind,
        payload=None,
        *,
        operation_id=None,
        group_id=None,
        expected_revision=None,
    ):
        app = self.app
        async with app._connection_lock:
            async with app._admin_lock:
                await app._expire_admin()
                saved = app.admin.get(username)
                if not saved:
                    raise ControlError("ADMIN_SESSION_REAUTHORIZE", 403)
                client, _, binding = saved
                config = await app.store.settings()
                status = await client.status()
                if (
                    binding != config["binding"]
                    or status["instance_id"] != config["instance_id"]
                ):
                    raise ControlError("OPERATION_BINDING_MISMATCH", 409)
                if operation_id:
                    original = await app.store.operation(
                        operation_id, check_binding=False
                    )
                    if original["instance_id"] != status["instance_id"]:
                        raise ControlError("OPERATION_INSTANCE_MISMATCH", 409)
                    kind = original["kind"]
                    if kind not in KINDS:
                        raise ControlError("OPERATION_NOT_IMPLEMENTED", 409)
                    saved_input = json.loads(original["original_input"])
                    request, group_id = (
                        saved_input["request"],
                        saved_input.get("group_id"),
                    )
                    action = KINDS[kind][1]
                else:
                    if expected_revision != config["revision"]:
                        raise ControlError(
                            "REVISION_CONFLICT", 409, revision=config["revision"]
                        )
                    if kind not in KINDS:
                        raise ControlError("OPERATION_NOT_IMPLEMENTED", 409)
                    request = dict(payload or {})
                    request["key"] = str(uuid.uuid4())
                    action = KINDS[kind][0]
                    validate_request(action, request)
                    if kind == "token_create":
                        identifier(group_id)
                        # Validate local placement before issuing irrevocable one-time material.
                        groups = config["groups"]
                        validate_group_topology(
                            groups, group_id, request["host_id"], request["entries"]
                        )
                        if len(groups) >= 2 and not any(
                            g["group_id"] == group_id for g in groups
                        ):
                            raise ControlError("GROUP_CAPACITY", 409)
                        if any(
                            g["group_id"] != group_id
                            and (
                                g["host_id"] != request["host_id"]
                                or set(g["entries"]) & set(request["entries"])
                            )
                            for g in groups
                        ):
                            raise ControlError("GROUP_TOPOLOGY_CONFLICT", 409)
                    original, _ = await app.store.create_operation(
                        binding,
                        status["instance_id"],
                        kind,
                        request["key"],
                        {"request": request, "group_id": group_id},
                    )
                    async with app.store.transaction() as db:
                        await db.execute(
                            "UPDATE settings SET revision=revision+1 WHERE id=1"
                        )
                    config = await app.store.settings()
                reply = await client.ingress(action, request)
                secret_obtained = False
                if kind == "token_create" and reply.observed:
                    exact_fields(reply.data, {"result", "token"})
                    result = reply.data["result"]
                    # result_envelope wraps this endpoint's native result rather than unwrapping it.
                    if type(result) is dict and set(result) == {"receipt", "source"}:
                        nested = Reply(200, "COMMITTED", reply.cleanup_pending, result)
                        committed(nested)
                        token = reply.data["token"]
                        if token is not None:
                            if result["source"] != "NEW":
                                raise ControlError("UNEXPECTED_TOKEN_REISSUE")
                            await app.sources.group(
                                config["revision"],
                                group_id,
                                request["host_id"],
                                request["entries"],
                                token,
                            )
                            secret_obtained = True
                        reply = nested
                    else:
                        # No nested success proof: retain original UNKNOWN, including native rejection.
                        reply = Reply(
                            reply.http_status,
                            "UNCONFIRMED",
                            reply.cleanup_pending,
                            None,
                        )
                elif kind != "token_create" and reply.outcome == "COMMITTED":
                    fact = committed(reply)
                    if kind == "source_register" and (
                        fact.get("entry_id") != request["entry_id"]
                        or fact.get("terminal") != "REGISTERED"
                    ):
                        raise ControlError("REGISTRATION_RECEIPT_MISMATCH")
                await app.store.finish_operation(original["id"], reply)
                return {
                    "id": original["id"],
                    "state": reply.outcome,
                    "cleanup_pending": reply.cleanup_pending,
                    "secret_obtained_this_response": secret_obtained,
                    "reason": "签发结果与秘密取得独立；丢失秘密不能自动重新签发"
                    if kind == "token_create"
                    else None,
                }
