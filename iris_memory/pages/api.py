"""Public AstrBot Web API adapter; no raw framework access or generic proxy."""

import asyncio

from astrbot.api.web import error_response, json_response, request

from .. import PLUGIN_NAME
from ..errors import ControlError
from ..validation import decode, exact_fields, integer

ROUTES = (
    ("overview", "GET"),
    ("features", "GET"),
    ("diagnostics", "GET"),
    ("connection/save", "POST"),
    ("connection/check", "POST"),
    ("controls/save", "POST"),
    ("admin/authorize", "POST"),
    ("admin/status", "POST"),
    ("admin/revoke", "POST"),
    ("sources", "GET"),
    ("sources/save", "POST"),
    ("groups/save", "POST"),
    ("groups/check", "POST"),
    ("ingress/limits", "POST"),
    ("ingress/register", "POST"),
    ("ingress/token", "POST"),
    ("ingress/revoke", "POST"),
    ("ingress/confirm", "POST"),
    ("ingress/list", "POST"),
    ("delivery", "GET"),
    ("delivery/confirm", "POST"),
    ("delivery/reauthorize-confirm", "POST"),
    ("delivery/resume", "POST"),
    ("connection/impact", "POST"),
)


class PagesAPI:
    def __init__(self, application):
        self.application = application

    def register(self, context):
        for endpoint, method in ROUTES:

            async def handler(endpoint=endpoint, method=method):
                return await self.handle(endpoint, method)

            context.register_web_api(
                f"/{PLUGIN_NAME}/{endpoint}",
                handler,
                [method],
                "Iris Memory local control",
            )

    async def handle(self, endpoint, method):
        app = self.application
        try:
            if request.plugin_name != PLUGIN_NAME:
                raise ControlError("HOST_PLUGIN_SCOPE_REQUIRED", 403)
            async with app.request(request.username):
                if request.method != method:
                    raise ControlError("METHOD_NOT_ALLOWED", 405)
                payload = {}
                if method == "POST":
                    if (
                        not request.content_type
                        or request.content_type.split(";")[0] != "application/json"
                    ):
                        raise ControlError("JSON_REQUIRED", 415)
                    length = request.headers.get("content-length")
                    if length is None or request.headers.get("transfer-encoding"):
                        raise ControlError("CONTENT_LENGTH_REQUIRED", 411)
                    if not length.isdigit() or int(length) > 16384:
                        raise ControlError("PAYLOAD_TOO_LARGE", 413)
                    try:
                        async with asyncio.timeout(3):
                            raw = await request.body()
                    except TimeoutError:
                        raise ControlError("REQUEST_BODY_TIMEOUT", 408) from None
                    if len(raw) != int(length):
                        raise ControlError("CONTENT_LENGTH_MISMATCH", 400)
                    payload = decode(raw, 16384)
                result = await self.dispatch(endpoint, payload, request.username)
                return json_response(result, headers={"Cache-Control": "no-store"})
        except asyncio.CancelledError:
            if app.state != "ready":
                return error_response("PLUGIN_STOPPING", status_code=503)
            raise
        except ControlError as error:
            app.record_error(error)
            return error_response(
                error.code,
                status_code=error.status,
                data={"revision": error.revision},
                headers={"Cache-Control": "no-store"},
            )
        except Exception:
            error = ControlError("LOCAL_CONTROL_FAILURE", 500)
            app.record_error(error)
            return error_response(error.code, status_code=500)

    async def dispatch(self, endpoint, payload, username):
        app = self.application
        if endpoint == "overview":
            return await app.overview()
        if endpoint == "sources":
            return await app.source_status()
        if endpoint == "delivery":
            return await app.delivery.status(
                integer(int(request.query.get("offset", "0")), 0, 3048)
            )
        if endpoint == "sources/save":
            exact_fields(payload, {"expected_revision", "source"})
            return await app.source_save(
                integer(payload["expected_revision"]), payload["source"]
            )
        if endpoint == "groups/save":
            exact_fields(
                payload,
                {"expected_revision", "group_id", "host_id", "entries", "token"},
            )
            return await app.group_save(
                payload["expected_revision"],
                payload["group_id"],
                payload["host_id"],
                payload["entries"],
                payload["token"],
            )
        if endpoint == "groups/check":
            exact_fields(payload, {"group_id"})
            return await app.group_check(payload["group_id"])
        if endpoint == "ingress/limits":
            exact_fields(payload, set())
            return await app.read_ingress_limits(username)
        if endpoint in {"ingress/register", "ingress/token", "ingress/revoke"}:
            exact_fields(payload, {"expected_revision", "input"}, {"group_id"})
            kind = {
                "ingress/register": "source_register",
                "ingress/token": "token_create",
                "ingress/revoke": "token_revoke",
            }[endpoint]
            return await app.registration.perform(
                username,
                kind,
                payload["input"],
                group_id=payload.get("group_id"),
                expected_revision=payload["expected_revision"],
            )
        if endpoint == "ingress/confirm":
            exact_fields(payload, {"id"})
            return await app.registration.perform(
                username, "", operation_id=payload["id"]
            )
        if endpoint == "ingress/list":
            exact_fields(payload, {"kind", "after"})
            return await app.management_list(
                username, payload["kind"], payload["after"]
            )
        if endpoint == "delivery/reauthorize-confirm":
            from ..delivery.confirmation import confirm_with_current_group

            exact_fields(payload, {"id", "group_id", "expected_revision"})
            return await confirm_with_current_group(
                app,
                username,
                payload["id"],
                payload["group_id"],
                integer(payload["expected_revision"]),
            )
        if endpoint == "delivery/confirm":
            exact_fields(payload, {"id"})
            return await app.delivery_confirm(payload["id"])
        if endpoint == "delivery/resume":
            exact_fields(payload, {"id"})
            return await app.delivery_resume(payload["id"])
        if endpoint == "connection/impact":
            exact_fields(payload, set())
            config = await app.store.settings()
            return {
                "revision": config["revision"],
                "groups": len(config["groups"]),
                "sources": len(config["sources"]),
                "delivery": (await app.delivery.status())["states"],
                "effect": "切换连接／凭据后，旧绑定操作保留且停止派发；须逐项核实原实例与权限，不自动改用新凭据",
            }
        if endpoint == "features":
            try:
                offset = integer(int(request.query.get("offset", "0")), 0, 1000)
                limit = integer(int(request.query.get("limit", "100")), 1, 100)
            except ValueError:
                raise ControlError("INVALID_PAGINATION") from None
            items = await app.features()
            return {
                "items": items[offset : offset + limit],
                "total": len(items),
                "offset": offset,
                "limit": limit,
            }
        if endpoint == "diagnostics":
            try:
                offset, limit = (
                    int(request.query.get("offset", "0")),
                    int(request.query.get("limit", "50")),
                )
            except ValueError:
                raise ControlError("INVALID_PAGINATION") from None
            return await app.diagnostics(username, offset, limit)
        if endpoint == "connection/save":
            exact_fields(payload, {"expected_revision", "origin"}, {"token"})
            return await app.save_connection(
                integer(payload["expected_revision"]),
                payload["origin"],
                payload.get("token"),
            )
        if endpoint == "controls/save":
            exact_fields(payload, {"expected_revision", "action_key", "desired"})
            return await app.save_intent(
                integer(payload["expected_revision"]),
                payload["action_key"],
                payload["desired"],
            )
        if endpoint == "admin/authorize":
            exact_fields(payload, {"session", "csrf", "expected_revision"})
            return await app.authorize_admin(
                username,
                payload["session"],
                payload["csrf"],
                integer(payload["expected_revision"]),
            )
        exact_fields(payload, set())
        if endpoint == "connection/check":
            return await app.check_connection()
        if endpoint == "admin/status":
            return await app.admin_status(username)
        if endpoint == "admin/revoke":
            return await app.revoke_admin(username)
        raise ControlError("ENDPOINT_NOT_FOUND", 404)
