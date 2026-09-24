"""Application lifecycle and the finite local management use cases."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from .control.barrier import BindingBarrier
from .control.catalog import catalog
from .control.operations import OriginalOperations
from .control.registration import Registration
from .control.sources import Sources
from .control.store import Store
from .core_client.http import Admission, HostClient, HTTPTransport, ManagementClient
from .delivery.engine import DeliveryEngine
from .errors import ControlError
from .validation import origin, secret

LOCAL_BASELINE = {
    "plugin": "4.0.0.dev1",
    "astrbot": "f8728adad897005e2e2cc2800f37c112835bb09d",
    "core_head": "1f2cc1e52d6f3a6fff2d41e97b161804c2295ccc",
    "core_engineering_sha256": "b2ecf46968922eb6d70318b59224409b7a5c324b3c92ef59082b5d88d9571565",
    "core_http_sha256": "c3b0788247b52ed8a52634745bb44661c3029dd7ec9993e4c14df5c0fcefea09",
    "core_ws_sha256": "9a929c0a9cef7289bb69450320ace83881c622f6474057861de7e4dd35ed1eb3",
    "remote_build": "unknown",
    "core_final_qualification": "accepted_engineering_with_recorded_limitations",
    "host_pages_qualification": "historical_local_verification",
    "host_pages_tested_plugin_sha256": "c6265405d59a6e6502d8a9a6c20d46cf19d174545d225568c46845bf8d3d5649",
    "actual_plugin_core_integration": "not_run",
    "current_environment_qualification": "unknown",
    "qualification_scope": "本地历史宿主验证与 Core 工程验收记录，不证明远端构建或当前环境合格",
}


class Application:
    def __init__(self, directory: Path, transport_factory=HTTPTransport):
        self.directory, self.transport_factory = directory, transport_factory
        self.state = "new"
        self.store = Store(directory)
        self.admission = Admission()
        self.transport = None
        self.host = None
        self.admin = {}
        self._requests = set()
        self._lifecycle_lock = asyncio.Lock()
        self._admin_lock = asyncio.Lock()
        self._connection_lock = BindingBarrier()
        self.errors = []
        self.operations = OriginalOperations(self.store)
        self.sources = Sources(self.store)
        self.registration = Registration(self)
        self.delivery = None
        self.release_failures = []

    async def initialize(self):
        async with self._lifecycle_lock:
            if self.state == "ready":
                return
            if self.state != "new":
                raise ControlError("LIFECYCLE_CLOSED", 503)
            self.state = "starting"
            try:
                await self.store.open()
                self.transport = self.transport_factory(self.admission)
                self.host = HostClient(self.transport)
                self._started_at = time.time()
                self.delivery = DeliveryEngine(self)
                await self.delivery.start()
                self.state = "ready"
            except BaseException:
                self.state = "failed"
                # Preserve the initialization exception; release diagnostics are separate.
                await self._finish_release()
                raise

    async def terminate(self):
        async with self._lifecycle_lock:
            if self.state == "closed":
                return
            self.state = "stopping"
            try:
                failed = await self._finish_release()
            finally:
                self.state = "closed" if self.released else "release_failed"
            if failed:
                raise ControlError("RESOURCE_RELEASE_FAILED", 503)

    @property
    def released(self):
        return (
            self.delivery is None
            and self.transport is None
            and not self.admin
            and self.store.db is None
            and not self._requests
            and not self.admission._tasks
            and self.admission.closed
        )

    async def _finish_release(self):
        # Own a single cleanup task and join it even if the lifecycle caller is
        # cancelled. No detached cleanup may outlive terminate/reload.
        task = asyncio.create_task(self._release(), name="iris-release")
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _release(self):
        failures = []

        async def attempt(name, close):
            try:
                await close()
                return True
            except BaseException as error:
                # Closed vocabulary only: exception messages may contain secrets.
                failures.append({"resource": name, "error_type": type(error).__name__})
                self.record_error(
                    ControlError("RELEASE_" + name.upper() + "_FAILED", 503)
                )
                return False

        # Drain ALL database users before the delivery clean marker or DB close.
        if self.delivery:
            self.delivery.closed = True
        await attempt("admission", self.admission.close)
        tasks = self._requests - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self.delivery:
            await attempt("delivery", self.delivery.close)
            if self.delivery.released:
                self.delivery = None
        for username, session in list(self.admin.items()):
            if await attempt("admin", session[0].close):
                self.admin.pop(username, None)
        if self.transport and await attempt("host", self.transport.close):
            self.transport = None
            self.host = None
        if (
            not self._requests
            and not self.admission._tasks
            and (self.delivery is None or self.delivery.quiescent)
        ):
            await attempt("store", self.store.close)
        else:
            failures.append({"resource": "store", "error_type": "UsersStillActive"})
        self.release_failures.extend(failures)
        del self.release_failures[:-100]
        return bool(failures)

    async def _clear_admin(self):
        for client, _, _ in self.admin.values():
            await client.close()
        self.admin.clear()

    @asynccontextmanager
    async def request(self, username):
        if not isinstance(username, str) or not username.strip() or len(username) > 128:
            raise ControlError("HOST_AUTHENTICATION_REQUIRED", 401)
        if self.state != "ready":
            raise ControlError("PLUGIN_NOT_READY", 503)
        if len(self._requests) >= 24:
            raise ControlError("PAGE_ADMISSION_FULL", 429)
        task = asyncio.current_task()
        self._requests.add(task)
        try:
            yield
        finally:
            self._requests.discard(task)

    def record_error(self, error):
        self.errors.append({"code": error.code, "at": time.time()})
        del self.errors[:-100]

    async def overview(self):
        config = await self.store.settings()
        observations = await self.store.observations()
        for observation in observations.values():
            observation["current"] = (
                observation["binding"] == config["binding"]
                and time.time() - observation["observed_at"] <= 60
            )
            # Stored observations survive a restart but are never current until this process probes.
            observation["current"] &= observation["observed_at"] >= getattr(
                self, "_started_at", float("inf")
            )
        return {
            "lifecycle": self.state,
            "settings": self.safe_settings(config),
            "local_baseline": LOCAL_BASELINE,
            "observations": observations,
            "model_usage": {"astrbot": None, "core": None, "state": "未接入／未知"},
            "limits": [
                "公开 handler 可达来源接入；学习、召回和主动发送未接入",
                "本地源码指纹不能证明远端构建",
                "公开消息 handler 不保证过滤前全量；发送后回调不证明送达",
                "一个 Core / SELF，5～10 个独立来源待验证，10 条提醒 route 至少两条 WS 连接",
            ],
        }

    @staticmethod
    def safe_settings(config):
        return {
            k: v for k, v in config.items() if k not in {"credential_ref", "groups"}
        } | {
            "groups": [
                {k: v for k, v in g.items() if k != "credential_ref"}
                for g in config["groups"]
            ],
            "credential_configured": config["credential_ref"] is not None,
        }

    async def save_connection(self, expected_revision, address, token=None):
        async with self._connection_lock:
            result = await self.store.connection(expected_revision, address, token)
            async with self._admin_lock:
                await self._clear_admin()
            return self.safe_settings(result)

    async def check_connection(self):
        if self._connection_lock.locked():
            raise ControlError("CONNECTION_BUSY", 409)
        async with self._connection_lock:
            config = await self.store.settings()
            if not config["origin"]:
                raise ControlError("CONNECTION_NOT_CONFIGURED", 409)
            result = {
                "connection": "unknown",
                "protocol": "unknown",
                "permission": "unknown",
                "health": None,
                "capabilities": None,
                "errors": [],
            }
            try:
                result["health"] = await self.host.health(config["origin"])
                result["connection"] = "reachable"
            except ControlError as error:
                result["errors"].append(error.code)
                self.record_error(error)
            token = await self.store.credential(config["binding"])
            if token:
                try:
                    reply = await self.host.capabilities(config["origin"], token)
                    result.update(
                        http_status=reply.http_status,
                        outcome=reply.outcome,
                        cleanup_pending=reply.cleanup_pending,
                    )
                    if reply.observed:
                        result.update(
                            protocol="compatible",
                            permission="granted",
                            capabilities=reply.data,
                            connection="reachable",
                        )
                    else:
                        result["permission"] = (
                            "denied" if reply.http_status in (401, 403) else "unknown"
                        )
                        result["errors"].append("HOST_CAPABILITIES_" + reply.outcome)
                except ControlError as error:
                    result["errors"].append(error.code)
                    self.record_error(error)
            else:
                result["permission"] = "credential_missing"
            await self.store.observe("connection", config["binding"], result)
            return result

    async def authorize_admin(self, username, session, csrf, expected_revision):
        secret(session)
        secret(csrf)
        async with self._connection_lock:
            config = await self.store.settings()
            if config["revision"] != expected_revision:
                raise ControlError(
                    "REVISION_CONFLICT", 409, revision=config["revision"]
                )
            address = origin(config["origin"])
            async with self._admin_lock:
                await self._expire_admin()
                if username not in self.admin and len(self.admin) >= 8:
                    raise ControlError("ADMIN_SESSION_CAPACITY", 429)
                client = ManagementClient(
                    self.transport_factory(self.admission), address, session, csrf
                )
                try:
                    status = await client.status()
                    updated = await self.store.observe(
                        "status",
                        config["binding"],
                        status,
                        instance_id=status["instance_id"],
                    )
                except BaseException:
                    await client.close()
                    raise
                previous = self.admin.pop(username, None)
                if previous:
                    await previous[0].close()
                # All sessions are invalidated if the observed instance changes.
                if updated["binding"] != config["binding"]:
                    await self._clear_admin()
                self.admin[username] = (
                    client,
                    time.monotonic() + 600,
                    updated["binding"],
                )
                return {
                    "authorized": True,
                    "expires_in": 600,
                    "status": status,
                    "revision": updated["revision"],
                }

    async def _expire_admin(self):
        for user, (client, expires, _) in list(self.admin.items()):
            if expires <= time.monotonic():
                self.admin.pop(user)
                await client.close()

    async def admin_status(self, username):
        async with self._connection_lock:
            async with self._admin_lock:
                await self._expire_admin()
                saved = self.admin.get(username)
                if saved is None:
                    raise ControlError("ADMIN_SESSION_REAUTHORIZE", 403)
                client, _, binding = saved
                try:
                    status = await client.status()
                    updated = await self.store.observe(
                        "status", binding, status, instance_id=status["instance_id"]
                    )
                    if updated["binding"] != binding:
                        await self._clear_admin()
                    return status
                except BaseException:
                    self.admin.pop(username, None)
                    await client.close()
                    raise

    async def revoke_admin(self, username):
        async with self._admin_lock:
            previous = self.admin.pop(username, None)
            if previous:
                await previous[0].close()
        return {"authorized": False}

    async def features(self):
        return catalog(await self.store.settings(), self.state)

    async def save_intent(self, revision, key, desired):
        async with self._connection_lock:
            result = await self.store.intent(revision, key, desired)
        if self.delivery:
            self.delivery.wake.set()
        return self.safe_settings(result)

    async def source_status(self):
        from .platforms.onebot import COVERAGE

        config = await self.store.settings()
        return {
            **Sources.safe(config, self._started_at),
            "coverage": COVERAGE,
            "intents": config["intents"],
            "limits": [
                "U01：过滤前全量无法保证",
                "U02：全宿主输出送达无法证明",
                "U03：复杂与超限原文无无损通用载体",
                "B01：离页不撤回父请求",
            ],
        }

    async def source_save(self, revision, source):
        async with self._connection_lock:
            await self.sources.save(revision, source)
        self.delivery.wake.set()
        return await self.source_status()

    async def group_save(self, revision, group_id, host_id, entries, token):
        async with self._connection_lock:
            await self.sources.group(revision, group_id, host_id, entries, token)
        return await self.source_status()

    async def group_check(self, group_id):
        async with self._connection_lock.read():
            config = await self.store.settings()
            group = next(
                (g for g in config["groups"] if g["group_id"] == group_id), None
            )
            if (
                not group
                or group["connection_binding"] != config["binding"]
                or group["instance_id"] != config["instance_id"]
            ):
                raise ControlError("GROUP_BINDING_CHANGED", 409)
            token = await self.sources.credential(group["credential_ref"])
            reply = await self.host.capabilities(config["origin"], token)
            matched = reply.observed and set(reply.data["entries"]) == set(
                group["entries"]
            )
            await self.sources.observe(
                group_id,
                group["binding"],
                {
                    "permission": "granted" if matched else "unknown",
                    "http_status": reply.http_status,
                    "outcome": reply.outcome,
                    "cleanup_pending": reply.cleanup_pending,
                    "capabilities": reply.data if matched else None,
                },
            )
        return await self.source_status()

    async def read_ingress_limits(self, username):
        from dataclasses import asdict

        from .core_client.receipts import configuration_limits

        async with self._connection_lock:
            async with self._admin_lock:
                await self._expire_admin()
                saved = self.admin.get(username)
                if not saved:
                    raise ControlError("ADMIN_SESSION_REAUTHORIZE", 403)
                client, _, binding = saved
                config = await self.store.settings()
                status = await client.status()
                if (
                    binding != config["binding"]
                    or status["instance_id"] != config["instance_id"]
                ):
                    raise ControlError("CONNECTION_CHANGED", 409)
                version, limits = configuration_limits(
                    await client.ingress("configuration/read", {"version_id": None})
                )
                await self.sources.limits(binding, version, asdict(limits))
        return await self.source_status()

    async def management_list(self, username, kind, after):
        if kind not in {"connections/hosts/list", "tokens/list"}:
            raise ControlError("HTTP_ROUTE_NOT_ALLOWED", 403)
        async with self._connection_lock.read():
            async with self._admin_lock:
                await self._expire_admin()
                saved = self.admin.get(username)
                if not saved:
                    raise ControlError("ADMIN_SESSION_REAUTHORIZE", 403)
                reply = await saved[0].ingress(kind, {"after": after})
                if not reply.observed:
                    raise ControlError("MANAGEMENT_READ_UNAVAILABLE", 502)
                from .validation import exact_fields

                exact_fields(reply.data, {"items", "after"})
                if (
                    type(reply.data["items"]) is not list
                    or len(reply.data["items"]) > 64
                ):
                    raise ControlError("MANAGEMENT_PAGE_LIMIT", 502)
                # Only reviewed metadata fields leave the backend, never arbitrary remote fields.
                allowed = (
                    {"entry_id", "host_id", "platform_id", "external_entry_id"}
                    if kind.startswith("connections")
                    else {
                        "object_id",
                        "revision",
                        "host_id",
                        "entries",
                        "operations",
                        "expires_at_us",
                        "revoked",
                        "route_ids",
                        "event_types",
                        "state",
                    }
                )
                return {
                    "items": [
                        {k: row[k] for k in allowed if k in row}
                        for row in reply.data["items"]
                    ],
                    "after": reply.data["after"],
                }

    async def delivery_confirm(self, event_id):
        result = await self.delivery.queue.schedule_confirmation(event_id)
        if result["scheduled"]:
            self.delivery.wake.set()
        return result

    async def delivery_resume(self, event_id):
        async with self._connection_lock:
            async with self.store.transaction() as db:
                async with db.execute(
                    "SELECT state,cleanup_pending FROM delivery_events WHERE id=?",
                    (event_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None or row[0] != "NOT_COMMITTED" or row[1]:
                    raise ControlError("ORIGINAL_ABSENCE_NOT_PROVEN", 409)
                await db.execute(
                    "UPDATE delivery_events SET state='SAVED',retry_authorized=1,next_attempt=0 WHERE id=?",
                    (event_id,),
                )
        self.delivery.wake.set()
        return {
            "scheduled": True,
            "operation": "same_key_same_input_after_not_committed",
        }

    async def diagnostics(self, username, offset=0, limit=50):
        async with self._admin_lock:
            await self._expire_admin()
            authorized = username in self.admin
        config = await self.store.settings()
        return {
            "revision": config["revision"],
            "errors": self.errors.copy(),
            "operations": await self.store.operations(offset, limit),
            "resources": {
                "http_active": self.admission.active,
                "http_waiting": self.admission.waiting,
                "http_limit": 4,
                "waiting_limit": 16,
                "admin_sessions": len(self.admin),
                "page_requests": len(self._requests),
            },
            "admin_authorized_for_current_user": authorized,
            "credential_storage": "仅后端受限本地文件（目录 0700、数据库 0600），未加密；不包含于页面或普通导出",
            "model_usage": "未接入／未知",
        }
