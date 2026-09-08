"""Thin public-API adapters. Importing this module imports neither Core nor SDK."""

import asyncio
import os
import time
from dataclasses import asdict

from .errors import IrisError
from .identity import digest

CAPABILITIES = (
    "active-surface.v1",
    "claims.v1",
    "episodes.v1",
    "observe.batch.v1",
    "recall.v1",
    "recall.usage.v1",
    "recall.revalidate.v1",
    "recall.graph.v1",
    "recall.vector.v1",
    "search.fts.v1",
    "memory-forget.v1",
    "identities.v1",
    "profile.v1",
    "persona.read.v1",
    "persona.mirror.v1",
    "persona.state.v1",
    "tasks.v1",
    "focus-items.v1",
    "notes.v1",
    "state.v1",
    "recent-context.v1",
    "embedding.v1",
    "outbox.jobs.v1",
    "cognitive-events.v1",
    "schedules.v1",
    "contract.negotiation",
    "error-envelope.v1",
)


class Backend:
    def __init__(self, control):
        self.control = control
        self.client = None
        self.capabilities = {}

    async def call(self, method, *args, **kwargs):
        try:
            return await getattr(self.client, method)(*args, **kwargs)
        except Exception as exc:
            envelope = getattr(exc, "envelope", None)
            code = getattr(envelope, "code", getattr(exc, "code", "core_error"))
            self.control.logs.emit(
                "core.call", level="ERROR", error=exc, method=method, code=code
            )
            raise IrisError(
                code,
                "Core 调用失败，请查看诊断",
                details={
                    "request_id": getattr(
                        envelope, "request_id", getattr(exc, "request_id", None)
                    ),
                    "result_unknown": getattr(exc, "result_unknown", False),
                },
            ) from exc

    async def proof(self, agent_id):
        return {}

    def require(self, capability):
        if capability not in self.capabilities.get("capabilities", []):
            raise IrisError("capability_missing", f"Core 未提供 {capability}")

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None


class LocalBackend(Backend):
    async def start(self):
        if os.name != "posix":
            raise IrisError(
                "platform_unsupported",
                "当前 Core 本地模式需要 Unix 目录锁；此平台请使用远程模式",
            )
        try:
            from iris_memory_core.embedded import (
                EmbeddedConfig,
                EmbeddedMemory,
                LocalBootstrap,
            )
        except ImportError as exc:
            raise IrisError(
                "dependency_missing",
                "请安装包含 EmbeddedMemory 的 Core wheel，见 README",
            ) from exc
        settings = self.control.settings
        embedding, cognitive = await self.control.host.providers(self.control)
        self.client = EmbeddedMemory(
            EmbeddedConfig(
                self.control.directory / "core",
                bootstrap=LocalBootstrap(
                    app_instance_id="iris-plugin",
                    tenant_id="iris-local",
                    agent_name="Iris",
                    manage_identities=True,
                    manage_persona=True,
                    manage_agents=True,
                    manage_indexes=True,
                    capabilities=CAPABILITIES,
                    surface_mode="off",
                ),
                background=False,
                max_pending=32,
                allow_local_sqlite=settings["allow_development_sqlite"],
            ),
            embedding=embedding,
            cognitive=cognitive,
        )
        try:
            await self.client.start()
        except Exception as exc:
            if getattr(exc, "code", None) == "sqlite_runtime_not_allowed":
                raise IrisError(
                    "sqlite_runtime_not_allowed",
                    "SQLite 运行库不在 Core 的支持范围；请升级运行库。仅开发验证可在 Pages 显式开启 SQLite 例外。",
                ) from exc
            raise
        self.capabilities = await self.call("capabilities")
        self.require("recall.v1")

    async def scope(self, identity, persona):
        # One independent Agent/Space per conversation/persona: the current Core
        # API cannot provision conversation Spaces. Never share a default scope.
        key = digest(identity.key, persona["id"] if persona else "default")
        scope = await self.call("provision_agent", "Iris " + key[:16], key=key)
        actor = await self.call(
            "register_actor",
            identity.platform,
            identity.user,
            realm=identity.realm,
            display_name=identity.display_name,
            idempotency_key="actor:" + digest(identity.realm, identity.user),
        )
        return {**scope, "entity_id": actor["entity_id"]}

    async def operation(self, operation, *, body=None, path=None, query=None, key=None):
        return await self.call(
            "execute",
            operation,
            body,
            path_parameters=path or {},
            query_parameters=query or {},
            idempotency_key=key,
        )

    async def maintain(self):
        return await self.call("run_pending")


class RemoteBackend(Backend):
    def __init__(self, control):
        super().__init__(control)
        self.leases = {}
        self.lease_lock = asyncio.Lock()

    async def proof(self, agent_id):
        app = self.control.settings["remote_app_instance"]
        if not app or not agent_id:
            return {}
        async with self.lease_lock:
            cached = self.leases.get(agent_id)
            if cached and cached["until"] > time.monotonic():
                return cached["proof"]
            if cached and cached["until"] + 10 > time.monotonic():
                value = await super().call(
                    "heartbeat_surface_lease",
                    cached["proof"]["lease_id"],
                    lease_epoch=cached["proof"]["lease_epoch"],
                    holder_app_instance_id=app,
                    ttl_us=30000000,
                )
            else:
                value = await super().call(
                    "acquire_surface_lease",
                    agent_id,
                    holder_app_instance_id=app,
                    ttl_us=30000000,
                )
            lease = value.get("lease", value)
            proof = {"lease_id": lease["lease_id"], "lease_epoch": lease["lease_epoch"]}
            self.leases[agent_id] = {"until": time.monotonic() + 20, "proof": proof}
            return proof

    async def call(self, method, *args, **kwargs):
        if method == "observe_batch":
            proof = await self.proof(args[0][0]["agent_id"]) if args[0] else {}
            kwargs = {**kwargs, **proof}
        elif method == "recall":
            args = (
                {**args[0], **await self.proof(args[0]["scope"]["agent_id"])},
                *args[1:],
            )
        return await super().call(method, *args, **kwargs)

    async def close(self):
        if self.client:
            try:
                # Release is best effort. A dead server must not turn N leases
                # into N sequential request timeouts during plugin unload.
                async with asyncio.timeout(2):
                    for entry in self.leases.values():
                        try:
                            await super().call(
                                "release_surface_lease",
                                entry["proof"]["lease_id"],
                                lease_epoch=entry["proof"]["lease_epoch"],
                                holder_app_instance_id=self.control.settings[
                                    "remote_app_instance"
                                ],
                            )
                        except IrisError:
                            pass
            except TimeoutError:
                self.control.logs.emit(
                    "lease.release_timeout",
                    level="WARNING",
                    reason="Unreleased leases expire server-side",
                )
            self.leases.clear()
        await super().close()

    async def start(self):
        try:
            from iris_memory_sdk import AsyncIrisMemoryClient
        except ImportError as exc:
            raise IrisError(
                "dependency_missing", "请安装 iris-memory-sdk wheel，见 README"
            ) from exc
        settings = self.control.settings
        if not settings["remote_url"] or not settings["remote_token"]:
            raise IrisError(
                "remote_unconfigured", "请先在 Pages 配置远程地址和应用 Token"
            )
        self.client = AsyncIrisMemoryClient(
            settings["remote_url"],
            bearer_token=settings["remote_token"],
            timeout_seconds=settings["request_timeout"],
            max_in_flight=4,
            max_pending=32,
        )
        if not callable(getattr(self.client, "aclose", None)):
            raise IrisError(
                "dependency_outdated", "SDK 缺少异步关闭 API，请安装已验证的开发 wheel"
            )
        self.capabilities = asdict(
            await self.call(
                "negotiate", required_capabilities=("recall.v1", "observe.batch.v1")
            )
        )

    async def scope(self, identity, persona):
        key = identity.key + "/" + (persona["id"] if persona else "default")
        binding = self.control.settings["remote_bindings"].get(key)
        if not binding:
            raise IrisError("scope_unconfigured", f"请预配置远程绑定：{key}")
        result = dict(binding)
        actor_key = result.get("realm", identity.realm) + "/" + identity.user
        entity = self.control.settings["remote_actors"].get(actor_key)
        if entity:
            result["entity_id"] = entity
        elif identity.is_group:
            result.pop("entity_id", None)
        return result

    async def operation(self, operation, *, body=None, path=None, query=None, key=None):
        path, query = path or {}, query or {}
        if operation == "dismissFocusItem":
            return await self.call(
                "focus_transition",
                path["focus_item_id"],
                "dismiss",
                **body,
                idempotency_key=key,
            )
        mapping = {
            "createFocusItem": ("create_focus_item", (body,)),
            "listFocusItems": ("list_focus_items", (query.get("agent_id"),)),
            "getCurrentPersona": ("current_persona", (path.get("agent_id"),)),
            "publishPersonaRevision": (
                "publish_persona_revision",
                (path.get("agent_id"), body),
            ),
            "reportRecallUsage": (
                "report_recall_usage",
                (path.get("request_id"), body),
            ),
            "getClaim": ("get_claim", (path.get("claim_id"),)),
            "createTask": ("create_task", (body,)),
            "listTasks": ("list_tasks", (query.get("agent_id"),)),
            "transitionTask": ("transition_task", (path.get("task_id"), body)),
            "getEntityProfile": ("get_entity_profile", (path.get("entity_id"),)),
        }
        if operation not in mapping:
            raise IrisError("unsupported", f"远程适配尚不支持 {operation}")
        method, args = mapping[operation]
        if operation == "createTask":
            args = ({**body, **await self.proof(body["agent_id"])},)
        kwargs = {k: v for k, v in query.items() if k != "agent_id"}
        if operation == "getEntityProfile":
            kwargs = query
        if key is not None:
            kwargs["idempotency_key"] = key
        return await self.call(method, *args, **kwargs)

    async def maintain(self):
        # Server owns its worker lifecycle.
        return {"managed_by": "server"}
