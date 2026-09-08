"""Shared observations, explicit memories, recall and tasks through one backend."""

import asyncio
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from ..backends import LocalBackend, RemoteBackend
from ..errors import IrisError
from ..identity import Identity, digest
from ..outbox import Outbox


def utc():
    return datetime.now(timezone.utc).isoformat()


class Module:
    def __init__(self, control):
        self.control = control
        factory = control.backend_factory or (
            LocalBackend if control.mode == "local" else RemoteBackend
        )
        self.backend = factory(control)
        self.outbox = Outbox(control)
        self.worker = None
        self.mirror_lock = asyncio.Lock()

    async def start(self):
        await self.backend.start()
        self.worker = asyncio.create_task(
            self._drain(), name="iris-observation-delivery"
        )

    async def close(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None
        await self.backend.close()

    async def _drain(self):
        while True:
            try:
                for row in await self.outbox.pending():
                    try:
                        payload = row["payload"]
                        identity = Identity(**payload["identity"])
                        if not self.control.allowed(identity):
                            await self.outbox.finish(row["id"], "cancelled")
                            continue
                        await self.observe(
                            identity,
                            payload["text"],
                            key=row["id"],
                            occurred_us=payload["occurred_us"],
                            role=payload.get("role", "user"),
                            persona_id=payload.get("persona_id"),
                        )
                        await self.outbox.finish(row["id"], "accepted")
                    except IrisError as exc:
                        # The original body/key remains unchanged on unknown results.
                        state = (
                            "retry"
                            if exc.code
                            in {
                                "deadline_exceeded",
                                "socket_timeout",
                                "transport_error",
                                "busy",
                                "service_unavailable",
                            }
                            else "failed"
                        )
                        await self.outbox.finish(row["id"], state, exc.code)
                        self.control.logs.emit(
                            "observation.delivery_failed",
                            level="WARNING",
                            error=exc,
                            key=row["id"],
                            state=state,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.control.logs.emit(
                    "observation.worker_failed", level="ERROR", error=exc
                )
            await asyncio.sleep(5)

    async def persona(self, identity):
        if "persona" not in self.control.modules:
            return None
        return await self.control.personas.resolve(
            identity, self.control.settings["default_persona"]
        )

    async def scope(self, identity, persona=None):
        if not self.control.allowed(identity):
            raise IrisError(
                "conversation_disabled", "此会话未启用，请在 Pages 中添加会话键"
            )
        return await self.backend.scope(identity, persona)

    async def capture(self, identity, text, *, role="user", key=None):
        if not self.control.allowed(identity) or not text.strip():
            return {"captured": False}
        if len(text.encode()) > 65536:
            raise IrisError("message_too_large", "消息超过采集上限，未截断写入")
        persona = await self.persona(identity)
        key = key or "observe:" + digest(
            identity.realm,
            identity.key,
            identity.user,
            identity.message_id or uuid.uuid4().hex,
            role,
        )
        added = await self.outbox.enqueue(
            key,
            {
                "identity": asdict(identity),
                "text": text,
                "role": role,
                "occurred_us": int(time.time() * 1e6),
                "persona_id": persona["id"] if persona else None,
            },
        )
        return {"captured": added, "key": key}

    async def observe(
        self, identity, text, *, key, occurred_us=None, role="user", persona_id=None
    ):
        persona = (
            await self.control.personas.published(persona_id) if persona_id else None
        )
        scope = await self.scope(identity, persona)
        stamp_key = digest(key)
        stamp = await self.control.store.get("observation_stamps", stamp_key)
        fingerprint = digest(scope["agent_id"], scope["space_id"], role, text)
        if stamp:
            if stamp["value"]["fingerprint"] != fingerprint:
                raise IrisError(
                    "idempotency_conflict", "同一操作键不能写入不同正文或作用域"
                )
            now = stamp["value"]["time"]
        else:
            now = occurred_us or int(time.time() * 1e6)
            try:
                await self.control.store.put(
                    "observation_stamps",
                    stamp_key,
                    {"time": now, "fingerprint": fingerprint},
                    expected_revision=0,
                )
            except IrisError as exc:
                if exc.code != "revision_conflict":
                    raise
                stamp = await self.control.store.get("observation_stamps", stamp_key)
                if stamp["value"]["fingerprint"] != fingerprint:
                    raise IrisError(
                        "idempotency_conflict", "同一操作键已被使用"
                    ) from exc
                now = stamp["value"]["time"]
        record = {
            "agent_id": scope["agent_id"],
            "space_id": scope["space_id"],
            "role": role,
            "kind": "message.text",
            "effect_state": "committed",
            "idempotency_key": key,
            "occurred_us": now,
            "committed_us": now,
            "content": text,
        }
        result = await self.backend.call("observe_batch", [record], idempotency_key=key)
        self.control.logs.emit("observation.accepted", key=key, role=role, body=text)
        return result

    async def mirror(self, scope, persona):
        if not persona:
            return None
        self.backend.require("persona.mirror.v1")
        async with self.mirror_lock:
            path = {"agent_id": scope["agent_id"]}
            current = (await self.backend.operation("getCurrentPersona", path=path))[
                "revision"
            ]
            identity = current.get("core", {}).get("identity", {})
            if (
                isinstance(identity, dict)
                and identity.get("plugin_persona_id") == persona["id"]
                and identity.get("plugin_revision") == persona["revision"]
            ):
                chunks = identity.get("text_chunks", [identity.get("text", "")])
                if "".join(chunks) != persona["text"]:
                    raise IrisError("mirror_conflict", "Core 人格正文与插件修订不符")
                return current
            row = await self.control.store.get("mirrors", scope["agent_id"])
            if row and (
                row["value"]["core_revision"] != current["revision"]
                or row["value"]["content_hash"] != current["content_hash"]
            ):
                raise IrisError("mirror_conflict", "Core 人格已被外部修改，未覆盖")
            if not row and current["revision"] > 1:
                raise IrisError(
                    "mirror_conflict", "远程人格已有修订，请先由运维确认独立 Agent"
                )
            body = {
                "expected_revision": current["revision"],
                "core": {
                    "identity": {
                        "plugin_persona_id": persona["id"],
                        "plugin_revision": persona["revision"],
                        "text_chunks": [
                            persona["text"][i : i + 4096]
                            for i in range(0, len(persona["text"]), 4096)
                        ],
                    }
                },
                "traits": {},
                "narrative": {},
                "reason": "plugin_publish",
            }
            await self.backend.operation(
                "publishPersonaRevision",
                path=path,
                body=body,
                key="persona:"
                + digest(scope["agent_id"], persona["id"], str(persona["revision"])),
            )
            check = (await self.backend.operation("getCurrentPersona", path=path))[
                "revision"
            ]
            if check.get("core") != body["core"]:
                raise IrisError("mirror_conflict", "人格镜像回读不一致，暂停注入")
            await self.control.store.put(
                "mirrors",
                scope["agent_id"],
                {
                    "plugin_id": persona["id"],
                    "plugin_revision": persona["revision"],
                    "core_revision": check["revision"],
                    "content_hash": check["content_hash"],
                },
                expected_revision=row["revision"] if row else 0,
            )
            return check

    async def recall(self, identity, query, budget=None):
        persona = await self.persona(identity)
        scope = await self.scope(identity, persona)
        mirror = await self.mirror(scope, persona)
        record = {
            "schema_version": 1,
            "request_id": uuid.uuid4().hex,
            "scope": {k: scope[k] for k in ("agent_id", "space_id")},
            "actors": [
                {
                    "provider": identity.platform,
                    "realm": scope.get("realm", identity.realm),
                    "external_id": identity.user,
                }
            ],
            "topic": (query.strip() or "recent context")[:512],
            "purpose": "reply",
            "token_budget": self.control.settings["context_tokens"]
            if budget is None
            else budget,
            "deadline_at": (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.control.settings["request_timeout"])
            ).isoformat(),
            "include_trace": True,
        }
        result = await self.backend.call("recall", record)
        if mirror and (
            result["persona_revision"] != mirror["revision"]
            or result["persona_content_hash"] != mirror["content_hash"]
        ):
            raise IrisError("mirror_changed", "召回期间人格镜像发生变化，请重试")
        self.control.logs.emit(
            "memory.recalled",
            request_id=result["request_id"],
            candidates=len(result["candidates"]),
            partial=result.get("partial"),
            degraded=result.get("degraded_routes"),
        )
        return result

    async def remember(self, identity, text, *, key=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise IrisError("invalid_memory", "记忆正文须为 1–4000 字符")
        persona = await self.persona(identity)
        scope = await self.scope(identity, persona)
        if not scope.get("entity_id"):
            raise IrisError(
                "identity_unconfigured", "远程显式记忆需要预配置当前用户 entity_id"
            )
        key = key or uuid.uuid4().hex
        observed = await self.observe(
            identity,
            text,
            key="evidence:" + key,
            persona_id=persona["id"] if persona else None,
        )
        evidence = [
            {
                "source_type": "observation",
                "source_id": observed["accepted_observation_ids"][0],
                "relation": "supports",
                "source_authority": "user_statement",
            }
        ]
        return await self.backend.call(
            "remember_claim",
            {
                "agent_id": scope["agent_id"],
                "space_id": scope["space_id"],
                **await self.backend.proof(scope["agent_id"]),
                "subject_entity_id": scope["entity_id"],
                "predicate": "stated",
                "category": "fact",
                "value": {"text": text},
                "canonical_text": text,
                "evidence": evidence,
            },
            idempotency_key="remember:" + key,
        )

    async def claim(self, identity, claim_id):
        scope = await self.scope(identity, await self.persona(identity))
        row = await self.backend.operation("getClaim", path={"claim_id": claim_id})
        if (
            row.get("agent_id") != scope["agent_id"]
            or row.get("scope", {}).get("space_id") != scope["space_id"]
        ):
            raise IrisError("scope_denied", "记忆不属于当前会话/人格")
        return row

    async def correct(self, identity, claim_id, text, expected_revision, *, key=None):
        claim = await self.claim(identity, claim_id)
        key = key or uuid.uuid4().hex
        persona = await self.persona(identity)
        observed = await self.observe(
            identity,
            text,
            key="correct-evidence:" + key,
            persona_id=persona["id"] if persona else None,
        )
        return await self.backend.call(
            "correct_claim",
            claim_id,
            {
                **await self.backend.proof(claim["agent_id"]),
                "expected_revision": expected_revision,
                "reason": "explicit correction",
                "mode": "supersede",
                "value": {"text": text},
                "canonical_text": text,
                "evidence": [
                    {
                        "source_type": "observation",
                        "source_id": observed["accepted_observation_ids"][0],
                        "relation": "corrects",
                        "source_authority": "explicit_correction",
                    }
                ],
            },
            idempotency_key="correct:" + key,
        )

    async def forget(self, identity, claim_id, *, key=None):
        claim = await self.claim(identity, claim_id)
        return await self.backend.call(
            "forget_memory",
            {
                "selector": {
                    "kind": "resource",
                    "resource_type": "claim",
                    "resource_id": claim_id,
                },
                "reason": "explicit user request",
                **await self.backend.proof(claim["agent_id"]),
            },
            idempotency_key="forget:" + (key or uuid.uuid4().hex),
        )

    async def tasks(self, identity):
        self.backend.require("tasks.v1")
        scope = await self.scope(identity, await self.persona(identity))
        return await self.backend.operation(
            "listTasks",
            query={"agent_id": scope["agent_id"], "space_id": scope["space_id"]},
        )

    async def create_task(self, identity, title, due_at_us, *, key=None):
        self.backend.require("tasks.v1")
        scope = await self.scope(identity, await self.persona(identity))
        return await self.backend.operation(
            "createTask",
            body={
                "agent_id": scope["agent_id"],
                "space_id": scope["space_id"],
                "title": title,
                "due_at_us": int(due_at_us),
                "origin": "explicit_tool",
            },
            key=key or uuid.uuid4().hex,
        )

    async def transition_task(
        self, identity, task_id, target, expected_revision, *, key=None
    ):
        rows = await self.tasks(identity)
        items = rows.get("items", rows.get("tasks", []))
        task = next((t for t in items if t["task_id"] == task_id), None)
        if task is None:
            raise IrisError("scope_denied", "任务不属于当前会话/人格")
        return await self.backend.operation(
            "transitionTask",
            path={"task_id": task_id},
            body={
                **await self.backend.proof(task["agent_id"]),
                "target": target,
                "expected_revision": expected_revision,
                "reason": "plugin user action",
                "origin": "explicit_tool",
            },
            key=key or uuid.uuid4().hex,
        )

    async def profile(self, identity):
        self.backend.require("profile.v1")
        scope = await self.scope(identity, await self.persona(identity))
        if not scope.get("entity_id"):
            raise IrisError("identity_unconfigured", "需配置当前用户的实体映射")
        return await self.backend.operation(
            "getEntityProfile",
            path={"entity_id": scope["entity_id"]},
            query={"agent_id": scope["agent_id"], "space_id": scope["space_id"]},
        )

    async def focuses(self, identity):
        self.backend.require("focus-items.v1")
        scope = await self.scope(identity, await self.persona(identity))
        return await self.backend.operation(
            "listFocusItems",
            query={"agent_id": scope["agent_id"], "space_id": scope["space_id"]},
        )

    async def create_focus(self, identity, summary, *, key=None):
        self.backend.require("focus-items.v1")
        scope = await self.scope(identity, await self.persona(identity))
        return await self.backend.operation(
            "createFocusItem",
            body={
                "agent_id": scope["agent_id"],
                "space_id": scope["space_id"],
                "kind": "concern",
                "summary": summary,
                **await self.backend.proof(scope["agent_id"]),
            },
            key=key or uuid.uuid4().hex,
        )

    async def dismiss_focus(
        self, identity, focus_item_id, expected_revision, *, key=None
    ):
        items = (await self.focuses(identity)).get("items", [])
        item = next((f for f in items if f["focus_item_id"] == focus_item_id), None)
        if not item:
            raise IrisError("scope_denied", "关注点不属于当前会话/人格")
        return await self.backend.operation(
            "dismissFocusItem",
            path={"focus_item_id": focus_item_id},
            body={
                "expected_revision": expected_revision,
                "reason": "plugin user action",
                **await self.backend.proof(item["agent_id"]),
            },
            key=key or uuid.uuid4().hex,
        )

    async def usage(self, envelope, selected, *, visible):
        self.backend.require("recall.usage.v1")
        return await self.backend.operation(
            "reportRecallUsage",
            path={"request_id": envelope["request_id"]},
            body={
                "host_cycle_id": envelope["request_id"],
                "persona_revision": envelope["persona_revision"],
                "returned_candidate_ids": [
                    c["candidate_id"] for c in envelope["candidates"]
                ],
                "host_selected_candidate_ids": selected,
                "model_visible_candidate_ids": selected if visible else [],
                "reported_at": utc(),
            },
            key="usage:" + envelope["request_id"],
        )
