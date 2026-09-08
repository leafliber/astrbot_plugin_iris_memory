"""Authenticated Pages commands, separate from transport and UI rendering."""

import json

from .config import validate, public_settings
from .errors import IrisError
from .identity import Identity


class PagesAPI:
    def __init__(self, control):
        self.control = control

    async def identity(self, key):
        row = await self.control.store.get("conversations", key)
        if not row:
            raise IrisError("conversation_unknown", "请先在目标会话执行 /iris status")
        return Identity(**{k: row["value"][k] for k in Identity.__dataclass_fields__})

    async def dispatch(self, action, data=None, *, username):
        if not username:
            raise IrisError("unauthenticated", "请登录 AstrBot 管理后台")
        if not self.control.started:
            raise IrisError("plugin_stopped", "插件已卸载或正在启动")
        data = data or {}
        if (
            not isinstance(data, dict)
            or len(json.dumps(data, ensure_ascii=False).encode()) > 262144
        ):
            raise IrisError("invalid_request", "请求须为不超过 256 KiB 的对象")
        c = self.control
        c.logs.emit("pages.request", action=action, operator=username)
        if action == "status":
            return c.status()
        if action == "providers":
            return c.host.provider_list()
        if action == "config.validate":
            return public_settings(validate(c.settings, data["changes"]))
        if action == "config.apply":
            return await c.apply(data["changes"], data["expected_revision"])
        if action == "conversations":
            return await c.store.list("conversations", limit=500)
        if action == "bindings":
            return await c.store.list("bindings", limit=500)
        if action == "personas":
            return await c.personas.list()
        if action == "persona.save":
            return await c.personas.save(
                persona_id=data.get("persona_id"),
                name=data["name"],
                text=data["text"],
                expected_revision=data.get("expected_revision", 0),
            )
        if action == "persona.publish":
            row = await c.store.get("personas", data["persona_id"])
            for ref in row["value"].get("source_refs", []) if row else []:
                identity = await self.identity(ref["conversation"])
                claim = await c.run("memory", "claim", identity, ref["resource_id"])
                if claim["revision"] != ref["revision"] or claim["status"] != "active":
                    raise IrisError(
                        "evidence_changed", "学习证据已失效，请重新生成或手动编辑草稿"
                    )
            return await c.personas.publish(
                data["persona_id"], data["expected_revision"]
            )
        if action == "persona.rollback":
            return await c.personas.rollback(
                data["persona_id"], data["target_revision"], data["expected_revision"]
            )
        if action == "persona.history":
            return await c.store.history("personas", data["persona_id"])
        if action == "persona.disable":
            if c.settings["default_persona"] == data["persona_id"]:
                raise IrisError("persona_in_use", "请先清空默认人格配置")
            return await c.personas.disable(
                data["persona_id"], data["expected_revision"]
            )
        if action == "persona.bind":
            return await c.personas.bind(
                data["key"], data["persona_id"], data.get("expected_revision", 0)
            )
        if action == "persona.export":
            row = await c.store.get("personas", data["persona_id"])
            if not row:
                raise IrisError("not_found", "人格不存在")
            return {
                "format": "iris-persona-v4",
                "name": row["value"]["name"],
                "text": row["value"]["text"],
            }
        if action == "persona.import":
            payload = data["document"]
            if (
                not isinstance(payload, dict)
                or payload.get("format") != "iris-persona-v4"
            ):
                raise IrisError(
                    "invalid_import", "仅接受 iris-persona-v4 格式；导入为新草稿"
                )
            return await c.personas.save(name=payload["name"], text=payload["text"])
        if action == "persona.release":
            await c.host.release_personas(c, data["umo"])
            return {"released": True}
        if action == "logs":
            return await c.logs.query(
                limit=data.get("limit", 100),
                before=data.get("before"),
                operation=data.get("operation"),
                level=data.get("level"),
            )
        if action == "delivery.status":
            return await c.store.run(
                lambda db: {
                    "queue": [
                        dict(r)
                        for r in db.execute(
                            "SELECT state,count(*) AS count FROM deliveries GROUP BY state"
                        )
                    ],
                    "effects": [
                        dict(r)
                        for r in db.execute(
                            "SELECT key,value FROM documents WHERE collection='effects' ORDER BY rowid DESC LIMIT 100"
                        )
                    ],
                }
            )
        if action == "maintenance.run":
            return await c.run("maintenance", "run")
        if action == "maintenance.rebuild":
            return await c.run("maintenance", "rebuild", data["kind"])
        if action.startswith(("memory.", "task.", "learning.")):
            identity = await self.identity(data["conversation"])
            key = data.get("idempotency_key")
            if key is not None and (
                not isinstance(key, str) or not 1 <= len(key) <= 128
            ):
                raise IrisError("invalid_key", "操作幂等键无效")
            if action == "memory.profile":
                return await c.run("memory", "profile", identity)
            if action == "task.focus.list":
                return await c.run("memory", "focuses", identity)
            if action == "task.focus.create":
                return await c.run(
                    "memory", "create_focus", identity, data["summary"], key=key
                )
            if action == "task.focus.dismiss":
                return await c.run(
                    "memory",
                    "dismiss_focus",
                    identity,
                    data["focus_item_id"],
                    data["expected_revision"],
                    key=key,
                )
            if action == "memory.recall":
                return await c.run("memory", "recall", identity, data["query"])
            if action == "memory.remember":
                return await c.run(
                    "memory", "remember", identity, data["text"], key=key
                )
            if action == "memory.claim":
                return await c.run("memory", "claim", identity, data["claim_id"])
            if action == "memory.correct":
                return await c.run(
                    "memory",
                    "correct",
                    identity,
                    data["claim_id"],
                    data["text"],
                    data["expected_revision"],
                    key=key,
                )
            if action == "memory.forget":
                return await c.run(
                    "memory", "forget", identity, data["claim_id"], key=key
                )
            if action == "task.list":
                return await c.run("memory", "tasks", identity)
            if action == "task.create":
                return await c.run(
                    "memory",
                    "create_task",
                    identity,
                    data["title"],
                    data["due_at_us"],
                    key=key,
                )
            if action == "task.transition":
                return await c.run(
                    "memory",
                    "transition_task",
                    identity,
                    data["task_id"],
                    data["target"],
                    data["expected_revision"],
                    key=key,
                )
            if action == "learning.draft":
                return await c.run(
                    "learning",
                    "draft",
                    identity,
                    data["persona_id"],
                    data["query"],
                    data["expected_revision"],
                )
        raise IrisError("not_found", "未知操作")
