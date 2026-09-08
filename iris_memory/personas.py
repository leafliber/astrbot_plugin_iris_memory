"""Plugin-owned immutable persona revisions and explicit publication."""

import hashlib
import time
import uuid

from .errors import Conflict, IrisError


class Personas:
    def __init__(self, store):
        self.store = store

    async def list(self):
        return await self.store.list("personas", limit=100)

    async def save(
        self,
        *,
        persona_id=None,
        name,
        text,
        expected_revision=0,
        reason="edit",
        source_refs=None,
    ):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise IrisError("invalid_persona", "人格名称须为 1–100 个字符")
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 28000:
            raise IrisError(
                "invalid_persona", "人格正文不能为空且不得超过 28000 UTF-8 字节"
            )
        persona_id = persona_id or uuid.uuid4().hex
        if not isinstance(persona_id, str) or not 1 <= len(persona_id) <= 64:
            raise IrisError("invalid_persona", "人格 ID 须为 1–64 字符")
        existing = await self.store.get("personas", persona_id)
        if (existing["revision"] if existing else 0) != expected_revision:
            raise Conflict()
        old = existing["value"] if existing else {}
        value = {
            "id": persona_id,
            "name": name.strip(),
            "text": text,
            "hash": hashlib.sha256(text.encode()).hexdigest(),
            "published": old.get("published"),
            "disabled": False,
            "created": time.time(),
            "reason": str(reason)[:256],
            "source_refs": source_refs or [],
        }
        return await self.store.put(
            "personas",
            persona_id,
            value,
            expected_revision=expected_revision,
            history=True,
        )

    async def publish(self, persona_id, expected_revision):
        row = await self.store.get("personas", persona_id)
        if not row or row["revision"] != expected_revision:
            raise Conflict()
        value = {
            **row["value"],
            "published": {
                "revision": row["revision"],
                "text": row["value"]["text"],
                "hash": row["value"]["hash"],
                "name": row["value"]["name"],
            },
        }
        return await self.store.put(
            "personas",
            persona_id,
            value,
            expected_revision=expected_revision,
            history=True,
        )

    async def rollback(self, persona_id, target_revision, expected_revision):
        rows = await self.store.history("personas", persona_id)
        target = next((r for r in rows if r["revision"] == target_revision), None)
        if not target:
            raise IrisError("not_found", "人格修订不存在或不在最近 100 个修订中")
        created = await self.save(
            persona_id=persona_id,
            name=target["value"]["name"],
            text=target["value"]["text"],
            expected_revision=expected_revision,
            reason=f"rollback:{target_revision}",
        )
        return await self.publish(persona_id, created["revision"])

    async def disable(self, persona_id, expected_revision):
        in_use = await self.store.run(
            lambda db: db.execute(
                "SELECT 1 FROM documents WHERE collection='bindings' AND json_extract(value, '$.persona_id')=? LIMIT 1",
                (persona_id,),
            ).fetchone()
            is not None
        )
        if in_use:
            raise IrisError("persona_in_use", "请先解除人格绑定")
        row = await self.store.get("personas", persona_id)
        if not row:
            raise IrisError("not_found", "人格不存在")
        return await self.store.put(
            "personas",
            persona_id,
            {**row["value"], "disabled": True},
            expected_revision=expected_revision,
            history=True,
        )

    async def bind(self, key, persona_id, expected_revision=0):
        if not isinstance(key, str) or not 1 <= len(key) <= 256:
            raise IrisError("invalid_binding", "绑定会话键无效")
        if persona_id:
            await self.published(persona_id)
        return await self.store.put(
            "bindings",
            key,
            {"persona_id": persona_id},
            expected_revision=expected_revision,
        )

    async def published(self, persona_id):
        row = await self.store.get("personas", persona_id)
        if not row or row["value"]["disabled"] or not row["value"].get("published"):
            raise IrisError("persona_unpublished", "人格尚未发布或已停用")
        return {"id": persona_id, **row["value"]["published"]}

    async def resolve(self, identity, default=""):
        for key in (identity.session_key, identity.key):
            binding = await self.store.get("bindings", key)
            if binding and binding["value"].get("persona_id"):
                return await self.published(binding["value"]["persona_id"])
        return await self.published(default) if default else None
