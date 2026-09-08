"""Core-backed reminders and optional quiet-conversation followups."""

import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ..errors import IrisError
from ..identity import Identity, digest


class Module:
    def __init__(self, control):
        self.control = control
        self.worker = None
        self.lock = asyncio.Lock()

    async def start(self):
        self.control.modules["memory"].backend.require("tasks.v1")
        self.worker = asyncio.create_task(self.loop(), name="iris-proactive")

    async def close(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None

    async def loop(self):
        while True:
            await asyncio.sleep(15)
            try:
                await self.tick()
            except Exception as exc:
                self.control.logs.emit("proactive.failed", level="ERROR", error=exc)

    def quiet(self, now):
        settings = self.control.settings
        hour = datetime.fromtimestamp(now, ZoneInfo(settings["timezone"])).hour
        start, end = settings["quiet_start"], settings["quiet_end"]
        return (
            start <= hour < end
            if start < end
            else hour >= start or hour < end
            if start > end
            else False
        )

    async def tick(self):
        async with self.lock:
            now = time.time()
            if self.quiet(now):
                return
            for conversation in await self.control.store.list(
                "conversations", limit=500
            ):
                identity = Identity(
                    **{
                        k: conversation["value"][k]
                        for k in Identity.__dataclass_fields__
                    }
                )
                if not self.control.allowed(identity):
                    continue
                try:
                    await self.conversation(identity, conversation, now)
                except Exception as exc:
                    self.control.logs.emit(
                        "proactive.conversation_failed",
                        level="WARNING",
                        error=exc,
                        conversation=identity.key,
                    )

    async def conversation(self, identity, snapshot, now):
        settings = self.control.settings
        receipt = await self.control.store.get("last_send", identity.key)
        if receipt and now - receipt["value"]["time"] < settings["proactive_cooldown"]:
            return
        memory = self.control.require("memory")
        envelope = await memory.tasks(identity)
        tasks = envelope.get("items", envelope.get("tasks", []))
        for task in tasks:
            due = (task.get("due_at_us") or 0) / 1e6
            if task["status"] != "active" or not due or due > now:
                continue
            if now - due > settings["catchup_seconds"]:
                continue
            key = "reminder:" + digest(identity.key, task["task_id"], str(due))
            if await self.control.store.get("effects", key):
                continue
            # Reminders use their Core task text verbatim, with no LLM charge.
            await self.deliver(
                identity,
                snapshot,
                "提醒：" + task["title"],
                key,
                now,
                expected_task=task,
            )
            return
        interval = settings["initiate_after_seconds"]
        if (
            not interval
            or identity.is_group
            or now - snapshot["value"]["last_seen"] < interval
        ):
            return
        key = "followup:" + digest(identity.key, str(snapshot["revision"]))
        if await self.control.store.get("effects", key):
            return
        persona = await memory.persona(identity)
        if not persona:
            raise IrisError("persona_required", "主动发起需要插件已发布人格")
        prompt = (
            "结合相关记忆做一次简短自然的关心。不要制造紧迫感或声称用户有未表达的感受。"
        )
        builder = self.control.require("context")
        plan = await builder.build(identity, prompt, system=persona["text"])
        response = await self.control.host.generate(
            self.control, prompt + "\n" + plan["text"], system=persona["text"]
        )
        await builder.report(plan, visible=True)
        await self.deliver(
            identity, snapshot, response, key, now, expected_persona=persona
        )

    async def deliver(
        self,
        identity,
        snapshot,
        text,
        key,
        now,
        *,
        expected_task=None,
        expected_persona=None,
    ):
        latest = await self.control.store.get("conversations", identity.key)
        if (
            not latest
            or latest["revision"] != snapshot["revision"]
            or not self.control.accepting
            or not self.control.allowed(identity)
        ):
            raise IrisError("conversation_changed", "会话已有新消息，取消主动发言")
        persona = await self.control.require("memory").persona(identity)
        if expected_persona and (
            not persona
            or (persona["id"], persona["revision"])
            != (expected_persona["id"], expected_persona["revision"])
        ):
            raise IrisError("persona_changed", "人格已变更，取消旧版本主动回复")
        if expected_task:
            current_tasks = await self.control.require("memory").tasks(identity)
            current = next(
                (
                    item
                    for item in current_tasks.get("items", [])
                    if item["task_id"] == expected_task["task_id"]
                ),
                None,
            )
            if (
                not current
                or current["revision"] != expected_task["revision"]
                or current["status"] != "active"
            ):
                raise IrisError("task_changed", "任务已取消或修订，取消提醒")
        # A persistent sending intent is never blindly resent after a crash.
        row = await self.control.store.put(
            "effects", key, {"state": "sending", "time": now}, expected_revision=0
        )
        state = "unknown"
        try:
            latest = await self.control.store.get("conversations", identity.key)
            if latest["revision"] != snapshot["revision"] or not self.control.accepting:
                state = "cancelled"
                raise IrisError("conversation_changed", "提交前出现新消息，取消发送")
            accepted = await self.control.host.send(identity.umo, text)
            state = "accepted" if accepted else "rejected"
        finally:
            await self.control.store.put(
                "effects",
                key,
                {"state": state, "time": now},
                expected_revision=row["revision"],
            )
        if state == "accepted":
            previous = await self.control.store.get("last_send", identity.key)
            await self.control.store.put(
                "last_send",
                identity.key,
                {"time": now},
                expected_revision=previous["revision"] if previous else 0,
            )
            # Platform acceptance is not delivery confirmation. Keep it out of
            # committed observations until a platform receipt can prove delivery.
            self.control.logs.emit(
                "proactive.accepted",
                key=key,
                persona_id=persona["id"] if persona else None,
                body=text,
            )
        else:
            self.control.logs.emit(
                "proactive.send_unconfirmed", level="WARNING", key=key, reason=state
            )
