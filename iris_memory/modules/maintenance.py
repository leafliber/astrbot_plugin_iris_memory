"""Optional single bounded maintenance loop; no independent memory engine."""

import asyncio
import time


class Module:
    def __init__(self, control):
        self.control = control
        self.worker = None

    async def start(self):
        self.worker = asyncio.create_task(self.loop(), name="iris-maintenance")

    async def close(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None

    async def loop(self):
        while True:
            await asyncio.sleep(30)
            try:
                await self.run()
            except Exception as exc:
                self.control.logs.emit("maintenance.failed", level="ERROR", error=exc)

    async def run(self):
        result = await self.control.require("memory").backend.maintain()
        await self.control.store.run(
            lambda db: db.execute(
                "DELETE FROM deliveries WHERE expires < ?", (time.time(),)
            )
        )
        self.control.logs.emit("maintenance.completed", result=result)
        return result

    async def rebuild(self, kind):
        from ..errors import IrisError

        if kind not in {"fts", "graph", "profile", "vector", "recent_context"}:
            raise IrisError("invalid_index", "索引类型无效")
        backend = self.control.require("memory").backend
        backend.require("admin.index-rebuild.v1")
        if self.control.mode == "remote":
            return await backend.call(
                "rebuild_index", kind, reason="plugin manual rebuild"
            )
        import uuid

        return await backend.operation(
            "rebuildIndex",
            body={"reason": "plugin manual rebuild"},
            path={"kind": kind},
            key=uuid.uuid4().hex,
        )
