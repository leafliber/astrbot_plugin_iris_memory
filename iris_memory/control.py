"""Always-on control plane; optional modules are loaded only when enabled."""

import asyncio
import importlib
import time
from pathlib import Path

from .budget import Budget
from .config import MODULES, defaults, describe, public_settings, validate
from .diagnostics import Diagnostics
from .errors import Conflict, IrisError
from .personas import Personas
from .storage import Store

ORDER = (
    "diagnostics",
    "memory",
    "persona",
    "context",
    "media",
    "learning",
    "maintenance",
    "proactive",
)


class Control:
    def __init__(self, directory, host, mode="local", *, backend_factory=None):
        self.directory = Path(directory)
        self.host, self.mode = host, mode
        self.backend_factory = backend_factory
        self.store = Store(self.directory)
        self.personas = Personas(self.store)
        self.logs = Diagnostics(self.directory, getattr(host, "logger", None))
        self.budget = Budget(self)
        self.settings, self.revision = defaults(), 0
        self.modules, self.failures = {}, {}
        self.lock = asyncio.Lock()
        self.tasks = set()
        self.generation = 0
        self.accepting = False
        self.started = False

    async def start(self):
        async with self.lock:
            if self.started:
                return
            await self.store.start()
            row = await self.store.get("config", "active")
            if row:
                self.settings, self.revision = (
                    validate(defaults(), row["value"]),
                    row["revision"],
                )
            await self._start_modules(tolerate=True)
            self.accepting = self.started = True

    async def _start_modules(self, *, tolerate=False):
        self.logs.secrets = [self.settings["remote_token"]]
        for name in ORDER:
            if not self.settings["modules"][name]:
                continue
            try:
                if any(dep not in self.modules for dep in MODULES[name][1]):
                    raise IrisError("dependency_failed", "依赖模块未启动")
                if name == "diagnostics":
                    module = self.logs
                    self.modules[name] = module
                    await module.start(self.settings)
                else:
                    module = importlib.import_module(
                        f"{__package__}.modules.{name}"
                    ).Module(self)
                    self.modules[name] = module
                    await module.start()
                self.failures.pop(name, None)
                self.logs.emit(
                    "module.started", module=name, generation=self.generation
                )
            except Exception as exc:
                self.failures[name] = {
                    "code": getattr(exc, "code", "startup_failed"),
                    "message": str(exc),
                }
                self.logs.emit(
                    "module.start_failed", level="ERROR", error=exc, module=name
                )
                module = self.modules.get(name)
                if module:
                    try:
                        await module.close()
                    except Exception:
                        # Retain ownership of resources that could not drain.
                        raise
                    del self.modules[name]
                if not tolerate:
                    raise

    async def _stop_modules(self):
        self.accepting = False
        self.generation += 1
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        for name in reversed(ORDER):
            module = self.modules.get(name)
            if module:
                await module.close()
                del self.modules[name]

    async def apply(self, changes, expected_revision):
        async with self.lock:
            if expected_revision != self.revision:
                raise Conflict()
            updated = validate(self.settings, changes)
            if updated["default_persona"]:
                await self.personas.published(updated["default_persona"])
            old = self.settings
            # No persistent "active" update until all new resources have started.
            await self._stop_modules()
            self.settings = updated
            try:
                await self._start_modules()
                row = await self.store.put(
                    "config", "active", updated, expected_revision=self.revision
                )
            except BaseException:
                await self._stop_modules()
                self.settings = old
                await self._start_modules(tolerate=True)
                self.accepting = True
                raise
            self.revision = row["revision"]
            self.accepting = True
            self.logs.emit("config.applied", revision=self.revision, keys=list(changes))
            return self.status()

    def require(self, name):
        if not self.accepting or name not in self.modules:
            raise IrisError("module_unavailable", f"{name} 未启用或正在重载")
        return self.modules[name]

    async def run(self, name, method, *args, **kwargs):
        module = self.require(name)
        generation = self.generation
        if len(self.tasks) >= 32:
            raise IrisError("busy", "插件请求过多")
        task = asyncio.create_task(
            getattr(module, method)(*args, **kwargs), name=f"iris-{name}-{method}"
        )
        self.tasks.add(task)
        try:
            result = await task
            if generation != self.generation:
                raise IrisError("generation_changed", "配置已变更，本次结果未采用")
            return result
        except asyncio.CancelledError:
            if generation != self.generation:
                raise IrisError(
                    "generation_changed", "模块已停用，本次操作已取消"
                ) from None
            raise
        finally:
            self.tasks.discard(task)

    def allowed(self, identity):
        return identity.key in self.settings["allowed_conversations"]

    async def remember_conversation(self, identity):
        from dataclasses import asdict

        from .storage import encode

        value = encode({**asdict(identity), "last_seen": time.time()})

        def upsert(db):
            db.execute(
                "INSERT INTO documents VALUES('conversations',?,1,?) ON CONFLICT(collection,key) DO UPDATE SET revision=documents.revision+1,value=excluded.value",
                (identity.key, value),
            )
            db.execute(
                "DELETE FROM documents WHERE collection='conversations' AND key NOT IN (SELECT key FROM documents WHERE collection='conversations' ORDER BY json_extract(value,'$.last_seen') DESC LIMIT 512)"
            )

        await self.store.run(upsert)

    def status(self):
        return {
            "mode": self.mode,
            "revision": self.revision,
            "generation": self.generation,
            "accepting": self.accepting,
            "settings": public_settings(self.settings),
            "fields": describe(),
            "modules": {
                name: {
                    "label": label,
                    "dependencies": list(deps),
                    "enabled": self.settings["modules"][name],
                    "state": "running"
                    if name in self.modules
                    else "failed"
                    if name in self.failures
                    else "disabled",
                    "error": self.logs.sanitize(self.failures.get(name)),
                }
                for name, (label, deps) in MODULES.items()
            },
            "capabilities": getattr(
                getattr(self.modules.get("memory"), "backend", None), "capabilities", {}
            ),
        }

    async def close(self):
        async with self.lock:
            await self._stop_modules()
            await self.store.close()
            self.started = False
