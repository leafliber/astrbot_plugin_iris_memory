"""Installed SDK -> actual Core ASGI routes, using a temporary service database."""

from functools import partial

import httpx
import pytest

from iris_memory.control import Control
from iris_memory.errors import IrisError


@pytest.mark.parametrize("surface_mode", ["off", "required"])
async def test_remote_public_client_chain(
    tmp_path, host, identity, other, monkeypatch, surface_mode
):
    import iris_memory_sdk
    from iris_memory_core.api import create_app
    from iris_memory_core.application.security import CredentialService
    from iris_memory_core.embedded import EmbeddedConfig, EmbeddedMemory, LocalBootstrap
    from iris_memory_core.storage.runtime import SQLiteRuntime, sqlite_runtime_version
    from iris_memory_core.storage.uow import Store
    from iris_memory.backends import CAPABILITIES

    # Service-side provisioning fixture is not part of the plugin runtime.
    service_dir = tmp_path / "server"
    async with EmbeddedMemory(
        EmbeddedConfig(
            service_dir,
            allow_local_sqlite=True,
            bootstrap=LocalBootstrap(
                surface_mode=surface_mode,
                manage_identities=True,
                manage_persona=True,
                capabilities=CAPABILITIES,
            ),
        )
    ) as memory:
        scope = await memory.start()
        actor = await memory.register_actor(
            identity.platform,
            identity.user,
            realm=identity.realm,
            display_name=identity.display_name,
            idempotency_key="fixture-actor",
        )
    store = Store(
        SQLiteRuntime(
            service_dir / "canonical.sqlite3",
            allowed_versions=(sqlite_runtime_version(),),
        )
    )
    credentials = CredentialService(store, store.clock)
    token = "fixture-token-for-plugin-integration"
    credentials.issue(
        token,
        tenant_id=scope["tenant_id"],
        app_instance_id="iris-plugin",
        plane="application",
        agent_ids=[scope["agent_id"]],
        space_ids=[scope["space_id"]],
        entity_ids=[actor["entity_id"]],
        data_purposes=["reply"],
        capabilities=CAPABILITIES,
        expires_us=store.clock.now_us() + 3600000000,
    )
    app = create_app(store, credentials=credentials)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://core.test"
    ) as http:
        original = iris_memory_sdk.AsyncIrisMemoryClient
        monkeypatch.setattr(
            iris_memory_sdk,
            "AsyncIrisMemoryClient",
            partial(original, http_client=http),
        )
        c = Control(tmp_path / "plugin", host, "remote")
        await c.start()
        try:
            await c.apply(
                {
                    "remote_app_instance": "iris-plugin"
                    if surface_mode == "required"
                    else "",
                    "remote_url": "http://core.test",
                    "remote_token": token,
                    "allowed_conversations": [identity.key, other.key],
                    "remote_bindings": {
                        identity.key + "/default": {
                            "agent_id": scope["agent_id"],
                            "space_id": scope["space_id"],
                            "entity_id": actor["entity_id"],
                        }
                    },
                    "modules": {"memory": True, "context": True},
                },
                0,
            )
            assert not (tmp_path / "plugin" / "core").exists()
            m = c.modules["memory"]
            result = await m.remember(identity, "远程用户喜欢茶", key="remote-remember")
            assert (
                await m.remember(identity, "远程用户喜欢茶", key="remote-remember")
            ) == result
            assert (await m.claim(identity, result["claim_id"]))["revision"] == 1
            plan = await c.modules["context"].build(identity, "茶")
            assert "远程用户喜欢茶" in plan["text"]
            await c.modules["context"].report(plan, visible=True)
            corrected = await m.correct(
                identity,
                result["claim_id"],
                "远程用户喜欢咖啡",
                1,
                key="remote-correct",
            )
            assert corrected["revision"] == 2
            with pytest.raises(IrisError, match="远程绑定"):
                await m.recall(other, "茶")
            task = await m.create_task(identity, "Remote task", 1900000000000000)
            assert (await m.tasks(identity))["items"][0]["task_id"] == task["task_id"]
            assert (
                await m.transition_task(
                    identity, task["task_id"], "cancel", task["revision"]
                )
            )["status"] == "cancelled"
            assert (await m.forget(identity, result["claim_id"]))["erased_count"] == 1
            # A transport failure never starts a local engine.
            await c.close()
            assert not http.is_closed
        finally:
            await c.close()
