# ruff: noqa: F811
import json
import time

import pytest

from iris_memory.application import Application
from iris_memory.errors import ControlError
from tests.test_ingress_delivery import core  # noqa: F401


@pytest.mark.parametrize("lost", [False, True])
async def test_token_commit_and_possession_are_separate(tmp_path, core, lost):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await app.save_connection(0, core.base)
        c = await app.store.settings()
        await app.authorize_admin(
            "alice", "synthetic-session", "synthetic-csrf", c["revision"]
        )
        c = await app.store.settings()
        request = {
            "host_id": "host",
            "entries": ["entry-0"],
            "operations": ["accept", "confirm", "media_upload", "media_inspect"],
            "expires_at_us": int((time.time() + 3600) * 1000000),
        }
        core.drop_token = lost
        if lost:
            with pytest.raises(ControlError, match="HTTP_UNAVAILABLE"):
                await app.registration.perform(
                    "alice",
                    "token_create",
                    request,
                    group_id="group-0",
                    expected_revision=c["revision"],
                )
            operations = await app.store.operations()
            original = operations["items"][0]
            result = await app.registration.perform(
                "alice", None, operation_id=original["id"]
            )
            assert (
                result["state"] == "COMMITTED"
                and not result["secret_obtained_this_response"]
            )
            assert not (await app.store.settings())["groups"]
        else:
            result = await app.registration.perform(
                "alice",
                "token_create",
                request,
                group_id="group-0",
                expected_revision=c["revision"],
            )
            assert (
                result["state"] == "COMMITTED"
                and result["secret_obtained_this_response"]
            )
            assert (
                await app.sources.credential(
                    (await app.store.settings())["groups"][0]["credential_ref"]
                )
            ) == "new-synthetic-token"
        assert core.token_count == 1
        safe = json.dumps(await app.diagnostics("alice"))
        assert "new-synthetic-token" not in safe and "synthetic-session" not in safe
    finally:
        await app.terminate()


async def test_register_confirm_cas_and_session_isolation(tmp_path, core):
    app = Application(tmp_path)
    await app.initialize()
    try:
        await app.save_connection(0, core.base)
        c = await app.store.settings()
        await app.authorize_admin(
            "alice", "synthetic-session", "synthetic-csrf", c["revision"]
        )
        c = await app.store.settings()
        request = {
            "entry_id": "entry-0",
            "host_id": "host",
            "platform_id": "onebot",
            "external_entry_id": "stable-source",
        }
        first = await app.registration.perform(
            "alice", "source_register", request, expected_revision=c["revision"]
        )
        assert first["state"] == "COMMITTED"
        with pytest.raises(ControlError, match="REVISION_CONFLICT"):
            await app.registration.perform(
                "alice", "source_register", request, expected_revision=c["revision"]
            )
        with pytest.raises(ControlError, match="REAUTHORIZE"):
            await app.registration.perform("bob", None, operation_id=first["id"])
        assert (
            await app.registration.perform("alice", None, operation_id=first["id"])
        )["state"] == "COMMITTED"
        config = await app.store.settings()
        await app.save_connection(
            config["revision"], core.base, "new-legacy-credential"
        )
        config = await app.store.settings()
        await app.authorize_admin(
            "alice", "synthetic-session", "synthetic-csrf", config["revision"]
        )
        assert (
            await app.registration.perform("alice", None, operation_id=first["id"])
        )["state"] == "COMMITTED"
        assert core.registration_count == 1
    finally:
        await app.terminate()
