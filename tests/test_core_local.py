import asyncio
import threading

import pytest

from iris_memory.control import Control
from iris_memory.errors import IrisError


async def local(tmp_path, host, identity, other):
    c = Control(tmp_path, host)
    await c.start()
    await c.apply(
        {
            "allow_development_sqlite": True,
            "allowed_conversations": [identity.key, other.key],
            "modules": {"memory": True, "context": True, "persona": True},
        },
        0,
    )
    return c


async def test_installed_core_write_recall_correct_forget_scope_restart(
    tmp_path, host, identity, other
):
    c = await local(tmp_path, host, identity, other)
    try:
        m = c.modules["memory"]
        first = await m.remember(identity, "用户喜欢中国茶", key="remember-one")
        assert (
            await m.remember(identity, "用户喜欢中国茶", key="remember-one")
        ) == first
        with pytest.raises(IrisError):
            await m.remember(identity, "another text", key="remember-one")
        with pytest.raises(IrisError, match="不属于"):
            await m.claim(other, first["claim_id"])
        assert not (await m.recall(other, "中国茶"))["candidates"]
        plan = await c.modules["context"].build(identity, "茶")
        assert "中国茶" in plan["text"]
        await c.modules["context"].report(plan, visible=True)
        corrected = await m.correct(
            identity, first["claim_id"], "用户现在喜欢咖啡", 1, key="correct-one"
        )
        assert corrected["revision"] == 2
        with pytest.raises(IrisError):
            await m.correct(identity, first["claim_id"], "outdated", 1)
        await c.close()
        await c.start()
        m = c.modules["memory"]
        assert (await m.claim(identity, first["claim_id"]))["revision"] == 2
        assert (await m.forget(identity, first["claim_id"]))["erased_count"] == 1
        assert not (await c.modules["context"].build(identity, "咖啡"))["text"]
    finally:
        await c.close()


async def test_core_persona_complete_text_mirror_cas(tmp_path, host, identity, other):
    c = await local(tmp_path, host, identity, other)
    try:
        m = c.modules["memory"]
        text = "这是 Iris 的完整人格。" * 500
        p = await c.personas.save(name="Iris", text=text)
        p = await c.personas.publish(p["value"]["id"], p["revision"])
        persona = await c.personas.published(p["value"]["id"])
        await c.personas.bind(identity.key, persona["id"])
        scope = await m.scope(identity, persona)
        mirrored = await m.mirror(scope, persona)
        assert "".join(mirrored["core"]["identity"]["text_chunks"]) == text
        recall = await m.recall(identity, "hello")
        assert recall["persona_revision"] == mirrored["revision"]
        # An external CAS edit is detected even without a local config reload.
        await m.backend.operation(
            "publishPersonaRevision",
            path={"agent_id": scope["agent_id"]},
            body={
                "expected_revision": mirrored["revision"],
                "core": {"identity": "external"},
                "traits": {},
                "narrative": {},
                "reason": "test external",
            },
            key="external",
        )
        with pytest.raises(IrisError, match="外部"):
            await m.mirror(scope, persona)
    finally:
        await c.close()


async def test_task_and_proactive_receipt_are_not_duplicate_task_store(
    tmp_path, host, identity, other
):
    import time

    c = await local(tmp_path, host, identity, other)
    try:
        await c.remember_conversation(identity)
        t = await c.modules["memory"].create_task(
            identity, "喝水", int(time.time() * 1e6) - 1000000, key="task-one"
        )
        assert t["status"] == "active"
        await c.apply(
            {"quiet_start": 0, "quiet_end": 0, "modules": {"proactive": True}},
            c.revision,
        )
        await c.modules["proactive"].tick()
        await c.modules["proactive"].tick()
        assert len(host.sent) == 1 and host.sent[0][1] == "提醒：喝水"
        assert (await c.modules["memory"].tasks(identity))["items"][0][
            "status"
        ] == "active"
        assert (
            await c.modules["memory"].transition_task(
                identity, t["task_id"], "cancel", t["revision"]
            )
        )["status"] == "cancelled"
        assert not (await c.modules["memory"].tasks(identity))["items"]
    finally:
        await c.close()


async def test_fifty_embedded_cycles_leave_no_worker_threads(
    tmp_path, host, identity, other
):
    c = await local(tmp_path, host, identity, other)
    await c.close()
    for _ in range(50):
        await c.start()
        await c.close()
    assert not [
        t
        for t in threading.enumerate()
        if t.name.startswith(("iris-embedded", "iris-plugin-db"))
    ]
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("iris-")]


async def test_cancelled_task_cannot_send_after_preparation(
    tmp_path, host, identity, other
):
    import time
    from iris_memory.modules.proactive import Module

    c = await local(tmp_path, host, identity, other)
    try:
        await c.remember_conversation(identity)
        snapshot = await c.store.get("conversations", identity.key)
        task = await c.modules["memory"].create_task(
            identity, "Cancelled reminder", int(time.time() * 1e6)
        )
        await c.modules["memory"].transition_task(
            identity, task["task_id"], "cancel", task["revision"]
        )
        with pytest.raises(IrisError, match="任务已取消"):
            await Module(c).deliver(
                identity,
                snapshot,
                "must not send",
                "cancelled",
                time.time(),
                expected_task=task,
            )
        assert not host.sent
    finally:
        await c.close()


async def test_focus_and_profile_are_consumed_from_core(
    tmp_path, host, identity, other
):
    c = await local(tmp_path, host, identity, other)
    try:
        m = c.modules["memory"]
        f = await m.create_focus(identity, "关注下周的学习进度")
        assert (await m.focuses(identity))["items"][0]["focus_item_id"] == f[
            "focus_item_id"
        ]
        assert (await m.dismiss_focus(identity, f["focus_item_id"], f["revision"]))[
            "status"
        ] == "dismissed"
        profile = await m.profile(identity)
        assert isinstance(profile, dict)
    finally:
        await c.close()
