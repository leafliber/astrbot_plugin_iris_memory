import pytest

from iris_memory.control import Control
from iris_memory.errors import IrisError
from iris_memory.modules.context import Module, tokens


async def test_persona_drafts_publish_rollback_and_bind(tmp_path, host, identity):
    c = Control(tmp_path, host)
    await c.start()
    p = await c.personas.save(name="Iris", text="Full identity and boundaries.")
    key = p["value"]["id"]
    with pytest.raises(IrisError):
        await c.personas.bind(identity.key, key)
    p = await c.personas.publish(key, p["revision"])
    await c.personas.bind(identity.key, key)
    old = await c.personas.resolve(identity)
    p = await c.personas.save(
        persona_id=key,
        name="Iris",
        text="A new full identity.",
        expected_revision=p["revision"],
    )
    assert (await c.personas.resolve(identity))["text"] == old["text"]
    p = await c.personas.publish(key, p["revision"])
    assert (await c.personas.resolve(identity))["text"] == "A new full identity."
    await c.personas.rollback(key, 1, p["revision"])
    assert (await c.personas.resolve(identity))["text"] == old["text"]
    with pytest.raises(IrisError):
        await c.personas.disable(key, (await c.store.get("personas", key))["revision"])
    await c.close()


async def test_builder_preserves_host_context_and_skips_oversize(
    tmp_path, host, identity
):
    c = Control(tmp_path, host)
    await c.start()
    c.settings.update(context_tokens=500, model_window=4096, output_reserve=512)

    def candidate(i, text, kind="claim"):
        return {
            "text": text,
            "candidate_id": f"cand:{i:016x}",
            "resource_ref": {
                "resource_type": kind,
                "resource_id": str(i),
                "revision": 1,
            },
        }

    envelope = {
        "request_id": "q",
        "persona_revision": 1,
        "candidates": [
            candidate(1, "长" * 500),
            candidate(2, "喜欢茶"),
            candidate(3, "喜欢茶"),
            candidate(4, "过时对话", "observation"),
        ],
    }

    class Memory:
        async def recall(self, *args):
            return envelope

    c.modules["memory"] = Memory()
    history = [{"role": "user", "content": "宿主历史"}]
    builder = Module(c)
    result = await builder.build(
        identity,
        "茶",
        system="宿主规则",
        history=history,
        tools=[{"name": "existing_tool"}],
    )
    assert result["selected"] == ["cand:0000000000000002"]
    assert tokens(result["text"]) <= 500
    assert history == [{"role": "user", "content": "宿主历史"}]
    assert "过时" not in result["text"]
    full = await builder.build(identity, "茶", system="x" * 4096)
    assert not full["text"] and full["envelope"] is None
    del c.modules["memory"]
    await c.close()


async def test_usage_from_old_generation_is_rejected(tmp_path, host):
    c = Control(tmp_path, host)
    await c.start()
    plan = {
        "generation": c.generation,
        "envelope": {"request_id": "stale"},
        "selected": [],
    }
    c.generation += 1
    with pytest.raises(IrisError, match="旧上下文"):
        await Module(c).report(plan, visible=True)
    await c.close()
