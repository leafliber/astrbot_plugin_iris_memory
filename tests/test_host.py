"""Public AstrBot adapter contract fixtures; no AstrBot global process started."""

import copy
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from iris_memory.control import Control
from iris_memory.errors import IrisError
from iris_memory.host import AstrBotHost


@pytest.fixture
def astrbot_context(monkeypatch):
    config = {}

    async def get(*args, **kwargs):
        return copy.deepcopy(config.get(args[1], {}))

    async def put(scope, umo, key, value):
        config[umo] = copy.deepcopy(value)

    api = ModuleType("astrbot.api")
    api.sp = SimpleNamespace(get_async=get, put_async=put)
    monkeypatch.setitem(sys.modules, "astrbot", ModuleType("astrbot"))
    monkeypatch.setitem(sys.modules, "astrbot.api", api)

    class Manager:
        def __init__(self):
            self.items = {
                "original": {
                    "name": "original",
                    "prompt": "Original full persona",
                    "tools": ["safe_tool"],
                    "skills": [],
                    "begin_dialogs": [],
                    "custom_error_message": "error",
                }
            }

        async def resolve_selected_persona(self, *, umo, **kwargs):
            selected = config.get(umo, {}).get("persona_id", "original")
            return selected, self.items.get(selected), selected, False

        def get_persona_v3_by_id(self, key):
            return self.items.get(key)

        async def create_persona(self, persona_id, system_prompt, **kwargs):
            self.items[persona_id] = {
                "name": persona_id,
                "prompt": system_prompt,
                **kwargs,
            }

        async def update_persona(self, persona_id, system_prompt, **kwargs):
            self.items[persona_id] = {
                "name": persona_id,
                "prompt": system_prompt,
                **kwargs,
            }

        async def delete_persona(self, key):
            del self.items[key]

    async def cid(umo):
        return None

    context = SimpleNamespace(
        persona_manager=Manager(),
        conversation_manager=SimpleNamespace(get_curr_conversation_id=cid),
    )
    return context, config


async def test_full_persona_adoption_restores_forced_selector_and_permissions(
    tmp_path, identity, astrbot_context
):
    context, config = astrbot_context
    config[identity.umo] = {"persona_id": "original", "other_setting": 42}
    host = AstrBotHost(context, logging.getLogger("test-host"))
    c = Control(tmp_path, host)
    await c.start()
    try:
        await host.check_persona_support()
        p = await c.personas.save(name="Iris", text="Full plugin persona")
        await c.personas.publish(p["value"]["id"], p["revision"])
        persona = await c.personas.published(p["value"]["id"])
        await c.personas.bind(identity.key, persona["id"])
        await host.adopt_persona(c, identity, persona)
        selected = config[identity.umo]["persona_id"]
        assert selected.startswith("iris_v4_")
        carrier = context.persona_manager.items[selected]
        assert carrier["prompt"] == persona["text"]
        assert carrier["tools"] == ["safe_tool"] and carrier["skills"] == []
        await host.verify_persona(
            c, identity, SimpleNamespace(system_prompt="Host rules\n" + persona["text"])
        )
        await host.release_personas(c)
        assert config[identity.umo] == {"persona_id": "original", "other_setting": 42}
        assert selected not in context.persona_manager.items
    finally:
        await c.close()


async def test_external_host_edit_is_not_overwritten(
    tmp_path, identity, astrbot_context
):
    context, config = astrbot_context
    host = AstrBotHost(context, logging.getLogger("test-host"))
    c = Control(tmp_path, host)
    await c.start()
    persona = {"id": "iris", "revision": 1, "text": "Managed text"}
    try:
        await host.adopt_persona(c, identity, persona)
        carrier = config[identity.umo]["persona_id"]
        context.persona_manager.items[carrier]["prompt"] = "User edited externally"
        with pytest.raises(IrisError, match="外部"):
            await host.adopt_persona(c, identity, persona)
        config[identity.umo]["persona_id"] = "another-selected-persona"
        await host.release_personas(c)
        assert config[identity.umo]["persona_id"] == "another-selected-persona"
        assert (
            context.persona_manager.items[carrier]["prompt"] == "User edited externally"
        )
    finally:
        await c.close()


async def test_embedding_normalization_budget_and_borrowed_provider(tmp_path):
    class Provider:
        closed = False

        def get_model(self):
            return "test"

        async def get_embeddings(self, texts):
            return [[3.0, 4.0] for _ in texts]

    provider = Provider()
    host = AstrBotHost(
        SimpleNamespace(get_provider_by_id=lambda key: provider),
        logging.getLogger("test"),
    )
    c = Control(tmp_path, host)
    await c.start()
    c.settings.update(embedding_provider="embedding", embedding_dimension=2)
    try:
        adapter, cognitive = await host.providers(c)
        assert await adapter.embed(["茶"]) == [[0.6, 0.8]]
        assert cognitive is None and not provider.closed
        c.settings["cognitive_provider"] = "incomplete-bridge"
        with pytest.raises(IrisError, match="候选 Scope"):
            await host.providers(c)
    finally:
        await c.close()
