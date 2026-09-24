# ruff: noqa: E402, F811, I001
from tests.test_host import astrbot_config, loaded  # noqa: F401

"""Real nine-stage host scheduler and adapter; protocol peer is synthetic, never QQ."""
import asyncio
import copy

import pytest
from aiocqhttp import Event
from astrbot.core.pipeline.context import PipelineContext
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
    AiocqhttpAdapter,
)

from tests.ingress_fixture import HTTPFixture, event
from tests.test_ingress_delivery import configure, until


async def test_real_pipeline_ordinary_group_dedup_and_prefilter_gap(loaded):
    manager, metadata, context = loaded
    app = metadata.star_cls.application
    core = HTTPFixture()
    await core.start()
    try:
        await configure(app, core)
        config = copy.deepcopy(dict(astrbot_config))
        config["provider_settings"]["enable"] = True
        context._config = config
        context.conversation_manager = None

        class EmptyProviders:
            async def get_using_provider_async(self, **kwargs):
                return None

        context.provider_manager = EmptyProviders()
        scheduler = PipelineScheduler(PipelineContext(config, manager, "default"))
        await scheduler.initialize()
        adapter = AiocqhttpAdapter(
            {"id": "onebot-test", "ws_reverse_host": "127.0.0.1", "ws_reverse_port": 0},
            config["platform_settings"],
            asyncio.Queue(maxsize=1),
        )

        # No socket/server is started. Unexpected API/send activity is a test failure.
        class Peer:
            calls = []

            async def call_action(self, name, **kwargs):
                self.calls.append(name)
                raise AssertionError("unexpected OneBot API action")

        peer = Peer()
        messages = []
        for number in (1, 1, 2):
            raw = event(number).message_obj.raw_message
            message = await adapter.convert_message(Event(raw))
            item = AiocqhttpMessageEvent(
                message.message_str, message, adapter.meta(), "group-0", peer
            )
            await scheduler.execute(item)
            messages.append(item)

        async def drained():
            return (await app.delivery.status())["counts"].get("confirmed") == 2

        await until(drained)
        assert len(scheduler.stages) == 9 and peer.calls == []
        assert all(not x.call_llm and not x._has_send_oper for x in messages)
        assert all(x.is_wake and not x.is_at_or_wake_command for x in messages)
        assert [v["input"]["event"]["body"] for v in core.originals.values()] == [
            "  原文\n\t中文  "
        ] * 2
        assert (await app.delivery.status())["counts"]["duplicate"] == 1
        # Built-in self-message filtering runs before the public observer, preserving U01/U02.
        raw = event(3).message_obj.raw_message
        raw.update(user_id=100, post_type="message_sent")
        raw["sender"]["user_id"] = 100
        with pytest.raises(UnboundLocalError):
            await adapter.convert_message(Event(raw))
        raw["post_type"] = "message"
        message = await adapter.convert_message(Event(raw))
        item = AiocqhttpMessageEvent(
            message.message_str, message, adapter.meta(), "group-0", peer
        )
        await scheduler.execute(item)
        assert len(core.accepts) == 2
        success, error = await manager.load(specified_dir_name="builtin_commands")
        assert success, error
        from astrbot.core.star.star import star_registry

        builtin = next(
            x for x in star_registry if x.root_dir_name == "builtin_commands"
        )
        try:
            c = await app.store.settings()
            core.tokens["secret-group-0"] = ["entry-0", "entry-private"]
            await app.group_save(
                c["revision"],
                "group-0",
                "host",
                core.tokens["secret-group-0"],
                "secret-group-0",
            )
            private = {
                "platform_instance": "onebot-test",
                "bot_self": "100",
                "kind": "private",
                "conversation_id": "200",
                "entry_id": "entry-private",
                "group_id": "group-0",
                "enabled": False,
            }
            for enabled in (False, True):
                c = await app.store.settings()
                await app.source_save(c["revision"], {**private, "enabled": enabled})

            class CommandPeer:
                def __init__(self):
                    self.sent = []

                async def send(self, **kwargs):
                    self.sent.append(kwargs["message"])
                    return {"message_id": 901}

                async def send_private_msg(self, **kwargs):
                    self.sent.append(kwargs["message"])
                    return {"message_id": 902}

            command_peer = CommandPeer()
            for private_chat in (False, True):
                raw = event(4 + int(private_chat), text="/sid").message_obj.raw_message
                if private_chat:
                    raw["message_type"] = "private"
                    raw.pop("group_id")
                for capture in (True, False):
                    config["plugin_set"] = ["*"] if capture else [builtin.name]
                    message = await adapter.convert_message(Event(raw))
                    item = AiocqhttpMessageEvent(
                        message.message_str,
                        message,
                        adapter.meta(),
                        "200" if private_chat else "group-0",
                        command_peer,
                    )
                    await scheduler.execute(item)
                    assert item._has_send_oper and not item.call_llm
                assert command_peer.sent[-1] == command_peer.sent[-2]

            async def commands_drained():
                return (await app.delivery.status())["counts"].get("confirmed") == 4

            await until(commands_drained)
            assert len(command_peer.sent) == 4
            assert all(
                v["input"]["event"]["event_kind"] == "MESSAGE"
                for v in core.originals.values()
            )
            assert (await app.delivery.status())["counts"]["after_send_callback"] == 2
        finally:
            await manager._terminate_plugin(builtin)
            await manager._unbind_plugin(builtin.name, builtin.module_path)
    finally:
        await core.close()
