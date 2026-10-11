import asyncio
from types import SimpleNamespace

import pytest

from iris_memory.config import Config
from iris_memory.forwarded import attach_forwarded
from iris_memory.mapping import input_message, map_entry
from tests.conftest import HostEvent
from tests.test_bridge import component


def node(name, *parts):
    return component("node", name=name, uin="quoted-user", content=list(parts))


async def expand(event, **options):
    config = Config.from_mapping(options)
    message = input_message(event, map_entry(event, config))
    logs = []
    await attach_forwarded(event, message, config, logs.append)
    return message, logs


async def test_inline_nodes_keep_authors_order_images_and_nested_data():
    event = HostEvent()
    event.messages = [
        component("plain", text="分享给大家"),
        component(
            "nodes",
            nodes=[
                node(
                    "甲",
                    component("plain", text="前面"),
                    component("image", url="https://example.test/private.png"),
                    component("image", file="base64://DO_NOT_COPY"),
                    component("nodes", nodes=[node("乙", component("plain", text="内层原话"))]),
                    component("plain", text="后面"),
                ),
                node("丙", component("plain", text="最后")),
            ],
        ),
    ]
    message, logs = await expand(event)
    assert message["content"] == "分享给大家"
    assert message["quote_author"] == "合并转发"
    assert not message.get("quote_author_account_id")
    assert message["quote_content"] == "甲：前面[图片×2]\n乙：内层原话\n甲：后面\n丙：最后"
    assert "media_ids" not in message and logs == ["forward_expanded"]


async def test_onebot_raw_forward_api_routing_title_and_reply():
    calls = []

    async def call_action(action, **kwargs):
        calls.append((action, kwargs))
        return {
            "data": {
                "title": "转发标题",
                "messages": [
                    {
                        "sender": {"card": "群名片", "nickname": "昵称", "user_id": "u2"},
                        "content": [{"type": "text", "data": {"text": "原话"}}],
                    },
                    {
                        "sender": {"nickname": "乙"},
                        "content": "正文[CQ:image,file=DO_NOT_COPY][CQ:record,file=DO_NOT_COPY]",
                    },
                ],
            }
        }

    event = HostEvent(platform="aiocqhttp")
    event.message_obj.self_id = "bot-test"
    event.message_obj.raw_message = {"message": [{"type": "forward", "data": {"id": "f1"}}]}
    event.messages = [
        component("reply", sender_nickname="原引用者", sender_id="u9", message_str="原引用内容")
    ]
    event.bot = SimpleNamespace(call_action=call_action)
    message, _ = await expand(event)
    assert calls == [("get_forward_msg", {"message_id": "f1", "self_id": "bot-test"})]
    assert message["quote_author"] == "转发标题" and not message.get("quote_author_account_id")
    assert "原引用者：原引用内容" in message["quote_content"]
    assert "群名片：原话" in message["quote_content"]
    assert "乙：正文[图片×1][音频]" in message["quote_content"]
    assert "DO_NOT_COPY" not in message["quote_content"]
    assert message["content"] == "[合并转发]"


async def test_duplicate_reference_cached_and_cycle_bounded():
    calls = []

    async def call_action(action, **kwargs):
        calls.append(kwargs["message_id"])
        return {
            "messages": [
                {
                    "sender": {"nickname": "甲"},
                    "content": [
                        {"type": "text", "data": {"text": "只说一次"}},
                        {"type": "forward", "data": {"id": "f1"}},
                    ],
                }
            ]
        }

    event = HostEvent(platform="aiocqhttp")
    event.bot = SimpleNamespace(call_action=call_action)
    event.messages = [component("forward", id="f1"), component("forward", id="f1")]
    message, _ = await expand(event)
    assert calls == ["f1"]
    assert message["quote_content"].count("甲：只说一次") == 2
    assert "已截断" in message["quote_content"]


@pytest.mark.parametrize("options", [{"forward_max_depth": 1}, {"forward_max_nodes": 1}])
async def test_depth_and_node_limits_have_explicit_marker(options):
    event = HostEvent()
    event.messages = [
        node("甲", component("plain", text="可见"), node("乙", component("plain", text="深层"))),
        node("丙", component("plain", text="尾部")),
    ]
    message, _ = await expand(event, **options)
    assert "甲：可见" in message["quote_content"] and "已截断" in message["quote_content"]
    assert "深层" not in message["quote_content"]


async def test_utf8_limit_and_titles_are_bounded():
    event = HostEvent()
    event.messages = [
        component(
            "nodes",
            title="标题" * 300,
            nodes=[node("甲" * 500, component("plain", text="🙂中" * 16000))],
        )
    ]
    message, _ = await expand(event)
    assert len(message["quote_author"]) <= 200
    assert len(message["quote_content"].encode()) <= 32768
    assert "已截断" in message["quote_content"] and "�" not in message["quote_content"]


@pytest.mark.parametrize("failure", ["exception", "timeout", "malformed"])
async def test_fetch_failure_retains_message_and_never_logs_exception(failure):
    async def call_action(*args, **kwargs):
        if failure == "timeout":
            await asyncio.sleep(1)
        if failure == "exception":
            raise RuntimeError("iris_test_token")
        return {"messages": "invalid"}

    event = HostEvent(platform="aiocqhttp")
    event.messages += [component("forward", id="f1")]
    event.bot = SimpleNamespace(call_action=call_action)
    message, logs = await expand(event, forward_timeout=0.01)
    assert message["content"] == "你好"
    assert "合并转发" in message["quote_content"]
    assert "iris_test_token" not in str(logs) + message["quote_content"]
    assert any(code == "forward_degraded" for code in logs)


async def test_unsupported_platform_does_not_call_onebot_api():
    async def call_action(*args, **kwargs):
        pytest.fail("No OneBot API on an unrelated platform")

    event = HostEvent(platform="qq_official")
    event.bot = SimpleNamespace(call_action=call_action)
    event.messages = [component("forward", id="f1")]
    message, _ = await expand(event)
    assert "未展开" in message["quote_content"]


async def test_ordinary_quote_unchanged():
    event = HostEvent()
    event.messages += [component("reply", sender_nickname="甲", sender_id="u2", message_str="引用")]
    message, logs = await expand(event)
    assert message["quote_content"] == "引用" and message["quote_author_account_id"] == "u2"
    assert logs == []


async def test_http_forward_expansion_does_not_upload_nested_images(plugin_factory, fake_iris):
    plugin = await plugin_factory()
    event = HostEvent()
    event.messages = [
        node(
            "甲",
            component("plain", text="看看"),
            component("image", url="https://example.test/never-download"),
        )
    ]
    await plugin.receive(event)
    await plugin.bridge.queue.join()
    message = fake_iris.messages["astrbot:bot1:group:g1"][0]
    assert message["sender"] == "小林" and message["account_id"] == "u1"
    assert message["quote_content"] == "甲：看看[图片×1]"
    assert not fake_iris.media


async def test_disabled_forward_expansion(plugin_factory, fake_iris):
    plugin = await plugin_factory(forward_enabled=False)
    event = HostEvent()
    event.messages += [node("甲", component("plain", text="不展开"))]
    await plugin.receive(event)
    await plugin.bridge.queue.join()
    message = fake_iris.messages["astrbot:bot1:group:g1"][0]
    assert "quote_content" not in message


@pytest.mark.parametrize(
    "values",
    [
        {"forward_enabled": "true"},
        {"forward_max_depth": 0},
        {"forward_max_nodes": 0},
        {"forward_timeout": 0},
    ],
)
def test_invalid_forward_config(values):
    with pytest.raises(ValueError):
        Config.from_mapping(values)
