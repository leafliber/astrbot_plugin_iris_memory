from dataclasses import replace
from types import SimpleNamespace

import pytest

from iris_memory.config import Config
from iris_memory.extras import decorate_result, friendly_error, pure_mention, strip_markdown
from iris_memory.mapping import input_message, map_entry
from iris_memory.vendor.iris_client.compat import entry_path
from tests.conftest import HostEvent, Result
from tests.test_bridge import component


@pytest.mark.parametrize(
    "bad",
    [
        "astrbot:a?b",
        "astrbot:a#b",
        "astrbot:a\x00b",
        "astrbot:a\x7fb",
        "a\x85b",
        "a/b",
        "a\\b",
        ".",
        "..",
        " ",
    ],
)
def test_invalid_ids_rejected_by_client_mapping_and_config(bad):
    event = HostEvent()
    with pytest.raises(ValueError):
        entry_path(bad, "prepare")
    with pytest.raises(ValueError):
        map_entry(event, Config(entry_overrides={event.unified_msg_origin: bad}))
    with pytest.raises(ValueError):
        Config.from_mapping({"entry_overrides": {event.unified_msg_origin: bad}})


def test_platform_components_are_escaped_before_mapping():
    event = HostEvent(group="a?#b\n")
    entry = map_entry(event, Config())
    assert "?" not in entry.id and "#" not in entry.id and "\n" not in entry.id
    assert "%3F%23" in entry.id


@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
def test_qq_identity_spaces_and_unknown_names(platform):
    group, private, guild = (
        HostEvent(platform=platform, group=g, user="same-openid") for g in ("g", "", "channel")
    )
    for event in (group, private, guild):
        event.get_sender_name = lambda: ""
    guild.message_obj.raw_message = SimpleNamespace(
        raw_data={"channel_id": "channel", "guild_id": "guild"}
    )
    entries = [map_entry(e, Config()) for e in (group, private, guild)]
    assert len({e.platform for e in entries}) == 3
    assert input_message(group, entries[0])["sender"].startswith("成员_")
    assert input_message(private, entries[1])["sender"].startswith("用户_")
    assert input_message(group, entries[0])["account_id"] == "same-openid"


@pytest.mark.parametrize(
    "platform", ["aiocqhttp", "qq_official", "qq_official_webhook", "telegram"]
)
def test_quote_and_images_use_common_astrbot_components(platform):
    event = HostEvent(platform=platform)
    event.messages = [
        component("plain", text="原话"),
        component("reply", sender_id="u2", sender_nickname="小赵", text="引用"),
        component("image", url="https://example.invalid/private-image"),
    ]
    result = input_message(event, map_entry(event, Config()))
    assert result["quote_author_account_id"] == "u2" and result["quote_content"] == "引用"
    assert "private-image" not in result["content"]


def test_plain_markdown_keeps_code_and_links():
    assert (
        strip_markdown(
            "# 标题\n**重点**\n* 列表\n`a_b * c`\n```py\na_b ** c\n```\n[网站](https://example.invalid)"
        )
        == "标题\n重点\n• 列表\na_b * c\na_b ** c\n\n网站（https://example.invalid）"
    )
    assert strip_markdown("user_name and 2 * 3") == "user_name and 2 * 3"


def test_extras_switches_t2i_and_framework_error():
    event = HostEvent()
    event.result = Result("**重点**", llm=True)
    decorate_result(event, Config(), {"t2i": False})
    assert event.result.chain[0].text == "重点"
    for settings, host, override in [
        (replace(Config(), strip_markdown=False), {}, None),
        (Config(), {}, True),
        (Config(), {"t2i": True, "t2i_word_threshold": 50}, None),
    ]:
        event.result = Result("**" + "长" * 100 + "**", llm=True)
        event.result.use_t2i_ = override
        decorate_result(event, settings, host)
        assert event.result.chain[0].text.startswith("**")
    error = "AstrBot 请求失败\n错误类型: NetworkTimeout\n错误信息: iris_test_token"
    event.result = Result(error)
    decorate_result(event, Config(), {})
    assert "iris_test_token" not in event.result.chain[0].text
    assert "网络" in event.result.chain[0].text
    event.result = Result(error)
    decorate_result(event, replace(Config(), error_friendly=False), {})
    assert event.result.chain[0].text == error
    assert friendly_error("HTTP API 返回 400 时，应该检查输入。") is None


def test_pure_mention_requires_only_self():
    event = HostEvent()
    event.messages = [component("at", qq="bot-self"), component("plain", text=" ")]
    assert pure_mention(event)
    event.messages.append(component("image"))
    assert not pure_mention(event)
    event.messages = [component("at", qq="someone-else")]
    assert not pure_mention(event)
