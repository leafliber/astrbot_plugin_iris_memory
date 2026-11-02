"""main.py 贴纸接线：门控前改写、标记剥离、只在文字回执后发送、频道/群边界。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_iris_memory.iris_memory.stickers.library import StickerLibrary
from astrbot_plugin_iris_memory.main import IrisMemoryPlugin

from .test_integration_main import CHANNEL, GROUP, OWNER, SCOPE, FakeEvent, FakeResult, plugin  # noqa: F401


class FakeLibrary:
    def __init__(self, labels=("开心",), cooldown_ok=True):
        self.labels = list(labels)
        self.sent = []
        self.cooldown_ok = cooldown_ok

    def allowed_for(self, scope, owner_ids, groups_enabled, *, blocked_scopes=()):
        if scope != f"private:{OWNER}":
            return []
        return [{"label": l, "note": "高兴的时候"} for l in self.labels]

    def should_send(self, scope, now=None):
        return self.cooldown_ok

    async def send(self, client, peer, entry, *, scope_key=None, reply_to=None):
        self.sent.append((peer, entry["label"], scope_key))
        return [5555]


def _resp(text):
    return SimpleNamespace(completion_text=text, usage=None)


@pytest.fixture
def sticker_plugin(plugin):  # noqa: F811
    lib = FakeLibrary()
    plugin._sticker_library = lib
    plugin._sticker_understanding = None
    client = object()
    inst = plugin.context.get_platform_inst("userbot")
    inst.get_client = lambda: client
    return plugin, lib


@pytest.mark.asyncio
async def test_marker_stripped_and_sticker_sent_only_after_text_receipt(sticker_plugin):
    p, lib = sticker_plugin
    ev = FakeEvent(text="今天好开心", msg_id=400)
    resp = _resp("真好呀！[表情:开心]")
    await p.on_llm_response(ev, resp)
    assert "表情" not in resp.completion_text
    assert ev.extras["iris_pending_assistant"]["content"] == "真好呀！"
    assert lib.sent == []

    ev.result = FakeResult("真好呀！")
    ev.extras["telethon_sent_message_ids"] = [9100]
    await p.on_message_sent(ev)
    assert lib.sent == [(int(OWNER), "开心", SCOPE)]
    rows = p._continuity.store.recent_outbound(SCOPE)
    assert any(r["text"] == "[贴纸：开心]" and r["tg_msg_ids"] == [5555] for r in rows)


@pytest.mark.asyncio
async def test_no_text_receipt_means_no_sticker(sticker_plugin):
    p, lib = sticker_plugin
    ev = FakeEvent(text="在吗", msg_id=401)
    await p.on_llm_response(ev, _resp("在的[表情:开心]"))
    ev.result = FakeResult("在的")
    await p.on_message_sent(ev)  # 无回执
    assert lib.sent == []


@pytest.mark.asyncio
async def test_unknown_label_dropped(sticker_plugin):
    p, lib = sticker_plugin
    ev = FakeEvent(text="hi", msg_id=402)
    resp = _resp("嗨[表情:不存在的]")
    await p.on_llm_response(ev, resp)
    assert resp.completion_text == "嗨"
    assert ev.extras.get("iris_sticker_pending") is None


@pytest.mark.asyncio
async def test_group_and_channel_never_get_stickers(sticker_plugin):
    p, lib = sticker_plugin
    for gid in (GROUP, CHANNEL):
        ev = FakeEvent(text="hi", sender="123", group_id=gid, msg_id=403)
        resp = _resp("嗨[表情:开心]")
        await p.on_llm_response(ev, resp)
        assert "表情" not in resp.completion_text
        assert ev.extras.get("iris_sticker_pending") is None
    assert lib.sent == []


@pytest.mark.asyncio
async def test_decorating_backstop_strips_marker(sticker_plugin):
    p, _ = sticker_plugin
    p._error_processor = None
    p._markdown_stripper = None
    from astrbot.api.message_components import Plain

    ev = FakeEvent(text="hi", msg_id=404)
    ev.result = SimpleNamespace(chain=[Plain("好的【表情：开心】")])
    await p.on_decorating_result(ev)
    assert ev.result.chain[0].text == "好的"


def test_sticker_description_still_counts_as_media_only():
    ev = FakeEvent(text="[贴纸 😭：一只猫趴在桌上哭]", msg_id=405)
    ev.extras["iris_sticker"] = {"doc_id": 1}
    assert IrisMemoryPlugin._is_media_only_message(ev) is True
    ev2 = FakeEvent(text="[贴纸 😭：一只猫趴在桌上哭] 今天好累", msg_id=406)
    ev2.extras["iris_sticker"] = {"doc_id": 1}
    assert IrisMemoryPlugin._is_media_only_message(ev2) is False


@pytest.mark.asyncio
async def test_sticker_prompt_has_no_digits(sticker_plugin):
    p, _ = sticker_plugin
    ev = FakeEvent(text="hi", msg_id=407)
    req = SimpleNamespace(system_prompt="base")
    await p._inject_sticker_prompt(ev, req)
    assert "开心" in req.system_prompt
    added = req.system_prompt[len("base"):]
    assert not any(ch.isdigit() for ch in added)


def test_real_library_prompt_block_no_digits(tmp_path):
    lib = StickerLibrary(tmp_path / "s.db")
    assert lib.prompt_block([{"label": "开心", "note": "高兴"}]) and not any(
        c.isdigit() for c in lib.prompt_block([{"label": "开心", "note": "高兴"}])
    )
    lib.close()


def test_photo_description_only_counts_as_media_only():
    ev = FakeEvent(text="[图片：一盘番茄炒蛋]", msg_id=410)
    ev.extras["iris_photo"] = {"photo_id": 1}
    assert IrisMemoryPlugin._is_media_only_message(ev) is True
    ev2 = FakeEvent(text="[图片：一盘番茄炒蛋] 看我做的", msg_id=411)
    ev2.extras["iris_photo"] = {"photo_id": 1}
    assert IrisMemoryPlugin._is_media_only_message(ev2) is False


CODE_REPLY = (
    "可以这样统计表情数量：\n\n```python\ndef count(msgs):\n    n = 0\n    for m in msgs:\n"
    "        if  \"表情\" in m:\n            n += 1\n    return n\n```\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("send_enable", [False, True])
async def test_reply_mentioning_biaoqing_without_marker_is_byte_identical(sticker_plugin, send_enable):
    """F2：只提到“表情”、没有标记的回复（代码块）原样保留，不论开关。"""
    p, lib = sticker_plugin
    p.config.values["continuity.sticker_send_enable"] = send_enable
    ev = FakeEvent(text="怎么统计表情", msg_id=420)
    resp = _resp(CODE_REPLY)
    await p.on_llm_response(ev, resp)
    assert resp.completion_text == CODE_REPLY
    assert ev.extras.get("iris_sticker_pending") is None


@pytest.mark.asyncio
async def test_marker_removal_keeps_code_indentation(sticker_plugin):
    p, _ = sticker_plugin
    ev = FakeEvent(text="怎么统计表情", msg_id=421)
    resp = _resp(CODE_REPLY + "[表情:开心]")
    await p.on_llm_response(ev, resp)
    assert resp.completion_text == CODE_REPLY.rstrip()
    assert ev.extras.get("iris_sticker_pending") == "开心"


@pytest.mark.asyncio
async def test_decorating_backstop_leaves_unmarked_plain_untouched(sticker_plugin):
    p, _ = sticker_plugin
    p._error_processor = None
    p._markdown_stripper = None
    from astrbot.api.message_components import Plain

    ev = FakeEvent(text="hi", msg_id=422)
    ev.result = SimpleNamespace(chain=[Plain(CODE_REPLY)])
    await p.on_decorating_result(ev)
    assert ev.result.chain[0].text == CODE_REPLY
