"""StickerLibrary 单元测试（tmp_path DB + 可控时钟 + 假 Telethon 对象，无网络）"""

import re
import sqlite3
from types import SimpleNamespace

import pytest
from telethon import errors as tg_errors
from telethon.tl.types import (
    DocumentAttributeSticker,
    InputDocument,
    InputStickerSetEmpty,
)

from astrbot_plugin_iris_memory.iris_memory.stickers import library as lib_mod
from astrbot_plugin_iris_memory.iris_memory.stickers.library import (
    StickerError,
    StickerLibrary,
    StickerSendError,
    parse_marker,
    prompt_block,
)

OWNER = "100000001"
OWNER_SCOPE = f"private:{OWNER}"
GROUP = "-1001000000003"
CHANNEL = "-1001000000001"


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def fake_doc(doc_id=111, access_hash=222, ref=b"ref-old", emoji="😺"):
    return SimpleNamespace(
        id=doc_id,
        access_hash=access_hash,
        file_reference=ref,
        attributes=[DocumentAttributeSticker(alt=emoji, stickerset=InputStickerSetEmpty())],
    )


def fake_msg(doc=None, chat_id=int(OWNER), msg_id=50):
    return SimpleNamespace(id=msg_id, chat_id=chat_id, sticker=doc)


class FakeClient:
    def __init__(self, fail_first=None, refetch=None, next_id=900):
        self.fail_first = fail_first
        self.refetch = refetch
        self.next_id = next_id
        self.sent = []
        self.get_calls = []

    async def send_file(self, peer, file=None, **kw):
        self.sent.append((peer, file, kw))
        if self.fail_first is not None:
            exc, self.fail_first = self.fail_first, None
            raise exc
        self.next_id += 1
        return SimpleNamespace(id=self.next_id)

    async def get_messages(self, chat, ids=None):
        self.get_calls.append((chat, ids))
        return self.refetch


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def lib(tmp_path, clock):
    lb = StickerLibrary(tmp_path / "stickers" / lib_mod.DB_FILENAME, clock)
    yield lb
    lb.close()


# --------------------------------------------------------------------------- schema


def test_schema_wal_and_version(lib, tmp_path):
    assert lib.schema_version() == lib_mod.SCHEMA_VERSION
    mode = lib._conn().execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    cols = {r[1] for r in lib._conn().execute("PRAGMA table_info(library)")}
    assert {"label", "note", "doc_id", "access_hash", "file_reference", "source_chat_id",
            "source_msg_id", "emoji", "scope", "enabled", "added_by", "created_at",
            "updated_at", "use_count", "last_used_at"} <= cols
    # 重开不重复迁移
    lib2 = StickerLibrary(lib.db_path, Clock())
    assert lib2.schema_version() == lib_mod.SCHEMA_VERSION
    lib2.close()


# --------------------------------------------------------------------------- add/list/del/scope


def test_add_list_del_scope(lib, clock):
    e = lib.add_from_message("摸摸头", "安慰的时候", fake_msg(fake_doc()), OWNER)
    assert e["label"] == "摸摸头" and e["note"] == "安慰的时候"
    assert e["doc_id"] == 111 and e["access_hash"] == 222 and e["file_reference"] == b"ref-old"
    assert e["source_chat_id"] == int(OWNER) and e["source_msg_id"] == 50
    assert e["emoji"] == "😺" and e["scope"] == "private" and e["enabled"] is True
    assert e["added_by"] == OWNER and e["use_count"] == 0

    clock.advance(1)
    lib.add_from_message("Hi", "", fake_msg(fake_doc(doc_id=5), msg_id=51), OWNER)
    labels = [x["label"] for x in lib.list()]
    assert labels == ["摸摸头", "hi"]

    # 同标签覆盖；scope 保留
    lib.set_scope("摸摸头", "all")
    clock.advance(10)
    e2 = lib.add_from_message("摸摸头", "新说明", fake_msg(fake_doc(doc_id=999), msg_id=60), OWNER)
    assert e2["doc_id"] == 999 and e2["note"] == "新说明" and e2["scope"] == "all"
    assert len(lib.list()) == 2

    with pytest.raises(StickerError):
        lib.set_scope("摸摸头", "everyone")
    with pytest.raises(StickerError):
        lib.set_scope("不存在", "all")

    assert lib.remove("HI") is True
    assert lib.remove("hi") is False
    assert [x["label"] for x in lib.list()] == ["摸摸头"]


def test_add_requires_sticker(lib):
    with pytest.raises(StickerError):
        lib.add_from_message("abc", "", fake_msg(None), OWNER)
    with pytest.raises(StickerError):
        lib.add_from_message("abc", "", None, OWNER)


@pytest.mark.parametrize("bad", ["", "   ", "a" * 13, "坏_标签", "hi!", "[表情]", "a:b"])
def test_label_validation_rejects(lib, bad):
    with pytest.raises(StickerError):
        lib.add_from_message(bad, "", fake_msg(fake_doc()), OWNER)


def test_label_normalization_and_note_rules(lib):
    e = lib.add_from_message(" ＡＢｃ 12 ", "  一个  说明 ", fake_msg(fake_doc()), OWNER)
    assert e["label"] == "abc12" and e["note"] == "一个 说明"
    e = lib.add_from_message("开心", "【括号】会被去掉", fake_msg(fake_doc()), OWNER)
    assert e["note"] == "括号会被去掉"
    with pytest.raises(StickerError):
        lib.add_from_message("开心", "长" * 41, fake_msg(fake_doc()), OWNER)


# --------------------------------------------------------------------------- privacy


def test_allowed_for_privacy(lib):
    lib.add_from_message("私密", "", fake_msg(fake_doc(1)), OWNER)
    lib.add_from_message("公开", "", fake_msg(fake_doc(2)), OWNER)
    lib.set_scope("公开", "all")
    lib.add_from_message("停用", "", fake_msg(fake_doc(3)), OWNER)
    lib.set_scope("停用", "all")
    lib.set_enabled("停用", False)

    owners = [int(OWNER)]
    labels = lambda rows: sorted(r["label"] for r in rows)  # noqa: E731

    assert labels(lib.allowed_for(OWNER_SCOPE, owners, False)) == ["公开", "私密"]
    assert labels(lib.allowed_for(OWNER_SCOPE, owners, True)) == ["公开", "私密"]
    # 群：从不出现 private；未开启群表情则什么都没有
    assert lib.allowed_for(GROUP, owners, False) == []
    assert labels(lib.allowed_for(GROUP, owners, True)) == ["公开"]
    assert labels(lib.allowed_for(GROUP + "#7", owners, True)) == ["公开"]
    # 非主人私聊不看到 private
    assert labels(lib.allowed_for("private:1234", owners, True)) == ["公开"]
    assert lib.allowed_for("private:1234", owners, False) == []
    # 频道 / 静默会话：永远为空（包括被误当成主人私聊的 silent）
    assert lib.allowed_for(CHANNEL, owners, True) == []
    assert lib.allowed_for(CHANNEL + "#3", owners, True) == []
    assert lib.allowed_for(GROUP, owners, True, blocked_scopes=[GROUP]) == []
    assert lib.allowed_for(OWNER_SCOPE, owners, True, blocked_scopes=[OWNER_SCOPE]) == []
    assert lib.allowed_for("", owners, True) == []


# --------------------------------------------------------------------------- prompt


def test_prompt_block_no_numbers(lib, clock):
    lib.add_from_message("摸摸头", "安慰的时候", fake_msg(fake_doc(1)), OWNER)
    lib.add_from_message("得意", "", fake_msg(fake_doc(2)), OWNER)
    lib.record_send(OWNER_SCOPE, "摸摸头", [77])
    text = prompt_block(lib.allowed_for(OWNER_SCOPE, [OWNER], False))
    assert "摸摸头：安慰的时候" in text and "- 得意" in text
    assert "[表情:标签]" in text
    assert "不要自己编" in text and "不能代替回答" in text and "最末尾" in text
    assert not re.search(r"\d", text)
    assert "use_count" not in text
    assert prompt_block([]) == ""
    assert lib.prompt_block is prompt_block or lib.prompt_block([]) == ""


# --------------------------------------------------------------------------- parse_marker


@pytest.mark.parametrize(
    "text,expected_text,expected_label",
    [
        ("好呀～ [表情:摸摸头]", "好呀～", "摸摸头"),
        ("好呀【表情：摸摸头】", "好呀", "摸摸头"),
        ("好呀［表情：摸摸头］", "好呀", "摸摸头"),
        ("好呀 [ 表情 : 摸摸头 ]", "好呀", "摸摸头"),
        ("好呀 [表情包:摸摸头]", "好呀", "摸摸头"),
        ("好呀 [表情:乱编的] 嗯", "好呀 嗯", None),
        ("[表情:乱编的] 好呀 [表情:得意] [表情:摸摸头]", "好呀", "得意"),
        ("好呀 [表情]", "好呀", None),
        ("好呀 [表情:摸摸头", "好呀", "摸摸头"),
        ("好呀\n\n[表情:HI]", "好呀", "hi"),
        ("没有标记的回复", "没有标记的回复", None),
        ("", "", None),
    ],
)
def test_parse_marker_variants(text, expected_text, expected_label):
    clean, label = parse_marker(text, ["摸摸头", "得意", "hi"])
    assert clean == expected_text
    assert label == expected_label


def test_parse_marker_disallowed_label_is_stripped():
    # 群里只允许 all，私密标签不得被返回
    clean, label = parse_marker("哈哈[表情:私密]", ["公开"])
    assert clean == "哈哈" and label is None


# --------------------------------------------------------------------------- send


@pytest.mark.asyncio
async def test_send_builds_input_document_and_records(lib, clock):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc()), OWNER)
    client = FakeClient()
    ids = await lib.send(client, int(OWNER), e, scope_key=OWNER_SCOPE)
    assert ids == [901]
    peer, file, _ = client.sent[0]
    assert peer == int(OWNER)
    assert isinstance(file, InputDocument)
    assert (file.id, file.access_hash, file.file_reference) == (111, 222, b"ref-old")
    got = lib.get("摸摸头")
    assert got["use_count"] == 1 and got["last_used_at"] == clock.t
    rows = lib._conn().execute("SELECT scope_key,label,tg_msg_id,status FROM sends").fetchall()
    assert [tuple(r) for r in rows] == [(OWNER_SCOPE, "摸摸头", 901, "sent")]


@pytest.mark.asyncio
async def test_send_refreshes_expired_file_reference_once(lib):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc(), msg_id=50), OWNER)
    fresh = fake_msg(fake_doc(doc_id=111, access_hash=333, ref=b"ref-new"), msg_id=50)
    client = FakeClient(
        fail_first=tg_errors.FileReferenceExpiredError(request=None), refetch=fresh
    )
    ids = await lib.send(client, int(OWNER), e)
    assert ids == [901]
    assert client.get_calls == [(int(OWNER), 50)]
    assert len(client.sent) == 2
    retry_file = client.sent[1][1]
    assert retry_file.file_reference == b"ref-new" and retry_file.access_hash == 333
    got = lib.get("摸摸头")
    assert got["file_reference"] == b"ref-new" and got["access_hash"] == 333


@pytest.mark.asyncio
async def test_send_refetch_missing_raises_and_records_failed(lib):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc()), OWNER)
    client = FakeClient(fail_first=tg_errors.FileReferenceExpiredError(request=None), refetch=None)
    with pytest.raises(StickerSendError):
        await lib.send(client, int(OWNER), e, scope_key=OWNER_SCOPE)
    assert lib.get("摸摸头")["use_count"] == 0
    st = lib._conn().execute("SELECT status FROM sends").fetchall()
    assert [r[0] for r in st] == ["failed"]
    # 失败不触发冷却
    assert lib.should_send(OWNER_SCOPE) is True


@pytest.mark.asyncio
async def test_send_other_error_not_retried(lib):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc()), OWNER)
    client = FakeClient(fail_first=RuntimeError("boom"))
    with pytest.raises(StickerSendError):
        await lib.send(client, int(OWNER), e)
    assert len(client.sent) == 1 and client.get_calls == []


@pytest.mark.asyncio
async def test_send_without_returned_message_raises(lib):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc()), OWNER)

    class NoneClient(FakeClient):
        async def send_file(self, peer, file=None, **kw):
            return None

    with pytest.raises(StickerSendError):
        await lib.send(NoneClient(), int(OWNER), e)
    with pytest.raises(StickerSendError):
        await lib.send(None, int(OWNER), e)


# --------------------------------------------------------------------------- cooldown


@pytest.mark.asyncio
async def test_should_send_cooldown(lib, clock):
    e = lib.add_from_message("摸摸头", "", fake_msg(fake_doc()), OWNER)
    assert lib.should_send(OWNER_SCOPE) is True
    await lib.send(FakeClient(), int(OWNER), e, scope_key=OWNER_SCOPE)
    assert lib.should_send(OWNER_SCOPE, clock.t + 119) is False
    assert lib.should_send(OWNER_SCOPE, clock.t + 120) is True
    # 不同会话互不影响
    assert lib.should_send(GROUP, clock.t + 1) is True
    clock.advance(5)
    assert lib.should_send(OWNER_SCOPE) is False


# --------------------------------------------------------------------------- commands


def test_commands_owner_flow(lib):
    reply = fake_msg(fake_doc())
    out = lib.handle_command(["add", "摸摸头", "安慰", "的时候"], reply, OWNER, [int(OWNER)])
    assert "已添加" in out and "摸摸头" in out and "安慰 的时候" in out
    out = lib.handle_command(["add", "摸摸头", "改了"], reply, OWNER, [OWNER])
    assert "已更新" in out
    out = lib.handle_command(["add", "摸摸头"], None, OWNER, [OWNER])
    assert "回复一张贴纸" in out
    out = lib.handle_command(["add", "坏_标签"], reply, OWNER, [OWNER])
    assert "标签只能用" in out
    out = lib.handle_command(["add", "abc"], fake_msg(None), OWNER, [OWNER])
    assert "不是贴纸" in out
    out = lib.handle_command(["list"], None, OWNER, [OWNER])
    assert "摸摸头" in out and "仅私聊" in out
    out = lib.handle_command(["scope", "摸摸头", "all"], None, OWNER, [OWNER])
    assert "私聊和群" in out
    out = lib.handle_command(["scope", "摸摸头", "xx"], None, OWNER, [OWNER])
    assert "private 或 all" in out
    out = lib.handle_command(["del", "摸摸头"], None, OWNER, [OWNER])
    assert "已删除" in out
    out = lib.handle_command(["del", "摸摸头"], None, OWNER, [OWNER])
    assert "没有叫" in out
    assert "/sticker add" in lib.handle_command(["help"], None, OWNER, [OWNER])
    assert "/sticker add" in lib.handle_command([], None, OWNER, [OWNER])
    assert "空的" in lib.handle_command(["list"], None, OWNER, [OWNER])


def test_commands_non_owner_and_group_refused(lib):
    reply = fake_msg(fake_doc())
    out = lib.handle_command(["add", "摸摸头"], reply, "1234", [OWNER])
    assert "只有主人" in out
    out = lib.handle_command(["add", "摸摸头"], reply, OWNER, [OWNER], is_group=True)
    assert "私聊" in out
    assert lib.list() == []


def test_parse_marker_no_marker_returns_text_byte_identical():
    text = "统计表情：\n\n```python\ndef f(x):\n    if x:\n        return  1\n```\n\n\n  尾部  "
    clean, label = parse_marker(text, ["摸摸头"])
    assert clean is text or clean == text
    assert label is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("```py\ndef f():\n    return 1\n``` [表情:摸摸头]", "```py\ndef f():\n    return 1\n```"),
        ("[表情:摸摸头]\n    缩进第一行\n        第二行", "    缩进第一行\n        第二行"),
        ("a  b\n    c [表情:摸摸头] d\n\n\n  e", "a  b\n    c d\n\n\n  e"),
        ("第一行\n[表情:摸摸头]\n第二行", "第一行\n第二行"),
        ("哈哈[表情:摸摸头]嗯", "哈哈嗯"),
    ],
)
def test_parse_marker_only_tidies_whitespace_next_to_marker(text, expected):
    clean, label = parse_marker(text, ["摸摸头"])
    assert clean == expected and label == "摸摸头"
