"""贴纸理解：缓存、并发去重、输出校验、事件改写、每日上限。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot.api.message_components import Image, Plain
from astrbot_plugin_iris_memory.iris_memory.stickers.store import StickerStore, day_key
from astrbot_plugin_iris_memory.iris_memory.stickers.understanding import (
    EXTRA_KEY,
    KIND_STATIC,
    KIND_TGS,
    KIND_VIDEO,
    StickerMeta,
    StickerUnderstanding,
    build_prompt,
    extract_sticker_meta,
    frame_getter_from_event,
    make_describe_image,
    rewrite_event,
    sanitize_description,
)


# -- Telethon 形状的假对象（类名与 telethon.tl.types 一致） ---------------------


class DocumentAttributeSticker:
    def __init__(self, alt, stickerset):
        self.alt = alt
        self.stickerset = stickerset


class InputStickerSetID:
    def __init__(self, id, access_hash=0):
        self.id = id
        self.access_hash = access_hash


class InputStickerSetShortName:
    def __init__(self, short_name):
        self.short_name = short_name


class DocumentAttributeVideo:
    pass


class DocumentAttributeImageSize:
    pass


def _raw(doc_id=111, mime="image/webp", alt="😂", stickerset=None, extra_attrs=()):
    doc = SimpleNamespace(
        id=doc_id,
        mime_type=mime,
        attributes=[
            DocumentAttributeImageSize(),
            DocumentAttributeSticker(alt, stickerset or InputStickerSetID(42)),
            *extra_attrs,
        ],
    )
    return SimpleNamespace(document=doc)


class Clock:
    def __init__(self, t=1_762_070_400.0):  # 2025-11-02 16:00 +08
        self.t = t

    def __call__(self):
        return self.t


class FakeVision:
    def __init__(self, reply="一只橘猫捂嘴偷笑，旁边写着「笑死」", delay=0.0):
        self.reply = reply
        self.delay = delay
        self.calls = []
        self.last_model = "fake-model"

    async def __call__(self, image_path, prompt, timeout):
        self.calls.append((image_path, prompt, timeout))
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def frame(tmp_path: Path) -> Path:
    p = tmp_path / "s.frame.png"
    p.write_bytes(b"png")
    return p


@pytest.fixture
def store(tmp_path: Path):
    clock = Clock()
    s = StickerStore.for_data_dir(tmp_path / "data", clock)
    yield s
    s.close()


def _getter(path, counter=None):
    async def get():
        if counter is not None:
            counter.append(1)
        return str(path)

    return get


META = StickerMeta(doc_id=111, emoji="😂", kind=KIND_STATIC, mime_type="image/webp", set_id=42)


# -- 元数据 -------------------------------------------------------------------


def test_extract_sticker_meta_kinds_and_sets():
    m = extract_sticker_meta(_raw())
    assert m == StickerMeta(111, "😂", KIND_STATIC, "image/webp", 42, None)

    m = extract_sticker_meta(_raw(mime="application/x-tgsticker", stickerset=InputStickerSetShortName("cats")))
    assert m.kind == KIND_TGS and m.set_short_name == "cats" and m.set_id is None

    m = extract_sticker_meta(_raw(mime="video/webm", extra_attrs=(DocumentAttributeVideo(),)))
    assert m.kind == KIND_VIDEO

    plain_doc = SimpleNamespace(document=SimpleNamespace(id=1, mime_type="image/png", attributes=[]))
    assert extract_sticker_meta(plain_doc) is None
    assert extract_sticker_meta(SimpleNamespace(document=None, media=None)) is None
    assert extract_sticker_meta(None) is None


def test_store_path_and_schema(tmp_path: Path, store):
    assert store.db_path == tmp_path / "data" / "stickers" / "stickers_desc.db"
    assert store.schema_version() == 1
    jm = store._conn().execute("PRAGMA journal_mode").fetchone()[0]
    assert jm.lower() == "wal"


# -- 描述 ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_avoids_llm(store, frame):
    vision = FakeVision()
    u = StickerUnderstanding(store, vision, Clock())
    first = await u.describe(META, _getter(frame))
    assert first == "一只橘猫捂嘴偷笑，旁边写着「笑死」"
    assert len(vision.calls) == 1
    assert vision.calls[0][0] == str(frame)
    assert "40字" in vision.calls[0][1]

    frames = []
    second = await u.describe(META, _getter(frame, frames))
    assert second == first
    assert len(vision.calls) == 1
    assert frames == []  # 命中缓存不下载
    cached = store.get(111, count_hit=False)
    assert cached.model == "fake-model" and cached.set_id == 42 and cached.hits == 1

    # 新实例（重启）同样命中
    u2 = StickerUnderstanding(store, vision, Clock())
    assert await u2.describe(META, _getter(frame)) == first
    assert len(vision.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_same_sticker_single_call(store, frame):
    vision = FakeVision(delay=0.05)
    u = StickerUnderstanding(store, vision, Clock())
    results = await asyncio.gather(*[u.describe(META, _getter(frame)) for _ in range(5)])
    assert len(set(results)) == 1 and results[0]
    assert len(vision.calls) == 1
    assert u._inflight == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "抱歉，我无法看到这张图片",
        "这张贴纸可爱度 9/10",
        "表情很开心，打分 8 分",
        "第一行\n第二行\n第三行",
        "猫" * 81,
        None,
    ],
)
async def test_invalid_output_rejected_not_cached(store, frame, reply):
    vision = FakeVision(reply=reply)
    u = StickerUnderstanding(store, vision, Clock())
    assert await u.describe(META, _getter(frame)) is None
    assert store.get(111) is None
    # 失败后短时间内不重复调用模型
    assert await u.describe(META, _getter(frame)) is None
    assert len(vision.calls) == 1


def test_sanitize_strips_wrapping():
    assert sanitize_description('"一只柴犬歪头。"') == "一只柴犬歪头"
    assert sanitize_description("**描述：** 小熊抱着枕头打哈欠") == "小熊抱着枕头打哈欠"
    assert sanitize_description("```\n兔子挥手说[你好]\n```") == "兔子挥手说「你好」"
    assert sanitize_description("「笑死」") == "笑死"
    assert sanitize_description("猫猫举牌「上班」，旁边写着「好累」") == "猫猫举牌「上班」，旁边写着「好累」"
    assert sanitize_description("熊猫举着写有「666」的牌子") == "熊猫举着写有「666」的牌子"


@pytest.mark.asyncio
async def test_vision_error_and_timeout_return_none(store, frame):
    u = StickerUnderstanding(store, FakeVision(reply=RuntimeError("boom")), Clock())
    assert await u.describe(META, _getter(frame)) is None

    slow = FakeVision(delay=1.0)
    u2 = StickerUnderstanding(store, slow, Clock(), timeout=0.1)
    meta2 = StickerMeta(222, "", KIND_STATIC, "image/webp")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await u2.describe(meta2, _getter(frame)) is None
    assert loop.time() - t0 < 0.5
    await u2.aclose()


@pytest.mark.asyncio
async def test_frame_fallback_to_animated_original_skips_llm(store, tmp_path):
    tgs = tmp_path / "s.tgs"
    tgs.write_bytes(b"tgs")
    vision = FakeVision()
    u = StickerUnderstanding(store, vision, Clock())
    meta = StickerMeta(333, "", KIND_TGS, "application/x-tgsticker")
    assert await u.describe(meta, _getter(tgs)) is None
    assert vision.calls == []
    assert store.misses_on(day_key(Clock()())) == 0


@pytest.mark.asyncio
async def test_daily_cap(store, frame):
    clock = Clock()
    vision = FakeVision()
    u = StickerUnderstanding(store, vision, clock, daily_cap=2)
    metas = [StickerMeta(1000 + i, "", KIND_STATIC, "image/webp") for i in range(3)]
    assert await u.describe(metas[0], _getter(frame))
    assert await u.describe(metas[1], _getter(frame))
    assert await u.describe(metas[2], _getter(frame)) is None
    assert len(vision.calls) == 2
    # 命中缓存不受上限影响
    assert await u.describe(metas[0], _getter(frame))
    assert store.misses_on(day_key(clock())) == 2
    # 第二天恢复（失败退避也已过期）
    clock.t += 86400
    assert await u.describe(metas[2], _getter(frame))
    assert len(vision.calls) == 3


# -- 事件改写 -----------------------------------------------------------------


class FakeEvent:
    def __init__(self, chain, message_str=""):
        self.message_obj = SimpleNamespace(message=chain, message_str=message_str)
        self.message_str = message_str
        self.extras = {}

    def set_extra(self, k, v):
        self.extras[k] = v


def test_rewrite_event_replaces_image_and_label():
    img = Image(file="telethon-lazy://image")
    ev = FakeEvent([img, Plain(text="[贴纸] 😂")])
    rewrite_event(ev, "一只橘猫捂嘴偷笑", META)
    assert len(ev.message_obj.message) == 1
    assert isinstance(ev.message_obj.message[0], Plain)
    assert ev.message_obj.message[0].text == "[贴纸 😂：一只橘猫捂嘴偷笑]"
    assert ev.message_str == "[贴纸 😂：一只橘猫捂嘴偷笑]"
    assert ev.message_obj.message_str == ev.message_str
    assert ev.extras[EXTRA_KEY]["doc_id"] == 111
    assert not any(isinstance(c, Image) for c in ev.message_obj.message)


def test_rewrite_event_keeps_other_text_and_no_emoji():
    meta = StickerMeta(5, "", KIND_STATIC, "image/webp")
    ev = FakeEvent([Image(file="x"), Plain(text="[贴纸]"), Plain(text="你看这个")])
    rewrite_event(ev, "小狗摇尾巴", meta)
    texts = [c.text for c in ev.message_obj.message]
    assert texts == ["[贴纸：小狗摇尾巴]", "你看这个"]
    assert ev.message_str == "[贴纸：小狗摇尾巴] 你看这个"


def test_rewrite_event_none_leaves_event_unchanged():
    img = Image(file="x")
    label = Plain(text="[贴纸] 😂")
    ev = FakeEvent([img, label])
    rewrite_event(ev, None, META)
    assert ev.message_obj.message == [img, label]
    assert ev.message_str == ""
    assert ev.extras == {}


@pytest.mark.asyncio
async def test_frame_getter_from_event_calls_convert(frame):
    class LazyImage(Image):
        async def convert_to_file_path(self):
            return str(frame)

    ev = FakeEvent([Plain(text="hi"), LazyImage(file="telethon-lazy://image")])
    assert await frame_getter_from_event(ev)() == str(frame)
    assert await frame_getter_from_event(FakeEvent([Plain(text="x")]))() is None


# -- LLMManager.generate_direct 适配 ---------------------------------------------


@pytest.mark.asyncio
async def test_make_describe_image_uses_llm_manager(frame):
    calls = []

    class Manager:
        async def generate_direct(self, **kw):
            calls.append(kw)
            return "兔子比心"

    async def pid():
        return "main-provider"

    fn = make_describe_image(Manager(), pid)
    out = await fn(str(frame), build_prompt(META), 5.0)
    assert out == "兔子比心"
    assert calls[0]["provider_id"] == "main-provider"
    assert calls[0]["image_urls"] == [str(frame)]
    assert calls[0]["module"] == "sticker_understanding"
    assert "😂" in calls[0]["prompt"]
    assert fn.last_model == "main-provider"

    out2 = await make_describe_image(Manager(), lambda: None)(str(frame), "p", 1.0)
    assert out2 == "兔子比心" and calls[-1]["provider_id"] is None

    class Broken:
        async def generate_direct(self, **kw):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await make_describe_image(Broken(), lambda: "p")(str(frame), "p", 1.0)


def test_prompt_has_no_numeric_scoring_request():
    p = build_prompt(META)
    assert "不要猜测" in p and "不要打分" in p


# -- 照片理解 -------------------------------------------------------------------

from astrbot_plugin_iris_memory.iris_memory.stickers.understanding import (  # noqa: E402
    KIND_PHOTO,
    PHOTO_EXTRA_KEY,
    extract_photo_meta,
)


def test_extract_photo_meta():
    raw = SimpleNamespace(photo=SimpleNamespace(id=9876), document=None)
    meta = extract_photo_meta(raw)
    assert meta.doc_id == 9876 and meta.kind == KIND_PHOTO
    assert extract_photo_meta(SimpleNamespace(photo=None)) is None
    assert extract_sticker_meta(raw) is None  # 照片不是贴纸


def test_photo_rewrite_keeps_image_and_caption():
    img = Image(file="telethon-lazy://image")
    ev = FakeEvent([img, Plain(text="看我今天做的饭")], message_str="看我今天做的饭")
    meta = extract_photo_meta(SimpleNamespace(photo=SimpleNamespace(id=1)))
    rewrite_event(ev, "一盘番茄炒蛋和米饭，放在木桌上", meta)
    chain = ev.message_obj.message
    assert isinstance(chain[0], Plain) and chain[0].text == "[图片：一盘番茄炒蛋和米饭，放在木桌上]"
    assert any(isinstance(c, Image) for c in chain)  # 原图保留给主模型
    assert ev.message_str == "[图片：一盘番茄炒蛋和米饭，放在木桌上] 看我今天做的饭"
    assert ev.extras[PHOTO_EXTRA_KEY]["photo_id"] == 1


def test_photo_prompt_and_longer_limit():
    meta = extract_photo_meta(SimpleNamespace(photo=SimpleNamespace(id=1)))
    assert "图片" in build_prompt(meta) and "贴纸" not in build_prompt(meta)
    long_desc = "终端截图，显示一段 Python 报错「ModuleNotFoundError: No module named foo」，下方是 pip 安装命令的输出，提示依赖冲突需要先升级 setuptools"
    assert sanitize_description(long_desc) is None  # 贴纸上限 80
    assert sanitize_description(long_desc, 160, 4) is not None


@pytest.mark.asyncio
async def test_photo_described_once_and_cached(store, frame):
    vision = FakeVision(reply="一张海边日落的照片，沙滩上有两个人影")
    su = StickerUnderstanding(store, vision, clock=Clock())
    meta = extract_photo_meta(SimpleNamespace(photo=SimpleNamespace(id=4242)))
    d1 = await su.describe(meta, _getter(frame))
    d2 = await su.describe(meta, _getter(frame))
    assert d1 == d2 == "一张海边日落的照片，沙滩上有两个人影"
    assert len(vision.calls) == 1
    assert "图片" in vision.calls[0][1]
