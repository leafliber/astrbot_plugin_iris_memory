import base64
import io
import stat
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from iris_memory.config import Config
from iris_memory.image_store import ImageStore, inspect_image
from iris_memory.store import Inbox
from iris_memory.vendor.iris_client import IrisError
from tests.test_bridge import Event, component
from tests.test_bridge import bridge_factory as bridge_factory


def picture(*, size=(64, 64), solid=False, reverse=False, tag=None):
    im = Image.new("RGB", size)
    im.putdata(
        [
            (100, 100, 100)
            if solid
            else (
                (255 - x * 255 // size[0] if reverse else x * 255 // size[0]),
                y * 255 // size[1],
                90,
            )
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    data = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if tag:
        info.add_text("test", tag)
    im.save(data, format="PNG", pnginfo=info)
    return data.getvalue()


def image_event(data, *, group="g1", message_id="m1", **attributes):
    event = Event(group=group, message_id=message_id)
    event.messages = [
        component("image", file="base64://" + base64.b64encode(data).decode(), **attributes)
    ]
    return event


def test_persistent_exact_scope_expiry_and_description(tmp_path):
    now = [100000.0]
    config = Config(image_cache_ttl_hours=1)
    fingerprint = inspect_image(picture(), config)
    store = ImageStore(tmp_path, config, "service-scope", clock=lambda: now[0])
    store.save("entry-a", fingerprint, "cat", "media-a")
    assert store.lookup("entry-a", fingerprint, "cat") == "media-a"
    assert store.lookup("entry-b", fingerprint, "cat") is None
    assert store.lookup("entry-a", fingerprint, "dog") is None
    store.close()
    store = ImageStore(tmp_path, config, "different-scope", clock=lambda: now[0])
    assert store.lookup("entry-a", fingerprint, "cat") is None
    store.close()
    store = ImageStore(tmp_path, config, "service-scope", clock=lambda: now[0])
    assert store.lookup("entry-a", fingerprint, "cat") == "media-a"
    now[0] += 3601
    assert store.lookup("entry-a", fingerprint, "cat") is None
    store.close()


def test_perceptual_opt_in_threshold_and_lru_bound(tmp_path):
    config = Config(image_cache_max_entries=2)
    store = ImageStore(tmp_path, config, "scope")
    a = inspect_image(picture(tag="one"), config)
    b = inspect_image(picture(tag="two"), config)
    c = inspect_image(picture(reverse=True), config)
    store.save("entry", a, "", "media-a")
    assert store.lookup("entry", b, "") is None
    store.config = replace(config, image_perceptual_reuse=True, image_hash_threshold=0)
    assert store.lookup("entry", b, "") == "media-a"
    assert store.lookup("entry", replace(b, dhash=b.dhash ^ 1), "") is None
    store.config = replace(store.config, image_hash_threshold=1)
    assert store.lookup("entry", replace(b, dhash=b.dhash ^ 1), "") == "media-a"
    assert store.lookup("entry", c, "") is None
    store.save("entry", b, "b", "media-b")
    store.lookup("entry", a, "")
    store.save("entry", c, "c", "media-c")
    assert store.lookup("entry", b, "b") is None
    assert store.lookup("entry", a, "") == "media-a"
    assert (tmp_path / "images.sqlite3").stat().st_size <= config.image_cache_max_mb * 1024**2
    store.close()


@pytest.mark.parametrize(
    "data,changes",
    [
        (b"\x89PNG\r\n\x1a\nBROKEN", {}),
        (picture(size=(8, 8)), {}),
        (picture(solid=True), {}),
        (picture(), {"image_max_pixels": 100}),
    ],
)
def test_invalid_filters(data, changes):
    with pytest.raises(ValueError):
        inspect_image(data, replace(Config(), **changes))


def test_filter_switch_and_animation():
    assert inspect_image(picture(solid=True), Config(image_skip_solid=False))
    im = Image.new("RGB", (64, 64), "red")
    output = io.BytesIO()
    im.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (64, 64), "blue")],
        duration=100,
    )
    assert inspect_image(output.getvalue(), Config()).animated
    with pytest.raises(ValueError):
        inspect_image(output.getvalue(), Config(image_skip_stickers=True))


def test_persistent_daily_quota_global_scope_and_zero(tmp_path):
    now = [1791676800.0]
    config = Config(image_entry_daily_quota=1, image_global_daily_quota=2)
    one = ImageStore(tmp_path, config, "one", clock=lambda: now[0])
    two = ImageStore(tmp_path, config, "two", clock=lambda: now[0])
    assert one.reserve("a")
    assert not two.reserve("a")
    assert two.reserve("b")
    assert not one.reserve("c")
    one.close()
    one = ImageStore(tmp_path, config, "one", clock=lambda: now[0])
    assert not one.reserve("c")
    now[0] += 86400
    assert one.reserve("a")
    one.config = replace(config, image_global_daily_quota=0)
    assert not one.reserve("b")
    one.close()
    two.close()


async def test_bridge_reuses_same_entry_but_never_cross_entry(bridge_factory):
    bridge, client, logs, _ = bridge_factory(image_entry_daily_quota=1, image_global_daily_quota=2)
    uploads = []

    async def upload(body):
        uploads.append(body)
        return {"id": f"media-{len(uploads)}"}

    client.upload_media = upload
    for i, group in enumerate(["g1", "g1", "g2", "g3"]):
        await bridge.receive(image_event(picture(), group=group, message_id=str(i)))
        await bridge.queue.join()
    assert len(uploads) == 2
    assert [item[1].get("media_ids") for item in client.inputs] == [
        ["media-1"],
        ["media-1"],
        ["media-2"],
        None,
    ]
    assert client.inputs[-1][1]["content"] == "[图片，未理解]"
    assert "Iris: image_quota_exhausted" in logs


async def test_stale_cached_reference_evicted_text_kept_then_reuploaded(bridge_factory):
    bridge, client, logs, _ = bridge_factory()
    uploads = []

    async def upload(body):
        uploads.append(body)
        return {"id": f"media-{len(uploads)}"}

    client.upload_media = upload
    await bridge.receive(image_event(picture()))
    await bridge.queue.join()
    original = client.receive_message

    async def receive(entry, message):
        if message.get("media_ids") == ["media-1"]:
            raise IrisError("not_found", status_code=404)
        return await original(entry, message)

    client.receive_message = receive
    for i in range(2):
        await bridge.receive(image_event(picture(), message_id=str(i + 1)))
        await bridge.queue.join()
    assert "media_ids" not in client.inputs[1][1]
    assert client.inputs[2][1]["media_ids"] == ["media-2"]
    assert len(uploads) == 2


async def test_failure_consumes_reservation_and_sticker_skips(bridge_factory):
    bridge, client, logs, _ = bridge_factory(image_entry_daily_quota=1, image_skip_stickers=True)
    uploads = []

    async def upload(body):
        uploads.append(body)
        raise RuntimeError("iris_test_token")

    client.upload_media = upload
    for i, attributes in enumerate([{"sub_type": 1}, {}, {}]):
        await bridge.receive(image_event(picture(), message_id=str(i), **attributes))
        await bridge.queue.join()
    assert len(uploads) == 1 and len(client.inputs) == 3
    assert "iris_test_token" not in str(logs)


@pytest.mark.parametrize(
    "changes",
    [
        {"image_cache_max_entries": 0},
        {"image_cache_max_mb": 0},
        {"image_cache_ttl_hours": 0},
        {"image_cache_ttl_hours": 25},
        {"image_hash_threshold": 65},
        {"image_min_side": True},
        {"image_max_pixels": 0},
        {"image_entry_daily_quota": -1},
        {"image_global_daily_quota": -1},
        {"image_perceptual_reuse": "true"},
        {"image_skip_stickers": 1},
        {"image_skip_solid": None},
    ],
)
def test_image_config_validation(changes):
    with pytest.raises(ValueError):
        Config.from_mapping(changes)


def test_storage_cap_and_metadata_only(tmp_path):
    import sqlite3

    config = Config(image_cache_max_mb=1, image_cache_max_entries=10000)
    store = ImageStore(tmp_path, config, "scope")
    fp = inspect_image(picture(), config)
    full = False
    for i in range(4000):
        try:
            store.save(
                "entry", replace(fp, sha256=f"{i:064x}"), "PRIVATE_IMAGE_DESCRIPTION", f"media-{i}"
            )
        except sqlite3.DatabaseError:
            full = True
            break
    assert full
    store.close()
    content = (tmp_path / "images.sqlite3").read_bytes()
    assert len(content) <= 1024**2
    assert b"PRIVATE_IMAGE_DESCRIPTION" not in content


async def test_actual_http_cache_and_filter(plugin_factory, fake_iris):
    from tests.conftest import HostEvent

    plugin = await plugin_factory()
    for i, group in enumerate(["g1", "g1", "g2"]):
        event = HostEvent(group=group, message_id=str(i))
        event.messages = image_event(picture()).messages
        await plugin.receive(event)
        await plugin.bridge.queue.join()
    assert len(fake_iris.media) == 2
    assert [m["media_ids"] for m in fake_iris.messages["astrbot:bot1:group:g1"]] == [
        ["media-1"],
        ["media-1"],
    ]
    event = HostEvent(message_id="small")
    event.messages = image_event(picture(size=(4, 4))).messages
    await plugin.receive(event)
    await plugin.bridge.queue.join()
    assert len(fake_iris.media) == 2
    assert "media_ids" not in fake_iris.messages["astrbot:bot1:group:g1"][-1]


def test_exact_match_precedes_more_recent_perceptual_match(tmp_path):
    now = [100000.0]
    config = Config(image_perceptual_reuse=True, image_hash_threshold=0)
    exact = inspect_image(picture(tag="exact"), config)
    similar = inspect_image(picture(tag="similar"), config)
    assert exact.sha256 != similar.sha256 and exact.dhash == similar.dhash
    store = ImageStore(tmp_path, config, "scope", clock=lambda: now[0])
    try:
        store.save("entry", exact, "", "media-exact")
        now[0] += 1
        store.save("entry", similar, "", "media-similar")
        assert store.lookup("entry", exact, "") == "media-exact"
        assert store.lookup("entry", similar, "") == "media-similar"
    finally:
        store.close()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("kind", ["images", "inbox"])
def test_database_files_are_private_on_create_and_reopen(tmp_path, existing, kind):
    path = tmp_path / f"{kind}.sqlite3"
    if existing:
        path.touch(mode=0o644)
        path.chmod(0o644)
    store = ImageStore(tmp_path, Config(), "scope") if kind == "images" else Inbox(path)
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()


async def test_missing_pillow_has_distinct_fault_and_keeps_placeholder(bridge_factory, monkeypatch):
    event = image_event(picture())
    bridge, client, logs, _ = bridge_factory()
    monkeypatch.setitem(sys.modules, "PIL", None)
    await bridge.receive(event)
    await bridge.queue.join()
    assert client.inputs[0][1]["content"] == "[图片，未理解]"
    assert "media_ids" not in client.inputs[0][1]
    assert bridge.counts["image_processing_unavailable"] == 1
    assert bridge.counts.get("image_filtered", 0) == 0
    assert bridge.counts.get("image_upload_failed", 0) == 0


def test_image_dependency_and_ttl_contract():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert any(item.lower().startswith("pillow") for item in project["dependency-groups"]["dev"])
    assert Config.from_mapping({"image_cache_ttl_hours": 24}).image_cache_ttl_hours == 24


@pytest.mark.parametrize("limit_mb", [0, 65])
def test_pragma_page_limit_rejects_out_of_range_before_sql(tmp_path, limit_mb):
    with pytest.raises(ValueError, match="invalid image cache page limit"):
        ImageStore(tmp_path, Config(image_cache_max_mb=limit_mb), "scope")
