import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest
import respx

from iris_memory.config import Config
from iris_memory.media import MAX_BYTES, image_bytes, image_type
from iris_memory.prompt import safe_hints
from tests.test_bridge import Event, component
from tests.test_bridge import bridge_factory as bridge_factory

GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")


@pytest.mark.parametrize("source", ["url", "file_url", "file", "path", "base64"])
async def test_astrbot_4282_image_sources(source, tmp_path):
    path = tmp_path / "image.gif"
    path.write_bytes(GIF)
    image = SimpleNamespace(url="", file="", path="")
    if source == "url":
        image.url = "https://media.example.test/image"
    elif source == "file_url":
        image.file = path.as_uri()
    elif source == "file":
        image.file = str(path)
    elif source == "path":
        image.path = str(path)
    else:
        image.file = "base64://" + base64.b64encode(GIF).decode()
    with respx.mock as router:
        route = router.get("https://media.example.test/image").respond(content=GIF)
        async with httpx.AsyncClient(trust_env=False) as http:
            assert await image_bytes(image, http) == GIF
        if source == "url":
            assert "authorization" not in route.calls[0].request.headers


async def test_upload_then_reference_with_host_description(bridge_factory):
    bridge, client, logs, _ = bridge_factory()
    uploaded = []

    async def upload(body):
        uploaded.append(body)
        assert client.inputs == []
        return {"id": "media-test"}

    client.upload_media = upload
    event = Event()
    event.messages = [component("image", url="", file="base64://" + base64.b64encode(GIF).decode())]
    event.set_extra("iris_image_descriptions", ["窗边有猫"])
    await bridge.receive(event)
    await bridge.queue.join()
    assert uploaded[0]["understanding_text"] == "窗边有猫"
    assert uploaded[0]["content_type"] == "image/gif"
    assert client.inputs[0][1]["media_ids"] == ["media-test"]
    assert "窗边有猫" in client.inputs[0][1]["content"]
    assert not logs


@pytest.mark.parametrize("mode", ["error", "timeout", "invalid"])
async def test_failed_upload_is_placeholder_and_ingest_still_succeeds(bridge_factory, mode):
    bridge, client, logs, _ = bridge_factory(image_timeout=0.01)

    async def upload(body):
        if mode == "timeout":
            await asyncio.sleep(1)
        raise ValueError("iris_test_token")

    client.upload_media = upload
    event = Event()
    event.messages = [
        component(
            "image",
            url="",
            file="base64://"
            + base64.b64encode(b"not an image" if mode == "invalid" else GIF).decode(),
        )
    ]
    await bridge.prepare(event, SimpleNamespace(system_prompt="host"))
    assert len(client.inputs) == 1
    assert "media_ids" not in client.inputs[0][1]
    assert "[图片，未理解]" == client.inputs[0][1]["content"]
    assert client.prepares
    assert logs and "iris_test_token" not in str(logs)


async def test_oversize_image_rejected(tmp_path):
    path = tmp_path / "too-large.gif"
    with path.open("wb") as stream:
        stream.truncate(MAX_BYTES + 1)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ValueError, match="invalid image file"):
            await image_bytes(SimpleNamespace(url="", file=str(path)), http)


@pytest.mark.parametrize(
    "data,mime",
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (GIF, "image/gif"),
        (b"RIFF0000WEBP", "image/webp"),
    ],
)
def test_image_mime_from_header(data, mime):
    assert image_type(data) == mime


def test_r02_whitelist_and_default_judgment_budget():
    config = Config()
    assert (config.judge, config.prepare_timeout, config.judge_budget_seconds) == (True, 8, 5)
    hint = {
        "code": "other_entries_pending",
        "entry_count": 2,
        "time_basis": "received_at",
        "time_from": "2026-10-10T12:10:00+00:00",
        "time_to": "2026-10-10T12:13:00+00:00",
        "message": "DO_NOT_INJECT",
        "entry_id": "SECRET_ENTRY",
    }
    text = str(safe_hints([hint, {"code": "new_future_code", "message": "DO_NOT_INJECT"}]))
    assert "2 个入口" in text and "12:10" in text and "12:13" in text
    assert "DO_NOT_INJECT" not in text and "SECRET_ENTRY" not in text
    assert safe_hints([{**hint, "time_from": "arbitrary diagnostic"}]) == []
    assert safe_hints([{**hint, "entry_count": True}]) == []


async def test_dedup_notice_never_sent_as_reminder(bridge_factory):
    bridge, client, logs, sends = bridge_factory()
    client.page = {
        "items": [{"id": 1, "kind": "goal_dedup_result", "content": "internal dedup"}],
        "next_cursor": 1,
        "has_more": False,
    }
    await bridge.poll()
    assert bridge.inbox.count() == 0 and bridge.inbox.cursor == 1
    assert not sends and "Iris: goal_dedup_result" in logs
