"""AstrBot 4.28.2 Image.url / file / path; bounded, credential-free transfer."""

import asyncio
import base64
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

import httpx

from .config import Config
from .image_store import ImageStore, inspect_image
from .mapping import component_kind
from .vendor.iris_client import AsyncIrisClient

MAX_BYTES = 10 * 1024 * 1024


def image_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image")


def read_local(path: str) -> bytes:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_BYTES:
        raise ValueError("invalid image file")
    with source.open("rb") as stream:
        return stream.read(MAX_BYTES + 1)


async def image_bytes(component: Any, http: httpx.AsyncClient) -> bytes:
    source = (
        getattr(component, "url", "")
        or getattr(component, "file", "")
        or getattr(component, "path", "")
    )
    if not isinstance(source, str) or not source:
        raise ValueError("missing image")
    url = urlsplit(source)
    if url.scheme in {"http", "https"}:
        if url.username or url.password:
            raise ValueError("invalid image URL")
        async with http.stream("GET", source) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > MAX_BYTES:
                    raise ValueError("image too large")
                data.extend(chunk)
            return bytes(data)
    if source.startswith("base64://"):
        encoded = source.removeprefix("base64://")
        if len(encoded) > (MAX_BYTES + 2) // 3 * 4:
            raise ValueError("image too large")
        return base64.b64decode(encoded, validate=True)
    if url.scheme == "file" and url.netloc in {"", "localhost"}:
        source = unquote(url.path)
    elif url.scheme:
        raise ValueError("unsupported image source")
    return await asyncio.to_thread(read_local, source)


async def attach_images(
    event: Any,
    entry_id: str,
    message: dict,
    client: AsyncIrisClient,
    config: Config,
    record: Callable[[str], None],
    store: ImageStore | None = None,
) -> set[str]:
    descriptions = event.get_extra("iris_image_descriptions", [])
    if not isinstance(descriptions, list):
        descriptions = []
    images = [c for c in event.get_messages() if component_kind(c) == "image"]
    if not images:
        return set()
    if store is None:
        record("image_storage_unavailable")
        return set()
    ids, cached = [], set()
    try:
        # One total budget for download AND upload across all images. Never share
        # the Iris HTTP client (and its Authorization header) with media hosts.
        async with asyncio.timeout(config.image_timeout):
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=True, max_redirects=3
            ) as http:
                for index, component in enumerate(images[: config.max_images]):
                    try:
                        if config.image_skip_stickers and (
                            getattr(component, "sub_type", None) == 1
                            or getattr(component, "subType", None) == 1
                            or getattr(component, "_type", None) in {"face", "sticker"}
                        ):
                            record("image_filtered")
                            continue
                        data = await image_bytes(component, http)
                        if len(data) > MAX_BYTES:
                            raise ValueError("image too large")
                        content_type = image_type(data)
                        try:
                            fingerprint = await asyncio.to_thread(inspect_image, data, config)
                        except ValueError:
                            record("image_filtered")
                            continue
                        description = (
                            descriptions[index]
                            if index < len(descriptions) and isinstance(descriptions[index], str)
                            else ""
                        )
                        reused = store.lookup(entry_id, fingerprint, description)
                        if reused:
                            ids.append(reused)
                            cached.add(reused)
                            record("image_cache_hit")
                            continue
                        if not store.reserve(entry_id):
                            record("image_quota_exhausted")
                            continue
                        upload = {
                            "entry_id": entry_id,
                            "content_type": content_type,
                            "data_base64": base64.b64encode(data).decode("ascii"),
                        }
                        if index < len(descriptions) and isinstance(descriptions[index], str):
                            upload["understanding_text"] = descriptions[index]
                        media = await client.upload_media(upload)
                        ids.append(media["id"])
                        store.save(entry_id, fingerprint, description, media["id"])
                    except Exception:
                        record("image_upload_failed")
    except Exception:
        record("image_upload_degraded")
    # Text already carries host descriptions/placeholders, including failed images.
    if ids:
        message["media_ids"] = list(dict.fromkeys(ids))

    return cached
