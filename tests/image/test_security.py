"""图片输入安全边界回归测试。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from iris_memory.image import ImageInfo, ImageParser
from iris_memory.image.security import (
    fetch_safe_image_bytes,
    local_image_to_data_url,
)
from iris_memory.l1_buffer.buffer import L1Buffer


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test-image-data"


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


@pytest.mark.asyncio
async def test_ssrf_literal_loopback_is_rejected_before_request():
    """直接使用环回地址时不得发出 HTTP 请求。"""
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, content=PNG_BYTES, request=request)

    with patch(
        "iris_memory.image.security._create_safe_client",
        return_value=_mock_client(handler),
    ):
        result = await fetch_safe_image_bytes("http://127.0.0.1/private.png")

    assert result is None
    assert request_count == 0


@pytest.mark.asyncio
async def test_redirect_to_private_address_is_rejected():
    """公网入口重定向到内网时，只允许请求入口，不得请求重定向目标。"""
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data"},
            request=request,
        )

    safe_check = AsyncMock(side_effect=[True, False])
    with (
        patch(
            "iris_memory.image.security._create_safe_client",
            return_value=_mock_client(handler),
        ),
        patch("iris_memory.image.security.is_safe_remote_url", safe_check),
    ):
        result = await fetch_safe_image_bytes("https://images.example/start")

    assert result is None
    assert requested_urls == ["https://images.example/start"]


@pytest.mark.asyncio
async def test_safe_relative_redirect_is_followed_and_revalidated():
    """合法相对重定向可跟随，且每一跳都会重新做安全检查。"""
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"location": "/final.png"},
                request=request,
            )
        return httpx.Response(200, content=PNG_BYTES, request=request)

    safe_check = AsyncMock(return_value=True)
    with (
        patch(
            "iris_memory.image.security._create_safe_client",
            return_value=_mock_client(handler),
        ),
        patch("iris_memory.image.security.is_safe_remote_url", safe_check),
    ):
        result = await fetch_safe_image_bytes("https://images.example/start")

    assert result == (PNG_BYTES, "image/png")
    assert requested_urls == [
        "https://images.example/start",
        "https://images.example/final.png",
    ]
    assert safe_check.await_count == 2


@pytest.mark.asyncio
async def test_streaming_response_larger_than_limit_is_rejected():
    """即使 Content-Length 缺失，实际流式字节超过上限也必须中止。"""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_ChunkedStream([PNG_BYTES, b"x" * 32]),
            request=request,
        )

    with (
        patch(
            "iris_memory.image.security._create_safe_client",
            return_value=_mock_client(handler),
        ),
        patch(
            "iris_memory.image.security.is_safe_remote_url",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await fetch_safe_image_bytes(
            "https://images.example/large.png", max_bytes=len(PNG_BYTES)
        )

    assert result is None


def test_local_image_requires_real_path_containment_and_valid_magic(tmp_path: Path):
    """本地图片必须在允许目录内，且不能只靠伪造扩展名通过。"""
    cache_root = tmp_path / "data" / "image_cache"
    cache_root.mkdir(parents=True)
    inside = cache_root / "inside.png"
    inside.write_bytes(PNG_BYTES)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_BYTES)
    fake = cache_root / "fake.png"
    fake.write_text("not an image", encoding="utf-8")

    assert local_image_to_data_url(inside, allowed_root=cache_root).startswith(
        "data:image/png;base64,"
    )
    assert local_image_to_data_url(outside, allowed_root=cache_root) is None
    assert local_image_to_data_url(fake, allowed_root=cache_root) is None


@pytest.mark.asyncio
async def test_parser_rejects_untrusted_absolute_file_path(tmp_path: Path):
    """消息携带的任意绝对路径不能越过插件 image_cache。"""
    data_dir = tmp_path / "data"
    (data_dir / "image_cache").mkdir(parents=True)
    outside = tmp_path / "secret.png"
    outside.write_bytes(PNG_BYTES)
    parser = ImageParser(Mock())

    with patch(
        "iris_memory.config.get_config",
        return_value=SimpleNamespace(data_dir=data_dir),
    ):
        result = await parser._resolve_image_url(ImageInfo(file_path=str(outside)))

    assert result is None


def test_cleanup_only_deletes_files_inside_configured_cache(tmp_path: Path):
    """路径中带 image_cache 字样不足以授权删除，必须真实位于配置目录内。"""
    data_dir = tmp_path / "plugin-data"
    cache_root = data_dir / "image_cache"
    cache_root.mkdir(parents=True)
    cached = cache_root / "cached.png"
    cached.write_bytes(PNG_BYTES)

    outside_dir = tmp_path / "untrusted" / "image_cache"
    outside_dir.mkdir(parents=True)
    outside = outside_dir / "keep.png"
    outside.write_bytes(PNG_BYTES)
    symlink_escape = cache_root / "escape.png"
    symlink_escape.symlink_to(outside)

    items = [
        SimpleNamespace(image_info=ImageInfo(file_path=str(cached))),
        SimpleNamespace(image_info=ImageInfo(file_path=str(outside))),
        SimpleNamespace(image_info=ImageInfo(file_path=str(symlink_escape))),
    ]
    with patch(
        "iris_memory.l1_buffer.buffer.get_config",
        return_value=SimpleNamespace(data_dir=data_dir),
    ):
        L1Buffer._cleanup_image_cache_files(items)

    assert not cached.exists()
    assert outside.exists()
    assert symlink_escape.is_symlink()
