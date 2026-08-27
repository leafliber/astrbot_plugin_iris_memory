"""图片输入安全边界。

集中处理远程图片下载和本地图片读取，确保所有调用方使用相同的 SSRF、
重定向、响应大小、文件类型与路径包含规则。
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from iris_memory.core import get_logger

logger = get_logger("image.security")

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_REDIRECTS = 3
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def detect_image_mime(data: bytes) -> Optional[str]:
    """根据魔数识别允许的图片类型，拒绝仅伪造扩展名/MIME 的文件。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _host_all_global(host: str) -> bool:
    """主机的全部解析地址是否均为全局可达地址。"""
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except (ValueError, IndexError):
            return False
    return bool(addresses) and all(address.is_global for address in addresses)


async def is_safe_remote_url(url: str) -> bool:
    """仅允许解析结果全部为公网地址的 HTTP(S) URL。"""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return await asyncio.to_thread(_host_all_global, parsed.hostname)


def _resolve_and_pin(url: str) -> Optional[tuple[tuple[str, ...], str, str]]:
    """解析并校验 URL 主机，返回 ``(pinned_urls, host_header, sni_hostname)``。

    所有解析地址均须为公网地址；pinned_urls 保留解析顺序并逐个用于
    直连，既消除 DNS rebinding TOCTOU 窗口，也保留双栈/多 A 记录的
    故障转移能力。Host 头与 TLS SNI/证书校验仍按原主机名进行。

    Returns:
        无法解析、含非全局地址或协议不合法时返回 None。
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None

    host = parsed.hostname
    try:
        literal = ipaddress.ip_address(host)
        candidates = [literal] if literal.is_global else []
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return None
        candidates = []
        for info in infos:
            try:
                address = ipaddress.ip_address(info[4][0])
            except (ValueError, IndexError):
                return None
            candidates.append(address)

    # 任一解析记录非全局地址即整体拒绝，防止多 A 记录混入内网地址
    if not candidates or not all(address.is_global for address in candidates):
        return None

    try:
        port = parsed.port
    except ValueError:
        return None

    # getaddrinfo 常会为不同 socket type 返回重复地址；按解析顺序去重，
    # 连接失败时依次尝试下一候选，避免首个 AAAA 不可达时误判整站不可用。
    pinned_urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        ip_str = str(candidate)
        if ip_str in seen:
            continue
        seen.add(ip_str)
        if ":" in ip_str:
            netloc = f"[{ip_str}]" if port is None else f"[{ip_str}]:{port}"
        else:
            netloc = ip_str if port is None else f"{ip_str}:{port}"
        pinned_urls.append(
            urlunparse(
                (
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
        )

    host_header = f"[{host}]" if ":" in host else host
    if port is not None:
        host_header = f"{host_header}:{port}"
    return tuple(pinned_urls), host_header, host


class _GlobalOnlyTransport(httpx.AsyncBaseTransport):
    """在每次实际请求交给网络栈前重新执行公网地址校验。"""

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not host or not await asyncio.to_thread(_host_all_global, host):
            raise httpx.ConnectError(
                f"目标主机解析含非全局地址，拒绝连接: {host or '<empty>'}",
                request=request,
            )
        return await self._wrapped.handle_async_request(request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


def _create_safe_client(timeout: float) -> httpx.AsyncClient:
    transport = _GlobalOnlyTransport(httpx.AsyncHTTPTransport(verify=True))
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        transport=transport,
    )


async def fetch_safe_image_bytes(
    url: str,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_redirects: int = MAX_IMAGE_REDIRECTS,
    timeout: float = 15.0,
) -> Optional[tuple[bytes, str]]:
    """安全下载图片，返回 ``(内容, MIME)``。

    初始 URL 与每一跳重定向都必须指向公网 HTTP(S) 地址；响应按流读取，
    同时校验 Content-Length 与实际解压后的累计字节，防止大响应占满内存。
    """
    current_url = url

    try:
        async with _create_safe_client(timeout) as client:
            for redirect_count in range(max_redirects + 1):
                pinned = await asyncio.to_thread(_resolve_and_pin, current_url)
                if pinned is None:
                    logger.warning(
                        f"图片 URL 主机不安全（内网/保留地址或无法解析），拒绝下载："
                        f"{current_url[:80]}"
                    )
                    return None
                pinned_urls, host_header, sni_host = pinned

                # 用已校验的 IP 依次直连；Host 头与 SNI 保留原主机名，
                # 证书校验同样按原主机名进行。仅连接/传输错误尝试下一 IP，
                # 一旦收到 HTTP 响应便交给下方状态码逻辑处理。
                response: Optional[httpx.Response] = None
                last_error: Optional[httpx.RequestError] = None
                for pinned_url in pinned_urls:
                    request = client.build_request("GET", pinned_url)
                    request.headers["Host"] = host_header
                    request.extensions["sni_hostname"] = sni_host
                    try:
                        response = await client.send(request, stream=True)
                        break
                    except httpx.RequestError as e:
                        last_error = e

                if response is None:
                    logger.debug(
                        f"图片所有安全解析地址均连接失败：{current_url[:80]}，"
                        f"最后错误：{last_error}"
                    )
                    return None
                try:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location or redirect_count >= max_redirects:
                            logger.debug("图片重定向缺少 Location 或跳数超限")
                            return None
                        # 用原始（非 pin 后的）URL 解析相对重定向，
                        # 保留域名信息供下一跳重新校验与 pin
                        current_url = urljoin(current_url, location)
                        continue

                    if response.status_code >= 400:
                        logger.debug(
                            f"图片 URL 返回 {response.status_code}：{current_url[:80]}"
                        )
                        return None

                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > max_bytes:
                                logger.warning("图片响应声明长度超限，已拒绝下载")
                                return None
                        except ValueError:
                            pass

                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            logger.warning(f"图片响应超过 {max_bytes} 字节，已中止下载")
                            return None
                        chunks.append(chunk)

                    content = b"".join(chunks)
                    mime = detect_image_mime(content)
                    if not content or mime is None:
                        logger.debug("远程响应不是受支持的图片格式")
                        return None
                    return content, mime
                finally:
                    await response.aclose()
    except Exception as e:
        logger.debug(f"安全下载图片失败：{e}")
    return None


def is_path_within(path: Path, root: Path) -> bool:
    """使用解析后的真实路径判断 ``path`` 是否位于 ``root`` 内。"""
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def local_image_to_data_url(
    file_path: Path,
    *,
    allowed_root: Optional[Path] = None,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> Optional[str]:
    """受限读取本地图片；可选要求真实路径位于指定根目录。"""
    try:
        resolved = file_path.resolve(strict=True)
        if allowed_root is not None and not is_path_within(resolved, allowed_root):
            logger.warning(f"本地图片越过允许目录，拒绝读取：{file_path}")
            return None
        if not resolved.is_file():
            return None
        size = resolved.stat().st_size
        if size <= 0 or size > max_bytes:
            logger.warning(f"本地图片大小不合法（{size} 字节），拒绝读取")
            return None
        with resolved.open("rb") as file:
            image_data = file.read(max_bytes + 1)
        if len(image_data) > max_bytes:
            return None
        mime = detect_image_mime(image_data)
        if mime is None:
            logger.warning(f"本地文件不是受支持的图片格式：{file_path.name}")
            return None
        encoded = base64.b64encode(image_data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except (OSError, ValueError) as e:
        logger.debug(f"读取本地图片失败：{e}")
        return None
