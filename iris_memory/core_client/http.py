"""Finite asynchronous HTTP transport; no redirects, ambient proxy or cookie sharing."""

import asyncio
from contextlib import asynccontextmanager
from typing import Protocol

import aiohttp

from ..errors import ControlError
from ..validation import decode, origin, reject_secret_echo, secret
from .protocol import capabilities, envelope, health, management_status


class Admission:
    def __init__(self, active=4, waiting=16):
        self.limit, self.wait_limit = active, waiting
        self.active = self.waiting = 0
        self.closed = False
        self._semaphore = asyncio.Semaphore(active)
        self._tasks = set()

    @asynccontextmanager
    async def slot(self):
        if self.closed:
            raise ControlError("CLIENT_CLOSED", 503)
        if self.active + self.waiting >= self.limit + self.wait_limit:
            raise ControlError("HTTP_ADMISSION_FULL", 429)
        task = asyncio.current_task()
        self._tasks.add(task)
        self.waiting += 1
        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            self.waiting -= 1
            self.active += 1
            if self.closed:
                raise ControlError("CLIENT_CLOSED", 503)
            yield
        finally:
            if acquired:
                self.active -= 1
                self._semaphore.release()
            else:
                self.waiting -= 1
            self._tasks.discard(task)

    async def close(self):
        self.closed = True
        tasks = self._tasks - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class Transport(Protocol):
    async def request(
        self, base: str, method: str, path: str, headers: dict, body: dict | None
    ) -> tuple[int, dict]: ...
    async def close(self) -> None: ...


class HTTPTransport:
    def __init__(
        self, admission: Admission, *, timeout=10.0, response_limit=256 * 1024
    ):
        self.admission, self.timeout, self.response_limit = (
            admission,
            timeout,
            response_limit,
        )
        self._session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            trust_env=False,
            connector=aiohttp.TCPConnector(limit=4, ssl=True),
            timeout=aiohttp.ClientTimeout(total=timeout, connect=3),
        )

    async def request(self, base, method, path, headers, body=None):
        base = origin(base)
        # Only the two clients below select routes. No page accepts a path or method.
        if (method, path) not in {
            ("GET", "/health"),
            ("POST", "/api/host/capabilities"),
            ("GET", "/api/status"),
        }:
            raise ControlError("HTTP_ROUTE_NOT_ALLOWED", 403)
        try:
            async with asyncio.timeout(self.timeout):
                async with self.admission.slot():
                    async with self._session.request(
                        method,
                        base + path,
                        headers=headers,
                        json=body,
                        allow_redirects=False,
                    ) as response:
                        if 300 <= response.status < 400:
                            raise ControlError("REDIRECT_REFUSED", 502)
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(8192):
                            if len(raw) + len(chunk) > self.response_limit:
                                raise ControlError("RESPONSE_TOO_LARGE", 502)
                            raw.extend(chunk)
                        return response.status, decode(bytes(raw), self.response_limit)
        except TimeoutError:
            raise ControlError("HTTP_TIMEOUT", 504) from None
        except (aiohttp.ClientError, OSError):
            raise ControlError("HTTP_UNAVAILABLE", 502) from None

    async def close(self):
        await self._session.close()


class HostClient:
    def __init__(self, transport: Transport):
        self.transport = transport

    async def health(self, base):
        status, value = await self.transport.request(base, "GET", "/health", {}, None)
        return health(status, value)

    async def capabilities(self, base, token):
        status, value = await self.transport.request(
            base,
            "POST",
            "/api/host/capabilities",
            {"Authorization": f"Bearer {secret(token)}"},
            {},
        )
        reject_secret_echo(value, (token,))
        reply = envelope(status, value)
        if reply.observed:
            capabilities(reply.data)
        return reply


class ManagementClient:
    """One short-lived explicit Core session owned by one authenticated host user."""

    def __init__(self, transport: Transport, base, session, csrf):
        self.transport, self.base = transport, origin(base)
        self._session, self._csrf = secret(session), secret(csrf)

    async def status(self):
        status, value = await self.transport.request(
            self.base,
            "GET",
            "/api/status",
            {
                "Cookie": f"iris_session={self._session}; iris_csrf={self._csrf}",
                "X-CSRF-Token": self._csrf,
                "Origin": self.base,
            },
            None,
        )
        reject_secret_echo(value, (self._session, self._csrf))
        reply = envelope(status, value)
        if not reply.observed:
            raise ControlError(
                "ADMIN_SESSION_REAUTHORIZE"
                if status in (401, 403)
                else "ADMIN_STATUS_UNAVAILABLE",
                403 if status in (401, 403) else 502,
            )
        return management_status(reply.data)

    async def close(self):
        self._session = self._csrf = ""
        await self.transport.close()
