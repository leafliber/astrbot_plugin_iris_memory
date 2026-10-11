"""One serial, cancellable SSE consumer with pull repair and durable checkpoints."""

from __future__ import annotations

import asyncio
import codecs
import json
import math
import random
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

import httpx

from . import compat
from .errors import IrisError
from .types import Notification, NotificationPage


@dataclass(frozen=True)
class EventCheckpoint:
    """Keep these three namespaces separate; scope this state to one server/token."""

    last_event_id: str | None = None
    pull_cursor: int = 0
    seen_ids: tuple[int, ...] = ()


class EventStateStore(Protocol):
    """Host adapter: save atomically; contains IDs only, never notification prose."""

    async def load(self) -> EventCheckpoint | None: ...
    async def save(self, checkpoint: EventCheckpoint) -> None: ...


class MemoryEventState:
    def __init__(self, checkpoint: EventCheckpoint | None = None) -> None:
        self.checkpoint = checkpoint

    async def load(self) -> EventCheckpoint | None:
        return self.checkpoint

    async def save(self, checkpoint: EventCheckpoint) -> None:
        self.checkpoint = checkpoint


@dataclass(frozen=True)
class EventOptions:
    heartbeat_timeout: float = 45.0
    poll_interval: float = 60.0
    backoff_initial: float = 0.5
    backoff_max: float = 30.0
    dedup_capacity: int = 2048
    max_frame_chars: int = 262144

    def __post_init__(self) -> None:
        values = (
            self.heartbeat_timeout,
            self.poll_interval,
            self.backoff_initial,
            self.backoff_max,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("event intervals must be finite and positive")
        if self.backoff_initial > self.backoff_max:
            raise ValueError("backoff_initial must not exceed backoff_max")
        if any(type(v) is not int or v < 1 for v in (self.dedup_capacity, self.max_frame_chars)):
            raise ValueError("event bounds must be positive integers")


EventReason = Literal[
    "connected",
    "events_unsupported",
    "invalid_content_type",
    "cursor_reset",
    "resync",
    "disconnected",
    "heartbeat_timeout",
    "network_error",
    "invalid_event",
    "rate_limited",
    "unavailable",
    "poll_error",
    "unauthorized",
    "forbidden",
    "request_rejected",
    "closed",
]


@dataclass(frozen=True)
class EventStatus:
    mode: Literal["sse", "poll", "stopped"]
    reason: EventReason


@dataclass(frozen=True)
class _Frame:
    event: str
    data: str
    id: str | None


class _Retry(Exception):
    def __init__(self, reason: EventReason, delay: float = 0.0) -> None:
        self.reason, self.delay = reason, delay


def _cursor(value: object) -> bool:
    return type(value) is int and value >= 0


def _event_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 1024
        and all(33 <= ord(char) <= 126 for char in value)
    )


async def _frames(response: httpx.Response, limit: int) -> AsyncGenerator[_Frame, None]:
    # Incremental UTF-8 and CR/LF parsing; httpx.aiter_lines alone does not bound a long line.
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    line, event = "", "message"
    data: list[str] = []
    event_id: str | None = None
    size, skip_lf = 0, False
    try:
        async for chunk in response.aiter_bytes():
            for char in decoder.decode(chunk):
                if skip_lf and char == "\n":
                    skip_lf = False
                    continue
                skip_lf = char == "\r"
                if char not in "\r\n":
                    line += char
                    size += 1
                    if size > limit:
                        raise _Retry("invalid_event")
                    continue
                if not line:
                    if data:
                        yield _Frame(event, "\n".join(data), event_id)
                    data, event, event_id, size = [], "message", None, 0
                elif not line.startswith(":"):
                    field, _, value = line.partition(":")
                    if value.startswith(" "):
                        value = value[1:]
                    if field == "data":
                        data.append(value)
                    elif field == "event":
                        event = value
                    elif field == "id" and "\x00" not in value:
                        event_id = value
                line = ""
        decoder.decode(b"", final=True)
        # EOF is not a blank line: an unfinished event must not move the checkpoint.
    except UnicodeError:
        raise _Retry("invalid_event") from None


class NotificationSubscription:
    """Await run() in one background task; callback success means durably accepted.

    The callback receives both goal_reminder and notifications found by pull repair.
    It must implement platform delivery/retry separately. aclose() cancels in-flight I/O.
    """

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        base_url: str,
        token: str,
        timeout: float,
        pull: Callable[[int], Awaitable[NotificationPage]],
        handler: Callable[[Notification], Awaitable[None]],
        state: EventStateStore | None = None,
        options: EventOptions = EventOptions(),
        on_status: Callable[[EventStatus], None] | None = None,
    ) -> None:
        self._http, self._base, self._token, self._timeout = http, base_url, token, timeout
        self._pull, self._handler = pull, handler
        self._store = state if state is not None else MemoryEventState()
        self._options, self._on_status = options, on_status
        self._checkpoint = EventCheckpoint()
        self._seen: set[int] = set()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._poll_only = False
        self._attempt = 0
        self._next_poll = 0.0
        self._poll_not_before = 0.0

    def _status(self, reason: EventReason, *, stopped: bool = False) -> None:
        if self._on_status is not None:
            # Observability must neither expose server text nor break consumption.
            with suppress(Exception):
                self._on_status(
                    EventStatus(
                        "stopped" if stopped else "poll" if self._poll_only else "sse", reason
                    )
                )

    async def _save(self, checkpoint: EventCheckpoint) -> None:
        try:
            await self._store.save(checkpoint)
        except Exception:
            raise IrisError("event_state_failed") from None
        self._checkpoint = checkpoint
        self._seen = set(checkpoint.seen_ids)

    async def _load(self) -> None:
        try:
            value = await self._store.load() or EventCheckpoint()
            if (
                not isinstance(value, EventCheckpoint)
                or not _cursor(value.pull_cursor)
                or (value.last_event_id is not None and not _event_id(value.last_event_id))
                or any(not _cursor(n) or n == 0 for n in value.seen_ids)
            ):
                raise ValueError()
            seen = tuple(dict.fromkeys(value.seen_ids))[-self._options.dedup_capacity :]
        except Exception:
            raise IrisError("event_state_failed") from None
        await self._save(replace(value, seen_ids=seen))

    async def _deliver(self, value: object) -> None:
        if (
            not isinstance(value, dict)
            or not _cursor(value.get("id"))
            or value["id"] == 0
            or not isinstance(value.get("kind"), str)
        ):
            raise _Retry("invalid_event")
        notification = cast(Notification, value)
        nid = notification["id"]
        # Pull's watermark also suppresses replay older than the bounded ID set.
        if nid <= self._checkpoint.pull_cursor or nid in self._seen:
            return
        try:
            await self._handler(notification)
        except Exception:
            raise IrisError("event_handler_failed") from None
        ids = (*self._checkpoint.seen_ids, nid)[-self._options.dedup_capacity :]
        await self._save(replace(self._checkpoint, seen_ids=ids))

    async def _repair(self) -> None:
        if asyncio.get_running_loop().time() < self._poll_not_before:
            raise _Retry("poll_error", self._poll_not_before - asyncio.get_running_loop().time())
        while not self._closed:
            after = self._checkpoint.pull_cursor
            try:
                page = await self._pull(after)
            except IrisError as error:
                self._fatal(error)
                delay = max(self._options.poll_interval, error.retry_after or 0.0)
                self._poll_not_before = asyncio.get_running_loop().time() + delay
                raise _Retry("poll_error", error.retry_after or 0.0) from None
            if (
                not isinstance(page, dict)
                or not isinstance(page.get("items"), list)
                or not _cursor(page.get("next_cursor"))
                or page["next_cursor"] < after
                or type(page.get("has_more")) is not bool
                or (page["has_more"] and page["next_cursor"] == after)
            ):
                raise _Retry("invalid_event")
            for item in page["items"]:
                if self._closed:
                    return
                await self._deliver(item)
            await self._save(replace(self._checkpoint, pull_cursor=page["next_cursor"]))
            if not page["has_more"]:
                self._next_poll = asyncio.get_running_loop().time() + self._options.poll_interval
                return

    def _fatal(self, error: IrisError) -> None:
        reason: EventReason | None = None
        if error.status_code in {401, 403}:
            reason = "unauthorized" if error.status_code == 401 else "forbidden"
        elif error.status_code is not None and error.status_code != 429 and error.status_code < 500:
            reason = "request_rejected"
        if reason:
            self._status(reason, stopped=True)
            # Fixed classifications only: no server-controlled error body/code in logs.
            raise IrisError(reason, status_code=error.status_code) from None

    async def _frame(self, frame: _Frame) -> None:
        if frame.event not in {"reminder", "heartbeat", "resync"}:
            return
        if not _event_id(frame.id):
            raise _Retry("invalid_event")
        try:
            data = json.loads(frame.data)
        except ValueError:
            raise _Retry("invalid_event") from None
        if not isinstance(data, dict):
            raise _Retry("invalid_event")
        if frame.event == "reminder":
            if data.get("kind") != "goal_reminder":
                raise _Retry("invalid_event")
            await self._deliver(data)
            await self._repair()
        elif frame.event == "resync":
            self._status("resync")
            # Never follow pull_url: repair uses the authenticated, fixed v1 endpoint.
            await self._repair()
        await self._save(replace(self._checkpoint, last_event_id=frame.id))

    async def _stream(self) -> None:
        headers = {"Authorization": "Bearer " + self._token, "Accept": "text/event-stream"}
        if self._checkpoint.last_event_id is not None:
            headers["Last-Event-ID"] = self._checkpoint.last_event_id
        request = self._http.build_request(
            "GET",
            self._base + compat.PREFIX + "/events",
            headers=headers,
            timeout=httpx.Timeout(self._timeout, read=None),
        )
        async with asyncio.timeout(self._timeout):
            response = await self._http.send(request, stream=True, follow_redirects=False)
        try:
            if response.status_code != 200:
                # Bound error response reads by the ordinary request timeout.
                async with asyncio.timeout(self._timeout):
                    await response.aread()
                try:
                    body = response.json()
                except ValueError:
                    body = None
                error = compat.error_response(
                    response.status_code, body, response.headers.get("Retry-After"), self._token
                )
                if error.status_code in {404, 405, 501}:
                    self._poll_only = True
                    self._status("events_unsupported")
                    return
                if error.status_code == 400 and self._checkpoint.last_event_id is not None:
                    await self._save(replace(self._checkpoint, last_event_id=None))
                    raise _Retry("cursor_reset")
                self._fatal(error)
                if error.status_code == 429 and error.code not in {
                    "event_token_limit",
                    "event_global_limit",
                }:
                    self._poll_not_before = max(
                        self._poll_not_before,
                        asyncio.get_running_loop().time() + (error.retry_after or 0.0),
                    )
                raise _Retry(
                    "rate_limited" if error.status_code == 429 else "unavailable",
                    error.retry_after or 0.0,
                )
            if (
                response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                != "text/event-stream"
            ):
                self._poll_only = True
                self._status("invalid_content_type")
                return
            frames = _frames(response, self._options.max_frame_chars)
            pending = asyncio.create_task(anext(frames))
            clock = asyncio.get_running_loop().time
            last_frame, opened = clock(), clock()
            try:
                await self._repair()
                self._status("connected")
                while not self._closed:
                    poll_due = max(self._next_poll, self._poll_not_before)
                    deadline = min(last_frame + self._options.heartbeat_timeout, poll_due)
                    done, _ = await asyncio.wait({pending}, timeout=max(0.0, deadline - clock()))
                    if done:
                        try:
                            frame = pending.result()
                        except StopAsyncIteration:
                            raise _Retry("disconnected") from None
                        last_frame = clock()
                        await self._frame(frame)
                        if clock() - opened >= self._options.heartbeat_timeout:
                            self._attempt = 0
                        pending = asyncio.create_task(anext(frames))
                    elif clock() >= last_frame + self._options.heartbeat_timeout:
                        raise _Retry("heartbeat_timeout")
                    else:
                        await self._repair()
            finally:
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration, _Retry, httpx.HTTPError):
                    await pending
                await frames.aclose()
        finally:
            await response.aclose()

    async def _wait(self, seconds: float) -> None:
        # Continue low-frequency pull repair even during a long SSE Retry-After.
        clock = asyncio.get_running_loop().time
        until = clock() + seconds
        while not self._closed and clock() < until:
            due = max(self._next_poll, self._poll_not_before)
            await asyncio.sleep(max(0.0, min(until, due) - clock()))
            if clock() >= due:
                await self._try_repair()

    async def _try_repair(self) -> None:
        try:
            await self._repair()
        except _Retry as error:
            self._status(error.reason)
            self._next_poll = asyncio.get_running_loop().time() + self._options.poll_interval

    async def run(self) -> None:
        if self._task is not None:
            raise RuntimeError("subscription is already running")
        if self._closed:
            return
        self._task = asyncio.current_task()
        try:
            await self._load()
            while not self._closed:
                if self._poll_only:
                    if asyncio.get_running_loop().time() >= max(
                        self._next_poll, self._poll_not_before
                    ):
                        await self._try_repair()
                    await self._wait(self._options.poll_interval)
                    continue
                try:
                    await self._stream()
                    if self._poll_only:
                        continue
                    break
                except (httpx.HTTPError, TimeoutError):
                    error = _Retry("network_error")
                except _Retry as caught:
                    error = caught
                self._status(error.reason)
                await self._try_repair()
                base = min(
                    self._options.backoff_max,
                    self._options.backoff_initial * 2 ** min(self._attempt, 30),
                )
                self._attempt += 1
                await self._wait(max(error.delay, random.uniform(base / 2, base)))
        except asyncio.CancelledError:
            if not self._closed:
                raise
        finally:
            self._closed = True
            self._task = None
            self._status("closed", stopped=True)

    async def aclose(self) -> None:
        self._closed = True
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
