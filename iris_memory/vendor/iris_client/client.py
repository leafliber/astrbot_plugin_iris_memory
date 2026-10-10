"""Bounded HTTP transport; synchronous and asynchronous APIs share the wire policy."""

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self, cast
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from . import compat
from .errors import IrisError, TransportError
from .types import (
    CurrentState,
    FeedbackResult,
    Goal,
    GoalCreate,
    GoalList,
    GoalPatch,
    GoalQuery,
    GoalReceipt,
    JSONObject,
    LearnResult,
    Media,
    MediaUpload,
    Message,
    NotificationPage,
    Prepared,
    PrepareOptions,
    ReceiveResult,
    SearchOptions,
    SearchResult,
    ServiceStatus,
    StatePatch,
    StatePut,
    StateSource,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 0.25
    max_wait_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        if any(
            not math.isfinite(v) or v < 0 for v in (self.backoff_seconds, self.max_wait_seconds)
        ):
            raise ValueError("retry delays must be finite and nonnegative")

    def delay(self, attempt: int, error: IrisError) -> float | None:
        if attempt >= self.max_retries:
            return None
        delay = error.retry_after
        if delay is None:
            delay = self.backoff_seconds * 2**attempt
        # Do not retry earlier than the server asks. Long waits return to caller.
        return delay if delay <= self.max_wait_seconds else None


def _config(base_url: str, token: str, timeout: float) -> str:
    url = urlsplit(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError("base_url must be an HTTP origin without credentials")
    if not token or "\r" in token or "\n" in token:
        raise ValueError("a nonempty bearer token is required")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    return base_url.rstrip("/")


def _result(response: httpx.Response, token: str) -> JSONObject:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not 200 <= response.status_code < 300:
        raise compat.error_response(
            response.status_code, body, response.headers.get("Retry-After"), token
        )
    return compat.object_response(body)


def _params(query: Mapping[str, object] | None) -> dict[str, str]:
    return {
        k: str(v).lower() if isinstance(v, bool) else str(v)
        for k, v in (query or {}).items()
        if v is not None
    }


def _goal_body(goal: GoalCreate) -> GoalCreate:
    result = goal.copy()
    if not result.get("host_key"):
        result["host_key"] = "iris-client:" + uuid4().hex
    return result


class IrisClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        token: str,
        timeout: float = 10.0,
        retry: RetryPolicy = RetryPolicy(),
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base = _config(base_url, token, timeout)
        self._token, self._timeout, self._retry, self._sleep = token, timeout, retry, sleep
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(trust_env=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def _request(
        self, method: str, path: str, body: object = None, query: Mapping[str, object] | None = None
    ) -> JSONObject:
        encoded = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        params = _params(query)
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = self._http.request(
                    method,
                    self._base + path,
                    content=encoded,
                    params=params,
                    headers={
                        "Authorization": "Bearer " + self._token,
                        "Content-Type": "application/json",
                    },
                    timeout=self._timeout,
                    follow_redirects=False,
                )
                return _result(response, self._token)
            except httpx.RequestError:
                if method in {"PUT", "PATCH", "DELETE"}:
                    # An uncertain mutable write must be read back before replay.
                    raise TransportError() from None
                error: IrisError = TransportError()
            except IrisError as caught:
                if caught.status_code not in {429, 503}:
                    raise caught from None
                error = caught
            delay = self._retry.delay(attempt, error)
            if delay is None:
                raise error from None
            self._sleep(delay)
        raise AssertionError("unreachable")

    def receive_message(self, entry_id: str, message: Message) -> ReceiveResult:
        return cast(
            ReceiveResult, self._request("POST", compat.entry_path(entry_id, "messages"), message)
        )

    def receive_messages(self, entry_id: str, messages: Sequence[Message]) -> ReceiveResult:
        return cast(
            ReceiveResult,
            self._request("POST", compat.entry_path(entry_id, "messages"), list(messages)),
        )

    def upload_media(self, upload: MediaUpload) -> Media:
        """Upload standard Base64; a retry can leave an extra orphan object."""
        return cast(Media, self._request("POST", compat.PREFIX + "/media", upload))

    def learn(self, entry_id: str) -> LearnResult:
        return cast(LearnResult, self._request("POST", compat.entry_path(entry_id, "learn")))

    def prepare(self, entry_id: str, options: PrepareOptions | None = None) -> Prepared:
        return cast(
            Prepared, self._request("POST", compat.entry_path(entry_id, "prepare"), options or {})
        )

    def search(
        self, options: SearchOptions | None = None, *, entry_id: str | None = None
    ) -> SearchResult:
        return cast(
            SearchResult,
            self._request(
                "POST", compat.PREFIX + "/memories/search", compat.search_body(options, entry_id)
            ),
        )

    def feedback(self, recall_id: str, memory_ids: Sequence[int]) -> FeedbackResult:
        return cast(
            FeedbackResult,
            self._request(
                "POST",
                compat.PREFIX + "/feedback",
                {"recall_id": recall_id, "memory_ids": list(memory_ids)},
            ),
        )

    def get_state(self) -> CurrentState:
        return cast(CurrentState, self._request("GET", compat.PREFIX + "/state"))

    def put_state(self, state: StatePut) -> CurrentState:
        return cast(CurrentState, self._request("PUT", compat.PREFIX + "/state", state))

    def patch_state(self, patch: StatePatch | None = None) -> CurrentState:
        return cast(CurrentState, self._request("PATCH", compat.PREFIX + "/state", patch or {}))

    def delete_state(self, source: StateSource | None = None) -> CurrentState:
        return cast(CurrentState, self._request("DELETE", compat.PREFIX + "/state", source))

    def list_goals(self, query: GoalQuery | None = None) -> GoalList:
        return cast(GoalList, self._request("GET", compat.PREFIX + "/goals", query=query))

    def create_goal(self, goal: GoalCreate) -> GoalReceipt:
        return cast(GoalReceipt, self._request("POST", compat.PREFIX + "/goals", _goal_body(goal)))

    def update_goal(self, goal_id: int, patch: GoalPatch) -> Goal:
        return cast(Goal, self._request("PATCH", compat.PREFIX + f"/goals/{goal_id}", patch))

    def pull_notifications(self, *, after: int = 0, limit: int = 30) -> NotificationPage:
        return cast(
            NotificationPage,
            self._request(
                "GET", compat.PREFIX + "/notifications", query={"after": after, "limit": limit}
            ),
        )

    def status(self) -> ServiceStatus:
        return cast(ServiceStatus, self._request("GET", compat.PREFIX + "/status"))


class AsyncIrisClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        token: str,
        timeout: float = 10.0,
        retry: RetryPolicy = RetryPolicy(),
        http_client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._base = _config(base_url, token, timeout)
        self._token, self._timeout, self._retry, self._sleep = token, timeout, retry, sleep
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(trust_env=False)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def _request(
        self, method: str, path: str, body: object = None, query: Mapping[str, object] | None = None
    ) -> JSONObject:
        encoded = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        params = _params(query)
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await self._http.request(
                    method,
                    self._base + path,
                    content=encoded,
                    params=params,
                    headers={
                        "Authorization": "Bearer " + self._token,
                        "Content-Type": "application/json",
                    },
                    timeout=self._timeout,
                    follow_redirects=False,
                )
                return _result(response, self._token)
            except httpx.RequestError:
                if method in {"PUT", "PATCH", "DELETE"}:
                    raise TransportError() from None
                error: IrisError = TransportError()
            except IrisError as caught:
                if caught.status_code not in {429, 503}:
                    raise caught from None
                error = caught
            delay = self._retry.delay(attempt, error)
            if delay is None:
                raise error from None
            await self._sleep(delay)
        raise AssertionError("unreachable")

    async def receive_message(self, entry_id: str, message: Message) -> ReceiveResult:
        return cast(
            ReceiveResult,
            await self._request("POST", compat.entry_path(entry_id, "messages"), message),
        )

    async def receive_messages(self, entry_id: str, messages: Sequence[Message]) -> ReceiveResult:
        return cast(
            ReceiveResult,
            await self._request("POST", compat.entry_path(entry_id, "messages"), list(messages)),
        )

    async def upload_media(self, upload: MediaUpload) -> Media:
        """Upload standard Base64; a retry can leave an extra orphan object."""
        return cast(Media, await self._request("POST", compat.PREFIX + "/media", upload))

    async def learn(self, entry_id: str) -> LearnResult:
        return cast(LearnResult, await self._request("POST", compat.entry_path(entry_id, "learn")))

    async def prepare(self, entry_id: str, options: PrepareOptions | None = None) -> Prepared:
        return cast(
            Prepared,
            await self._request("POST", compat.entry_path(entry_id, "prepare"), options or {}),
        )

    async def search(
        self, options: SearchOptions | None = None, *, entry_id: str | None = None
    ) -> SearchResult:
        return cast(
            SearchResult,
            await self._request(
                "POST", compat.PREFIX + "/memories/search", compat.search_body(options, entry_id)
            ),
        )

    async def feedback(self, recall_id: str, memory_ids: Sequence[int]) -> FeedbackResult:
        return cast(
            FeedbackResult,
            await self._request(
                "POST",
                compat.PREFIX + "/feedback",
                {"recall_id": recall_id, "memory_ids": list(memory_ids)},
            ),
        )

    async def get_state(self) -> CurrentState:
        return cast(CurrentState, await self._request("GET", compat.PREFIX + "/state"))

    async def put_state(self, state: StatePut) -> CurrentState:
        return cast(CurrentState, await self._request("PUT", compat.PREFIX + "/state", state))

    async def patch_state(self, patch: StatePatch | None = None) -> CurrentState:
        return cast(
            CurrentState, await self._request("PATCH", compat.PREFIX + "/state", patch or {})
        )

    async def delete_state(self, source: StateSource | None = None) -> CurrentState:
        return cast(CurrentState, await self._request("DELETE", compat.PREFIX + "/state", source))

    async def list_goals(self, query: GoalQuery | None = None) -> GoalList:
        return cast(GoalList, await self._request("GET", compat.PREFIX + "/goals", query=query))

    async def create_goal(self, goal: GoalCreate) -> GoalReceipt:
        return cast(
            GoalReceipt, await self._request("POST", compat.PREFIX + "/goals", _goal_body(goal))
        )

    async def update_goal(self, goal_id: int, patch: GoalPatch) -> Goal:
        return cast(Goal, await self._request("PATCH", compat.PREFIX + f"/goals/{goal_id}", patch))

    async def pull_notifications(self, *, after: int = 0, limit: int = 30) -> NotificationPage:
        return cast(
            NotificationPage,
            await self._request(
                "GET", compat.PREFIX + "/notifications", query={"after": after, "limit": limit}
            ),
        )

    async def status(self) -> ServiceStatus:
        return cast(ServiceStatus, await self._request("GET", compat.PREFIX + "/status"))
