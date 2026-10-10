import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
import respx

from iris_memory.vendor.iris_client import (
    AsyncIrisClient,
    IrisClient,
    IrisError,
    NotificationCursor,
    ProtocolError,
    RetryPolicy,
    TransportError,
)

BASE = "http://127.0.0.1:8080/api/v1"
TOKEN = "iris_test_token"
MESSAGE = {
    "sender": "小林",
    "account_id": "lin-1",
    "content": "你好",
    "occurred_at": "2026-10-10T10:00:00+08:00",
    "dedupe_key": "test-message",
    "quote_author": "小赵",
    "quote_author_account_id": "zhao-2",
    "quote_content": "早上好",
}


async def invoke(client, name, *args, **kwargs):
    result = getattr(client, name)(*args, **kwargs)
    return await result if asyncio.iscoroutine(result) else result


@pytest.fixture(params=[IrisClient, AsyncIrisClient], ids=["sync", "async"])
async def client(request):
    value = request.param(token=TOKEN)
    yield value
    if isinstance(value, AsyncIrisClient):
        await value.aclose()
    else:
        value.close()


@pytest.mark.parametrize(
    "name,args,kwargs,method,path,body,query",
    [
        (
            "receive_message",
            ["群 A", MESSAGE],
            {},
            "POST",
            "/entries/%E7%BE%A4%20A/messages",
            MESSAGE,
            {},
        ),
        (
            "receive_messages",
            ["e", [MESSAGE, {**MESSAGE, "dedupe_key": "second"}]],
            {},
            "POST",
            "/entries/e/messages",
            [MESSAGE, {**MESSAGE, "dedupe_key": "second"}],
            {},
        ),
        ("learn", ["e"], {}, "POST", "/entries/e/learn", None, {}),
        (
            "prepare",
            [
                "e",
                {
                    "text": "",
                    "participants": [],
                    "known_memory_ids": [3],
                    "recent_limit": 0,
                    "memory_limit": 2,
                    "goal_limit": 1,
                    "token_budget": 400,
                    "judge": False,
                },
            ],
            {},
            "POST",
            "/entries/e/prepare",
            {
                "text": "",
                "participants": [],
                "known_memory_ids": [3],
                "recent_limit": 0,
                "memory_limit": 2,
                "goal_limit": 1,
                "token_budget": 400,
                "judge": False,
            },
            {},
        ),
        ("prepare", ["e"], {}, "POST", "/entries/e/prepare", {}, {}),
        (
            "search",
            [
                {
                    "text": "茶",
                    "people": ["p1"],
                    "kinds": ["偏好"],
                    "stances": ["转述"],
                    "include_forgotten": True,
                    "include_goals": True,
                    "include_state": True,
                    "limit": 5,
                }
            ],
            {},
            "POST",
            "/memories/search",
            {
                "text": "茶",
                "people": ["p1"],
                "kinds": ["偏好"],
                "stances": ["转述"],
                "include_forgotten": True,
                "include_goals": True,
                "include_state": True,
                "limit": 5,
            },
            {},
        ),
        (
            "feedback",
            ["recall", [1, 2]],
            {},
            "POST",
            "/feedback",
            {"recall_id": "recall", "memory_ids": [1, 2]},
            {},
        ),
        ("get_state", [], {}, "GET", "/state", None, {}),
        (
            "put_state",
            [{"activity": "散步", "details": {"place": "公园"}, "mood": None}],
            {},
            "PUT",
            "/state",
            {"activity": "散步", "details": {"place": "公园"}, "mood": None},
            {},
        ),
        (
            "patch_state",
            [{"details": {"place": None}, "mood": None}],
            {},
            "PATCH",
            "/state",
            {"details": {"place": None}, "mood": None},
            {},
        ),
        ("patch_state", [], {}, "PATCH", "/state", {}, {}),
        ("delete_state", [], {}, "DELETE", "/state", None, {}),
        ("delete_state", [{"entry_id": "e"}], {}, "DELETE", "/state", {"entry_id": "e"}, {}),
        (
            "list_goals",
            [{"overdue": False, "due_soon": True, "state": "open", "entry_id": "e", "offset": 2}],
            {},
            "GET",
            "/goals",
            None,
            {
                "overdue": "false",
                "due_soon": "true",
                "state": "open",
                "entry_id": "e",
                "offset": "2",
            },
        ),
        (
            "create_goal",
            [{"content": "喝水", "host_key": "test-goal"}],
            {},
            "POST",
            "/goals",
            {"content": "喝水", "host_key": "test-goal"},
            {},
        ),
        (
            "update_goal",
            [4, {"state": "completed", "expected_revision": 2, "deadline": None}],
            {},
            "PATCH",
            "/goals/4",
            {"state": "completed", "expected_revision": 2, "deadline": None},
            {},
        ),
        (
            "pull_notifications",
            [],
            {"after": 12, "limit": 5},
            "GET",
            "/notifications",
            None,
            {"after": "12", "limit": "5"},
        ),
        ("status", [], {}, "GET", "/status", None, {}),
    ],
)
async def test_all_methods(client, name, args, kwargs, method, path, body, query):
    with respx.mock as router:
        route = router.route(method=method, url=BASE + path, params=query).respond(
            json={"extra": "retained"}
        )
        assert await invoke(client, name, *args, **kwargs) == {"extra": "retained"}
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        assert (json.loads(request.content) if request.content else None) == body


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "invalid_request"),
        (401, "unauthorized"),
        (403, "forbidden"),
        (404, "not_found"),
        (409, "revision_conflict"),
        (413, "payload_too_large"),
        (500, "http_error"),
    ],
)
async def test_errors_no_retry(client, status, code):
    with respx.mock as router:
        route = router.get(BASE + "/status").respond(status, text="not JSON")
        with pytest.raises(IrisError) as caught:
            await invoke(client, "status")
        assert caught.value.status_code == status
        assert caught.value.code == code
        assert route.call_count == 1


async def test_error_details_are_redacted(client):
    with respx.mock as router:
        router.get(BASE + "/status").respond(
            409,
            json={
                "error": {
                    "code": "goal_merged",
                    "canonical_id": 5,
                    "message": TOKEN,
                    "fields": [{"field": TOKEN}],
                }
            },
        )
        with pytest.raises(IrisError) as caught:
            await invoke(client, "status")
        assert caught.value.code == "goal_merged"
        assert caught.value.details["canonical_id"] == 5
        assert TOKEN not in str(caught.value) + repr(caught.value.details)


@pytest.mark.parametrize("status,header", [(429, "2"), (503, "0")])
async def test_retries_honor_header(client, status, header):
    delays = []

    async def asleep(delay):
        delays.append(delay)

    client._sleep = asleep if isinstance(client, AsyncIrisClient) else delays.append
    with respx.mock as router:
        route = router.get(BASE + "/status").mock(
            side_effect=[
                httpx.Response(status, headers={"Retry-After": header}),
                httpx.Response(200, json={"service": "ready"}),
            ]
        )
        assert (await invoke(client, "status"))["service"] == "ready"
        assert delays == [float(header)]
        assert route.call_count == 2


async def test_http_date_retry_after(client):
    delays = []

    async def asleep(delay):
        delays.append(delay)

    client._sleep = asleep if isinstance(client, AsyncIrisClient) else delays.append
    header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
    with respx.mock as router:
        router.get(BASE + "/status").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": header}),
                httpx.Response(200, json={}),
            ]
        )
        await invoke(client, "status")
    assert 8 <= delays[0] <= 10


async def test_long_retry_after_returns_instead_of_retrying_early(client):
    with respx.mock as router:
        route = router.get(BASE + "/status").respond(429, headers={"Retry-After": "300"})
        with pytest.raises(IrisError) as caught:
            await invoke(client, "status")
        assert caught.value.retry_after == 300
        assert route.call_count == 1


async def test_finite_network_retries_and_no_raw_error(client):
    delays = []

    async def asleep(delay):
        delays.append(delay)

    client._sleep = asleep if isinstance(client, AsyncIrisClient) else delays.append
    with respx.mock as router:
        route = router.get(BASE + "/status").mock(side_effect=httpx.ConnectError(TOKEN))
        with pytest.raises(TransportError) as caught:
            await invoke(client, "status")
        assert route.call_count == 3
        assert delays == [0.25, 0.5]
        assert TOKEN not in str(caught.value)


async def test_goal_key_stable_within_retry(client):
    async def asleep(delay):
        pass

    client._sleep = asleep if isinstance(client, AsyncIrisClient) else lambda _: None
    with respx.mock as router:
        route = router.post(BASE + "/goals").mock(
            side_effect=[httpx.Response(503), httpx.Response(201, json={})]
        )
        goal = {"content": "喝水"}
        await invoke(client, "create_goal", goal)
        assert route.calls[0].request.content == route.calls[1].request.content
        assert json.loads(route.calls[0].request.content)["host_key"].startswith("iris-client:")
        assert "host_key" not in goal


async def test_retry_snapshots_body_before_await(client):
    message = dict(MESSAGE)

    def mutate(delay):
        message["content"] = "changed"

    async def asleep(delay):
        mutate(delay)

    client._sleep = asleep if isinstance(client, AsyncIrisClient) else mutate
    with respx.mock as router:
        route = router.post(BASE + "/entries/e/messages").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json={})]
        )
        await invoke(client, "receive_message", "e", message)
        assert route.calls[0].request.content == route.calls[1].request.content
        assert json.loads(route.calls[1].request.content)["content"] == MESSAGE["content"]


async def test_no_unscoped_fallback(client):
    with respx.mock as router:
        route = router.post(BASE + "/memories/search").respond(
            403, json={"error": {"code": "forbidden"}}
        )
        with pytest.raises(IrisError, match="forbidden"):
            await invoke(client, "search", {}, entry_id="private")
        assert route.call_count == 1
        assert json.loads(route.calls[0].request.content) == {"entry_id": "private"}


@pytest.mark.parametrize("body", [None, [], "text"])
async def test_invalid_success_body(client, body):
    with respx.mock as router:
        router.get(BASE + "/status").respond(json=body)
        with pytest.raises(ProtocolError):
            await invoke(client, "status")


async def test_no_redirect_with_token(client):
    with respx.mock as router:
        route = router.get(BASE + "/status").respond(
            307, headers={"Location": "https://example.invalid/"}
        )
        with pytest.raises(IrisError):
            await invoke(client, "status")
        assert route.call_count == 1


async def test_async_cancellation():
    async def transport(request):
        raise asyncio.CancelledError()

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        async with AsyncIrisClient(token=TOKEN, http_client=http) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.status()
        assert not http.is_closed


def test_cursor_explicit_commit_and_restart(tmp_path, monkeypatch):
    path = tmp_path / "cursor.json"
    cursor = NotificationCursor(path)
    page = {"items": [], "next_cursor": 8, "has_more": False}
    assert cursor.value == 0
    cursor.commit(page)
    assert NotificationCursor(path).value == 8
    with pytest.raises(ProtocolError):
        cursor.commit({**page, "next_cursor": 1})
    monkeypatch.setattr("os.replace", lambda *args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        cursor.commit({**page, "next_cursor": 9})
    assert cursor.value == NotificationCursor(path).value == 8


@pytest.mark.parametrize(
    "url", ["http://iris_test_token@localhost", "http://localhost/?q=1", "file:///tmp/a"]
)
def test_credential_free_origin(url):
    with pytest.raises(ValueError):
        IrisClient(url, token=TOKEN)


@pytest.mark.parametrize(
    "kwargs", [{"max_retries": -1}, {"backoff_seconds": float("nan")}, {"max_wait_seconds": -1}]
)
def test_invalid_retry_config(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)
