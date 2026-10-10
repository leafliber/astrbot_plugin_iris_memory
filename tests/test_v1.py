import asyncio
import json
from pathlib import Path

import pytest
import respx

from iris_memory.vendor.iris_client import AsyncIrisClient, IrisClient, IrisError, RetryPolicy

CONTRACT = json.loads((Path(__file__).parent / "fixtures/host-v1-examples.json").read_text())
METHODS = {
    ("POST", "/entries/{entry_id}/messages"): ("receive_message", ["group-a"]),
    ("POST", "/entries/{entry_id}/learn"): ("learn", ["group-a"]),
    ("POST", "/entries/{entry_id}/prepare"): ("prepare", ["group-a"]),
    ("POST", "/media"): ("upload_media", []),
    ("POST", "/memories/search"): ("search", []),
    ("POST", "/feedback"): ("feedback", ["recall-id", [1]]),
    ("GET", "/state"): ("get_state", []),
    ("PUT", "/state"): ("put_state", []),
    ("PATCH", "/state"): ("patch_state", []),
    ("DELETE", "/state"): ("delete_state", []),
    ("GET", "/goals"): ("list_goals", []),
    ("POST", "/goals"): ("create_goal", []),
    ("PATCH", "/goals/{goal_id}"): ("update_goal", [1]),
    ("GET", "/notifications"): ("pull_notifications", []),
    ("GET", "/status"): ("status", []),
}


@pytest.fixture(params=[IrisClient, AsyncIrisClient])
async def client(request):
    c = request.param(token="iris_test_token", retry=RetryPolicy(max_retries=0))
    yield c
    if isinstance(c, AsyncIrisClient):
        await c.aclose()
    else:
        c.close()


async def call(client, method, *args, **kwargs):
    value = getattr(client, method)(*args, **kwargs)
    return await value if asyncio.iscoroutine(value) else value


@pytest.mark.parametrize(
    "op", CONTRACT["operations"], ids=lambda op: op["method"] + " " + op["path"]
)
async def test_frozen_openapi_examples(client, op):
    path = op["path"].replace("{entry_id}", "group-a").replace("{goal_id}", "1")
    method, prefix = METHODS[op["method"], op["path"].removeprefix("/api/v1")]
    args = prefix[:]
    if op["request"] is not None and method != "feedback":
        args.append(op["request"])
    status, expected = next((int(k), v) for k, v in op["responses"].items() if k.startswith("2"))
    with respx.mock as router:
        route = router.route(method=op["method"], url="http://127.0.0.1:8080" + path).respond(
            status, json=expected
        )
        assert await call(client, method, *args) == expected
        if op["request"] is not None:
            assert json.loads(route.calls[0].request.content) == op["request"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 429, 503])
async def test_frozen_error_envelope(client, status):
    body = CONTRACT["operations"][0]["responses"][str(status)]
    with respx.mock as router:
        router.get("http://127.0.0.1:8080/api/v1/status").respond(
            status, json=body, headers={"Retry-After": "7"}
        )
        with pytest.raises(IrisError) as caught:
            await call(client, "status")
    error, expected = caught.value, body["error"]
    assert error.code == expected["code"]
    assert error.message == expected["message"]
    assert error.fields == expected["fields"]
    assert error.retry_after_seconds == expected["retry_after_seconds"]
    assert error.retry_at == expected["retry_at"]
    assert error.retry_after == 7


@pytest.mark.parametrize("scope", [None, "private"])
async def test_search_body_scope(client, scope):
    with respx.mock as router:
        route = router.post("http://127.0.0.1:8080/api/v1/memories/search").respond(json={})
        await call(client, "search", {"entry_id": scope})
        assert json.loads(route.calls[0].request.content) == {"entry_id": scope}
    with pytest.raises(ValueError, match="conflicting"):
        await call(client, "search", {"entry_id": "a"}, entry_id="b")


async def test_media_reference_and_unknown_hint(client):
    hint = {
        "code": "other_entries_pending",
        "entry_count": 2,
        "time_from": "2026-10-10T12:10:00+00:00",
        "time_to": "2026-10-10T12:13:00+00:00",
        "time_basis": "received_at",
        "message": "另有待学习消息",
    }
    with respx.mock as router:
        route = router.post("http://127.0.0.1:8080/api/v1/entries/e/messages").respond(json={})
        await call(client, "receive_message", "e", {"media_ids": ["media-object-id"]})
        assert json.loads(route.calls[0].request.content)["media_ids"] == ["media-object-id"]
        router.post("http://127.0.0.1:8080/api/v1/entries/e/prepare").respond(
            json={"hints": [hint, {"code": "future"}]}
        )
        assert (await call(client, "prepare", "e"))["hints"] == [hint, {"code": "future"}]


@pytest.mark.parametrize("status", [429, 503])
async def test_retry_uses_body_fallback_and_header_precedence(status):
    import httpx

    waits = []
    body = {
        "error": {
            "code": "unavailable",
            "message": "稍后重试",
            "fields": [],
            "retry_after_seconds": 2,
            "retry_at": "2026-10-10T08:00:00+08:00",
        }
    }
    with respx.mock as router, IrisClient(token="iris_test_token", sleep=waits.append) as client:
        router.get("http://127.0.0.1:8080/api/v1/status").mock(
            side_effect=[
                httpx.Response(status, json=body),
                httpx.Response(status, json=body, headers={"Retry-After": "3"}),
                httpx.Response(200, json={}),
            ]
        )
        client.status()
    assert waits == [2, 3]


@pytest.mark.parametrize(
    "method,args",
    [
        ("patch_state", [{}]),
        ("put_state", [{"activity": "a"}]),
        ("delete_state", []),
        ("update_goal", [1, {"expected_revision": 1}]),
    ],
)
async def test_uncertain_mutation_requires_readback(client, method, args):
    client._retry = RetryPolicy(max_retries=2, backoff_seconds=0)
    import httpx

    with respx.mock as router:
        route = router.route().mock(side_effect=httpx.ReadTimeout("iris_test_token"))
        with pytest.raises(IrisError) as error:
            await call(client, method, *args)
        assert error.value.code == "network_error"
        assert route.call_count == 1
