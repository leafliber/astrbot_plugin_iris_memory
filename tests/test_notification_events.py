"""SSE HTTP boundary plus durable host acceptance and one delivery channel."""

import asyncio
import json
import stat
from dataclasses import replace

import httpx
import pytest

from iris_memory.bridge import Bridge
from iris_memory.config import Config
from iris_memory.store import Inbox
from iris_memory.vendor.iris_client import AsyncIrisClient, RetryPolicy


async def eventually(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class EventStream(httpx.AsyncByteStream):
    def __init__(self):
        self.frames = asyncio.Queue()
        self.closed = False

    async def __aiter__(self):
        while (frame := await self.frames.get()) is not None:
            yield frame

    async def aclose(self):
        self.closed = True

    def emit(self, event, data, event_id):
        self.frames.put_nowait(
            f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data)}\n\n".encode()
        )


class FakeEvents:
    def __init__(self):
        self.streams = []
        self.resumes = []
        self.pulls = []
        self.notifications = []
        self.outputs = []
        self.event_status = 200
        self.pull_status = 200
        self.page_size = 100

    async def request(self, request):
        assert request.headers["Authorization"] == "Bearer iris_test_token"
        path = request.url.path
        if path.endswith("/events"):
            self.resumes.append(request.headers.get("Last-Event-ID"))
            if self.event_status != 200:
                return httpx.Response(self.event_status, json={"error": {"code": "test_failure"}})
            stream = EventStream()
            self.streams.append(stream)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)
        if path.endswith("/notifications"):
            after = int(request.url.params["after"])
            self.pulls.append(after)
            if self.pull_status != 200:
                return httpx.Response(self.pull_status, json={"error": {"code": "test_failure"}})
            items = [n for n in self.notifications if n["id"] > after]
            limit = min(self.page_size, int(request.url.params["limit"]))
            page = items[:limit]
            return httpx.Response(
                200,
                json={
                    "items": page,
                    "next_cursor": page[-1]["id"] if page else after,
                    "has_more": len(items) > len(page),
                },
            )
        if path.endswith("/messages"):
            self.outputs.append(json.loads(request.content))
            return httpx.Response(200, json={"message_ids": [1]})
        raise AssertionError("unexpected fake Iris endpoint")

    def publish(self, number=1, *, event_id="test-instance:99", content="喝水"):
        notice = {"id": number, "kind": "goal_reminder", "goal_id": number, "content": content}
        self.notifications.append(notice)
        for stream in self.streams:
            if not stream.closed:
                stream.emit("reminder", notice, event_id)
        return notice


@pytest.fixture
async def event_bridge(tmp_path):
    resources = []

    def create(*, service=None, send=None, directory=None, **overrides):
        service = service or FakeEvents()
        sends, logs = [], []

        async def default_send(target, text):
            sends.append((target, text))
            return True

        http = httpx.AsyncClient(transport=httpx.MockTransport(service.request))
        client = AsyncIrisClient(
            token="iris_test_token", http_client=http, timeout=1, retry=RetryPolicy(max_retries=0)
        )
        config = replace(
            Config(
                token="iris_test_token",
                images_enabled=False,
                reminder_interval=0.05,
                request_timeout=1,
                reminder_mode="active",
                reminder_target="bot:GroupMessage:g",
                reminder_entry_id="astrbot:bot:group:g",
            ),
            **overrides,
        )
        bridge = Bridge(
            config,
            client,
            directory or tmp_path / str(len(resources)),
            logs.append,
            send or default_send,
        )
        bridge.start()
        resources.append((bridge, http))
        return bridge, service, sends, logs

    yield create
    for bridge, http in reversed(resources):
        await bridge.close()
        await http.aclose()


async def test_sse_enqueue_before_delivery_and_poll_does_not_send_twice(event_bridge):
    release, sending = asyncio.Event(), asyncio.Event()
    sent = []

    async def send(target, text):
        sending.set()
        await release.wait()
        sent.append(text)
        return True

    bridge, service, _, logs = event_bridge(send=send, reminder_interval=60)
    await eventually(lambda: bridge.diagnostics.reminder_transport["mode"] == "sse")
    service.publish()
    await asyncio.wait_for(sending.wait(), 1)
    assert bridge.inbox.count() == 1  # Durable before platform delivery.
    replay = asyncio.create_task(bridge.poll())
    release.set()
    await replay
    await eventually(lambda: len(service.outputs) == 1)
    await bridge.poll()
    assert sent == ["提醒：喝水"]
    assert service.outputs[0]["kind"] == "self_output"
    assert not logs
    assert bridge.diagnostics.counts["reminder_enqueued"] == 1


async def test_context_push_is_persisted_until_a_reply_consumes_it(event_bridge):
    bridge, service, sends, _ = event_bridge(reminder_mode="context", reminder_interval=60)
    await eventually(lambda: bool(service.streams))
    service.publish(content="PRIVATE_REMINDER_BODY")
    await eventually(lambda: bridge.inbox.count() == 1)
    assert not sends
    reopened = Inbox(bridge.data_dir / next(bridge.data_dir.glob("notifications-*.sqlite3")).name)
    assert reopened.pending()[0]["content"] == "PRIVATE_REMINDER_BODY"
    reopened.close()
    await eventually(lambda: bridge.inbox.cursor == 1)
    for path in bridge.data_dir.glob("*.sqlite3"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert b"iris_test_token" not in path.read_bytes()
        if path.name.startswith("events-"):
            assert b"PRIVATE_REMINDER_BODY" not in path.read_bytes()


async def test_resume_separate_cursor_and_paged_resync(event_bridge):
    bridge, service, sends, _ = event_bridge(reminder_interval=60)
    await eventually(lambda: bool(service.streams))
    service.publish(event_id="test-instance:999")
    await eventually(lambda: len(sends) == 1 and bridge.inbox.cursor == 1)
    await eventually(
        lambda: bridge.reminders.state.load_sync().last_event_id == "test-instance:999"
    )
    service.streams[-1].frames.put_nowait(None)
    await eventually(lambda: len(service.resumes) == 2)
    assert service.resumes[-1] == "test-instance:999"
    assert service.pulls[-1] == 1
    service.page_size = 1
    service.notifications.extend(
        [
            {"id": 2, "kind": "goal_reminder", "content": "第二条"},
            {"id": 3, "kind": "goal_reminder", "content": "第三条"},
        ]
    )
    service.streams[-1].emit("resync", {}, "new-instance:1")
    await eventually(lambda: len(sends) == 3 and bridge.inbox.cursor == 3)
    await eventually(lambda: bridge.reminders.state.load_sync().last_event_id == "new-instance:1")
    assert service.pulls[-2:] == [1, 2]
    assert bridge.reminders.state.load_sync().pull_cursor == 3


@pytest.mark.parametrize("status", [404, 405, 501])
async def test_unsupported_sse_uses_client_poll_fallback(event_bridge, status):
    service = FakeEvents()
    service.event_status = status
    bridge, _, sends, _ = event_bridge(service=service)
    await eventually(lambda: bridge.diagnostics.reminder_transport["mode"] == "poll")
    service.publish()
    await eventually(lambda: len(sends) == 1)
    await asyncio.sleep(0.12)
    assert len(sends) == 1 and len(service.resumes) == 1
    assert bridge.diagnostics.reminder_transport["reason"] == "events_unsupported"


@pytest.mark.parametrize("status,reason", [(401, "unauthorized"), (403, "forbidden")])
@pytest.mark.parametrize("source", ["sse", "poll", "legacy_poll"])
async def test_auth_failure_stops_and_preserves_reason(event_bridge, status, reason, source):
    service = FakeEvents()
    if source == "sse":
        service.event_status = status
    else:
        service.pull_status = status
    bridge, _, _, logs = event_bridge(
        service=service, reminder_push_enabled=source != "legacy_poll"
    )
    await eventually(lambda: bridge.diagnostics.reminder_transport["reason"] == reason)
    before = len(service.resumes), len(service.pulls)
    await asyncio.sleep(0.15)
    assert (len(service.resumes), len(service.pulls)) == before
    assert bridge.diagnostics.reminder_transport["mode"] == "stopped"
    assert "检查令牌" in bridge.diagnostics.reminder_transport["hint"]
    assert logs == ["Iris: reminder_auth_failed"]


async def test_push_disabled_keeps_poll_and_unload_closes_subscription(event_bridge):
    bridge, service, sends, _ = event_bridge(reminder_push_enabled=False)
    service.publish()
    await eventually(lambda: len(sends) == 1)
    assert not service.streams
    assert bridge.diagnostics.reminder_transport["reason"] == "push_disabled"
    await bridge.close()
    assert all(task.done() for task in bridge.tasks + bridge.reminders.tasks)
    second, service2, _, _ = event_bridge()
    await eventually(lambda: bool(service2.streams))
    tasks = list(second.reminders.tasks)
    second.start()
    assert second.reminders.tasks == tasks
    await second.close()
    assert all(task.done() for task in second.tasks + tasks)
    assert all(stream.closed for stream in service2.streams)
    assert second.diagnostics.reminder_transport["mode"] == "stopped"


def test_inbox_acceptance_survives_crash_before_event_checkpoint(tmp_path):
    path = tmp_path / "notifications.sqlite3"
    notice = {"id": 5, "kind": "goal_reminder", "content": "喝水"}
    inbox = Inbox(path)
    assert inbox.accept(notice)
    inbox.consume([5])
    inbox.close()
    inbox = Inbox(path)
    assert not inbox.accept(notice)
    assert not inbox.pending()
    inbox.save_page({"items": [notice], "next_cursor": 5, "has_more": False})
    assert not inbox.accept(notice)
    assert not inbox.pending() and inbox.cursor == 5
    inbox.close()


async def test_restart_resumes_ids_without_replaying_a_sent_notice(event_bridge, tmp_path):
    directory = tmp_path / "restart"
    bridge, service, sends, _ = event_bridge(directory=directory, reminder_interval=60)
    await eventually(lambda: bool(service.streams))
    service.publish(event_id="instance:27")
    await eventually(lambda: len(sends) == 1 and bridge.inbox.cursor == 1)
    await eventually(lambda: bridge.reminders.state.load_sync().last_event_id == "instance:27")
    await bridge.close()
    second, _, second_sends, _ = event_bridge(service=service, directory=directory)
    await eventually(lambda: len(service.streams) == 2)
    service.streams[-1].emit("reminder", service.notifications[0], "instance:28")
    await eventually(lambda: second.reminders.state.load_sync().last_event_id == "instance:28")
    assert service.resumes == [None, "instance:27"]
    assert not second_sends and not second.inbox.pending()


async def test_failed_platform_send_stays_queued_and_does_not_backflow(event_bridge):
    allowed = False
    sends = []

    async def send(target, text):
        if not allowed:
            return False
        sends.append(text)
        return True

    bridge, service, _, _ = event_bridge(send=send)
    await eventually(lambda: bool(service.streams))
    service.publish()
    await eventually(lambda: bridge.counts.get("reminder_send_unconfirmed", 0) > 0)
    assert bridge.inbox.count() == 1 and not service.outputs
    allowed = True
    await eventually(lambda: len(service.outputs) == 1)
    await bridge.poll()
    assert sends == ["提醒：喝水"] and bridge.inbox.count() == 0


async def test_backpressure_does_not_acknowledge_or_lose_notifications(event_bridge):
    bridge, service, _, _ = event_bridge(reminder_mode="context")
    bridge.inbox.capacity = 1
    await eventually(lambda: bool(service.streams))
    service.publish(1)
    await eventually(lambda: bridge.inbox.cursor == 1)
    service.publish(2, event_id="instance:2")
    await eventually(lambda: bridge.reminders.fallback)
    assert bridge.inbox.cursor == 1
    assert bridge.reminders.state.load_sync().last_event_id != "instance:2"
    bridge.inbox.consume([1])
    await eventually(lambda: bridge.inbox.cursor == 2)
    assert [n["id"] for n in bridge.inbox.pending()] == [2]


async def test_state_metadata_is_bounded_and_independent_of_pending_bodies(tmp_path):
    from iris_memory.reminders import EventState
    from iris_memory.vendor.iris_client import EventCheckpoint

    inbox = Inbox(tmp_path / "notifications.sqlite3")
    store = EventState(tmp_path / "events.sqlite3", inbox)
    assert inbox.accept({"id": 2500, "kind": "goal_reminder", "content": "PRIVATE_BODY"})
    await store.save(EventCheckpoint("test-instance:9000", 100, tuple(range(1, 2501))))
    state = await store.load()
    assert state.last_event_id == "test-instance:9000" and state.pull_cursor == 100
    assert len(state.seen_ids) == 2048
    assert inbox.cursor == 100 and inbox.count() == 1
    inbox.advance(101)  # Legacy pull during a mode switch cannot regress.
    await store.save(EventCheckpoint("test-instance:9001", 100, (2500,)))
    assert (await store.load()).pull_cursor == 101
    store.close()
    inbox.close()
    assert b"PRIVATE_BODY" not in (tmp_path / "events.sqlite3").read_bytes()


async def test_storage_failure_falls_back_without_logging_error_text(event_bridge, monkeypatch):
    def fail(*args):
        raise OSError("iris_test_token PRIVATE_BODY")

    monkeypatch.setattr("iris_memory.reminders.EventState", fail)
    bridge, service, sends, logs = event_bridge()
    service.publish()
    await eventually(lambda: len(sends) == 1)
    assert not service.resumes
    assert bridge.diagnostics.reminder_transport == {
        "mode": "poll",
        "reason": "state_failed",
        "hint": "",
    }
    assert logs == ["Iris: reminder_state_failed"]


async def test_reminder_total_switch_disables_both_network_modes(event_bridge):
    bridge, service, _, _ = event_bridge(reminders_enabled=False)
    await asyncio.sleep(0.06)
    assert not service.resumes and not service.pulls and not bridge.reminders.tasks
    assert bridge.diagnostics.reminder_transport["reason"] == "disabled"


async def test_heartbeat_timeout_falls_back_then_recovers(event_bridge, monkeypatch):
    from iris_memory.vendor.iris_client import EventOptions

    monkeypatch.setattr(
        "iris_memory.reminders.EventOptions",
        lambda **kwargs: EventOptions(
            **kwargs, heartbeat_timeout=0.04, backoff_initial=0.05, backoff_max=0.05
        ),
    )
    bridge, service, _, _ = event_bridge(reminder_interval=60)
    await eventually(lambda: bridge.diagnostics.reminder_transport["reason"] == "heartbeat_timeout")
    assert bridge.diagnostics.reminder_transport["mode"] == "poll"
    await eventually(lambda: len(service.streams) > 1)
    service.streams[-1].emit("heartbeat", {}, "test-instance:10")
    await eventually(lambda: bridge.reminders.state.load_sync().last_event_id == "test-instance:10")
    assert bridge.diagnostics.reminder_transport["mode"] == "sse"


def test_full_sse_receipts_do_not_block_earlier_pull_items(tmp_path):
    from iris_memory.store import InboxFull

    inbox = Inbox(tmp_path / "notifications.sqlite3")
    inbox.receipt_capacity = 2
    for number in (2, 3):
        inbox.accept({"id": number, "kind": "goal_reminder"})
        inbox.consume([number])
    with pytest.raises(InboxFull):
        inbox.accept({"id": 4, "kind": "goal_reminder"})
    # Pull can fill an earlier gap and retire receipts in the same transaction.
    inbox.save_page({"items": [{"id": 1, "kind": "goal_dedup_result"}], "next_cursor": 2})
    assert inbox.cursor == 2 and [n["id"] for n in inbox.pending()] == [1]
    assert inbox.db.execute("SELECT id FROM accepted").fetchall() == [(3,)]
    assert inbox.accept({"id": 4, "kind": "goal_reminder"})
    inbox.close()
