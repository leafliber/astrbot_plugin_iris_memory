import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Config
from .diagnostics import ROUTINE_CODES, Diagnostics
from .extras import strip_markdown
from .image_store import ImageStore
from .input_cleanup import clean_message, clean_text
from .mapping import (
    Entry,
    content_text,
    input_message,
    map_entry,
    output_message,
    raw_message,
    stable_key,
)
from .media import attach_images
from .prompt import render
from .reminders import ReminderConsumer
from .store import Inbox
from .vendor.iris_client import AsyncIrisClient, IrisError


@dataclass
class Work:
    call: Callable[[], Awaitable[Any]]
    done: asyncio.Future[bool]
    timeout: float


class Bridge:
    def __init__(
        self,
        config: Config,
        client: AsyncIrisClient,
        data_dir: Path,
        log: Callable[[str], None],
        send_active: Callable[[str, str], Awaitable[bool]],
        diagnostics: Diagnostics | None = None,
    ) -> None:
        self.diagnostics = diagnostics if diagnostics is not None else Diagnostics(config)
        self.data_dir = data_dir
        self.config, self.client, self.log, self.send_active = config, client, log, send_active
        # Separate checkpoints when endpoint/token scope changes. Only a one-way
        # digest is persisted; credentials never enter the inbox or logs.
        namespace = stable_key(config.base_url.rstrip("/"), config.token)
        self.inbox = Inbox(data_dir / f"notifications-{namespace}.sqlite3")
        self.queue: asyncio.Queue[Work] = asyncio.Queue(config.queue_size)
        self.tasks: list[asyncio.Task[None]] = []
        self.closed = False
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.poll_lock = asyncio.Lock()
        self.counts: dict[str, int] = {}
        self.reminders = ReminderConsumer(self, namespace)
        self.images = None
        if config.images_enabled:
            try:
                self.images = ImageStore(data_dir, config, namespace)
            except Exception:
                self.record("image_storage_unavailable")

    def record(self, code: str) -> None:
        code = self.diagnostics.record(code)
        self.counts[code] = self.counts.get(code, 0) + 1
        # Codes come from this module, never exceptions, payloads, or config.
        if code not in ROUTINE_CODES:
            self.log("Iris: " + code)

    def start(self) -> None:
        if self.tasks or self.closed:
            return
        self.tasks.append(asyncio.create_task(self._worker()))
        self.reminders.start()

    def enqueue(
        self, call: Callable[[], Awaitable[Any]], *, timeout: float | None = None
    ) -> asyncio.Future[bool]:
        future = asyncio.get_running_loop().create_future()
        if self.closed:
            future.set_result(False)
            return future
        try:
            self.queue.put_nowait(Work(call, future, timeout or self.config.request_timeout))
        except asyncio.QueueFull:
            self.record("queue_full")
            future.set_result(False)
        return future

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            ok = False
            try:
                async with asyncio.timeout(item.timeout):
                    await item.call()
                ok = True
            except Exception:
                self.record("background_request_failed")
            finally:
                if not item.done.done():
                    item.done.set_result(ok)
                self.queue.task_done()

    async def receive(self, event: Any) -> None:
        if self.closed or event.get_extra("iris_received"):
            return
        try:
            entry = map_entry(event, self.config)
            self.entries[entry.origin] = entry
            self.entries.move_to_end(entry.origin)
            if len(self.entries) > 200:
                self.entries.popitem(last=False)
            message = input_message(event, entry)
            event.set_extra("iris_entry", entry)
            if self.config.force_non_streaming:
                # Verified per-event override in AstrBot 4.28.2. Native streaming
                # does not expose a portable delivery receipt to plugins.
                event.set_extra("enable_streaming", False)
            event.set_extra(
                "iris_received",
                self.enqueue(
                    lambda: self._ingest(event, entry, message),
                    timeout=self.config.image_timeout + 2 * self.config.request_timeout + 0.1,
                ),
            )
            self._observe_send(event, entry)
        except Exception:
            self.record("receive_failed")

    async def _ingest(self, event: Any, entry: Entry, message: dict) -> None:
        cached = set()
        if self.config.images_enabled:
            cached = await attach_images(
                event, entry.id, message, self.client, self.config, self.record, self.images
            )
        # Parsed components are ordinary text. Only a known OneBot wire string
        # may supply CQ syntax, and only when no component chain is available.
        if event.get_platform_name() in {"aiocqhttp", "onebot"} and not event.get_messages():
            wire = raw_message(event).get("message")
            if isinstance(wire, str):
                message["content"] = clean_text(wire, self.config, raw_cq=True)
        # Keep attachments in the cleaned copy, including any 404 retry below.
        message = clean_message(message, self.config)
        try:
            async with asyncio.timeout(self.config.request_timeout):
                await self.client.receive_message(entry.id, message)
        except IrisError as error:
            if error.status_code != 404 or not cached or self.images is None:
                raise
            self.images.discard(entry.id, cached)
            remaining = [value for value in message.get("media_ids", []) if value not in cached]
            if remaining:
                message["media_ids"] = remaining
            else:
                message.pop("media_ids", None)
            self.record("image_cache_stale")
            # Same message key: preserve text once, without extra uploads on failure.
            async with asyncio.timeout(self.config.request_timeout):
                await self.client.receive_message(entry.id, message)
        self.diagnostics.record("message_received")

    def _observe_send(self, event: Any, entry: Entry) -> None:
        if event.get_extra("iris_observing_send"):
            return
        original = event.send
        sequence = 0
        event_key = uuid4().hex

        async def send(chain: Any) -> Any:
            nonlocal sequence
            # Snapshot before adapters mutate message components.
            try:
                text = content_text(chain.chain)
            except Exception:
                text = ""
            result = await original(chain)  # failure/cancellation propagates to AstrBot
            try:
                if text and result is not False and not self.closed:
                    sequence += 1
                    self.submit_output(entry, text, f"{event_key}:{sequence}")
                    event.set_extra("iris_confirmed_send", True)
                    llm_result = event.get_result()
                    if llm_result is not None and llm_result.is_llm_result():
                        self._used(event)
            except Exception:
                self.record("output_capture_failed")
            return result

        event.send = send  # per-event only; no adapter/class/global monkey patch
        event.set_extra("iris_observing_send", True)

    def submit_output(self, entry: Entry, text: str, key: str) -> asyncio.Future[bool]:
        """Call only after a successful send; share dedupe and queue semantics."""
        message = clean_message(output_message(entry, text, key), self.config)
        return self.enqueue(lambda: self.client.receive_message(entry.id, message))

    def prepare_options(self) -> dict:
        return {
            "judge": self.config.judge,
            "memory_limit": self.config.memory_limit,
            "token_budget": self.config.token_budget,
            "goal_limit": self.config.goal_limit,
            "recent_limit": self.config.recent_limit,
            "judge_budget_seconds": self.config.judge_budget_seconds,
        }

    async def prepare_material(self, entry_id: str, options: dict, *, purpose="reply") -> dict:
        started = time.monotonic()
        prepared, outcome = None, "failed"
        try:
            prepared = await self.client.prepare(entry_id, options)
            outcome = "ok"
            return prepared
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self.diagnostics.prepared(
                entry_id,
                prepared,
                time.monotonic() - started,
                purpose=purpose,
                outcome=outcome,
                judge=options.get("judge", True) is True,
            )

    async def prepare(self, event: Any, request: Any) -> None:
        if (
            self.closed
            or not self.config.inject_memory
            or event.get_extra("iris_prepared_request") is request
        ):
            return
        # A new provider request supersedes the earlier one on this event. A
        # failed new prepare must not retain feedback IDs from an older request.
        event.set_extra("iris_prepared", None)
        event.set_extra("iris_usage_recorded", False)
        try:
            async with asyncio.timeout(self.config.prepare_timeout):
                await self.receive(event)
                received = event.get_extra("iris_received")
                if received is None or not await asyncio.shield(received):
                    self.record("prepare_without_ingest")
                    return
                entry = event.get_extra("iris_entry")
                options = self.prepare_options()
                # Optional host extensions are explicit, per request; no persistent
                # known-ID cache because AstrBot may evict or replace its context.
                for key in ("participants", "known_memory_ids", "text"):
                    value = event.get_extra("iris_" + key)
                    if value is not None:
                        options[key] = value
                prepared = await self.prepare_material(entry.id, options)
                goals = {g["id"] for g in prepared.get("goals", [])}
                reminders = [
                    n
                    for n in self.inbox.pending()
                    if self.config.reminder_mode == "context"
                    and n.get("kind") == "goal_reminder"
                    and n.get("goal_id") in goals
                    and (
                        not self.config.reminder_entry_id
                        or self.config.reminder_entry_id == entry.id
                    )
                ]
                prompt = render(prepared, reminders, self.config)
                recall_id = prepared["recall_id"]
                request.system_prompt = (request.system_prompt or "") + "\n\n" + prompt.text
                if self.config.take_over_context:
                    request.contexts = []
                event.set_extra("iris_prepared", (recall_id, prompt))
                event.set_extra("iris_prepared_request", request)
        except Exception:
            self.record("prepare_degraded")

    def _used(self, event: Any) -> None:
        prepared = event.get_extra("iris_prepared")
        if prepared is None or event.get_extra("iris_usage_recorded"):
            return
        recall_id, prompt = prepared
        self.inbox.consume(prompt.reminder_ids)
        event.set_extra("iris_usage_recorded", True)
        if self.config.assume_injected_used and prompt.memory_ids:
            self.enqueue(lambda: self.client.feedback(recall_id, prompt.memory_ids))

    async def after_sent(self, event: Any) -> None:
        # 4.28.2 calls this hook even after caught send errors. Never read the
        # result as proof of sending. The observer above owns actual backflow.
        if not event.get_extra("iris_confirmed_send"):
            self.record("send_unconfirmed")

    async def poll(self) -> None:
        async with self.poll_lock:
            capacity = self.inbox.capacity - self.inbox.count()
            if capacity > 0:
                async with asyncio.timeout(self.config.request_timeout):
                    page = await self.client.pull_notifications(
                        after=self.inbox.cursor, limit=min(30, capacity)
                    )
                self.inbox.save_page(page)
            else:
                self.record("reminder_inbox_full")
            await self.deliver_reminders()

    async def deliver_reminders(self) -> None:
        # Caller holds poll_lock: push and pull cannot send the same ID twice.
        # Dedup receipts are control notifications, never reminders to users.
        for notice in self.inbox.pending():
            if notice.get("kind") == "goal_dedup_result":
                self.record("goal_dedup_result")
                self.inbox.consume([notice["id"]])
        if self.config.reminder_mode == "active":
            for notice in self.inbox.pending():
                if notice.get("kind") != "goal_reminder":
                    continue
                text = "提醒：" + str(notice.get("content", ""))
                if self.config.strip_markdown:
                    text = strip_markdown(text)
                async with asyncio.timeout(self.config.request_timeout):
                    accepted = await self.send_active(self.config.reminder_target, text)
                if not accepted:
                    self.record("reminder_send_unconfirmed")
                    break
                self.inbox.consume([notice["id"]])
                self.record("reminder_sent")
                parts = self.config.reminder_target.split(":", 2)
                entry = Entry(
                    self.config.reminder_entry_id,
                    "astrbot:" + parts[0],
                    "private" if parts[1] == "FriendMessage" else "group",
                    self.config.reminder_target,
                )
                self.submit_output(entry, text, f"notification:{notice['id']}")

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.reminders.close()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.done.done():
                item.done.set_result(False)
            self.queue.task_done()
        await self.client.aclose()
        self.inbox.close()
        if self.images is not None:
            self.images.close()
