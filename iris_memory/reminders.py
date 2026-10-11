"""SSE/pull acceptance is independent of platform delivery and its cooldown policy."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from .store import Inbox, InboxFull, open_private_database
from .vendor.iris_client import EventCheckpoint, EventOptions, EventStatus, IrisError

if TYPE_CHECKING:
    from .bridge import Bridge


class EventState:
    """Three distinct ID namespaces; no notification body or credential is stored here."""

    def __init__(self, path: Path, inbox: Inbox) -> None:
        self.inbox = inbox
        self.db = open_private_database(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoint (
                id INTEGER PRIMARY KEY CHECK(id=1), last_event_id TEXT, pull_cursor INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO checkpoint VALUES(1,NULL,0);
            CREATE TABLE IF NOT EXISTS seen (position INTEGER PRIMARY KEY, notification_id INTEGER NOT NULL);
        """)

    def load_sync(self) -> EventCheckpoint:
        event_id, cursor = self.db.execute(
            "SELECT last_event_id,pull_cursor FROM checkpoint WHERE id=1"
        ).fetchone()
        return EventCheckpoint(
            last_event_id=event_id,
            pull_cursor=max(cursor, self.inbox.cursor),
            seen_ids=tuple(
                row[0]
                for row in self.db.execute("SELECT notification_id FROM seen ORDER BY position")
            ),
        )

    async def load(self) -> EventCheckpoint:
        return self.load_sync()

    async def save(self, checkpoint: EventCheckpoint) -> None:
        # All page callbacks have committed first. If interrupted between these
        # two databases, the inbox watermark/receipts still suppress replay.
        self.inbox.advance(checkpoint.pull_cursor)
        with self.db:
            self.db.execute(
                "UPDATE checkpoint SET last_event_id=?,pull_cursor=? WHERE id=1",
                (checkpoint.last_event_id, self.inbox.cursor),
            )
            self.db.execute("DELETE FROM seen")
            self.db.executemany(
                "INSERT INTO seen VALUES(?,?)", enumerate(checkpoint.seen_ids[-2048:])
            )

    def close(self) -> None:
        self.db.close()


class ReminderConsumer:
    def __init__(self, bridge: Bridge, namespace: str) -> None:
        self.bridge = bridge
        self.path = bridge.data_dir / f"events-{namespace}.sqlite3"
        self.tasks: list[asyncio.Task] = []
        self.state: EventState | None = None
        self.subscription = None
        self.wake = asyncio.Event()
        self.fallback = not bridge.config.reminder_push_enabled
        self.closed = False
        self.stopped = False

    def _status(self, mode: str, reason: str) -> None:
        self.bridge.diagnostics.set_reminder_transport(mode, reason)

    def _stop(self, reason: str) -> None:
        if not self.stopped:
            if reason in {"unauthorized", "forbidden"}:
                self.bridge.record("reminder_auth_failed")
            else:
                self.bridge.record("reminder_stream_failed")
        self.stopped = True
        self._status("stopped", reason)
        self.wake.set()

    def _on_status(self, status: EventStatus) -> None:
        if status.reason == "closed" or self.closed or self.stopped:
            # The client always emits closed in finally, including after an
            # authorization error. Keep the actionable terminal reason visible.
            return
        if status.mode == "stopped":
            self._stop(status.reason)
            return
        mode = status.mode if status.reason in {"connected", "resync"} else "poll"
        previous = self.bridge.diagnostics.reminder_transport
        if (previous["mode"], previous["reason"]) == (mode, status.reason):
            return
        self._status(mode, status.reason)
        if status.reason == "connected":
            self.bridge.record("reminder_sse_connected")
        elif status.reason in {"events_unsupported", "resync"}:
            self.bridge.record("reminder_transport_changed")
        else:
            self.bridge.record("reminder_stream_degraded")

    async def _accept(self, notification: dict) -> None:
        # This callback acknowledges local durability, never platform delivery.
        async with self.bridge.poll_lock:
            try:
                added = self.bridge.inbox.accept(notification)
            except InboxFull:
                self.bridge.record("reminder_inbox_full")
                raise
            if added:
                self.bridge.record("reminder_enqueued")
            else:
                self.bridge.record("reminder_duplicate")
            self.wake.set()

    def start(self) -> None:
        if self.tasks or self.closed or not self.bridge.config.reminders_enabled:
            return
        if self.bridge.config.reminder_push_enabled:
            try:
                self.state = EventState(self.path, self.bridge.inbox)
                self.subscription = self.bridge.client.subscribe_notifications(
                    self._accept,
                    state=self.state,
                    options=EventOptions(poll_interval=self.bridge.config.reminder_interval),
                    on_status=self._on_status,
                )
            except Exception:
                self.fallback = True
                self._status("poll", "state_failed")
                self.bridge.record("reminder_state_failed")
            else:
                self._status("poll", "connecting")
                self.tasks.append(asyncio.create_task(self._subscribe()))
        else:
            self._status("poll", "push_disabled")
        self.tasks.append(asyncio.create_task(self._consume()))

    async def _subscribe(self) -> None:
        try:
            await self.subscription.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Known client failures are deliberately not interpolated into logs.
            # Auth/rejection has already latched its terminal status above.
            if not self.stopped and not self.closed:
                self.bridge.record("reminder_stream_failed")
        finally:
            if not self.stopped and not self.closed:
                self.fallback = True
                self._status("poll", "stream_failed")
                self.wake.set()

    async def _consume(self) -> None:
        while not self.closed and not self.stopped:
            self.wake.clear()
            try:
                if self.fallback:
                    await self.bridge.poll()
                else:
                    # The client owns SSE plus initial/resync/periodic pull.
                    # Do not schedule a second network poller alongside it.
                    async with self.bridge.poll_lock:
                        await self.bridge.deliver_reminders()
            except IrisError as error:
                if error.status_code in {401, 403}:
                    self._stop("unauthorized" if error.status_code == 401 else "forbidden")
                else:
                    self.bridge.record("reminder_poll_failed")
            except Exception:
                self.bridge.record("reminder_poll_failed")
            if self.stopped:
                return
            try:
                await asyncio.wait_for(self.wake.wait(), self.bridge.config.reminder_interval)
            except TimeoutError:
                pass

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for task in self.tasks:
            task.cancel()
        if self.subscription is not None:
            await self.subscription.aclose()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.state is not None:
            self.state.close()
        if not self.stopped:
            self._status("stopped", "closed")
