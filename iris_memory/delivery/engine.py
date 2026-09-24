"""Two owned workers, source FIFO barriers and original-input confirmation."""

import asyncio
import time
from dataclasses import asdict

from ..core_client.ingress import IngressLimits, event_v2
from ..core_client.receipts import acceptance
from ..errors import ControlError
from ..platforms.onebot import identity, snapshot
from ..validation import encode
from .media import MediaDelivery
from .store import DeliveryStore


class DeliveryEngine:
    def __init__(self, application, capacity=None):
        self.app = application
        self.queue = DeliveryStore(application.store, capacity)
        self.media = None
        self.task = None
        self.jobs = {}
        self.captures = 0
        self.capture_completions = set()
        self.peak_jobs = self.peak_captures = 0
        self.closed = False
        self.wake = asyncio.Event()

    async def start(self):
        await self.queue.open()
        self.media = MediaDelivery(
            self.queue, self.app.host, self.app.directory / "delivery-media"
        )
        self.task = asyncio.create_task(self.run(), name="iris-delivery")

    @property
    def quiescent(self):
        return (
            (self.task is None or self.task.done())
            and not self.jobs
            and not self.capture_completions
        )

    @property
    def released(self):
        return self.quiescent and self.media is None

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        jobs = list(self.jobs.values())
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await asyncio.gather(*self.capture_completions, return_exceptions=True)
        failures = []
        if self.media:
            try:
                await self.media.close()
                self.media = None
            except BaseException as error:
                failures.append(error)
        if not failures:
            try:
                await self.queue.close()
            except BaseException as error:
                failures.append(error)
        else:
            # Retain the unclean marker, even if a later close releases the session.
            self.queue.close_attempted = True
            self.queue.storage_failed = True
        if failures:
            raise BaseExceptionGroup("Delivery release failed", failures)

    async def capture(self, host_event):
        # No waiting task is created for overload; do not retain another mutable host event.
        if self.closed or self.queue.storage_failed or self.captures >= 8:
            self.queue.memory_gaps += 1
            return
        self.captures += 1
        completion = asyncio.get_running_loop().create_future()
        self.capture_completions.add(completion)
        self.peak_captures = max(self.peak_captures, self.captures)
        source_id = "unidentified"
        try:
            try:
                source_id = identity(host_event)
            except ControlError:
                return
            try:
                material, external, reason = snapshot(
                    host_event, self.queue.capacity.record_bytes
                )
                encode(material, self.queue.capacity.record_bytes)
            except (ControlError, UnicodeError):
                # No truncated body or fabricated platform identifier is submitted.
                material, external, reason = (
                    {
                        "raw": None,
                        "event": None,
                        "media": [],
                        "full_raw_retained": False,
                    },
                    None,
                    "RAW_RECORD_NOT_RETAINED",
                )
            config = await self.app.store.settings()
            source = next(
                (s for s in config["sources"] if s["source_id"] == source_id), None
            )
            if (
                not source
                or not source["enabled"]
                or not config["intents"].get("plugin.enabled")
                or not config["intents"].get("observation.enabled")
            ):
                return
            group = next(
                g for g in config["groups"] if g["group_id"] == source["group_id"]
            )
            binding = {
                "connection": config["binding"],
                "instance_id": config["instance_id"],
                "origin": config["origin"],
                "group_id": group["group_id"],
                "group_binding": group["binding"],
                "group_version": group["version"],
                "credential_ref": group["credential_ref"],
                "entry_id": source["entry_id"],
                "source_version": source["version"],
                "host_id": group["host_id"],
                "limits": config["ingress_limits"],
            }
            if not binding["instance_id"] or not binding["limits"]:
                reason = reason or "INGRESS_BINDING_UNVERIFIED"
            await self.queue.admit(source_id, binding, material, external, reason)
            self.wake.set()
        except asyncio.CancelledError:
            self.queue.memory_gaps += 1
            raise
        except Exception:
            await self.queue.gap(source_id, "LOCAL_PERSISTENCE_FAILURE")
            self.queue.storage_failed = True
        finally:
            self.captures -= 1
            if not completion.done():
                completion.set_result(None)
            self.capture_completions.discard(completion)

    async def observe_stage(self, event, stage):
        """Bounded diagnostic counts only: neither callback proves platform delivery."""
        if self.closed or self.queue.storage_failed or self.captures >= 8:
            self.queue.memory_gaps += 1
            return
        self.captures += 1
        completion = asyncio.get_running_loop().create_future()
        self.capture_completions.add(completion)
        try:
            try:
                source_id = identity(event)
            except ControlError:
                return
            config = await self.app.store.settings()
            if not (
                config["intents"].get("plugin.enabled")
                and config["intents"].get("observation.enabled")
            ):
                return
            if any(
                s["source_id"] == source_id and s["enabled"] for s in config["sources"]
            ):
                async with self.app.store.transaction() as db:
                    await self.queue._count(db, stage)
        except Exception:
            await self.queue.gap("all", "OUTPUT_DIAGNOSTIC_NOT_SAVED")
        finally:
            self.captures -= 1
            if not completion.done():
                completion.set_result(None)
            self.capture_completions.discard(completion)

    async def run(self):
        while not self.closed:
            try:
                heads = await self.queue.heads()
                for head in heads:
                    if len(self.jobs) >= self.queue.capacity.workers:
                        break
                    if head["source_id"] not in self.jobs:
                        job = asyncio.create_task(
                            self.work(head["id"]), name="iris-source-delivery"
                        )
                        self.jobs[head["source_id"]] = job
                        job.add_done_callback(
                            lambda task, source=head["source_id"]: self.completed(
                                source, task
                            )
                        )
                        self.peak_jobs = max(self.peak_jobs, len(self.jobs))
            except asyncio.CancelledError:
                raise
            except Exception:
                self.queue.storage_failed = True
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1)
            except TimeoutError:
                pass

    def completed(self, source, task):
        self.jobs.pop(source, None)
        if not task.cancelled():
            task.exception()
        self.wake.set()

    async def work(self, event_id):
        async with self.app._connection_lock.read():
            try:
                async with asyncio.timeout(60):
                    await self._work(event_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                try:
                    row = await self.queue.get(event_id)
                    code = (
                        error.code
                        if isinstance(error, ControlError)
                        else "DELIVERY_IO_FAILURE"
                    )
                    permanent = code in {
                        "MEDIA_BLOB_LIMIT",
                        "EMPTY_MEDIA",
                        "MEDIA_RESOURCE_UNSUPPORTED",
                        "MEDIA_LOCAL_INTEGRITY_FAILURE",
                    }
                    if permanent and not row["submitted"]:
                        await self.queue.state(event_id, "BLOCKED", reason=code)
                        await self.queue.gap(row["source_id"], code)
                        return
                    # Unknown stays unknown. A malformed response is not a rejection receipt.
                    await self.queue.state(
                        event_id,
                        row["state"],
                        reason=code,
                        cleanup=bool(row["cleanup_pending"]),
                        delay=10,
                    )
                except Exception:
                    self.queue.storage_failed = True

    async def _work(self, event_id):
        row = await self.queue.get(event_id)
        # A scheduled job can wait behind an explicit reauthorization which already
        # resolved and erased this body. Re-read under the lease before any I/O.
        if row["state"] == "CONFIRMED" and not row["cleanup_pending"]:
            if row["material"] is not None:
                await self.media.cleanup(event_id)
            return
        if row["state"] == "REJECTED" and not row["cleanup_pending"]:
            return
        confirming = bool(row["submitted"] and not row["retry_authorized"])
        config = await self.app.store.settings()
        binding = row["binding"]
        source = next(
            (s for s in config["sources"] if s["source_id"] == row["source_id"]),
            None,
        )
        group = next(
            (g for g in config["groups"] if g["group_id"] == binding["group_id"]),
            None,
        )
        if (
            not group
            or binding["connection"] != config["binding"]
            or binding["instance_id"] != config["instance_id"]
            or group["binding"] != binding["group_binding"]
            or group["connection_binding"] != config["binding"]
            or group["instance_id"] != config["instance_id"]
        ):
            raise ControlError("ORIGINAL_BINDING_CHANGED")
        enabled = bool(
            source
            and source["enabled"]
            and config["intents"].get("plugin.enabled")
            and config["intents"].get("observation.enabled")
        )
        if row["state"] == "BLOCKED":
            await self.queue.state(
                event_id,
                "BLOCKED",
                reason=row["reason"],
                cleanup=bool(row["cleanup_pending"]),
                delay=60,
            )
            return
        if not confirming and not enabled:
            token = await self.app.sources.credential(binding["credential_ref"])
            await self.media.confirm_paused(row, token)
            pending = any(
                m["cleanup_pending"] for m in await self.queue.media(event_id)
            )
            await self.queue.state(event_id, "PAUSED", cleanup=pending, delay=5)
            return
        if row["state"] == "CONFIRMED" and not row["cleanup_pending"]:
            await self.media.cleanup(event_id)
            return
        observation = group["observation"]
        if (
            not observation
            or observation["at"] < self.app._started_at
            or time.time() - observation["at"] > 60
        ):
            token = await self.app.sources.credential(binding["credential_ref"])
            reply = await self.app.host.capabilities(binding["origin"], token)
            if not reply.observed:
                raise ControlError("GROUP_PERMISSION_UNVERIFIED")
            if set(reply.data["entries"]) != set(group["entries"]):
                raise ControlError("GROUP_PERMISSION_MISMATCH")
            observation = {
                "at": time.time(),
                "capabilities": reply.data,
                "permission": "granted",
            }
            await self.app.sources.observe(
                group["group_id"], group["binding"], observation
            )
        allowed = observation["capabilities"]["operations"]
        if ("confirm" if confirming else "accept") not in allowed:
            raise ControlError("GROUP_PERMISSION_DENIED")
        token = await self.app.sources.credential(binding["credential_ref"])
        limits = IngressLimits(**binding["limits"]["values"])
        material = row["material"]
        if not row["submitted"]:
            if material["media"] and not {
                "media_upload",
                "media_inspect",
                "confirm",
            } <= set(allowed):
                raise ControlError("GROUP_MEDIA_PERMISSION_DENIED")
            await self.queue.state(
                event_id, "MEDIA_PENDING" if material["media"] else "SAVED"
            )
            refs = []
            for medium in material["media"]:
                upload_id = await self.media.upload(row, medium, token, limits)
                refs.append(
                    {
                        "reference_id": upload_id,
                        "occurrence_id": medium["id"],
                        "modality": medium["modality"],
                        "interpretation": None,
                    }
                )
            material["event"]["media"] = refs
            try:
                event_v2(material["event"], limits)
            except ControlError as error:
                await self.queue.state(event_id, "BLOCKED", reason=error.code)
                await self.queue.gap(row["source_id"], error.code)
                return
            await self.queue.freeze_event(event_id, material)
        payload = {"key": row["original_key"], "event": material["event"]}
        await self.queue.dispatch(event_id, confirm=confirming)
        reply = await self.app.host.ingress(
            binding["origin"],
            token,
            binding["entry_id"],
            "accept/resolve" if confirming else "accept",
            payload,
            limits=limits,
        )
        confirmed = acceptance(reply, binding["entry_id"])
        if confirmed:
            await self.queue.state(
                event_id,
                "CONFIRMED",
                cleanup=reply.cleanup_pending,
                delay=5 if reply.cleanup_pending else 0,
            )
            if not reply.cleanup_pending:
                await self.media.cleanup(event_id)
        elif reply.outcome == "NOT_COMMITTED" and not reply.cleanup_pending:
            await self.queue.state(
                event_id, "NOT_COMMITTED", reason="ORIGINAL_NOT_COMMITTED", delay=60
            )
        elif (
            reply.outcome == "REJECTED"
            and not row["submitted"]
            and not reply.cleanup_pending
        ):
            await self.queue.state(event_id, "REJECTED", reason="CORE_REJECTED")
            await self.queue.gap(row["source_id"], "CORE_REJECTED")
        else:
            state = "ABSENT_UNKNOWN" if reply.outcome == "ABSENT" else "UNKNOWN"
            await self.queue.state(
                event_id,
                state,
                reason="CORE_" + reply.outcome,
                cleanup=reply.cleanup_pending,
                delay=10,
            )

    async def status(self, offset=0):
        result = await self.queue.status(offset)
        config = await self.app.store.settings()
        enabled = config["intents"].get("plugin.enabled") and config["intents"].get(
            "observation.enabled"
        )
        enabled_sources = {s["source_id"] for s in config["sources"] if s["enabled"]}
        for row in result["items"]:
            row["intake_paused"] = not row["submitted"] and not (
                enabled and row["source_id"] in enabled_sources
            )
        return {
            **result,
            "active_jobs": len(self.jobs),
            "capture_active": self.captures,
            "download_active": self.media.downloading if self.media else 0,
            "peak_jobs": self.peak_jobs,
            "peak_captures": self.peak_captures,
            "capacity": asdict(self.queue.capacity),
        }
