"""Owned, metered original files and the finite public upload lifecycle."""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from ..core_client.receipts import inspection, media_fact
from ..errors import ControlError


class MediaDelivery:
    def __init__(self, queue, host, directory):
        self.queue, self.host = queue, host
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ControlError("UNSAFE_MEDIA_DIRECTORY")
        self.downloading = 0
        self.session = aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar(),
            trust_env=False,
            connector=aiohttp.TCPConnector(limit=2),
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=60, connect=3, sock_read=10),
        )

    async def close(self):
        await self.session.close()

    def path(self, media_id):
        from ..validation import identifier

        identifier(media_id)
        return self.directory / (media_id + ".bin")

    async def download(self, event, medium, limit):
        await self.queue.reserve_media(event["id"], event["source_id"], medium["id"])
        record = next(
            m for m in await self.queue.media(event["id"]) if m["id"] == medium["id"]
        )
        target = self.path(medium["id"])
        if record["digest"]:
            return record
        if record["dispatched"]:
            raise ControlError("MEDIA_LOCAL_INTEGRITY_FAILURE")
        address = urlsplit(medium["url"])
        if (
            address.scheme not in {"http", "https"}
            or not address.hostname
            or address.username
            or address.password
            or address.fragment
        ):
            raise ControlError("MEDIA_RESOURCE_UNSUPPORTED")
        temporary = target.with_suffix(".part")
        # A crash after rename but before sealing leaves an owned, unsealed file.
        # Remove it before another download so one reservation never owns two originals.
        target.unlink(missing_ok=True)
        # The registered reservation owns this exact temporary path across restart.
        self.downloading += 1
        try:
            async with self.session.get(
                medium["url"], allow_redirects=False
            ) as response:
                if (
                    response.status != 200
                    or response.headers.get("Content-Encoding", "identity")
                    != "identity"
                ):
                    raise ControlError("MEDIA_FETCH_FAILED")
                maximum = min(limit, self.queue.capacity.blob_bytes)
                if (
                    response.content_length is not None
                    and response.content_length > maximum
                ):
                    raise ControlError("MEDIA_BLOB_LIMIT")
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                    0o600,
                )
                size, digest = 0, hashlib.sha256()
                with os.fdopen(fd, "wb") as stream:
                    async for chunk in response.content.iter_chunked(65536):
                        if size + len(chunk) > maximum:
                            raise ControlError("MEDIA_BLOB_LIMIT")
                        stream.write(chunk)
                        size += len(chunk)
                        digest.update(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if size == 0:
                    raise ControlError("EMPTY_MEDIA")
                os.replace(temporary, target)
                fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                await self.queue.update_media(
                    medium["id"],
                    bytes=size,
                    digest=digest.hexdigest(),
                    state="LOCAL_READY",
                )
                return next(
                    m
                    for m in await self.queue.media(event["id"])
                    if m["id"] == medium["id"]
                )
        finally:
            self.downloading -= 1

    async def upload(self, event, medium, token, limits):
        record = await self.download(event, medium, limits.blob_bytes)
        binding = event["binding"]
        base, entry = binding["origin"], binding["entry_id"]
        original = {"key": record["original_key"], "modality": medium["modality"]}

        async def call(action, payload):
            return await self.host.ingress(
                base, token, entry, "media/" + action, payload
            )

        if record["dispatched"]:
            observed = await call("inspect", original)
            if observed.cleanup_pending:
                raise ControlError("MEDIA_CLEANUP_PENDING")
            observed = inspection(observed)
            if observed is None:
                raise ControlError("MEDIA_UNCONFIRMED")
            upload_id = observed["upload_id"]
            if observed["state"] == "READY":
                return await self._ready(
                    record, upload_id, observed["completion"]["byte_count"]
                )
            if observed["state"] not in {"UPLOADING", "REUPLOAD_REQUIRED"}:
                raise ControlError("MEDIA_UNCONFIRMED")
            offset = 0 if observed["reupload_required"] else observed["volatile_offset"]
            if offset is None:
                raise ControlError("MEDIA_PROGRESS_UNKNOWN")
        else:
            await self.queue.update_media(
                record["id"], dispatched=1, state="BEGIN_UNKNOWN", cleanup_pending=1
            )
            reply = await call("begin", original)
            fact = media_fact(reply)
            if fact is None or reply.cleanup_pending:
                raise ControlError("MEDIA_UNCONFIRMED")
            upload_id = fact["upload_id"]
            if fact["state"] == "READY":
                return await self._ready(record, upload_id, fact["byte_count"])
            offset = 0
        await self.queue.update_media(
            record["id"], upload_id=upload_id, state="UPLOADING"
        )
        # Hash the actual retained file before uploading, not a path supplied by Pages.
        fd = os.open(self.path(record["id"]), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            digest, size = hashlib.sha256(), 0
            while chunk := stream.read(65536):
                size += len(chunk)
                digest.update(chunk)
                if size > limits.blob_bytes:
                    raise ControlError("MEDIA_LOCAL_INTEGRITY_FAILURE")
            if (
                size != record["bytes"]
                or digest.hexdigest() != record["digest"]
                or offset > size
            ):
                raise ControlError("MEDIA_LOCAL_INTEGRITY_FAILURE")
            stream.seek(offset)
            while chunk := stream.read(limits.chunk_bytes):
                reply = await call(
                    "chunk", {"upload_id": upload_id, "offset": offset, "data": chunk}
                )
                if not reply.observed or reply.data != {
                    "upload_id": upload_id,
                    "offset": offset + len(chunk),
                    "state": "VOLATILE_PROGRESS",
                }:
                    raise ControlError("MEDIA_PROGRESS_UNKNOWN")
                offset += len(chunk)
        await self.queue.update_media(
            record["id"], state="FINISH_UNKNOWN", cleanup_pending=1
        )
        reply = await call("finish", {"upload_id": upload_id})
        fact = media_fact(reply)
        if fact is None or fact["state"] != "READY":
            reply = await call("resolve", original)
            fact = media_fact(reply)
        if fact is None or fact["state"] != "READY" or reply.cleanup_pending:
            raise ControlError("MEDIA_UNCONFIRMED")
        return await self._ready(record, fact["upload_id"], fact["byte_count"])

    async def _ready(self, record, upload_id, byte_count):
        if byte_count != record["bytes"]:
            raise ControlError("MEDIA_SIZE_MISMATCH")
        await self.queue.update_media(
            record["id"], state="READY", upload_id=upload_id, cleanup_pending=0
        )
        return upload_id

    async def confirm_paused(self, event, token):
        for record in await self.queue.media(event["id"]):
            if not record["dispatched"] or record["state"] == "READY":
                continue
            medium = next(
                m for m in event["material"]["media"] if m["id"] == record["id"]
            )
            reply = await self.host.ingress(
                event["binding"]["origin"],
                token,
                event["binding"]["entry_id"],
                "media/inspect",
                {"key": record["original_key"], "modality": medium["modality"]},
            )
            observed = inspection(reply)
            if observed and observed["state"] == "READY" and not reply.cleanup_pending:
                await self._ready(
                    record, observed["upload_id"], observed["completion"]["byte_count"]
                )

    async def cleanup(self, event_id):
        for medium in await self.queue.media(event_id):
            if not medium["retained"]:
                continue
            self.path(medium["id"]).unlink(missing_ok=True)
            self.path(medium["id"]).with_suffix(".part").unlink(missing_ok=True)
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            await self.queue.update_media(medium["id"], retained=0, state="CLEANED")
        await self.queue.cleaned(event_id)
