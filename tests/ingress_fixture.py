"""HTTP branch fixtures from the reviewed runtime, never real-Core qualification."""

import hashlib
import struct
import time
import zlib
from types import SimpleNamespace

from aiohttp import web

from tests.fixtures import CAPABILITIES, STATUS, envelope


def event(number=1, conversation="group-0", text="  原文\n\t中文  ", media=None):
    raw = {
        "post_type": "message",
        "message_type": "group",
        "message_id": number,
        "self_id": 100,
        "user_id": 200,
        "group_id": conversation,
        "time": 1700000000,
        "sender": {"nickname": "合成参与者", "user_id": 200},
        "message": [{"type": "text", "data": {"text": text}}],
    }
    if media:
        raw["message"].append(
            {"type": "image", "data": {"url": media, "file": "original.png"}}
        )
    return SimpleNamespace(
        message_obj=SimpleNamespace(raw_message=raw),
        platform_meta=SimpleNamespace(id="onebot-test"),
        get_self_id=lambda: "100",
        get_platform_name=lambda: "aiocqhttp",
        is_private_chat=lambda: False,
        get_group_id=lambda: conversation,
        get_sender_id=lambda: "200",
        get_messages=lambda: [],
    )


def receipt(result, kind="accept_media_event"):
    return {
        "receipt": {
            "schema_version": 1,
            "identity": {
                "database_id": "database-test",
                "owner_namespace": "runtime",
                "operation_kind": kind,
                "scope_id": "instance-a",
                "operation_key": "server-original",
            },
            "command_version": 1,
            "fingerprint_version": 1,
            "fingerprint": "f" * 64,
            "commit_id": "commit-test",
            "recorded_at": "2026-09-19T00:00:00+00:00",
            "result_schema_version": 1,
            "result": result,
        },
        "source": "NEW",
    }


def accepted(entry):
    return receipt(
        {
            "operation_id": "message-test",
            "entry_id": entry,
            "batch_id": None,
            "candidate_id": None,
            "source_id": None,
            "terminal": "ACCEPTED",
            "object_refs": [],
            "history": [],
            "retired_source_ids": [],
            "targets": [],
            "storage_execution": "ACTUAL",
            "model_adapter": "REMOTE_PROVIDER",
            "candidate_origin": "MODEL_VALIDATED",
            "facts": {},
        }
    )


def media_result(upload, state, size=0):
    return {
        "upload_id": upload,
        "blob_id": "blob-test" if state == "READY" else None,
        "generation": int(state == "READY"),
        "state": state,
        "byte_count": size,
        "targets": [],
        "change": {},
    }


def png_bytes(width=64):
    """Valid deterministic PNG originals, below the actual per-blob ceiling."""

    def chunk(kind, value):
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
        )

    pixels = b"".join(
        b"\0" + bytes((x + y) % 256 for x in range(width * 3)) for y in range(width)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, width, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels, 0))
        + chunk(b"IEND", b"")
    )


class HTTPFixture:
    def __init__(self):
        self.tokens, self.accepts, self.confirms, self.originals = {}, [], [], {}
        self.token_hosts = {}
        self.status_instance = "instance-a"
        self.uploads, self.requests = {}, []
        self.unknown, self.offline, self.revoke = False, False, set()
        self.media_data = png_bytes()
        self.registration_count = self.token_count = 0
        self.admin_originals = {}
        self.drop_token = False
        self.drop_accept = False
        self.mid_chunk = False

    async def start(self):
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def close(self):
        await self.runner.cleanup()

    async def handle(self, request):
        path = request.path
        if path == "/original.png":
            return web.Response(body=self.media_data)
        if self.offline:
            request.transport.close()
            return web.Response()
        if path == "/api/status":
            return web.json_response(
                envelope({**STATUS, "instance_id": self.status_instance})
            )
        if path == "/api/configuration/read":
            return web.json_response(
                envelope(
                    {
                        "status": {
                            "state": "APPLIED",
                            "admission_closed": False,
                            "published_version": "v1",
                            "authoritative_version": "v1",
                        },
                        "version_id": "v1",
                        "birth_version": "v1",
                        "values": {
                            "runtime": {"ingress.event_max_bytes": 8192},
                            "content": {
                                "media.event_occurrence_limit": 2,
                                "media.interpretation_text_max_bytes": 512,
                                "media.blob_max_bytes": 1048576,
                                "media.upload_chunk_bytes": 65536,
                            },
                        },
                    }
                )
            )
        if path.startswith("/api/connections/") or path.startswith("/api/tokens/"):
            assert request.headers["Origin"] == self.base
            assert request.headers["X-CSRF-Token"] == "synthetic-csrf"
            assert "iris_session=synthetic-session" in request.headers["Cookie"]
            body = await request.json()
            if path.endswith("tokens/list"):
                return web.json_response(
                    envelope(
                        {
                            "items": [
                                {
                                    "object_id": hashlib.sha256(t.encode()).hexdigest(),
                                    "host_id": self.token_hosts.get(t, "host"),
                                    "entries": entries,
                                    "operations": ["confirm"],
                                    "expires_at_us": int(
                                        (time.time() + 3600) * 1000000
                                    ),
                                    "revoked": t in self.revoke,
                                }
                                for t, entries in self.tokens.items()
                            ],
                            "after": None,
                        }
                    )
                )
            key = body["key"]
            previous = self.admin_originals.get(key)
            if previous:
                assert previous == body
            else:
                self.admin_originals[key] = body
            result = receipt(
                {"entry_id": body.get("entry_id"), "terminal": "REGISTERED"},
                "register_content_entry",
            )
            if previous:
                result["source"] = "EXISTING"
            if path.endswith("tokens/create"):
                if not previous:
                    self.token_count += 1
                result = {
                    "result": result,
                    "token": None if previous else "new-synthetic-token",
                }
                if self.drop_token and not previous:
                    request.transport.close()
                    return web.Response()
                return web.json_response(envelope(result))
            if not previous:
                self.registration_count += 1
            return web.json_response(envelope(result, outcome="COMMITTED"))
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if token in self.revoke or token not in self.tokens:
            return web.json_response(envelope(outcome="REJECTED"), status=403)
        entries = self.tokens[token]
        if path.endswith("capabilities"):
            return web.json_response(
                envelope(
                    {
                        **CAPABILITIES,
                        "entries": entries,
                        "operations": [
                            "accept",
                            "confirm",
                            "media_upload",
                            "media_inspect",
                        ],
                    }
                )
            )
        if path.endswith("media/chunk"):
            assert request.content_type == "application/octet-stream"
            entry, upload, offset = (
                request.headers[h]
                for h in ("X-Iris-Entry", "X-Iris-Upload", "X-Iris-Offset")
            )
            if entry not in entries:
                return web.json_response(envelope(outcome="REJECTED"), status=403)
            data = await request.read()
            row = self.uploads[upload]
            assert (
                str(int(offset)) == offset
                and int(offset) == len(row["bytes"])
                and len(data) <= 65536
            )
            row["bytes"] += data
            row["state"] = "UPLOADING"
            self.requests.append("chunk")
            if self.mid_chunk:
                return web.json_response(
                    {"version": 1, "outcome": "UNCONFIRMED", "cleanup_pending": True},
                    status=202,
                )
            return web.json_response(
                envelope(
                    {
                        "upload_id": upload,
                        "offset": len(row["bytes"]),
                        "state": "VOLATILE_PROGRESS",
                    }
                )
            )
        body = await request.json()
        if body["entry_id"] not in entries:
            return web.json_response(envelope(outcome="REJECTED"), status=403)
        entry, value = body["entry_id"], body["input"]
        if path in {"/api/host/accept", "/api/host/accept/resolve"}:
            key = value["key"]
            if path.endswith("resolve"):
                self.confirms.append(key)
            else:
                self.accepts.append(key)
                self.originals[key] = body
                if self.drop_accept:
                    self.drop_accept = False
                    request.transport.close()
                    return web.Response()
            if self.unknown:
                return web.json_response(
                    {
                        "version": 1,
                        "outcome": "UNCONFIRMED",
                        "state": "UNCONFIRMED",
                        "operation_key": key,
                        "cleanup_pending": True,
                    },
                    status=202,
                )
            if key not in self.originals:
                return web.json_response(envelope(outcome="ABSENT"))
            assert body == self.originals[key]
            return web.json_response(envelope(accepted(entry), outcome="COMMITTED"))
        action = path.rsplit("/", 1)[1]
        upload = (
            value.get("upload_id")
            or "upload:" + hashlib.sha256(value["key"].encode()).hexdigest()
        )
        self.requests.append(action)
        if action == "begin":
            self.uploads.setdefault(upload, {"bytes": b"", "state": "UPLOADING"})
        row = self.uploads[upload]
        if action == "inspect":
            return web.json_response(
                envelope(
                    {
                        "upload_id": upload,
                        "state": row["state"],
                        "volatile_offset": len(row["bytes"]),
                        "reupload_required": row["state"] == "REUPLOAD_REQUIRED",
                        "completion": media_result(upload, "READY", len(row["bytes"]))
                        if row["state"] == "READY"
                        else None,
                        "observed_at_us": 1,
                        "progress_durable": False,
                    }
                )
            )
        if action == "finish":
            row["state"] = "READY"
        result = receipt(
            media_result(upload, row["state"], len(row["bytes"])),
            "publish_media_upload" if row["state"] == "READY" else "begin_media_upload",
        )
        return web.json_response(
            envelope(
                {"value": result["receipt"]} if action == "resolve" else result,
                outcome="OBSERVED" if action == "resolve" else "COMMITTED",
            )
        )
