import asyncio

import pytest
from aiohttp import web

from iris_memory.core_client.http import (
    Admission,
    HostClient,
    HTTPTransport,
    ManagementClient,
)
from iris_memory.errors import ControlError
from tests.fixtures import CAPABILITIES, HEALTH, STATUS, envelope


@pytest.fixture
async def server():
    seen = []
    mode = {"value": "normal"}
    block = asyncio.Event()

    async def handler(request):
        seen.append(
            {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "body": await request.read(),
            }
        )
        if mode["value"] == "slow":
            await block.wait()
        if mode["value"] == "large":
            response = web.StreamResponse()
            await response.prepare(request)
            try:
                for _ in range(128):
                    await response.write(b"x" * 8192)
                    await asyncio.sleep(0.001)
            except ConnectionResetError:
                pass
            return response
        if mode["value"] == "redirect":
            return web.Response(
                status=302, headers={"Location": "http://localhost:1/secret"}
            )
        if mode["value"] == "disconnect":
            request.transport.close()
            return web.Response()
        if mode["value"] == "denied":
            return web.json_response(envelope(outcome="REJECTED"), status=403)
        if request.path == "/health":
            value = {"version": 1, "health": HEALTH}
        elif request.path == "/api/host/capabilities":
            value = envelope(CAPABILITIES)
        else:
            value = envelope(STATUS)
        response = web.json_response(value)
        response.set_cookie("unwanted", "never-forward")
        return response

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", seen, mode, block
    block.set()
    await runner.cleanup()


async def test_real_http_post_health_and_authentication_isolation(server):
    base, seen, _, _ = server
    gate = Admission()
    host_transport, admin_transport = HTTPTransport(gate), HTTPTransport(gate)
    host = HostClient(host_transport)
    admin = ManagementClient(admin_transport, base, "admin-secret", "csrf-secret")
    try:
        assert (await host.health(base))["business_ready"]
        assert (await host.capabilities(base, "host-secret")).observed
        assert (await admin.status())["instance_id"] == "instance-a"
        await host.health(base)
        assert [(r["method"], r["path"]) for r in seen] == [
            ("GET", "/health"),
            ("POST", "/api/host/capabilities"),
            ("GET", "/api/status"),
            ("GET", "/health"),
        ]
        assert seen[1]["body"] == b"{}"
        assert seen[1]["headers"]["Authorization"] == "Bearer host-secret"
        assert (
            "Cookie" not in seen[0]["headers"]
            and "Authorization" not in seen[0]["headers"]
        )
        assert "Cookie" not in seen[1]["headers"]
        assert "Authorization" not in seen[2]["headers"]
        assert seen[2]["headers"]["Origin"] == base
        assert seen[2]["headers"]["X-CSRF-Token"] == "csrf-secret"
        assert "Cookie" not in seen[3]["headers"]
        assert gate.active == gate.waiting == 0
    finally:
        await gate.close()
        await host_transport.close()
        await admin.close()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("large", "RESPONSE_TOO_LARGE"),
        ("redirect", "REDIRECT_REFUSED"),
        ("disconnect", "HTTP_UNAVAILABLE"),
        ("slow", "HTTP_TIMEOUT"),
    ],
)
async def test_response_limits_redirect_disconnect_timeout_release(server, mode, code):
    base, _, current, _ = server
    current["value"] = mode
    gate = Admission()
    transport = HTTPTransport(gate, timeout=0.08)
    try:
        with pytest.raises(ControlError, match=code) as error:
            await HostClient(transport).capabilities(base, "host-secret")
        assert "host-secret" not in str(error.value)
        assert gate.active == gate.waiting == 0
    finally:
        await transport.close()


async def test_shared_four_active_sixteen_waiting_and_shutdown(server):
    base, _, mode, _ = server
    mode["value"] = "slow"
    gate = Admission()
    first, second = HTTPTransport(gate), HTTPTransport(gate)
    tasks = [
        asyncio.create_task(HostClient(first if i % 2 else second).health(base))
        for i in range(20)
    ]
    try:
        for _ in range(100):
            if gate.active == 4 and gate.waiting == 16:
                break
            await asyncio.sleep(0.005)
        assert (gate.active, gate.waiting) == (4, 16)
        with pytest.raises(ControlError, match="ADMISSION_FULL"):
            await HostClient(first).health(base)
        tasks[-1].cancel()
        await asyncio.gather(tasks[-1], return_exceptions=True)
        assert gate.waiting == 15
        await gate.close()
        assert gate.active == gate.waiting == 0
        assert all(t.done() for t in tasks)
        with pytest.raises(ControlError, match="CLOSED"):
            await HostClient(first).health(base)
    finally:
        await gate.close()
        await first.close()
        await second.close()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_permission_denied_is_not_connected_capability(server):
    base, _, mode, _ = server
    mode["value"] = "denied"
    transport = HTTPTransport(Admission())
    try:
        reply = await HostClient(transport).capabilities(base, "expired-token")
        assert reply.http_status == 403 and reply.outcome == "REJECTED"
        assert reply.data is None
        with pytest.raises(ControlError, match="REAUTHORIZE"):
            await ManagementClient(transport, base, "expired-admin", "csrf").status()
    finally:
        await transport.close()


async def test_application_shutdown_cancels_real_inflight_request(server, tmp_path):
    from iris_memory.application import Application

    base, _, mode, _ = server
    mode["value"] = "slow"
    app = Application(tmp_path)
    await app.initialize()
    await app.save_connection(0, base)

    async def query():
        async with app.request("alice"):
            return await app.check_connection()

    task = asyncio.create_task(query())
    for _ in range(100):
        if app.admission.active:
            break
        await asyncio.sleep(0.005)
    assert app.admission.active == 1
    await app.terminate()
    assert task.done() and task.cancelled()
    assert app.store.db is None and app.transport is None
    assert not app._requests
    assert app.admission.active == app.admission.waiting == 0
