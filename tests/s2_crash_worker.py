"""Abrupt process loss at owned delivery boundaries; run only by S2 tests."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from iris_memory.application import Application
from tests.ingress_fixture import event
from tests.test_ingress_delivery import configure, until


async def run(directory, base, phase):
    app = Application(Path(directory))
    await app.initialize()
    await configure(app, SimpleNamespace(base=base, tokens={}))
    if phase == "before":
        app.delivery.task.cancel()
        await asyncio.gather(app.delivery.task, return_exceptions=True)
    await app.delivery.capture(
        event(
            77,
            media=base + "/original.png"
            if phase in {"media_ready", "media_mid"}
            else None,
        )
    )
    if phase != "before":

        async def durable():
            status = await app.delivery.status()
            if phase == "media_mid":
                return (
                    status["items"]
                    and status["items"][0]["reason"] == "MEDIA_PROGRESS_UNKNOWN"
                )
            return (
                status["items"]
                and status["items"][0]["submitted"] == 1
                and status["items"][0]["state"] == "UNKNOWN"
            )

        await until(durable)
    os._exit(17)


if __name__ == "__main__":
    asyncio.run(run(*sys.argv[1:]))
