"""Fresh-process recovery audit, against the parent's loopback HTTP fixture."""

import asyncio
import json
import sys
from pathlib import Path

from iris_memory.application import Application
from tests.test_ingress_delivery import until


async def main(directory, ident):
    app = Application(Path(directory))
    await app.initialize()
    try:
        status = await app.delivery.status()
        unclean = any(
            g["reason"] == "UNCLEAN_SESSION_COVERAGE_UNKNOWN" for g in status["gaps"]
        )
        await app.delivery_confirm(ident)

        async def done():
            row = await app.delivery.queue.get(ident)
            return row["state"] == "CONFIRMED" and row["material"] is None

        await until(done)
        row = await app.delivery.queue.get(ident)
        print(
            json.dumps(
                {
                    "unclean": unclean,
                    "state": row["state"],
                    "submitted": row["submitted"],
                }
            )
        )
    finally:
        await app.terminate()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
