"""Temporary-data process RSS smoke; not a production/large-index benchmark."""

import argparse
import asyncio
import json
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iris_memory.control import Control


class Host:
    async def providers(self, control):
        return None, None


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1048576 if sys.platform == "darwin" else 1024)


async def measure(mode):
    with tempfile.TemporaryDirectory(prefix="iris-baseline-") as directory:
        c = Control(directory, Host())
        start = time.perf_counter()
        before = peak_mib()
        await c.start()
        if mode == "local":
            await c.apply(
                {"allow_development_sqlite": True, "modules": {"memory": True}}, 0
            )
        active = peak_mib()
        elapsed = time.perf_counter() - start
        await c.close()
        return {
            "mode": mode,
            "baseline_peak_mib": round(before, 2),
            "active_peak_mib": round(active, 2),
            "peak_increase_mib": round(active - before, 2),
            "start_seconds": round(elapsed, 3),
            "iris_threads_after_close": [
                t.name for t in threading.enumerate() if t.name.startswith("iris-")
            ],
            "heavy_modules_loaded": [
                name
                for name in ("faiss", "numpy", "uvicorn", "fastapi")
                if name in sys.modules
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "local"), required=True)
    print(json.dumps(asyncio.run(measure(parser.parse_args().mode)), indent=2))
