"""Consume only installed wheels in an otherwise clean environment."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iris_memory.control import Control
from iris_memory.identity import Identity


class Host:
    async def providers(self, control):
        return None, None


async def smoke(mode):
    if mode == "remote":
        assert importlib.util.find_spec("iris_memory_core") is None
        from iris_memory_sdk import AsyncIrisMemoryClient

        async with AsyncIrisMemoryClient(
            "http://127.0.0.1:1", bearer_token="test-not-used"
        ):
            pass
    else:
        assert importlib.util.find_spec("iris_memory_sdk") is None
        assert importlib.util.find_spec("fastapi") is None
        assert importlib.util.find_spec("faiss") is None
    with tempfile.TemporaryDirectory() as directory:
        c = Control(directory, Host(), mode)
        await c.start()
        try:
            if mode == "local":
                identity = Identity(
                    "chat:smoke",
                    "session:smoke",
                    "test",
                    "test",
                    "user",
                    "User",
                    "test:FriendMessage:user",
                    "msg",
                    False,
                )
                await c.apply(
                    {
                        "allow_development_sqlite": True,
                        "allowed_conversations": [identity.key],
                        "modules": {"memory": True},
                    },
                    0,
                )
                claim = await c.modules["memory"].remember(
                    identity, "Tea is preferred."
                )
                result = await c.modules["memory"].recall(identity, "tea")
                assert any(
                    item["resource_ref"]["resource_id"] == claim["claim_id"]
                    for item in result["candidates"]
                )
            else:
                assert not (Path(directory) / "core").exists()
        finally:
            await c.close()
    return {"mode": mode, "smoke": "passed", "dependency_isolation": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    print(json.dumps(asyncio.run(smoke(parser.parse_args().mode))))
