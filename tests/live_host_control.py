"""Finite controls for the explicitly configured disposable host, never production."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


async def run(action, output):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:18764/api/auth/login",
            json={"username": "iris-test", "password": "IrisSynthetic2026"},
        ) as response:
            value = await response.json()
            token = value["data"]["token"]
        headers = {"Authorization": "Bearer " + token}
        if action == "reload":
            async with session.post(
                "http://127.0.0.1:18764/api/v1/plugins/astrbot_plugin_iris_memory/reload",
                headers=headers,
            ) as response:
                result = {"http_status": response.status, "body": await response.json()}
        elif action == "inspect":
            base = "http://127.0.0.1:18764/api/v1/plugins/extensions/astrbot_plugin_iris_memory/"
            result = {}
            for endpoint in ("overview", "diagnostics", "features"):
                async with session.get(base + endpoint, headers=headers) as response:
                    result[endpoint] = {
                        "http_status": response.status,
                        "body": await response.json(),
                    }
        else:
            raise ValueError("unsupported action")
        await asyncio.to_thread(
            Path(output).write_text, json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(json.dumps({"action": action, "output": output, "completed": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reload", "inspect"))
    parser.add_argument("output")
    args = parser.parse_args()
    asyncio.run(run(args.action, args.output))
