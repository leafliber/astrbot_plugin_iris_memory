"""Read-only dependency probe; no data directory, listener or model is created."""

import argparse
import importlib.metadata
import json
import sqlite3
import sys


def check(mode):
    if sys.version_info < (3, 12):
        raise RuntimeError("Core / SDK development snapshots require Python >= 3.12")
    if mode == "local":
        from iris_memory_core.embedded import EmbeddedMemory
        from iris_memory_core.embedded_providers import AsyncEmbeddingAdapter

        required = (
            "start",
            "aclose",
            "execute",
            "register_actor",
            "provision_agent",
            "diagnostics",
        )
        if not all(callable(getattr(EmbeddedMemory, name, None)) for name in required):
            raise RuntimeError("Core wheel lacks the plugin public API")
        assert AsyncEmbeddingAdapter
        package = "iris-memory-core"
    else:
        from iris_memory_sdk import AsyncIrisMemoryClient

        required = (
            "aclose",
            "negotiate",
            "report_recall_usage",
            "get_entity_profile",
            "publish_persona_revision",
        )
        if not all(
            callable(getattr(AsyncIrisMemoryClient, name, None)) for name in required
        ):
            raise RuntimeError("SDK wheel lacks the required async API")
        package = "iris-memory-sdk"
    return {
        "mode": mode,
        "package": package,
        "version": importlib.metadata.version(package),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "feature_probe": "passed",
        "note": "Version alone does not identify the unpublished development artifact. Compare wheel SHA-256 with docs/DEPENDENCY_SNAPSHOT.json.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.mode), ensure_ascii=False, indent=2))
