import asyncio
from unittest.mock import MagicMock, patch

import pytest

from astrbot_plugin_iris_memory.iris_memory.image.coordinator import ImageParseCoordinator


@pytest.mark.asyncio
async def test_global_image_workers_and_per_session_serialization():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "image_queue_limit": 200,
        "image_max_concurrent_parse": 2,
    }.get(key, default)
    coordinator = ImageParseCoordinator()
    with patch("astrbot_plugin_iris_memory.iris_memory.image.coordinator.get_config", return_value=config):
        await coordinator.initialize()

    active = 0
    max_active = 0
    session_active: dict[str, int] = {}
    session_max: dict[str, int] = {}
    lock = asyncio.Lock()

    async def job(session_id: str) -> None:
        nonlocal active, max_active
        async with lock:
            active += 1
            session_active[session_id] = session_active.get(session_id, 0) + 1
            max_active = max(max_active, active)
            session_max[session_id] = max(
                session_max.get(session_id, 0), session_active[session_id]
            )
        await asyncio.sleep(0.005)
        async with lock:
            active -= 1
            session_active[session_id] -= 1

    completions = []
    for index in range(20):
        session_id = f"s{index % 4}"
        completion = coordinator.submit(
            key=f"job-{index}",
            session_id=session_id,
            runner=lambda session_id=session_id: job(session_id),
        )
        assert completion is not None
        completions.append(completion)

    await asyncio.gather(*(event.wait() for event in completions))
    assert max_active <= 2
    assert max(session_max.values()) == 1
    await coordinator.shutdown()
