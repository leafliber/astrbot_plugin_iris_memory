import asyncio
from datetime import datetime, timedelta

import pytest

from astrbot_plugin_iris_memory.iris_memory.image.models import ImageParseStatus, ImageQueueItem
from astrbot_plugin_iris_memory.iris_memory.l1_buffer.buffer import L1Buffer


@pytest.mark.asyncio
async def test_fifty_concurrent_claims_return_same_hash_once():
    buffer = L1Buffer()
    item = ImageQueueItem(image_hash="same")
    buffer.add_image("session", item)

    claims = await asyncio.gather(
        *(
            buffer.claim_pending_images("session", 1, f"claim-{index}")
            for index in range(50)
        )
    )

    assert sum(len(batch) for batch in claims) == 1
    assert item.status == ImageParseStatus.PROCESSING
    assert item.attempt_count == 1


@pytest.mark.asyncio
async def test_stale_processing_claim_is_safely_recovered():
    buffer = L1Buffer()
    item = ImageQueueItem(
        image_hash="stale",
        status=ImageParseStatus.PROCESSING,
        claim_token="dead-worker",
        claimed_at=datetime.now() - timedelta(seconds=120),
    )
    buffer.add_image("session", item)

    claimed = await buffer.claim_pending_images(
        "session", 1, "new-worker", stale_after_seconds=60
    )

    assert claimed == [item]
    assert item.claim_token == "new-worker"
    assert item.attempt_count == 1


@pytest.mark.asyncio
async def test_only_claim_owner_can_finish_image():
    buffer = L1Buffer()
    item = ImageQueueItem(image_hash="owned")
    buffer.add_image("session", item)
    await buffer.claim_pending_images("session", 1, "owner")

    assert not buffer.mark_image_parsed(
        "session", "owned", ImageParseStatus.SUCCESS, "intruder"
    )
    assert item.status == ImageParseStatus.PROCESSING
    assert buffer.mark_image_parsed(
        "session", "owned", ImageParseStatus.SUCCESS, "owner"
    )
    assert item.status == ImageParseStatus.SUCCESS
    assert item.claim_token == ""
