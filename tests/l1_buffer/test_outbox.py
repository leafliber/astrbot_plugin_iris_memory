from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_iris_memory.iris_memory.l1_buffer.buffer import L1Buffer
from astrbot_plugin_iris_memory.iris_memory.l1_buffer.models import ContextMessage
from astrbot_plugin_iris_memory.iris_memory.l1_buffer.outbox import SummaryOutbox


def _message() -> ContextMessage:
    return ContextMessage(
        role="user",
        content="测试",
        timestamp=datetime.now(),
        token_count=1,
        source="u1",
    )


def test_outbox_survives_reopen_after_rotate_boundary(tmp_path):
    path = tmp_path / "outbox.db"
    outbox = SummaryOutbox(path)
    job_id = outbox.enqueue(
        queue_key="session",
        group_id="g1",
        summary="summary",
        messages=[_message()],
    )
    outbox.close()

    recovered = SummaryOutbox(path)
    job = recovered.get(job_id)
    assert job is not None
    assert job.messages[0].content == "测试"
    assert recovered.count() == 1
    recovered.close()


@pytest.mark.asyncio
async def test_completed_l2_stage_is_not_repeated_when_profile_retries(tmp_path):
    buffer = L1Buffer()
    buffer._outbox = SummaryOutbox(tmp_path / "outbox.db")
    job_id = buffer._outbox.enqueue(
        queue_key="session",
        group_id="g1",
        summary="summary",
        messages=[_message()],
    )
    buffer._write_summary_to_l2 = AsyncMock(return_value="mem")
    buffer._update_profile_after_summary = AsyncMock(
        side_effect=[RuntimeError("profile down"), None]
    )

    await buffer._process_outbox_job(job_id)
    pending = buffer._outbox.get(job_id)
    assert pending is not None and pending.l2_done
    assert not pending.profile_done

    await buffer._process_outbox_job(job_id)
    buffer._write_summary_to_l2.assert_awaited_once()
    assert buffer._outbox.get(job_id) is None
    buffer._outbox.close()
