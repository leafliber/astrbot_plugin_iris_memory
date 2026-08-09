from unittest.mock import AsyncMock, Mock, patch

import pytest

from iris_memory.dream.reconciliation import ReconciliationPhase
from iris_memory.l2_memory.models import MemoryEntry, MemorySearchResult


def _config():
    values = {
        "dream_consolidation_similarity_threshold": 0.85,
        "dream_consolidation_batch_size": 10,
        "dream_consolidation_scan_budget": 200,
        "dream_consolidation_query_batch_size": 50,
        "dream_consolidation_max_group_size": 5,
        "dream_consolidation_query_top_k": 5,
        "dream_contradiction_similarity_floor": 0.55,
        "dream_contradiction_similarity_ceiling": 0.85,
        "dream_contradiction_max_groups": 20,
        "dream_contradiction_scan_budget": 200,
        "dream_contradiction_query_batch_size": 50,
        "dream_contradiction_llm_batch_size": 5,
        "isolation_config.enable_group_memory_isolation": False,
    }
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: values.get(key, default))
    return config


@pytest.mark.asyncio
async def test_reconciliation_reuses_one_neighbor_scan_for_both_policies():
    entries = [
        MemoryEntry(id="a", content="用户喜欢苹果", metadata={"user_id": "u1"}),
        MemoryEntry(id="b", content="用户很喜欢苹果", metadata={"user_id": "u1"}),
        MemoryEntry(id="c", content="用户喜欢咖啡", metadata={"user_id": "u1"}),
        MemoryEntry(id="d", content="用户不再喜欢咖啡", metadata={"user_id": "u1"}),
    ]

    def result(entry, score):
        return MemorySearchResult(entry=entry, score=score, distance=1 - score)

    l2 = Mock()
    l2.batch_retrieve_by_ids = AsyncMock(
        return_value=[
            [result(entries[1], 0.92)],
            [result(entries[0], 0.92)],
            [result(entries[3], 0.70)],
            [result(entries[2], 0.70)],
        ]
    )
    l2.add_memory = AsyncMock(return_value="merged")
    l2.delete_entries = AsyncMock(return_value=True)
    l2.update_content = AsyncMock(return_value=True)
    l2.update_metadata = AsyncMock(return_value=True)

    llm = Mock()
    llm.generate_direct = AsyncMock(
        side_effect=[
            "用户喜欢苹果",
            '[{"group": 1, "conflict": true, "keep": 2, '
            '"merged": "用户现在不再喜欢咖啡"}]',
        ]
    )

    with patch(
        "iris_memory.dream.reconciliation.get_config", return_value=_config()
    ):
        details = await ReconciliationPhase().execute(
            l2, None, llm, entries=entries, persona_id="default"
        )

    l2.batch_retrieve_by_ids.assert_awaited_once()
    assert llm.generate_direct.await_count == 2
    assert details["consolidation"]["merged_groups"] == 1
    assert details["contradiction"]["resolved"] == 1
