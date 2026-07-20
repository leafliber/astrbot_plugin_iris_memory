"""
ConsolidationPhase 合并重复项测试

测试核心功能：
- 并查集构建
- 话题级归拢
- LLM 合并调用
- 批量处理
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.dream.consolidation import ConsolidationPhase


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "dream_consolidation_similarity_threshold": 0.85,
            "dream_consolidation_batch_size": 10,
            "dream_consolidation_scan_budget": 500,
            "dream_consolidation_query_batch_size": 50,
            "dream_consolidation_max_group_size": 5,
        }.get(key, default)
    )
    return mock


class TestConsolidationPhase:
    @pytest.fixture
    def phase(self):
        return ConsolidationPhase()

    @pytest.mark.asyncio
    async def test_execute_l2_unavailable(self, phase):
        l2 = Mock()
        l2.is_available = False
        l3 = None
        llm = None

        with patch(
            "iris_memory.dream.consolidation.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["merged_groups"] == 0
        assert result["deleted_entries"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_entries(self, phase):
        l2 = Mock()
        l2.is_available = True
        l2.get_all_entries = AsyncMock(return_value=[])
        l3 = None
        llm = Mock()

        with patch(
            "iris_memory.dream.consolidation.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["merged_groups"] == 0
        assert result["deleted_entries"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_llm(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = None
        llm = None

        with patch(
            "iris_memory.dream.consolidation.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["merged_groups"] == 0
        assert result["deleted_entries"] == 0

    @pytest.mark.asyncio
    async def test_merge_memories_success(self, phase):
        llm = Mock()
        llm.generate_direct = AsyncMock(return_value="合并后的记忆内容")

        result = await phase._merge_memories("记忆1", "记忆2", llm)

        assert result == "合并后的记忆内容"
        llm.generate_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_memories_llm_failure(self, phase):
        llm = Mock()
        llm.generate_direct = AsyncMock(return_value=None)

        result = await phase._merge_memories("记忆1", "记忆2", llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_merge_memories_llm_exception(self, phase):
        llm = Mock()
        llm.generate_direct = AsyncMock(side_effect=Exception("LLM error"))

        result = await phase._merge_memories("记忆1", "记忆2", llm)

        assert result is None
