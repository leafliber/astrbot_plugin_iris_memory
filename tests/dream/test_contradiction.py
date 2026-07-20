"""
ContradictionPhase 矛盾消解测试

测试核心功能：
- 近邻矛盾检测
- LLM 矛盾判断
- 冲突解决策略
- _parse_resolved_index 边界检查
- _parse_merged_content 多行内容
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from iris_memory.dream.contradiction import ContradictionPhase


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "dream_contradiction_similarity_floor": 0.55,
            "dream_contradiction_similarity_ceiling": 0.85,
            "dream_contradiction_max_groups": 20,
        }.get(key, default)
    )
    return mock


class TestContradictionPhase:
    @pytest.fixture
    def phase(self):
        return ContradictionPhase()

    @pytest.mark.asyncio
    async def test_execute_no_llm(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = None
        llm = None

        with patch(
            "iris_memory.dream.contradiction.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["groups_checked"] == 0
        assert result["resolved"] == 0

    @pytest.mark.asyncio
    async def test_execute_no_entries(self, phase):
        l2 = Mock()
        l2.is_available = True
        l2.get_all_entries = AsyncMock(return_value=[])
        l3 = None
        llm = Mock()

        with patch(
            "iris_memory.dream.contradiction.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["groups_checked"] == 0
        assert result["resolved"] == 0

    def test_parse_resolved_index_valid(self, phase):
        response = "RESOLVED: 2\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response, group_size=3) == 1

    def test_parse_resolved_index_out_of_range(self, phase):
        response = "RESOLVED: 5\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response, group_size=3) is None

    def test_parse_resolved_index_zero(self, phase):
        response = "RESOLVED: 0\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response, group_size=3) is None

    def test_parse_resolved_index_negative(self, phase):
        response = "RESOLVED: -1\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response, group_size=3) is None

    def test_parse_resolved_index_no_group_size(self, phase):
        response = "RESOLVED: 100\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response) == 99

    def test_parse_resolved_index_invalid(self, phase):
        response = "RESOLVED: abc\nMERGED: 合并内容"
        assert phase._parse_resolved_index(response) is None

    def test_parse_merged_content_single_line(self, phase):
        response = "RESOLVED: 1\nMERGED: 合并后的内容"
        assert phase._parse_merged_content(response) == "合并后的内容"

    def test_parse_merged_content_multiline(self, phase):
        response = "RESOLVED: 1\nMERGED: 第一行\n第二行\n第三行"
        result = phase._parse_merged_content(response)
        assert "第一行" in result
        assert "第二行" in result
        assert "第三行" in result

    def test_parse_merged_content_stops_at_keyword(self, phase):
        response = "RESOLVED: 1\nMERGED: 合并内容\nPATTERN: 不应包含"
        result = phase._parse_merged_content(response)
        assert "合并内容" in result
        assert "PATTERN" not in result

    def test_parse_merged_content_no_merged(self, phase):
        response = "NO_CONFLICT"
        assert phase._parse_merged_content(response) is None
