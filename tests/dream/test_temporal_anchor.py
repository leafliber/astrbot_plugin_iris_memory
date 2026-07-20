"""
TemporalAnchorPhase 时间锚定测试

测试核心功能：
- 相对时间词识别
- 绝对日期转换
- 批量处理
- LLM 润色
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from iris_memory.dream.temporal_anchor import (
    TemporalAnchorPhase,
    _RELATIVE_TIME_PATTERN,
    _resolve_relative_time,
    _format_date,
)


class TestRelativeTimeResolution:
    def test_yesterday(self):
        match = _RELATIVE_TIME_PATTERN.search("昨天我们决定了方案")
        assert match is not None
        base = datetime(2026, 5, 25)
        result = _resolve_relative_time(match, base)
        assert result is not None
        assert "2026" in result
        assert "5" in result
        assert "24" in result

    def test_today(self):
        match = _RELATIVE_TIME_PATTERN.search("今天天气不错")
        assert match is not None
        base = datetime(2026, 5, 25)
        result = _resolve_relative_time(match, base)
        assert result is not None
        assert "2026年5月25日" in result

    def test_days_ago(self):
        match = _RELATIVE_TIME_PATTERN.search("3天前讨论过")
        assert match is not None
        base = datetime(2026, 5, 25)
        result = _resolve_relative_time(match, base)
        assert result is not None
        assert "5月22日" in result

    def test_no_relative_time(self):
        match = _RELATIVE_TIME_PATTERN.search("我们决定使用Redis")
        assert match is None

    def test_last_week(self):
        match = _RELATIVE_TIME_PATTERN.search("上周开会讨论了")
        assert match is not None
        base = datetime(2026, 5, 25)
        result = _resolve_relative_time(match, base)
        assert result is not None


def _mock_config():
    mock = Mock()
    mock.get = Mock(
        side_effect=lambda key, default=None: {
            "dream_temporal_anchor_batch_size": 50,
        }.get(key, default)
    )
    return mock


class TestTemporalAnchorPhase:
    @pytest.fixture
    def phase(self):
        return TemporalAnchorPhase()

    @pytest.mark.asyncio
    async def test_execute_no_llm(self, phase):
        l2 = Mock()
        l2.is_available = True
        l3 = None
        llm = None

        with patch(
            "iris_memory.dream.temporal_anchor.get_config", return_value=_mock_config()
        ):
            result = await phase.execute(l2, l3, llm)

        assert result["scanned"] == 0
        assert result["anchored"] == 0


class TestFormatDate:
    def test_format_date_no_leading_zero(self):
        date = datetime(2026, 5, 3)
        result = _format_date(date)
        assert "05月03日" not in result or "5月3日" in result

    def test_format_date_double_digit(self):
        date = datetime(2026, 12, 25)
        result = _format_date(date)
        assert "2026年" in result
        assert "12月" in result
        assert "25日" in result
