"""
DreamTask 梦境任务测试

测试梦境任务的核心功能：
- 六阶段流水线编排
- DreamReport 报告生成
- 各阶段开关控制
- 错误处理与降级
"""

import pytest
from unittest.mock import Mock, patch

from iris_memory.dream.dream_task import (
    DreamTask,
    DreamReport,
    DreamPhaseReport,
    _PHASES_THAT_MUTATE_ENTRIES,
)


class TestDreamReport:
    """DreamReport 测试类"""

    def test_summary_all_succeeded(self):
        report = DreamReport(total_duration_ms=1000)
        report.phases = [
            DreamPhaseReport(
                phase="consolidation", enabled=True, success=True, duration_ms=100
            ),
            DreamPhaseReport(
                phase="temporal_anchor", enabled=True, success=True, duration_ms=200
            ),
        ]
        assert "2 阶段成功" in report.summary
        assert "1000ms" in report.summary

    def test_summary_with_failures(self):
        report = DreamReport(total_duration_ms=500)
        report.phases = [
            DreamPhaseReport(
                phase="consolidation", enabled=True, success=True, duration_ms=100
            ),
            DreamPhaseReport(
                phase="contradiction",
                enabled=True,
                success=False,
                duration_ms=50,
                error="test error",
            ),
        ]
        assert "1 阶段成功" in report.summary
        assert "1 阶段失败" in report.summary

    def test_summary_with_skipped(self):
        report = DreamReport(total_duration_ms=200)
        report.phases = [
            DreamPhaseReport(
                phase="consolidation", enabled=True, success=True, duration_ms=100
            ),
            DreamPhaseReport(
                phase="pattern_discovery", enabled=False, success=True, duration_ms=0
            ),
        ]
        assert "1 阶段成功" in report.summary
        assert "1 阶段跳过" in report.summary


class TestDreamTask:
    """DreamTask 测试类"""

    @pytest.fixture
    def mock_component_manager(self):
        manager = Mock()
        manager.get_component = Mock(return_value=None)
        manager.check_component = Mock(return_value="unavailable")
        return manager

    @pytest.fixture
    def dream_task(self, mock_component_manager):
        return DreamTask(mock_component_manager)

    def test_init(self, dream_task):
        assert dream_task._component_manager is not None

    def test_temporal_anchor_in_mutating_phases(self):
        """回归：temporal_anchor 必须在 _PHASES_THAT_MUTATE_ENTRIES 中

        temporal_anchor 阶段会修改 L2 entries（将相对时间表达锚定为绝对时间），
        因此执行后必须使缓存的 entries 失效以重载，否则后续阶段使用过期缓存。
        """
        assert "temporal_anchor" in _PHASES_THAT_MUTATE_ENTRIES

    @pytest.mark.asyncio
    async def test_execute_dream_disabled(self, dream_task):
        with patch("iris_memory.dream.dream_task.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(return_value=False)
            mock_config.return_value = mock_config_instance

            report = await dream_task.execute()

            assert isinstance(report, DreamReport)
            assert len(report.phases) == 0

    @pytest.mark.asyncio
    async def test_execute_l2_unavailable(self, dream_task, mock_component_manager):
        mock_component_manager.get_component = Mock(return_value=None)

        with patch("iris_memory.dream.dream_task.get_config") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.get = Mock(
                side_effect=lambda key, default=None: {
                    "scheduled_tasks.enable_dream": True,
                }.get(key, default)
            )
            mock_config.return_value = mock_config_instance

            report = await dream_task.execute()

            assert isinstance(report, DreamReport)
            assert len(report.phases) == 0
