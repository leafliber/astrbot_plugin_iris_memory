"""
DreamTask 梦境任务测试

测试梦境任务的核心功能：
- 合并后的增量流水线编排
- DreamReport 报告生成
- 各阶段开关控制
- 错误处理与降级
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from iris_memory.dream.dream_task import (
    DreamTask,
    DreamReport,
    DreamPhaseReport,
    _GLOBAL_L3_CONFIG_KEY,
    _PHASE_CONFIG_KEYS,
    _PHASES_THAT_MUTATE_ENTRIES,
)
from iris_memory.dream.state import DreamCursorStore


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

    def test_cost_aggregates_phase_metrics(self):
        report = DreamReport()
        report.phases = [
            DreamPhaseReport(
                phase="reconciliation",
                enabled=True,
                success=True,
                duration_ms=1,
                llm_calls=2,
                input_tokens=100,
                output_tokens=20,
                embedding_requests=1,
                embedded_texts=5,
            ),
            DreamPhaseReport(
                phase="knowledge_induction",
                enabled=True,
                success=True,
                duration_ms=1,
                llm_calls=1,
                input_tokens=50,
                output_tokens=10,
                embedding_requests=2,
                embedded_texts=2,
            ),
        ]

        assert report.cost == {
            "llm_calls": 3,
            "input_tokens": 150,
            "output_tokens": 30,
            "embedding_requests": 3,
            "embedded_texts": 7,
        }


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

    def test_only_new_stage_switches_are_used(self):
        assert set(_PHASE_CONFIG_KEYS.values()) == {
            "scheduled_tasks.dream_stage_temporal_anchor_enabled",
            "scheduled_tasks.dream_stage_reconciliation_enabled",
            "scheduled_tasks.dream_stage_knowledge_induction_enabled",
            "scheduled_tasks.dream_stage_l2_pruning_enabled",
        }
        assert (
            _GLOBAL_L3_CONFIG_KEY
            == "scheduled_tasks.dream_stage_l3_maintenance_enabled"
        )

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

    @pytest.mark.asyncio
    async def test_global_l3_pruning_runs_once_for_multiple_personas(self, dream_task):
        l2 = Mock(is_available=True)
        l2.get_all_persona_ids = AsyncMock(return_value=["default", "alt"])
        l3 = Mock(is_available=True)
        dream_task._get_l2 = Mock(return_value=l2)
        dream_task._get_l3 = Mock(return_value=l3)
        dream_task._get_llm = Mock(return_value=None)
        dream_task._run_pipeline_for_persona = AsyncMock()
        dream_task._run_phase = AsyncMock(
            return_value=DreamPhaseReport(
                phase="pruning_l3_global",
                enabled=True,
                success=True,
                duration_ms=1,
            )
        )

        with patch("iris_memory.dream.dream_task.get_config") as mock_config:
            config = Mock()
            config.get = Mock(
                side_effect=lambda key, default=None: {
                    "scheduled_tasks.enable_dream": True,
                    "scheduled_tasks.dream_stage_l3_maintenance_enabled": True,
                }.get(key, default)
            )
            mock_config.return_value = config
            report = await dream_task.execute()

        assert dream_task._run_pipeline_for_persona.await_count == 2
        dream_task._run_phase.assert_awaited_once()
        assert report.phases[0].phase == "pruning_l3_global"

    @pytest.mark.asyncio
    async def test_stage_cursor_resumes_at_partial_stage(self, tmp_path, dream_task):
        store = DreamCursorStore(tmp_path / "dream-cursor.db")
        dream_task._cursor_store = store
        dream_task._cursor_store_initialized = True
        l2 = Mock()
        l2.get_all_entries = AsyncMock(return_value=[])
        phase_reports = [
            DreamPhaseReport("temporal_anchor", True, True, 1),
            DreamPhaseReport("reconciliation", True, True, 1),
            DreamPhaseReport(
                "knowledge_induction", True, True, 1, details={"has_more": True}
            ),
        ]
        dream_task._run_phase = AsyncMock(side_effect=phase_reports)

        with patch("iris_memory.dream.dream_task.get_config") as mock_config:
            config = Mock()
            config.get = Mock(return_value=True)
            mock_config.return_value = config
            first = await dream_task._run_pipeline_for_persona(
                "p1", l2, None, None, DreamReport(), cycle=1, cursor_store=store
            )

        assert first is False
        assert store.completed_stages(1, "p1") == {
            "temporal_anchor",
            "reconciliation",
        }

        dream_task._run_phase = AsyncMock(
            side_effect=[
                DreamPhaseReport("knowledge_induction", True, True, 1),
                DreamPhaseReport("pruning", True, True, 1),
            ]
        )
        with patch("iris_memory.dream.dream_task.get_config") as mock_config:
            config = Mock()
            config.get = Mock(return_value=True)
            mock_config.return_value = config
            second = await dream_task._run_pipeline_for_persona(
                "p1", l2, None, None, DreamReport(), cycle=1, cursor_store=store
            )

        assert second is True
        called_phases = [call.args[0] for call in dream_task._run_phase.await_args_list]
        assert called_phases == ["knowledge_induction", "pruning"]

    def test_cursor_store_survives_reopen_per_persona_and_stage(self, tmp_path):
        path = tmp_path / "dream-cursor.db"
        first = DreamCursorStore(path)
        first.mark_stage(1, "persona-a", "temporal_anchor", completed=True)
        first.set_next_persona("persona-b")

        reopened = DreamCursorStore(path)
        assert reopened.get_cycle() == 1
        assert reopened.get_next_persona() == "persona-b"
        assert reopened.completed_stages(1, "persona-a") == {"temporal_anchor"}
