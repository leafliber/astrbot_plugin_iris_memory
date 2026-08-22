"""proactive 统计同步链路回归测试

覆盖 group_state_summary → sync_stats_group_state → StatsCollector.update_group_state，
防止 summary 键与 update_group_state 形参再次漂移（曾因新增 initiate_rolling_24h_count
等字段导致周期保存反复抛 TypeError）。
"""

import inspect
import time

from iris_memory.proactive.api import group_state_summary, sync_stats_group_state
from iris_memory.proactive.stats import StatsCollector

GID = "g1"


def _prepare(state):
    state.add_to_whitelist(GID)
    data = state.get_state(GID)
    data.initiate_timestamps = [time.time() - 60, time.time() - 30]
    data.initiate_hazard_probability = 0.35
    data.initiate_next_check_at = time.time() + 120


class TestStatsSync:
    def test_summary_keys_match_update_group_state_signature(self, state):
        """summary 的键集合必须与 update_group_state 形参完全一致"""
        _prepare(state)
        params = set(inspect.signature(StatsCollector.update_group_state).parameters)
        params -= {"self", "group_id"}
        assert set(group_state_summary(state, GID)) == params

    def test_sync_populates_group_stats(self, state):
        _prepare(state)
        stats = StatsCollector()
        stats.enabled = True
        sync_stats_group_state(state, stats)
        summary = stats.get_group_summaries()[0]
        assert summary["group_id"] == GID
        assert summary["initiate_daily_count"] == 2
        assert summary["initiate_rolling_24h_count"] == 2
        assert summary["initiate_hazard_probability"] == 0.35
        assert summary["initiate_next_check_at"] > time.time()

    def test_sync_disabled_stats_is_noop(self, state):
        _prepare(state)
        stats = StatsCollector()
        sync_stats_group_state(state, stats)
        assert stats.get_group_summaries() == []
