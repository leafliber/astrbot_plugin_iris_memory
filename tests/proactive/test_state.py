"""proactive.state.StateManager 状态机测试

覆盖：backoff 升级与衰减、boost 窗口、冷却设置/检查、
锚点 TTL 过期、initiate pending 记账。
"""

import time

from iris_memory.proactive.prompts import MAX_BACKOFF_LEVEL
from iris_memory.proactive.state import GroupState, StateManager

GID = "g1"


def _state(nm_config, overrides=None):
    cm = nm_config(overrides=overrides)
    return StateManager(cm)


class TestCooldown:
    def test_set_and_check(self, state):
        minutes = state.set_cooldown(GID, 5)
        assert minutes == 5
        data = state.get_state(GID)
        assert data.state == GroupState.COOLDOWN
        assert data.cooldown_until > time.time()

    def test_minutes_clamped(self, state):
        assert state.set_cooldown(GID, 0) == 1
        assert state.set_cooldown(GID, 999) == 120

    def test_cooldown_auto_expires(self, state):
        state.set_cooldown(GID, 1)
        state.get_state(GID).cooldown_until = time.time() - 1
        assert state.get_state(GID).state == GroupState.IDLE


class TestBackoff:
    def test_escalation_raises_thresholds(self, nm_config):
        # medium 意愿：n = int(10 * 1.3^level * 0.85)，t = int(30 * 1.3^level * 0.85)
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        assert st.get_effective_thresholds(GID) == (8, 25)
        st.record_skip_reply(GID)
        st.record_skip_reply(GID)
        assert st.get_state(GID).backoff_level == 2
        assert st.get_effective_thresholds(GID) == (14, 43)

    def test_escalation_capped(self, state):
        for _ in range(MAX_BACKOFF_LEVEL + 5):
            state.record_skip_reply(GID)
        assert state.get_state(GID).backoff_level == MAX_BACKOFF_LEVEL

    def test_skip_resets_consecutive(self, state):
        state.get_state(GID).consecutive_replies = 3
        state.record_skip_reply(GID)
        assert state.get_state(GID).consecutive_replies == 0

    def test_decay_after_interval(self, nm_config):
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        data = st.get_state(GID)
        data.backoff_level = 3
        data.last_backoff_time = time.time() - 650  # BACKOFF_DECAY_INTERVAL=300 → 降 2 级
        st.get_effective_thresholds(GID)
        assert data.backoff_level == 1

    def test_no_decay_before_interval(self, nm_config):
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        data = st.get_state(GID)
        data.backoff_level = 3
        data.last_backoff_time = time.time() - 100
        st.get_effective_thresholds(GID)
        assert data.backoff_level == 3

    def test_actual_reply_reduces_backoff(self, state):
        data = state.get_state(GID)
        data.backoff_level = 2
        state.record_actual_reply(GID)
        assert data.backoff_level == 1
        assert data.consecutive_replies == 1

    def test_actual_reply_no_reduce_when_consecutive(self, state):
        data = state.get_state(GID)
        data.backoff_level = 2
        data.consecutive_replies = 2
        state.record_actual_reply(GID)
        assert data.backoff_level == 2
        assert data.consecutive_replies == 3


class TestBoost:
    def test_boost_window_set_on_reply(self, nm_config):
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        st.record_actual_reply(GID)
        data = st.get_state(GID)
        assert data.boost_initial == 0.6  # 默认 boost_factor
        assert data.boost_set_at > 0
        assert data.boost_until > time.time()
        # combined = 1.0 * 0.6：阈值临时降低
        assert st.get_effective_thresholds(GID) == (5, 15)

    def test_boost_expires(self, nm_config):
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        st.record_actual_reply(GID)
        data = st.get_state(GID)
        data.boost_until = time.time() - 1
        assert st.get_effective_thresholds(GID) == (8, 25)

    def test_boost_linear_decay(self, nm_config):
        st = _state(nm_config, {"default_n": 10, "default_t": 30})
        data = st.get_state(GID)
        # 进度 0.5：boost = 0.6 + (1 - 0.6) * 0.5 = 0.8
        data.boost_initial = 0.6
        data.boost_set_at = time.time() - 450
        data.boost_until = time.time() + 450
        assert st.get_effective_thresholds(GID) == (6, 20)

    def test_fatigue_after_max_boosted_replies(self, state):
        data = state.get_state(GID)
        data.consecutive_replies = 5  # max_boosted_replies 默认 5
        state.record_actual_reply(GID)
        assert data.consecutive_replies == 6
        assert data.boost_initial == 1.0  # 超出后不再 boost
        assert data.backoff_level == 1  # fatigue = min(6 - 5, 5)
        assert data.last_backoff_time > 0


class TestAnchor:
    def test_write_and_match(self, state):
        state.write_anchor(
            GID,
            kind="chime_in",
            topic="周末计划",
            bot_message="去哪玩",
            users=["u1"],
            keywords=["天气"],
            reason="感兴趣",
        )
        assert state.match_anchor_user(GID, "u1") is True
        assert state.match_anchor_user(GID, "u9") is False
        assert state.match_anchor_keyword(GID, "今天天气不错") == ["天气"]
        assert state.match_anchor_keyword(GID, "无关内容") == []
        anchor = state.get_anchor(GID)
        assert anchor.active is True
        assert anchor.kind == "chime_in"
        assert anchor.topic == "周末计划"

    def test_user_ttl_expiry(self, state):
        state.add_anchor_watch(GID, users=["u1"], ttl_minutes=-1)
        assert state.match_anchor_user(GID, "u1") is False
        assert state.get_anchor(GID).active is False

    def test_keyword_ttl_expiry(self, state):
        state.add_anchor_watch(GID, keywords=["天气"], ttl_minutes=-1)
        assert state.match_anchor_keyword(GID, "天气") == []
        assert state.get_anchor(GID).active is False

    def test_match_blocked_in_cooldown(self, state):
        state.add_anchor_watch(GID, users=["u1"], keywords=["天气"])
        state.set_cooldown(GID, 5)
        assert state.match_anchor_user(GID, "u1") is False
        assert state.match_anchor_keyword(GID, "天气") == []

    def test_write_replaces_old_anchor(self, state):
        state.write_anchor(GID, kind="chime_in", users=["u1"])
        state.write_anchor(GID, kind="follow_up", users=["u2"])
        assert state.match_anchor_user(GID, "u1") is False
        assert state.match_anchor_user(GID, "u2") is True
        assert state.get_anchor(GID).kind == "follow_up"

    def test_add_watch_merges_without_replace(self, state):
        state.write_anchor(GID, kind="chime_in", bot_message="hi", users=["u1"])
        state.add_anchor_watch(GID, users=["u2"], keywords=["k1"], reason="r")
        anchor = state.get_anchor(GID)
        assert anchor.participants == {"u1", "u2"}
        assert anchor.keywords == {"k1"}
        assert anchor.bot_message == "hi"
        assert anchor.reason == "r"

    def test_anchor_user_cap(self, state):
        state.add_anchor_watch(GID, users=[f"u{i}" for i in range(25)])
        assert len(state.get_anchor(GID).participants) == 20

    def test_anchor_keyword_cap(self, state):
        state.add_anchor_watch(GID, keywords=[f"k{i}" for i in range(15)])
        assert len(state.get_anchor(GID).keywords) == 10

    def test_remove_anchor_watch(self, state):
        state.add_anchor_watch(GID, users=["u1", "u2"], keywords=["k1"])
        state.remove_anchor_watch(GID, user_ids=["u1"])
        anchor = state.get_anchor(GID)
        assert anchor.participants == {"u2"}
        # 不带参数清空整个锚点
        state.remove_anchor_watch(GID)
        assert state.get_anchor(GID).active is False

    def test_close_anchor(self, state):
        state.write_anchor(GID, kind="chime_in", bot_message="hi", users=["u1"])
        state.close_anchor(GID)
        anchor = state.get_anchor(GID)
        assert anchor.active is False
        assert anchor.has_context is False

    def test_clear_decision_context_preserves_unrelated_state(self, state):
        state.set_willingness(GID, "high")
        state.get_state(GID).msg_count = 7
        state.set_observation(GID, "旧观察")
        state.write_anchor(
            GID,
            kind="follow_up",
            topic="旧话题",
            bot_message="旧回复",
            users=["u1"],
            keywords=["旧关键词"],
            reason="旧原因",
        )

        cleared = state.clear_decision_context(GID)

        assert cleared == {"observation": True, "anchor": True}
        assert state.get_observation(GID) == ""
        assert state.get_anchor(GID).has_context is False
        assert state.get_anchor(GID).topic == ""
        assert state.get_willingness(GID) == "high"
        assert state.get_state(GID).msg_count == 7


class TestInitiateAccounting:
    def test_record_initiate(self, state):
        state.record_initiate(GID, topic="新话题", bot_message="大家好", users=["u1"])
        data = state.get_state(GID)
        assert data.initiate_daily_count == 1
        assert data.last_initiate_time > 0
        assert data.initiate_pending_since > 0
        assert state.is_initiate_pending(GID) is True
        anchor = state.get_anchor(GID)
        assert anchor.kind == "initiate"
        assert anchor.bot_message == "大家好"

    def test_record_initiate_resets_drive(self, state):
        data = state.get_state(GID)
        data.initiate_score = 0.9
        data.initiate_target = 1.2
        state.record_initiate(GID)
        assert data.initiate_score == 0.0
        assert data.initiate_target == 0.0

    def test_consume_pending_resets_streak(self, state):
        state.record_initiate(GID)
        data = state.get_state(GID)
        data.initiate_no_reply_streak = 2
        assert state.consume_initiate_pending(GID) is True
        assert data.initiate_pending_since == 0.0
        assert data.initiate_no_reply_streak == 0
        assert state.consume_initiate_pending(GID) is False

    def test_unanswered_increments_streak(self, state):
        state.record_initiate(GID)
        state.record_initiate_unanswered(GID)
        data = state.get_state(GID)
        assert data.initiate_pending_since == 0.0
        assert data.initiate_no_reply_streak == 1
        # 无 pending 时重复调用不累计
        state.record_initiate_unanswered(GID)
        assert data.initiate_no_reply_streak == 1

    def test_cross_day_reset(self, state):
        data = state.get_state(GID)
        data.initiate_count_date = "2000-01-01"
        data.initiate_daily_count = 5
        data.initiate_no_reply_streak = 3
        refreshed = state.get_state(GID)
        assert refreshed.initiate_daily_count == 0
        assert refreshed.initiate_no_reply_streak == 0
        assert refreshed.initiate_count_date == time.strftime("%Y-%m-%d", time.localtime())


class TestInitiateDrive:
    """发起意愿值：积累 / 消退 / 冻结 / 采样 / 重置"""

    def test_target_sampled_lazily(self, state):
        now = time.time()
        state.update_initiate_drive(GID, now=now, quiet_seconds=100)
        target = state.get_state(GID).initiate_target
        assert StateManager.INITIATE_TARGET_MIN <= target <= StateManager.INITIATE_TARGET_MAX

    def test_first_tick_no_elapsed(self, state):
        # 首次结算（score_at=0）不积累
        now = time.time()
        fired = state.update_initiate_drive(GID, now=now, quiet_seconds=9999)
        assert fired is False
        assert state.get_state(GID).initiate_score == 0.0

    def test_accumulate_scales_with_willingness(self, state):
        now = time.time()
        data = state.get_state(GID)
        data.initiate_target = 1.5
        data.initiate_score_at = now - 600
        state.update_initiate_drive(GID, now=now, quiet_seconds=9999)
        medium_score = data.initiate_score
        assert medium_score > 0

        state.set_willingness(GID, "high")
        data.initiate_score = 0.0
        data.initiate_score_at = now - 600
        state.update_initiate_drive(GID, now=now, quiet_seconds=9999)
        assert data.initiate_score > medium_score

        state.set_willingness(GID, "low")
        data.initiate_score = 0.0
        data.initiate_score_at = now - 600
        state.update_initiate_drive(GID, now=now, quiet_seconds=9999)
        assert data.initiate_score < medium_score

    def test_score_capped_at_target(self, state):
        now = time.time()
        data = state.get_state(GID)
        data.initiate_score = 1.45  # 距阈值 0.05，一拍（≈0.098）即可越线
        data.initiate_target = 1.5
        data.initiate_score_at = now - 600
        assert state.update_initiate_drive(GID, now=now, quiet_seconds=9999) is True
        assert data.initiate_score == 1.5

    def test_decay_when_active(self, state):
        now = time.time()
        data = state.get_state(GID)
        data.initiate_score = 0.5
        data.initiate_target = 1.5
        data.initiate_score_at = now - 600
        fired = state.update_initiate_drive(GID, now=now, quiet_seconds=0)
        assert fired is False
        assert data.initiate_score < 0.5

    def test_freeze_keeps_score(self, state):
        now = time.time()
        data = state.get_state(GID)
        data.initiate_score = 0.5
        data.initiate_target = 1.5
        data.initiate_score_at = now - 600
        fired = state.update_initiate_drive(GID, now=now, quiet_seconds=9999, freeze=True)
        assert fired is False
        assert data.initiate_score == 0.5
        assert data.initiate_score_at == now

    def test_elapsed_capped_after_downtime(self, state):
        now = time.time()
        data = state.get_state(GID)
        data.initiate_target = 1.5
        data.initiate_score_at = now - 86400  # 一天未结算
        state.update_initiate_drive(GID, now=now, quiet_seconds=99999)
        cap = state._config.proactive_check_interval * 120.0
        base = 1.0 / (state._config.proactive_quiet_minutes * 60.0)
        # 最多按结算步长上限积累（medium t_factor=0.85），而非整天
        assert data.initiate_score <= cap * base / 0.85 + 1e-9

    def test_reset_initiate_drive(self, state):
        data = state.get_state(GID)
        data.initiate_score = 0.9
        data.initiate_target = 1.2
        state.reset_initiate_drive(GID)
        assert data.initiate_score == 0.0
        assert data.initiate_target == 0.0

    def test_drive_fields_serialize_round_trip(self, state):
        data = state.get_state(GID)
        data.initiate_score = 0.42
        data.initiate_target = 1.1
        data.initiate_score_at = 12345.0
        snapshot = state._serialize_group(data)
        restored = state._deserialize_group(snapshot)
        assert restored.initiate_score == 0.42
        assert restored.initiate_target == 1.1
        assert restored.initiate_score_at == 12345.0


class TestDetectRateLimit:
    def test_can_detect_fresh_group(self, state):
        assert state.can_detect(GID) is True

    def test_record_detect_time_blocks(self, state):
        state.record_detect_time(GID)
        assert state.can_detect(GID) is False

    def test_follow_up_half_interval(self, state):
        # 默认 trigger_min_interval=30s，follow_up 减半为 15s
        state.get_state(GID).last_detect_time = time.time() - 20
        assert state.can_detect(GID) is False
        assert state.can_detect(GID, follow_up=True) is True
