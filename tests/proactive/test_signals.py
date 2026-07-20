"""signals.SignalGate 门控矩阵测试

覆盖：停用 / 冷却 / 静音 / 白名单（Gatekeeper 层）/ 锚点用户与关键词命中 /
采样计数触发 / 采样时间触发，以及 evaluate_timer 的 initiate 门控。
"""

import time
from unittest.mock import Mock

from iris_memory.proactive.perception import Gatekeeper
from iris_memory.proactive.signals import SignalGate
from iris_memory.proactive.state import GroupState, StateManager

GID = "g1"


def _build(nm_config, cfg=None, overrides=None):
    cm = nm_config(cfg=cfg, overrides=overrides)
    st = StateManager(cm)
    return cm, st, SignalGate(cm, st)


def _event(group_id=GID, text="hello", private=False):
    ev = Mock()
    ev.message_str = text
    ev.is_private_chat.return_value = private
    ev.get_group_id.return_value = group_id
    return ev


class TestMessageGateBasic:
    def test_disabled_config_blocks_all(self, nm_config):
        _, st, gate = _build(nm_config, cfg={"proactive": {"enabled": False}})
        st.add_anchor_watch(GID, users=["u1"])
        assert gate.evaluate_message(GID, "u1", "hi") is None

    def test_cooldown_blocks(self, nm_config):
        _, st, gate = _build(nm_config)
        st.set_cooldown(GID, 5)
        st.add_anchor_watch(GID, users=["u1"])
        assert gate.evaluate_message(GID, "u1", "hi") is None

    def test_muted_blocks(self, nm_config):
        _, st, gate = _build(nm_config)
        st.is_muted = lambda: True
        assert gate.evaluate_message(GID, "u1", "hi") is None

    def test_no_signal_increments_count(self, gate, state):
        assert gate.evaluate_message(GID, "u1", "今天聊点什么") is None
        assert state.get_state(GID).msg_count == 1


class TestMessageGateAnchor:
    def test_anchor_user_hit_returns_follow_up(self, gate, state):
        state.increment_msg_count(GID)
        state.add_anchor_watch(GID, users=["u1"])
        assert gate.evaluate_message(GID, "u1", "我回来了") == "follow_up"
        # 命中后采样计数被重置
        assert state.get_state(GID).msg_count == 0

    def test_anchor_keyword_hit_returns_follow_up(self, gate, state):
        state.add_anchor_watch(GID, keywords=["天气"])
        assert gate.evaluate_message(GID, "u9", "今天天气真不错") == "follow_up"
        assert state.get_state(GID).msg_count == 0

    def test_anchor_miss_falls_through_to_sampling(self, gate, state):
        state.add_anchor_watch(GID, users=["u1"], keywords=["天气"])
        assert gate.evaluate_message(GID, "u9", "完全无关的内容") is None
        assert state.get_state(GID).msg_count == 1


class TestMessageGateSampling:
    def test_count_trigger(self, nm_config):
        # effective_n = max(5, int(5 * 0.85)) = 5（medium 意愿）
        _, st, gate = _build(nm_config, overrides={"default_n": 5})
        for i in range(4):
            assert gate.evaluate_message(GID, "u1", f"消息{i}") is None
        assert gate.evaluate_message(GID, "u1", "第五条") == "chime_in"
        # 触发后采样重置
        data = st.get_state(GID)
        assert data.msg_count == 0
        assert time.time() - data.last_sample_time < 5

    def test_time_trigger(self, gate, state):
        # effective_t = max(5, int(30 * 0.85)) = 25 分钟
        data = state.get_state(GID)
        data.last_sample_time = time.time() - 26 * 60
        assert gate.evaluate_message(GID, "u1", "一条新消息") == "chime_in"

    def test_time_trigger_requires_pending_messages(self, gate, state):
        # msg_count == 0 时即使超时也不触发
        data = state.get_state(GID)
        data.last_sample_time = time.time() - 26 * 60
        assert state.should_trigger_sampling(GID) is False


class TestTimerGate:
    def _timer_gate(self, nm_config, overrides=None):
        base = {"proactive_enabled": True, "proactive_quiet_minutes": 30}
        if overrides:
            base.update(overrides)
        return _build(nm_config, overrides=base)

    def test_initiate_on_quiet(self, nm_config, make_msg):
        _, _, gate = self._timer_gate(nm_config)
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) == "initiate"

    def test_not_quiet_enough(self, nm_config, make_msg):
        _, _, gate = self._timer_gate(nm_config)
        recent = make_msg(timestamp=time.time() - 60)
        assert gate.evaluate_timer(GID, [recent]) is None

    def test_proactive_disabled(self, nm_config, make_msg):
        _, _, gate = _build(nm_config)  # proactive_enabled 默认 False
        old = make_msg(timestamp=time.time() - 9999)
        assert gate.evaluate_timer(GID, [old]) is None

    def test_empty_messages(self, nm_config):
        _, _, gate = self._timer_gate(nm_config)
        assert gate.evaluate_timer(GID, []) is None

    def test_pending_blocks(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config)
        st.record_initiate(GID, bot_message="大家好")
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None

    def test_daily_cap_blocks(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config, {"proactive_max_per_day": 1})
        st.record_initiate(GID, bot_message="大家好")
        # 隔离 pending 因素，只验证每日上限
        st.get_state(GID).initiate_pending_since = 0.0
        st.get_state(GID).last_initiate_time = 0.0
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None

    def test_streak_cap_blocks(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config, {"proactive_max_streak": 1})
        st.get_state(GID).initiate_no_reply_streak = 1
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None

    def test_min_interval_blocks(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config, {"proactive_min_interval": 60})
        st.get_state(GID).last_initiate_time = time.time() - 100
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None
        # 超过最小间隔后放行
        st.get_state(GID).last_initiate_time = time.time() - 3700
        assert gate.evaluate_timer(GID, [old]) == "initiate"

    def test_drift_accelerates_initiate(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config, {"proactive_drift_delay": 5})
        # 静默 400s 未达 1800s 阈值，但话题 100s 前刚结束 → 阈值收紧为 300s
        st.get_state(GID).last_drift_time = time.time() - 100
        quiet = make_msg(timestamp=time.time() - 400)
        assert gate.evaluate_timer(GID, [quiet]) == "initiate"

    def test_cooldown_blocks_timer(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config)
        st.set_cooldown(GID, 5)
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None

    def test_muted_blocks_timer(self, nm_config, make_msg):
        _, st, gate = self._timer_gate(nm_config)
        st.is_muted = lambda: True
        old = make_msg(timestamp=time.time() - 1900)
        assert gate.evaluate_timer(GID, [old]) is None


class TestGatekeeperWhitelist:
    """白名单门控在 perception.Gatekeeper 层（SignalGate 上游）"""

    def test_whitelist_blocks_unknown_group(self, nm_config):
        cm = nm_config()
        st = StateManager(cm)
        gk = Gatekeeper(cm, st)
        assert gk.should_process(_event()) is False
        st.add_to_whitelist(GID)
        assert gk.should_process(_event()) is True

    def test_empty_command_and_private_blocked(self, nm_config):
        cm = nm_config()
        st = StateManager(cm)
        st.add_to_whitelist(GID)
        gk = Gatekeeper(cm, st)
        assert gk.should_process(_event(text="")) is False
        assert gk.should_process(_event(text="/iris_reply status")) is False
        assert gk.should_process(_event(private=True)) is False

    def test_muted_blocks(self, nm_config):
        cm = nm_config()
        st = StateManager(cm)
        st.add_to_whitelist(GID)
        st.is_muted = lambda: True
        gk = Gatekeeper(cm, st)
        assert gk.should_process(_event()) is False

    def test_state_enum_values(self):
        assert GroupState.IDLE.value == "idle"
        assert GroupState.COOLDOWN.value == "cooldown"
