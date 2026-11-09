"""Frisk 群聊门控：主人发言交给决策；群友只在明确找她后的有限窗口内跟进。"""

from astrbot_plugin_iris_memory.iris_memory.proactive.decision import DecisionRequest
from astrbot_plugin_iris_memory.iris_memory.proactive.prompts import TRIGGER_INSTRUCTIONS
from astrbot_plugin_iris_memory.iris_memory.proactive.signals import SignalGate
from astrbot_plugin_iris_memory.iris_memory.proactive.state import StateManager

GID = "-100g"
OWNER = "100000001"
OWNERS = {OWNER}


def _build(nm_config, cfg=None):
    cm = nm_config(cfg=cfg)
    st = StateManager(cm)
    st.add_to_whitelist(GID)
    return cm, st, SignalGate(cm, st)


def test_owner_message_triggers_decision_with_debounce(nm_config):
    _, _, gate = _build(nm_config)
    assert gate.evaluate_group_message(GID, OWNER, "今天好累", OWNERS, now=1000.0) == ("chime_in", "owner")
    assert gate.evaluate_group_message(GID, OWNER, "真的", OWNERS, now=1030.0) == (None, "")
    assert gate.evaluate_group_message(GID, OWNER, "算了", OWNERS, now=1061.0) == ("chime_in", "owner")


def test_other_member_never_sampled(nm_config):
    _, _, gate = _build(nm_config)
    for i in range(200):
        assert gate.evaluate_group_message(GID, "u1", f"闲聊 {i}", OWNERS, now=1000.0 + i * 60) == (None, "")


def test_other_member_name_mention_triggers_and_engages(nm_config):
    _, st, gate = _build(nm_config)
    assert gate.evaluate_group_message(GID, "u1", "真由理你怎么看", OWNERS, now=1000.0) == ("chime_in", "name")
    assert gate.evaluate_group_message(GID, "u1", "FRISK 在吗", OWNERS, now=1001.0) == ("chime_in", "name")
    st.add_anchor_watch(GID, users=["u1"], ttl_minutes=240)
    assert gate.evaluate_group_message(GID, "u1", "接着说", OWNERS, now=1000.0 + 59 * 60) == ("follow_up", "follow")
    # 超过 1 小时：即使锚点仍在，也不再跟进
    assert gate.evaluate_group_message(GID, "u1", "还在吗", OWNERS, now=1000.0 + 61 * 60) == (None, "")


def test_anchor_on_unengaged_member_is_ignored(nm_config):
    _, st, gate = _build(nm_config)
    st.add_anchor_watch(GID, users=["u2"], keywords=["地铁"])
    assert gate.evaluate_group_message(GID, "u2", "地铁又坏了", OWNERS, now=1000.0) == (None, "")


def test_passive_engagement_window(nm_config):
    _, st, gate = _build(nm_config)
    gate.mark_engaged(GID, "u3", now=1000.0)
    st.add_anchor_watch(GID, users=["u3"], ttl_minutes=240)
    assert gate.evaluate_group_message(GID, "u3", "然后呢", OWNERS, now=1000.0 + 30 * 60) == ("follow_up", "follow")
    assert gate.is_engaged(GID, "u3", now=1000.0 + 60 * 60 + 1) is False


def test_owner_anchor_follow_up(nm_config):
    _, st, gate = _build(nm_config)
    st.add_anchor_watch(GID, users=[OWNER])
    assert gate.evaluate_group_message(GID, OWNER, "嗯", OWNERS, now=1000.0) == ("follow_up", "follow")


def test_cooldown_does_not_block_owner(nm_config):
    _, st, gate = _build(nm_config)
    st.set_cooldown(GID, 5)
    assert gate.evaluate_group_message(GID, OWNER, "hi", OWNERS) == ("chime_in", "owner")


def test_trigger_instructions_have_no_numeric_relationship():
    for text in TRIGGER_INSTRUCTIONS.values():
        assert "好感" not in text and "分" not in text.replace("分钟", "")
    req = DecisionRequest(group_id=GID, wake="message", motive="chime_in", trigger="owner", trigger_user=OWNER)
    assert req.trigger == "owner"
