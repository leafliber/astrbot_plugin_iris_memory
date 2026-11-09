"""Frisk 群聊修复回归：认领泄漏、冷却绕过、群内不定时发起、名字、发言提示、群内称呼。"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_iris_memory.main import IrisMemoryPlugin
from astrbot_plugin_iris_memory.iris_memory.proactive.parser import Decision
from astrbot_plugin_iris_memory.iris_memory.proactive.perception import WindowMessage
from astrbot_plugin_iris_memory.iris_memory.proactive.signals import SignalGate
from astrbot_plugin_iris_memory.iris_memory.proactive.state import StateManager
from astrbot_plugin_iris_memory.iris_memory.proactive.tickets import DecisionTicketRegistry

GID = "-100g"
OWNER = "100000001"


NAMES = ["真由理", "mayuri", "まゆり", "マユリ"]


def _gate(nm_config):
    cm = nm_config(cfg={"proactive": {"bot_name_keywords": NAMES}})
    st = StateManager(cm)
    st.add_to_whitelist(GID)
    return st, SignalGate(cm, st)


# ---- signals -------------------------------------------------------------

def test_cooldown_does_not_block_owner_or_name(nm_config):
    st, gate = _gate(nm_config)
    st.set_cooldown(GID, 15)
    assert gate.evaluate_group_message(GID, OWNER, "晚上好", {OWNER}, now=1000.0) == ("chime_in", "owner")
    assert gate.evaluate_group_message(GID, "u1", "mayuri 在吗", {OWNER}, now=1001.0) == ("chime_in", "name")


def test_cooldown_still_blocks_other_member_follow_up(nm_config):
    st, gate = _gate(nm_config)
    gate.mark_engaged(GID, "u1", now=1000.0)
    st.add_anchor_watch(GID, users=["u1"], ttl_minutes=60)
    st.set_cooldown(GID, 15)
    assert gate.evaluate_group_message(GID, "u1", "然后呢", {OWNER}, now=1010.0) == (None, "")


@pytest.mark.parametrize("text", ["Mayuri你好", "まゆり", "マユリ聞いて", "真由理"])
def test_name_variants(nm_config, text):
    _, gate = _gate(nm_config)
    assert gate.evaluate_group_message(GID, "u9", text, {OWNER}, now=1000.0) == ("chime_in", "name")


def test_group_timer_initiate_disabled(nm_config):
    cm = nm_config(
        cfg={
            "proactive": {
                "enabled": True,
                "proactive_enabled": True,
                "group_initiate_enabled": False,
            }
        }
    )
    st = StateManager(cm)
    gate = SignalGate(cm, st)
    msgs = [WindowMessage(sender_id="u1", sender_name="a", content="hi", timestamp=time.time() - 99999)]
    assert gate.evaluate_timer(GID, msgs) is None


# ---- main: claim release --------------------------------------------------

def _plugin():
    p = object.__new__(IrisMemoryPlugin)
    p._decision_tickets = DecisionTicketRegistry(120)
    p._triggering = {}
    p._reply_in_progress = {}
    p._passive_active = {}
    p._tool_ctx = Mock()
    p.component_manager = None
    p.config = {"continuity": {"enable": True}}
    return p


class _Ev:
    def __init__(self, text="[图片] [贴纸] 😍", group=GID):
        self.message_str = text
        self._extra = {}
        self.stopped = False
        self._group = group

    def get_extra(self, k, default=None):
        return self._extra.get(k, default)

    def set_extra(self, k, v):
        self._extra[k] = v

    def get_group_id(self):
        return self._group

    def is_private_chat(self):
        return not self._group

    def get_sender_id(self):
        return OWNER

    def stop_event(self):
        self.stopped = True


def _claim(p, ev):
    t = p._decision_tickets.claim(GID, "message:1")
    p._triggering[GID] = time.time()
    ev.set_extra("iris_decision", {"motive": "follow_up", "session_key": GID, "ticket_id": t.ticket_id})
    return t


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_enabled", [False, True])
async def test_group_message_path_follows_gate_switch(gate_enabled):
    p = _plugin()
    p._prepare_pure_at_request = AsyncMock(return_value=False)
    p._continuity_on_owner_message = AsyncMock()
    p._reply_config = Mock(
        enabled=True,
        quality_threshold=0,
        group_gate_enabled=gate_enabled,
        private_whitelist=[],
    )
    p._gatekeeper = Mock()
    p._gatekeeper.should_process.return_value = True
    p._gatekeeper.quality_score.return_value = 1.0
    p._state = Mock()
    p._state.get_lock.return_value = asyncio.Lock()
    p._state.consume_initiate_pending.return_value = False
    p._state.match_anchor_user.return_value = False
    p._state.is_whitelisted.return_value = True
    p._group_umo = {}
    p._self_id = "bot"
    p._sliding_window = Mock()
    p._proactive = Mock()
    p._proactive.is_initiating.return_value = False
    p._signals = Mock()
    p._signals.evaluate_message.return_value = None
    p._signals.evaluate_group_message.return_value = (None, "")
    ev = _Ev(text="今天好累")
    ev.get_sender_name = lambda: "群友"
    ev.is_at_or_wake_command = False
    ev.unified_msg_origin = "umo"

    await p.on_message(ev)

    assert p._signals.evaluate_group_message.called is gate_enabled
    assert p._signals.evaluate_message.called is not gate_enabled


@pytest.mark.asyncio
async def test_media_with_pending_decision_skips_gate_and_keeps_claim():
    p = _plugin()
    p._should_reply_to_media = AsyncMock(return_value=False)
    ev = _Ev()
    _claim(p, ev)
    await p.on_all_message(ev)
    p._should_reply_to_media.assert_not_called()
    assert ev.stopped is False
    assert GID in p._triggering  # decision layer will release it


@pytest.mark.asyncio
async def test_media_gate_stop_releases_claim():
    p = _plugin()
    p._should_reply_to_media = AsyncMock(return_value=False)
    ev = _Ev()
    ev.set_extra("iris_mode", "passive")
    p._passive_active[GID] = time.time()
    await p.on_all_message(ev)
    assert ev.stopped is True
    assert GID not in p._passive_active


@pytest.mark.asyncio
async def test_silent_session_stop_releases_claim():
    p = _plugin()
    p.config = {"proactive.silent_sessions": [GID]}
    ev = _Ev(text="hello")
    _claim(p, ev)
    await p.on_all_message(ev)
    assert ev.stopped is True
    assert GID not in p._triggering
    assert p._decision_tickets.get(GID) is None


def test_undecided_claim_released_lazily():
    p = _plugin()
    p._decision_tickets.claim(GID, "message:1")
    p._triggering[GID] = 1000.0
    p._release_undecided_claim(GID, 1030.0)
    assert GID in p._triggering
    p._release_undecided_claim(GID, 1050.0)
    assert GID not in p._triggering
    assert p._decision_tickets.get(GID) is None


def test_undecided_claim_kept_while_replying():
    p = _plugin()
    p._triggering[GID] = 1000.0
    p._reply_in_progress[GID] = 1000.0
    p._release_undecided_claim(GID, 2000.0)
    assert GID in p._triggering


# ---- main: group speak hint ----------------------------------------------

def test_group_speak_hint_carries_window_and_decision():
    p = _plugin()
    p._self_id = "200000002"
    window = [
        WindowMessage(sender_id=OWNER, sender_name="OwnerNick", content="还在修群聊的bug", timestamp=1.0),
        WindowMessage(sender_id="200000002", sender_name="我", content="加油", timestamp=2.0),
        WindowMessage(sender_id=OWNER, sender_name="OwnerNick", content="mayuri你好呀", timestamp=3.0),
    ]
    p._sliding_window = SimpleNamespace(get_messages=lambda gid: window)
    d = Decision(action="speak", message="修得怎么样了？", why="对方在修 bug", observation="obs")
    hint = p._build_group_speak_hint(GID, "chime_in", d)
    assert "还在修群聊的bug" in hint and "mayuri你好呀" in hint
    assert "[我(200000002)] 加油" in hint
    assert "对方在修 bug" in hint and "修得怎么样了？" in hint
    assert "群里其他人都看得到" in hint and "不提私聊" in hint
    assert "好感" not in hint


def test_group_speak_hint_empty_window_has_no_false_claim():
    p = _plugin()
    p._self_id = ""
    p._sliding_window = SimpleNamespace(get_messages=lambda gid: [])
    hint = p._build_group_speak_hint(GID, "follow_up", Decision(action="speak"))
    assert "最近的消息" not in hint and "以上是群聊" not in hint


# ---- profile: group 称呼 ---------------------------------------------------

def test_group_profile_drops_descriptive_relationship(monkeypatch):
    import astrbot_plugin_iris_memory.iris_memory.config as config_mod
    from astrbot_plugin_iris_memory.iris_memory.core.llm_request_hook import (
        _format_profiles_for_injection,
    )

    class _Cfg:
        def get(self, k, default=None):
            return default

    monkeypatch.setattr(config_mod, "get_config", lambda: _Cfg())
    user = SimpleNamespace(
        user_id=OWNER, user_name="OwnerNick", historical_names=[],
        bot_relationship="把频道当作可以自言自语的地方，希望有人愿意陪她聊",
        personality_tags=[], interests=[], occupation="", language_style="",
        communication_style="", emotional_baseline="", favorability=0,
        important_dates=[], important_events=[], taboo_topics=[], custom_fields={},
    )
    group = SimpleNamespace(interests=[], atmosphere_tags=[], long_term_tags=[],
                            blacklist_topics=[], custom_fields={})
    text = _format_profiles_for_injection(group, user, is_group=True)
    assert "自言自语" not in text
    user.bot_relationship = "主人"
    assert "称呼: 主人" in _format_profiles_for_injection(group, user, is_group=True)
