"""ProactiveBridge / OutboundLedger / 引擎桥接通路测试

全部使用 tmp_path DB、可控时钟、假平台（send_with_receipt）、假 LLM，不触网。
StateManager 内部使用真实 time.time()，因此时钟以当前时间为起点。
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity import models as m
from astrbot_plugin_iris_memory.iris_memory.continuity.outbound import (
    classify_send_error,
    match_sent_message,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.proactive_bridge import (
    ProactiveBridge,
    SendPolicy,
    split_speech_parts,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.store import ContinuityStore
from astrbot_plugin_iris_memory.iris_memory.proactive.config import ConfigManager
from astrbot_plugin_iris_memory.iris_memory.proactive.decision import DecisionCore
from astrbot_plugin_iris_memory.iris_memory.proactive.perception import (
    ContextPackager,
    SlidingWindow,
    WindowMessage,
)
from astrbot_plugin_iris_memory.iris_memory.proactive.proactive import ProactiveEngine
from astrbot_plugin_iris_memory.iris_memory.proactive.signals import SignalGate
from astrbot_plugin_iris_memory.iris_memory.proactive.state import StateManager

OWNER = "100000001"
SCOPE = f"private:{OWNER}"
UMO = f"userbot:FriendMessage:{OWNER}"
GROUP = "-1001000000003"
GROUP_UMO = f"userbot:GroupMessage:{GROUP}"
CHANNEL = "-1001000000001"
SILENT = "-1009999999999"


class Clock:
    def __init__(self, t: float | None = None):
        self.t = time.time() if t is None else t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class RPCError(Exception):
    """模拟 telethon.errors.RPCError（按类名识别）"""


class FloodWaitError(RPCError):
    pass


class FakePlatform:
    """假 Telethon 适配器：behaviors 依次为 list[int]（回执）或 Exception（抛出）"""

    def __init__(self, behaviors=None, *, on_send=None):
        self.behaviors = list(behaviors or [])
        self.calls: list[tuple[str, str]] = []
        self.on_send = on_send

    def meta(self):
        return SimpleNamespace(id="userbot", name="telethon_userbot")

    async def send_with_receipt(self, session, chain):
        self.calls.append((str(session), chain.get_plain_text()))
        if self.on_send is not None:
            await self.on_send(len(self.calls))
        b = self.behaviors.pop(0) if self.behaviors else [1000 + len(self.calls)]
        if isinstance(b, BaseException):
            raise b
        return b


class NoReceiptPlatform:
    def __init__(self):
        self.send_by_session = AsyncMock()

    def meta(self):
        return SimpleNamespace(id="userbot")


class FakeClient:
    def __init__(self, messages=None, error: Exception | None = None):
        self.messages = list(messages or [])
        self.error = error
        self.calls = []

    async def get_messages(self, peer, limit=20):
        self.calls.append((peer, limit))
        if self.error is not None:
            raise self.error
        return list(self.messages)


def tg_msg(mid, text, ts, out=True):
    return SimpleNamespace(
        id=mid, out=out, message=text, date=datetime.fromtimestamp(ts, timezone.utc)
    )


def make_config(**proactive):
    cfg = {
        "mute_period": {"start_hour": 0, "start_minute": 0, "end_hour": 0, "end_minute": 0},
        "proactive": {
            "enabled": True,
            "proactive_enabled": True,
            "provider_id": "provider",
            "private_enabled": True,
            "private_whitelist": [OWNER],
            **proactive,
        },
    }
    return ConfigManager(cfg, hidden_get={}.get)


def default_policy():
    return SendPolicy(
        owner_ids=frozenset({OWNER}),
        silent_sessions=frozenset({SILENT}),
        forbidden_chats=frozenset({CHANNEL}),
        group_whitelist=frozenset({GROUP, CHANNEL, SILENT}),
    )


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "continuity" / "continuity.db"


@pytest.fixture
def store(db_path, clock):
    s = ContinuityStore(db_path, clock=clock)
    s.open()
    yield s
    s.close()


@pytest.fixture
def state():
    return StateManager(make_config())


@pytest.fixture
def platform():
    return FakePlatform()


def make_bridge(store, state, clock, platform=None, **kw):
    context = SimpleNamespace(
        get_platform_inst=lambda pid: platform if pid == "userbot" else None
    )
    kw.setdefault("policy_get", default_policy)
    return ProactiveBridge(
        store,
        state,
        context=context,
        clock=clock,
        umo_get=lambda sk: UMO if sk == SCOPE else (GROUP_UMO if sk == GROUP else None),
        reconcile_query_interval_s=0.0,
        **kw,
    )


@pytest.fixture
def bridge(store, state, clock, platform):
    return make_bridge(store, state, clock, platform)


def add_item(store, clock, **over):
    fields = dict(
        subject_id=OWNER,
        subject_type="user",
        scope_key=SCOPE,
        topic="周五面试",
        summary="她周五下午有个面试",
        kind=m.KIND_PLAN,
        source_kind=m.SOURCE_SHARED_DIALOGUE,
        source_refs=[],
        allowed_audiences=[SCOPE],
        not_before=clock() - 60,
        expires_at=clock() + 3 * 86400,
    )
    fields.update(over)
    return store.create_item(**fields)


# ===================================================================== helpers


class TestHelpers:
    def test_split_parts(self):
        assert split_speech_parts("a") == ["a"]
        assert split_speech_parts("a\n\nb\n \nc") == ["a", "b", "c"]
        assert split_speech_parts("a\n\nb\n\nc\n\nd") == ["a", "b", "c\n\nd"]
        assert split_speech_parts("  \n\n ") == []

    def test_classify(self):
        assert classify_send_error(FloodWaitError("x")) == m.OUT_FAILED
        assert classify_send_error(ConnectionError()) == m.OUT_UNCERTAIN
        assert classify_send_error(asyncio.TimeoutError()) == m.OUT_UNCERTAIN
        assert classify_send_error(RuntimeError()) == m.OUT_UNCERTAIN

    def test_match_sent_message(self, clock):
        now = clock()
        msgs = [
            tg_msg(5, "你好  呀", now, out=False),
            tg_msg(6, "你好 呀", now - 600),
            tg_msg(7, "你好 呀", now),
        ]
        assert match_sent_message(msgs, "你好\n呀", now - 60) == 7
        assert match_sent_message(msgs, "你好 呀", now - 60, exclude_ids=[7]) is None


# ===================================================================== target guard


class TestTargetGuard:
    def test_allowed(self, bridge):
        assert bridge.check_target(SCOPE, UMO) == ""
        assert bridge.check_target(GROUP, GROUP_UMO) == ""

    @pytest.mark.parametrize(
        "scope,umo,reason",
        [
            (CHANNEL, f"userbot:GroupMessage:{CHANNEL}", "target_forbidden_chat"),
            (GROUP, f"userbot:GroupMessage:{CHANNEL}", "target_forbidden_chat"),
            ("1000000001", "userbot:GroupMessage:1000000001", "target_forbidden_chat"),
            (SILENT, f"userbot:GroupMessage:{SILENT}", "target_silent_session"),
            ("private:111", "userbot:FriendMessage:111", "target_private_not_whitelisted"),
            (SCOPE, "userbot:FriendMessage:111", "target_umo_mismatch"),
            (SCOPE, f"userbot:GroupMessage:{OWNER}", "target_umo_mismatch"),
            (GROUP, "userbot:GroupMessage:-1000000000001", "target_umo_mismatch"),
            ("-1000000000001", "userbot:GroupMessage:-1000000000001", "target_group_not_whitelisted"),
            (SCOPE, "garbage", "target_umo_mismatch"),
        ],
    )
    def test_refused(self, bridge, scope, umo, reason):
        assert bridge.check_target(scope, umo) == reason

    def test_forbidden_even_if_config_omits_it(self, store, state, clock, platform):
        b = make_bridge(
            store, state, clock, platform,
            policy_get=lambda: SendPolicy(owner_ids=frozenset({OWNER}), forbidden_chats=frozenset(),
                                          group_whitelist=frozenset({CHANNEL})),
        )
        assert b.check_target(CHANNEL, f"userbot:GroupMessage:{CHANNEL}") == "target_forbidden_chat"

    def test_policy_failure_refuses_everything(self, store, state, clock, platform):
        def boom():
            raise RuntimeError("cfg")

        b = make_bridge(store, state, clock, platform, policy_get=boom)
        assert b.check_target(SCOPE, UMO) != ""
        assert b.check_target(GROUP, GROUP_UMO) != ""

    @pytest.mark.asyncio
    async def test_send_part_refuses_forbidden_without_network(self, store, state, clock, platform):
        state.add_to_whitelist(CHANNEL)
        b = make_bridge(store, state, clock, platform)
        start = await b.begin_turn(CHANNEL)
        out = await b.send_part(start.turn["turn_id"], f"userbot:GroupMessage:{CHANNEL}", "hi", 0)
        assert out.status == m.OUT_CANCELLED
        assert platform.calls == []
        pre = await b.presend_check(start.turn["turn_id"], f"userbot:GroupMessage:{CHANNEL}")
        assert not pre.ok and pre.reason == "target_forbidden_chat"


# ===================================================================== T03 send ledger


class TestSendLedger:
    @pytest.mark.asyncio
    async def test_success_records_ids_and_sending_persisted_first(self, store, state, clock):
        seen = {}

        async def on_send(n):
            rows = store.outbound_in_state((m.OUT_SENDING,))
            seen["sending"] = len(rows)

        platform = FakePlatform([[501, 502]], on_send=on_send)
        b = make_bridge(store, state, clock, platform)
        start = await b.begin_turn(SCOPE)
        tid = start.turn["turn_id"]
        out = await b.send_part(tid, UMO, "在吗", 0)
        assert seen["sending"] == 1  # 网络调用前已落盘 sending
        assert out.sent and out.tg_msg_ids == [501, 502]
        row = store.get_outbound(out.outbound_id)
        assert row["status"] == m.OUT_SENT and row["tg_msg_ids"] == [501, 502]
        assert store.outbound_by_tg_id(SCOPE, 502)["id"] == out.outbound_id
        res = await b.finish_turn(tid)
        assert res.state == m.TURN_SENT and res.quota_state == "charged"
        assert store.get_turn(tid)["parts_sent"] == 1
        # 额度由引擎 record_initiate 计；bridge 不动 state
        assert state.get_state(SCOPE).initiate_timestamps == []

    @pytest.mark.asyncio
    async def test_rpc_error_is_confirmed_failure(self, store, state, clock):
        b = make_bridge(store, state, clock, FakePlatform([FloodWaitError("wait")]))
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        out = await b.send_part(tid, UMO, "在吗", 0)
        assert out.status == m.OUT_FAILED
        res = await b.finish_turn(tid, stop_reason="send_failed")
        assert res.state == m.TURN_FAILED and res.quota_state == "none"
        data = state.get_state(SCOPE)
        assert data.initiate_timestamps == [] and data.initiate_pending_since == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("behavior", [ConnectionError("reset"), RuntimeError("x"), []])
    async def test_uncertain_reserves_without_pending(self, store, state, clock, behavior):
        b = make_bridge(store, state, clock, FakePlatform([behavior]))
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        out = await b.send_part(tid, UMO, "在吗", 0)
        assert out.status == m.OUT_UNCERTAIN
        res = await b.finish_turn(tid, stop_reason="send_uncertain")
        assert res.state == m.TURN_UNCERTAIN and res.quota_state == "reserved"
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1
        assert data.last_initiate_time > 0
        assert data.initiate_pending_since == 0  # 不计未回复

    @pytest.mark.asyncio
    async def test_timeout_is_uncertain(self, store, state, clock):
        async def hang(n):
            await asyncio.sleep(5)

        b = make_bridge(store, state, clock, FakePlatform(on_send=hang), send_timeout_s=0.05)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        out = await b.send_part(tid, UMO, "在吗", 0)
        assert out.status == m.OUT_UNCERTAIN and out.error == "timeout"

    @pytest.mark.asyncio
    async def test_duplicate_receipt_is_idempotent(self, store, state, clock):
        platform = FakePlatform([[700], [700]])
        b = make_bridge(store, state, clock, platform)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        first = await b.send_part(tid, UMO, "一", 0)
        # 回执重放到同一行：不重复登记
        again = await b.ledger.apply_receipt(first.outbound_id, SCOPE, [700])
        assert again.sent and store.get_outbound(first.outbound_id)["tg_msg_ids"] == [700]
        # 另一行拿到同一个 id：不能作为证据
        second = await b.send_part(tid, UMO, "二", 1)
        assert second.status == m.OUT_UNCERTAIN and second.error == "duplicate_receipt"
        assert store.outbound_by_tg_id(SCOPE, 700)["id"] == first.outbound_id
        n = store._conn().execute("SELECT COUNT(*) FROM outbound_ids").fetchone()[0]
        assert n == 1

    @pytest.mark.asyncio
    async def test_no_receipt_method_does_not_send(self, store, state, clock):
        platform = NoReceiptPlatform()
        b = make_bridge(store, state, clock, platform)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        out = await b.send_part(tid, UMO, "在吗", 0)
        assert out.status == m.OUT_FAILED and out.error == "no_receipt_path"
        platform.send_by_session.assert_not_called()
        res = await b.finish_turn(tid, stop_reason="send_failed")
        assert res.state == m.TURN_FAILED
        assert state.get_state(SCOPE).initiate_timestamps == []

    def test_record_reply_and_manual_idempotent(self, store, bridge):
        led = bridge.ledger
        r1 = led.record_reply(SCOPE, "好呀", [900])
        assert r1["status"] == m.OUT_SENT and r1["origin"] == "reply"
        assert led.record_reply(SCOPE, "好呀", [900])["id"] == r1["id"]
        failed = led.record_reply(SCOPE, "好呀", [])
        assert failed["status"] == m.OUT_FAILED
        unknown = led.record_reply(SCOPE, "好呀", [], receipt_supported=False)
        assert unknown["author"] == "unknown" and unknown["status"] == m.OUT_UNCERTAIN
        man = led.record_manual(SCOPE, 901, "手动")
        assert man["author"] == "manual" and man["status"] == m.OUT_SENT
        assert led.record_manual(SCOPE, 901, "手动")["id"] == man["id"]


# ===================================================================== T04 / T09 turns


class TestTurns:
    @pytest.mark.asyncio
    async def test_new_inbound_during_generation_cancels(self, store, bridge, platform, clock):
        store.advance_watermark(SCOPE, 10, clock())
        tid = (await bridge.begin_turn(SCOPE)).turn["turn_id"]
        assert store.get_turn(tid)["input_msg_id"] == 10
        store.advance_watermark(SCOPE, 11, clock())
        pre = await bridge.presend_check(tid, UMO)
        assert not pre.ok and pre.reason == "new_inbound"
        res = await bridge.finish_turn(tid, stop_reason=pre.reason)
        assert res.state == m.TURN_CANCELLED
        assert platform.calls == []

    @pytest.mark.asyncio
    async def test_replayed_inbound_does_not_cancel(self, store, bridge, clock):
        store.advance_watermark(SCOPE, 10, clock())
        tid = (await bridge.begin_turn(SCOPE)).turn["turn_id"]
        store.advance_watermark(SCOPE, 9, clock())  # 重放 / 补采旧消息
        assert (await bridge.presend_check(tid, UMO)).ok

    @pytest.mark.asyncio
    async def test_focus_item_closed_and_paused(self, store, bridge, clock):
        item = add_item(store, clock)
        tid = (await bridge.begin_turn(SCOPE)).turn["turn_id"]
        await bridge.mark_turn(tid, m.TURN_GENERATING, item_id=item["item_id"])
        assert (await bridge.presend_check(tid, UMO)).ok
        assert (await bridge.presend_check(tid, UMO, paused=True)).reason == "paused"
        store.update_item(item["item_id"], expected_revision=None, reason="t", status=m.STATUS_DONE)
        assert (await bridge.presend_check(tid, UMO)).reason == "focus_item_inactive"

    @pytest.mark.asyncio
    async def test_duplicate_tick_skips(self, store, state, clock, platform, bridge):
        first = await bridge.begin_turn(SCOPE)
        assert first.turn is not None
        second = await bridge.begin_turn(SCOPE)
        assert second.skipped and second.reason == "turn_active"
        # 另一个 bridge 实例（同库）也看得到进行中回合
        other = make_bridge(store, state, clock, platform)
        assert (await other.begin_turn(SCOPE)).reason == "turn_active"
        await bridge.close_turn(first.turn["turn_id"], m.TURN_SKIPPED, "llm_declined")
        assert store.get_turn(first.turn["turn_id"])["state"] == m.TURN_SKIPPED
        assert (await bridge.begin_turn(SCOPE)).turn is not None

    @pytest.mark.asyncio
    async def test_concurrent_begin_only_one_wins(self, bridge):
        a, b = await asyncio.gather(bridge.begin_turn(SCOPE), bridge.begin_turn(SCOPE))
        assert [x.turn is not None for x in (a, b)].count(True) == 1

    @pytest.mark.asyncio
    async def test_close_turn_is_idempotent(self, bridge, store):
        tid = (await bridge.begin_turn(SCOPE)).turn["turn_id"]
        await bridge.close_turn(tid, m.TURN_SKIPPED, "a")
        await bridge.close_turn(tid, m.TURN_CANCELLED, "b")
        t = store.get_turn(tid)
        assert t["state"] == m.TURN_SKIPPED and t["reason"] == "a"

    @pytest.mark.asyncio
    async def test_closure_cooldown(self, store, bridge, clock):
        store.put_closure(SCOPE, state="ended", ended_at=clock() - 600)
        res = await bridge.begin_turn(SCOPE)
        assert res.reason == "closure_cooldown"
        assert (await bridge.begin_turn(SCOPE, forced=True)).turn is not None

    @pytest.mark.asyncio
    async def test_closure_cooldown_bypassed_by_due_reminder(self, store, bridge, clock):
        add_item(store, clock, remind_requested=1)
        store.put_closure(SCOPE, state="ended", ended_at=clock() - 600)
        assert (await bridge.begin_turn(SCOPE)).turn is not None

    @pytest.mark.asyncio
    async def test_candidates_filtering_and_order(self, store, bridge, clock):
        now = clock()
        low = add_item(store, clock, topic="低", importance=1)
        rem = add_item(store, clock, topic="提醒", remind_requested=1)
        high = add_item(store, clock, topic="高", importance=3)
        add_item(store, clock, topic="未到", not_before=now + 3600)
        add_item(store, clock, topic="过期", expires_at=now - 1)
        add_item(store, clock, topic="不主动", proactive_allowed=0)
        add_item(store, clock, topic="问过两次", ask_count=2)
        add_item(store, clock, topic="刚提过", last_mentioned_at=now - 3600)
        add_item(store, clock, topic="群项", allowed_audiences=[GROUP])
        got = await bridge.candidates(SCOPE, now)
        assert [c["item_id"] for c in got] == [rem["item_id"], high["item_id"], low["item_id"]]

    @pytest.mark.asyncio
    async def test_candidates_disabled(self, store, state, clock, platform):
        add_item(store, clock)
        b = make_bridge(store, state, clock, platform, topics_enabled=lambda: False)
        assert await b.candidates(SCOPE) == []


# ===================================================================== reconcile / restart


class TestReconcile:
    async def _uncertain_turn(self, store, state, clock, text="在吗"):
        b = make_bridge(store, state, clock, FakePlatform([ConnectionError()]))
        item = add_item(store, clock)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        await b.mark_turn(tid, m.TURN_SENDING, item_id=item["item_id"], parts_total=1)
        await b.send_part(tid, UMO, text, 0)
        res = await b.finish_turn(tid, stop_reason="send_uncertain")
        assert res.state == m.TURN_UNCERTAIN
        return tid, item

    @pytest.mark.asyncio
    async def test_uncertain_blocks_next_turn(self, store, state, clock, bridge):
        await self._uncertain_turn(store, state, clock)
        assert (await bridge.begin_turn(SCOPE)).reason == "uncertain_pending"

    @pytest.mark.asyncio
    async def test_reconcile_to_sent(self, store, state, clock):
        tid, item = await self._uncertain_turn(store, state, clock, "在吗\n今天")
        client = FakeClient([tg_msg(42, "在吗 今天", clock() + 1), tg_msg(41, "在吗 今天", clock(), out=False)])
        b = make_bridge(store, state, clock, client_getter=lambda sk: client)
        stats = await b.reconcile()
        assert stats["sent"] == 1
        assert client.calls == [(int(OWNER), 20)]
        turn = store.get_turn(tid)
        assert turn["state"] == m.TURN_SENT and turn["quota_state"] == "charged"
        row = store.outbound_for_turn(tid)[0]
        assert row["status"] == m.OUT_SENT and row["tg_msg_ids"] == [42]
        it = store.get_item(item["item_id"])
        assert it["ask_count"] == 1 and it["last_outbound_msg_ids"] == [42]
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1  # 预占转为正式计费
        # F5：确认送达且仍在接话窗口内 → pending 从实际发送时间起算
        assert data.initiate_pending_since == pytest.approx(clock() + 1, abs=1.0)
        assert data.initiate_no_reply_streak == 0
        assert (await b.begin_turn(SCOPE)).turn is not None

    @pytest.mark.asyncio
    async def test_reconcile_to_failed_releases_quota(self, store, state, clock):
        tid, _ = await self._uncertain_turn(store, state, clock)
        client = FakeClient([tg_msg(42, "别的话", clock())])
        b = make_bridge(store, state, clock, client_getter=lambda sk: client)
        stats = await b.reconcile()
        assert stats["kept"] == 1 and store.get_turn(tid)["state"] == m.TURN_UNCERTAIN
        clock.advance(11 * 60)
        stats = await b.reconcile()
        assert stats["failed"] == 1
        turn = store.get_turn(tid)
        assert turn["state"] == m.TURN_FAILED and turn["quota_state"] == "released"
        data = state.get_state(SCOPE)
        assert data.initiate_timestamps == [] and data.last_initiate_time == 0

    @pytest.mark.asyncio
    async def test_reconcile_query_failure_keeps_uncertain(self, store, state, clock):
        tid, _ = await self._uncertain_turn(store, state, clock)
        clock.advance(3600)
        b = make_bridge(store, state, clock, client_getter=lambda sk: FakeClient(error=ConnectionError()))
        stats = await b.reconcile()
        assert stats["kept"] == 1
        assert store.get_turn(tid)["state"] == m.TURN_UNCERTAIN
        b2 = make_bridge(store, state, clock, client_getter=lambda sk: None)
        await b2.reconcile()
        assert store.get_turn(tid)["state"] == m.TURN_UNCERTAIN

    @pytest.mark.asyncio
    async def test_reconcile_cheap_when_nothing_pending(self, bridge):
        client = FakeClient()
        stats = await bridge.reconcile(client_getter=lambda sk: client)
        assert stats == {"stale": 0, "sent": 0, "failed": 0, "kept": 0}
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_reconcile_skips_live_turn(self, store, bridge, clock):
        tid = (await bridge.begin_turn(SCOPE)).turn["turn_id"]
        oid = await bridge.ledger.create_draft(SCOPE, "x", turn_id=tid, part_index=0)
        store.mark_outbound(oid, m.OUT_SENDING)
        await bridge.reconcile(client_getter=lambda sk: FakeClient())
        assert store.get_outbound(oid)["status"] == m.OUT_SENDING
        assert store.get_turn(tid)["state"] == m.TURN_DECIDING

    @pytest.mark.asyncio
    async def test_restart_keeps_state_and_does_not_burst(self, db_path, state, clock, platform):
        s1 = ContinuityStore(db_path, clock=clock)
        s1.open()
        b1 = make_bridge(s1, state, clock, platform)
        tid = (await b1.begin_turn(SCOPE)).turn["turn_id"]
        await b1.mark_turn(tid, m.TURN_SENDING, parts_total=1)
        oid = await b1.ledger.create_draft(SCOPE, "崩溃前", turn_id=tid, part_index=0)
        s1.mark_outbound(oid, m.OUT_SENDING)  # 进程在网络调用中途崩溃
        win = SlidingWindow(make_config())
        win.append(SCOPE, WindowMessage("u", "U", "旧的", clock() - 25 * 3600))
        win.append(SCOPE, WindowMessage("u", "U", "最近", clock() - 600))
        await b1.save_windows(win, {SCOPE})
        s1.close()

        s2 = ContinuityStore(db_path, clock=clock)
        s2.open()
        try:
            b2 = make_bridge(s2, state, clock, platform, client_getter=lambda sk: FakeClient())
            # 启动对账前：遗留进行中回合仍阻止新回合
            assert (await b2.begin_turn(SCOPE)).reason == "turn_active"
            stats = await b2.reconcile(force_query=True)
            assert stats["stale"] == 1
            assert s2.get_outbound(oid)["status"] == m.OUT_UNCERTAIN
            turn = s2.get_turn(tid)
            assert turn["state"] == m.TURN_UNCERTAIN and turn["quota_state"] == "reserved"
            assert len(state.get_state(SCOPE).initiate_timestamps) == 1
            assert (await b2.begin_turn(SCOPE)).reason == "uncertain_pending"
            assert platform.calls == []

            win2 = SlidingWindow(make_config())
            assert await b2.restore_windows(win2) == 1
            assert await b2.restore_windows(win2) == 0  # 幂等
            assert [x.content for x in win2.get_messages(SCOPE)] == ["最近"]
        finally:
            s2.close()

    @pytest.mark.asyncio
    async def test_save_windows_skips_unchanged(self, store, bridge, clock):
        win = SlidingWindow(make_config())
        win.append(SCOPE, WindowMessage("u", "U", "a", clock()))
        assert await bridge.save_windows(win, {SCOPE}) == 1
        assert await bridge.save_windows(win, {SCOPE}) == 0
        win.append(SCOPE, WindowMessage("u", "U", "b", clock()))
        assert await bridge.save_windows(win, {SCOPE}) == 1
        assert [d["content"] for d in store.load_windows()[SCOPE]] == ["a", "b"]


class TestReconcileFixes:
    """F5（对账确认送达后的接话记账）与 F9（_active 释放顺序 / 对账宽限）"""

    async def _uncertain_turn(self, store, state, clock, text="在吗"):
        b = make_bridge(store, state, clock, FakePlatform([ConnectionError()]))
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        await b.mark_turn(tid, m.TURN_SENDING, parts_total=1)
        await b.send_part(tid, UMO, text, 0)
        res = await b.finish_turn(tid, stop_reason="send_uncertain")
        assert res.state == m.TURN_UNCERTAIN
        return tid

    @pytest.mark.asyncio
    async def test_reconciled_after_window_counts_unanswered_once(self, store, state, clock):
        tid = await self._uncertain_turn(store, state, clock)
        sent_at = clock()
        clock.advance(31 * 60)  # 默认 pending_timeout=30min，期间本人没有新消息
        client = FakeClient([tg_msg(42, "在吗", sent_at)])
        b = make_bridge(store, state, clock, client_getter=lambda sk: client)
        assert (await b.reconcile())["sent"] == 1
        data = state.get_state(SCOPE)
        assert data.initiate_no_reply_streak == 1 and data.initiate_pending_since == 0
        assert store.get_turn(tid)["state"] == m.TURN_SENT
        # 幂等：再对账、再结算都不重复计数，也不再设 pending
        await b.reconcile(force_query=True)
        await b._settle_turn(tid, sent_at)
        data = state.get_state(SCOPE)
        assert data.initiate_no_reply_streak == 1 and data.initiate_pending_since == 0

    @pytest.mark.asyncio
    async def test_reconciled_within_window_sets_pending_once(self, store, state, clock):
        tid = await self._uncertain_turn(store, state, clock)
        sent_at = clock()
        clock.advance(5 * 60)
        client = FakeClient([tg_msg(42, "在吗", sent_at)])
        b = make_bridge(store, state, clock, client_getter=lambda sk: client)
        await b.reconcile()
        data = state.get_state(SCOPE)
        assert data.initiate_pending_since == pytest.approx(sent_at, abs=1.0)
        assert data.initiate_no_reply_streak == 0
        await b._settle_turn(tid, sent_at + 100)  # 已结算：不再改动
        assert state.get_state(SCOPE).initiate_pending_since == pytest.approx(sent_at, abs=1.0)

    @pytest.mark.asyncio
    async def test_reconciled_after_owner_reply_no_pending_no_unanswered(self, store, state, clock):
        await self._uncertain_turn(store, state, clock)
        sent_at = clock()
        clock.advance(40 * 60)
        store.advance_watermark(SCOPE, 700, clock() - 60)  # 本人已经回复过
        client = FakeClient([tg_msg(42, "在吗", sent_at)])
        b = make_bridge(store, state, clock, client_getter=lambda sk: client)
        await b.reconcile()
        data = state.get_state(SCOPE)
        assert data.initiate_pending_since == 0 and data.initiate_no_reply_streak == 0

    @pytest.mark.asyncio
    async def test_finish_turn_releases_active_after_terminal_persist(self, store, state, clock, platform):
        b = make_bridge(store, state, clock, platform)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        await b.mark_turn(tid, m.TURN_SENDING, parts_total=1)
        await b.send_part(tid, UMO, "嗨", 0)
        seen = []
        orig = store.update_turn

        def spy(turn_id, **changes):
            if changes.get("state") in (m.TURN_SENT, m.TURN_PARTIAL, m.TURN_FAILED, m.TURN_CANCELLED):
                seen.append(b._active.get(SCOPE) == tid)
            return orig(turn_id, **changes)

        store.update_turn = spy
        try:
            res = await b.finish_turn(tid)
        finally:
            store.update_turn = orig
        assert res.state == m.TURN_SENT
        assert seen == [True]  # 落盘终态时仍占位
        assert SCOPE not in b._active  # 落盘后释放

    @pytest.mark.asyncio
    async def test_close_turn_releases_active_after_terminal_persist(self, store, state, clock, platform):
        b = make_bridge(store, state, clock, platform)
        tid = (await b.begin_turn(SCOPE)).turn["turn_id"]
        seen = []
        orig = store.update_turn

        def spy(turn_id, **changes):
            if changes.get("state") == m.TURN_SKIPPED:
                seen.append(b._active.get(SCOPE) == tid)
            return orig(turn_id, **changes)

        store.update_turn = spy
        try:
            await b.close_turn(tid, m.TURN_SKIPPED, "llm_declined")
        finally:
            store.update_turn = orig
        assert seen == [True]
        assert SCOPE not in b._active

    @pytest.mark.asyncio
    async def test_reconcile_skips_recently_updated_inflight_turn(self, store, state, clock, platform):
        owner = make_bridge(store, state, clock, platform)  # 另一个协程 / 实例正在收尾的回合
        tid = (await owner.begin_turn(SCOPE)).turn["turn_id"]
        await owner.mark_turn(tid, m.TURN_SENDING, parts_total=1)
        oid = await owner.ledger.create_draft(SCOPE, "x", turn_id=tid, part_index=0)
        store.mark_outbound(oid, m.OUT_SENDING)
        other = make_bridge(store, state, clock, client_getter=lambda sk: FakeClient())
        stats = await other.reconcile()
        assert stats["stale"] == 0
        assert store.get_outbound(oid)["status"] == m.OUT_SENDING
        assert store.get_turn(tid)["state"] == m.TURN_SENDING
        clock.advance(61)
        stats = await other.reconcile()
        assert stats["stale"] == 1
        assert store.get_turn(tid)["state"] == m.TURN_UNCERTAIN


# ===================================================================== F6 avoid topics


def closed_item(store, clock, topic, status, *, age_s=0.0, audience=SCOPE):
    saved = clock.t
    clock.t = saved - age_s
    try:
        it = add_item(store, clock, topic=topic, allowed_audiences=[audience])
        store.update_item(it["item_id"], expected_revision=it["revision"], reason="test", status=status)
    finally:
        clock.t = saved
    return it


class TestAvoidTopics:
    @pytest.mark.asyncio
    async def test_avoid_topics_recent_closed_same_audience(self, store, bridge, clock):
        closed_item(store, clock, "面试", m.STATUS_DECLINED, age_s=2 * 86400)
        closed_item(store, clock, "交报告", m.STATUS_DONE)
        closed_item(store, clock, "体检", m.STATUS_CANCELLED, age_s=86400)
        closed_item(store, clock, "很久以前", m.STATUS_DECLINED, age_s=31 * 86400)
        closed_item(store, clock, "别的受众", m.STATUS_DECLINED, audience="private:111")
        add_item(store, clock, topic="还在进行")
        got = await bridge.avoid_topics(SCOPE)
        assert sorted(got) == sorted(["面试", "交报告", "体检"])


# ===================================================================== T10 resume


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_only_on_advanced_watermark(self, bridge, state, clock):
        data = state.get_state(SCOPE)
        data.initiate_no_reply_streak = 2
        data.initiate_pending_since = time.time() - 60
        assert await bridge.note_owner_message(SCOPE, 100, clock()) is True
        assert data.initiate_no_reply_streak == 0 and data.initiate_pending_since == 0
        data.initiate_no_reply_streak = 2
        assert await bridge.note_owner_message(SCOPE, 100, clock()) is False  # 重放
        assert await bridge.note_owner_message(SCOPE, 99, clock()) is False  # 补采旧消息
        assert data.initiate_no_reply_streak == 2

    def test_non_owner_and_group_do_not_resume(self, bridge, state):
        for sk in ("private:111", GROUP):
            state.get_state(sk).initiate_no_reply_streak = 2
            assert bridge.resume_if_advanced(sk, True) is False
            assert state.get_state(sk).initiate_no_reply_streak == 2

    def test_state_quota_helpers(self, state):
        ts = state.reserve_initiate(SCOPE)
        data = state.get_state(SCOPE)
        assert data.initiate_timestamps == [ts] and data.last_initiate_time == ts
        assert data.initiate_pending_since == 0
        assert state.release_initiate_reservation(SCOPE, ts) is True
        assert state.release_initiate_reservation(SCOPE, ts) is False
        assert data.initiate_timestamps == [] and data.last_initiate_time == 0


# ===================================================================== engine integration


DECISION_JSON = (
    '{{"action":"speak","topic":"问问面试","why_now":"面试已经结束",'
    '"topic_source":"recent_context","obs":"","watch":[],"watch_keywords":[],'
    '"why":"","drifted":false,"cooldown":0,"item_id":{item_id}}}'
)


def make_engine(store, state, clock, platform, *, speech, item_id="null", bridge_on=True, **bridge_kw):
    config = state._config
    window = SlidingWindow(config)
    packager = ContextPackager(config)
    core = DecisionCore(config, state, window, packager)
    prompts = []

    async def generate_direct(**kw):
        prompts.append(kw)
        if len(prompts) == 1:
            return DECISION_JSON.format(item_id=item_id)
        return speech

    context = SimpleNamespace(
        persona_manager=SimpleNamespace(get_default_persona_v3=AsyncMock(return_value={"prompt": "P"})),
        send_message=AsyncMock(return_value=True),
        get_config=lambda _umo=None: {},
        get_platform_inst=lambda pid: platform if pid == "userbot" else None,
    )
    speech_ctx = AsyncMock(return_value="<iris:continuity>CTX</iris:continuity>")
    bridge = None
    if bridge_on:
        bridge = ProactiveBridge(
            store,
            state,
            policy_get=default_policy,
            context=context,
            clock=clock,
            speech_context_fn=speech_ctx,
            umo_get=lambda sk: UMO,
            **bridge_kw,
        )
    on_sent = AsyncMock()
    engine = ProactiveEngine(
        context, config, state, window, SignalGate(config, state), core, SimpleNamespace(
            record_decision_error=lambda *a, **k: None, record_decision=lambda *a, **k: None
        ),
        llm_manager=SimpleNamespace(generate_direct=generate_direct),
        packager=packager,
        umo_get=lambda gid: UMO,
        is_busy=lambda gid: False,
        self_id_get=lambda: "bot",
        save_fn=AsyncMock(),
        on_initiate_sent=on_sent,
        bridge=bridge,
    )
    state.add_to_whitelist(SCOPE)
    old = time.time() - 7200
    window.append(SCOPE, WindowMessage(OWNER, "她", "周五下午面试", old))
    state.record_human_message(SCOPE, old)
    return SimpleNamespace(
        engine=engine, bridge=bridge, prompts=prompts, context=context,
        speech_ctx=speech_ctx, on_sent=on_sent, window=window,
    )


class TestEngineBridge:
    @pytest.mark.asyncio
    async def test_full_send_with_focus_item(self, store, state, clock):
        item = add_item(store, clock)
        platform = FakePlatform([[11]])
        env = make_engine(store, state, clock, platform, speech="面试怎么样？", item_id=f'"{item["item_id"]}"')
        result = await env.engine.attempt_initiate(SCOPE)
        assert result.startswith("已发起")
        # 决策 prompt 带候选与私聊措辞
        decision_prompt = env.prompts[0]
        assert "<candidates>" in decision_prompt["prompt"] and item["item_id"] in decision_prompt["prompt"]
        assert "私聊" in decision_prompt["system_prompt"]
        # 生成 prompt 使用连续性上下文（焦点事项）与私聊提示
        env.speech_ctx.assert_awaited_once_with(SCOPE, item["item_id"])
        assert "CTX" in env.prompts[1]["system_prompt"]
        assert "私聊" in env.prompts[1]["prompt"]
        env.context.send_message.assert_not_awaited()  # 旧直发路径不用
        assert platform.calls == [(UMO, "面试怎么样？")]
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1 and data.initiate_pending_since > 0
        turns = store.turns_in_state((m.TURN_SENT,))
        assert len(turns) == 1 and turns[0]["item_id"] == item["item_id"]
        assert turns[0]["llm_calls"] == 2 and turns[0]["quota_state"] == "charged"
        it = store.get_item(item["item_id"])
        assert it["ask_count"] == 1 and it["last_outbound_msg_ids"] == [11]
        env.on_sent.assert_awaited_once_with(SCOPE, "面试怎么样？")

    @pytest.mark.asyncio
    async def test_invalid_item_id_becomes_null(self, store, state, clock):
        add_item(store, clock)
        env = make_engine(store, state, clock, FakePlatform([[11]]), speech="嗨", item_id='"made-up"')
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
        env.speech_ctx.assert_awaited_once_with(SCOPE, None)
        assert store.turns_in_state((m.TURN_SENT,))[0]["item_id"] is None

    @pytest.mark.asyncio
    async def test_correction_between_parts_stops_rest(self, store, state, clock):
        async def on_send(n):
            if n == 1:  # 第一段发出时主人又发来一条（更正）
                store.advance_watermark(SCOPE, 500, clock())

        platform = FakePlatform([[21], [22]], on_send=on_send)
        env = make_engine(store, state, clock, platform, speech="第一段\n\n第二段\n\n第三段")
        result = await env.engine.attempt_initiate(SCOPE)
        assert result.startswith("已发起")
        assert [c[1] for c in platform.calls] == ["第一段"]
        turn = store.turns_in_state((m.TURN_PARTIAL,))[0]
        assert turn["parts_sent"] == 1 and turn["parts_total"] == 3
        assert turn["reason"] == "new_inbound" and turn["quota_state"] == "charged"
        rows = store.outbound_for_turn(turn["turn_id"])
        assert [(r["text"], r["status"]) for r in rows] == [("第一段", m.OUT_SENT)]
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1  # 部分成功只计一次
        # F3：本人已在首段之后回复 → 不设 pending（不会在超时后被计为未回复）
        assert data.initiate_pending_since == 0

    @pytest.mark.asyncio
    async def test_multi_part_success_counts_once(self, store, state, clock):
        platform = FakePlatform([[31], [32], [33]])
        env = make_engine(store, state, clock, platform, speech="一\n\n二\n\n三\n\n四")
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
        assert [c[1] for c in platform.calls] == ["一", "二", "三\n\n四"]
        assert len(state.get_state(SCOPE).initiate_timestamps) == 1
        assert store.turns_in_state((m.TURN_SENT,))[0]["parts_sent"] == 3

    @pytest.mark.asyncio
    async def test_failure_on_second_part_is_partial(self, store, state, clock):
        platform = FakePlatform([[41], FloodWaitError("x"), [43]])
        env = make_engine(store, state, clock, platform, speech="一\n\n二\n\n三")
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
        assert len(platform.calls) == 2
        turn = store.turns_in_state((m.TURN_PARTIAL,))[0]
        assert turn["parts_sent"] == 1
        assert len(state.get_state(SCOPE).initiate_timestamps) == 1

    @pytest.mark.asyncio
    async def test_confirmed_failure_no_quota_no_unanswered(self, store, state, clock):
        env = make_engine(store, state, clock, FakePlatform([FloodWaitError("x")]), speech="嗨")
        result = await env.engine.attempt_initiate(SCOPE)
        assert result.startswith("发送失败")
        data = state.get_state(SCOPE)
        assert data.initiate_timestamps == [] and data.initiate_pending_since == 0
        assert data.last_initiate_time == 0
        assert store.turns_in_state((m.TURN_FAILED,))[0]["quota_state"] == "none"
        env.on_sent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uncertain_reserves_and_blocks(self, store, state, clock):
        env = make_engine(store, state, clock, FakePlatform([ConnectionError()]), speech="嗨")
        result = await env.engine.attempt_initiate(SCOPE)
        assert "不确定" in result
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1 and data.initiate_pending_since == 0
        # min_interval 生效：定时门控不会点火
        assert env.engine._signals.evaluate_timer(SCOPE, env.window.get_messages(SCOPE)) is None
        assert state.get_next_initiate_check(SCOPE) >= time.time() + 59 * 60
        # 下一次尝试（例如管理命令强制）被 uncertain 回合挡住
        assert (await env.engine.attempt_initiate(SCOPE, force=True)) == "已跳过发起：uncertain_pending"

    @pytest.mark.asyncio
    async def test_new_inbound_during_generation_cancels(self, store, state, clock):
        platform = FakePlatform([[51]])
        env = make_engine(store, state, clock, platform, speech="嗨")
        orig = env.engine._generate_speech

        async def gen(*a, **k):
            store.advance_watermark(SCOPE, 900, clock())
            return await orig(*a, **k)

        env.engine._generate_speech = gen
        result = await env.engine.attempt_initiate(SCOPE)
        assert result == "已取消发起：new_inbound"
        assert platform.calls == []
        assert store.turns_in_state((m.TURN_CANCELLED,))[0]["reason"] == "new_inbound"
        assert state.get_state(SCOPE).initiate_timestamps == []

    @pytest.mark.asyncio
    async def test_declined_closes_turn(self, store, state, clock):
        env = make_engine(store, state, clock, FakePlatform(), speech="嗨")

        async def decline(**kw):
            return '{"action":"none"}'

        env.engine._llm_manager = SimpleNamespace(generate_direct=decline)
        assert await env.engine.attempt_initiate(SCOPE) == "LLM 决定暂不发起"
        assert store.turns_in_state((m.TURN_SKIPPED,))[0]["reason"] == "llm_declined"
        assert env.engine._bridge._active == {}

    @pytest.mark.asyncio
    async def test_no_receipt_platform_never_sends(self, store, state, clock):
        platform = NoReceiptPlatform()
        env = make_engine(store, state, clock, platform, speech="嗨")
        result = await env.engine.attempt_initiate(SCOPE)
        assert result == "发送失败：no_receipt_path"
        platform.send_by_session.assert_not_called()
        env.context.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_startup_reschedules_overdue_without_burst(self, store, state, clock):
        env = make_engine(store, state, clock, FakePlatform(), speech="嗨")
        past = time.time() - 3600
        state.set_next_initiate_check(SCOPE, past)
        await env.engine._bridge_startup()
        nxt = state.get_next_initiate_check(SCOPE)
        assert time.time() + 25 <= nxt <= time.time() + 185

    @pytest.mark.asyncio
    async def test_without_bridge_old_path(self, store, state, clock):
        env = make_engine(store, state, clock, FakePlatform(), speech="一\n\n二", bridge_on=False)
        result = await env.engine.attempt_initiate(SCOPE)
        assert result.startswith("已发起")
        env.context.send_message.assert_awaited_once()
        assert "<candidates>" not in env.prompts[0]["prompt"]
        assert "群" in env.prompts[1]["prompt"]
        assert store.turns_in_state(tuple(m.TURN_STATES)) == []


class TestEngineFixes:
    @pytest.mark.asyncio
    async def test_engine_with_bridge_disables_private_time_reset(self, store, state, clock):
        make_engine(store, state, clock, FakePlatform(), speech="嗨")
        data = state.get_state(SCOPE)
        data.initiate_no_reply_streak = 2
        data.last_initiate_time = time.time() - 25 * 3600
        assert state.get_state(SCOPE).initiate_no_reply_streak == 2  # F1：不按时间解除

    @pytest.mark.asyncio
    async def test_owner_reply_after_single_part_no_pending(self, store, state, clock):
        async def on_send(n):
            store.advance_watermark(SCOPE, 600, clock())  # 发出后本人立刻回复

        env = make_engine(store, state, clock, FakePlatform([[61]], on_send=on_send), speech="嗨")
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
        data = state.get_state(SCOPE)
        assert len(data.initiate_timestamps) == 1 and data.initiate_pending_since == 0

    @pytest.mark.asyncio
    async def test_avoided_topic_listed_and_decision_skipped(self, store, state, clock):
        closed_item(store, clock, "面试", m.STATUS_DECLINED, age_s=2 * 86400)
        platform = FakePlatform([[71]])
        env = make_engine(store, state, clock, platform, speech="面试怎么样？")
        result = await env.engine.attempt_initiate(SCOPE)
        assert result == "已跳过发起：avoided_topic"
        assert "<avoid_topics>\n- 面试\n</avoid_topics>" in env.prompts[0]["prompt"]
        assert len(env.prompts) == 1 and platform.calls == []
        assert store.turns_in_state((m.TURN_SKIPPED,))[0]["reason"] == "avoided_topic"
        assert state.get_state(SCOPE).initiate_timestamps == []

    @pytest.mark.asyncio
    async def test_avoid_list_in_speech_prompt_without_numbers(self, store, state, clock):
        import re

        closed_item(store, clock, "体检", m.STATUS_DONE)
        env = make_engine(store, state, clock, FakePlatform([[81]]), speech="面试怎么样？")
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
        block = re.search(r"<avoid_topics>.*?</avoid_topics>", env.prompts[0]["prompt"], re.S).group(0)
        assert "体检" in block and not re.search(r"\d", block)
        speech_sys = env.prompts[1]["system_prompt"]
        assert "不要主动提起" in speech_sys and "- 体检" in speech_sys
        assert not re.search(r"\d", speech_sys.split("【不要主动提起】", 1)[1])

    @pytest.mark.asyncio
    async def test_group_turn_has_no_avoid_block(self, store, state, clock):
        closed_item(store, clock, "面试", m.STATUS_DECLINED)
        env = make_engine(store, state, clock, FakePlatform(), speech="嗨", bridge_on=False)
        await env.engine.attempt_initiate(SCOPE)
        assert "<avoid_topics>" not in env.prompts[0]["prompt"]


class TestAvoidTopicsRegression:
    """复核发现：已完成“面试”不应挡住仍有效的“面试结果”候选"""

    @pytest.mark.asyncio
    async def test_overlapping_active_item_not_avoided(self, store, bridge, clock):
        closed_item(store, clock, "面试", m.STATUS_DONE)
        closed_item(store, clock, "体检", m.STATUS_DONE)
        add_item(store, clock, topic="面试结果", kind=m.KIND_PENDING_RESULT)
        got = await bridge.avoid_topics(SCOPE)
        assert got == ["体检"]

    @pytest.mark.asyncio
    async def test_chosen_candidate_not_skipped_by_avoid_check(self, store, state, clock):
        closed_item(store, clock, "面试", m.STATUS_DECLINED)  # 与候选不重叠时仍会列入回避
        item = add_item(store, clock, topic="交报告")
        env = make_engine(store, state, clock, FakePlatform([[91]]), speech="报告交了吗？", item_id=f'"{item["item_id"]}"')
        result = await env.engine.attempt_initiate(SCOPE)
        assert result.startswith("已发起"), result


class TestLegacyTargetGuard:
    """P7：未接入桥接时旧直发路径也拒绝禁发目标"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scope",
        [CHANNEL, "1000000001", f"{CHANNEL}#12", SILENT, "private:111"],
    )
    async def test_refuses_forbidden_targets(self, store, clock, scope):
        st = StateManager(make_config(silent_sessions=[SILENT]))
        env = make_engine(store, st, clock, FakePlatform(), speech="嗨", bridge_on=False)
        st.add_to_whitelist(scope)
        env.window.append(scope, WindowMessage("u", "U", "hi", time.time() - 7200))
        st.record_human_message(scope, time.time() - 7200)
        result = await env.engine.attempt_initiate(scope, force=True)
        assert result.startswith("已拒绝发起"), result
        env.context.send_message.assert_not_awaited()
        assert env.prompts == []

    @pytest.mark.asyncio
    async def test_allows_whitelisted_owner_and_group(self, store, clock):
        st = StateManager(make_config(silent_sessions=[SILENT]))
        env = make_engine(store, st, clock, FakePlatform(), speech="嗨", bridge_on=False)
        assert env.engine._legacy_target_refusal(SCOPE) == ""
        assert env.engine._legacy_target_refusal(GROUP) == ""
        assert (await env.engine.attempt_initiate(SCOPE)).startswith("已发起")
