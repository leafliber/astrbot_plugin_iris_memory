"""ContextAssembler 单元测试（tmp_path DB + 可控时钟 + 假画像/L1）"""

from pathlib import Path

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity import context as ctx_mod
from astrbot_plugin_iris_memory.iris_memory.continuity.context import (
    CLOSE_TAG,
    OPEN_TAG,
    ContextAssembler,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.store import ContinuityStore

OWNER_UID = "100000001"
OWNER = f"private:{OWNER_UID}"
GROUP = "-100777"
STRANGER = "private:111"
T0 = 1_800_000_000.0

pytestmark = pytest.mark.asyncio


class Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock):
    s = ContinuityStore(tmp_path / "continuity" / "continuity.db", clock=clock)
    s.open()
    yield s
    s.close()


def _assembler(store, clock, **kw):
    kw.setdefault("config", {"inject_max_items": 3, "inject_max_chars": 600})
    return ContextAssembler(store, lambda: {OWNER_UID}, clock=clock, **kw)


def _item(store, **over):
    base = dict(
        subject_id=OWNER_UID,
        subject_type="user",
        scope_key=OWNER,
        topic="周五面试",
        summary="她周五下午有个面试",
        kind="plan",
        source_kind="shared_dialogue",
        source_refs=[{"adapter": "telegram", "scope_key": OWNER, "msg_id": 10, "at": T0}],
    )
    base.update(over)
    return store.create_item(**base)


def _sent(store, scope, text, author="bot", status="sent", tg=()):
    oid = store.add_outbound(scope, "reply", text, "sending", author)
    store.mark_outbound(oid, status, tg_msg_ids=tg)
    return oid


def _seed_private(store):
    store.put_private_context("identity", "主人私密身份段落", "persona_split", OWNER)
    store.put_private_context("recent_dynamics", "她最近在频道发了很多猫的照片", "digest", OWNER)
    _item(store, topic="周五面试", summary="她周五下午有个面试")
    _item(
        store,
        topic="猫咪照片",
        summary="她在频道晒猫",
        kind="note",
        source_kind="channel_observation",
    )


# ---------------------------------------------------------------- T11 permissions


async def test_group_never_gets_private_context_or_owner_items(store, clock):
    _seed_private(store)
    # 即使事项被错误地授予了群 audience，私聊来源/频道观察也不得进群
    _item(store, topic="私聊来源误授权", allowed_audiences=[OWNER, GROUP])
    _item(
        store,
        scope_key=GROUP,
        topic="频道误授权",
        source_kind="channel_observation",
        allowed_audiences=[GROUP],
    )
    _item(store, scope_key=GROUP, subject_id="555", topic="群里约饭", summary="群友说周末约饭")
    a = _assembler(store, clock)

    out = await a.assemble(GROUP, purpose="reply", sender_id=OWNER_UID)
    assert "主人私密身份段落" not in out
    assert "猫" not in out
    assert "周五面试" not in out
    assert "私聊来源误授权" not in out
    assert "频道误授权" not in out
    assert "群里约饭" in out

    out_p = await a.assemble(GROUP, purpose="proactive", sender_id=None)
    assert "主人私密身份段落" not in out_p
    assert "周五面试" not in out_p


async def test_non_owner_private_gets_nothing_private(store, clock):
    _seed_private(store)
    a = _assembler(store, clock)
    assert await a.assemble(STRANGER, purpose="reply", sender_id="111") == ""
    # 主人私聊 scope 但 sender 不是主人（防御）
    out = await a.assemble(OWNER, purpose="reply", sender_id="111")
    assert "主人私密身份段落" not in out


async def test_owner_private_gets_private_layer_and_items(store, clock):
    _seed_private(store)
    a = _assembler(store, clock)
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert out.startswith(OPEN_TAG) and out.endswith(CLOSE_TAG)
    assert "【身份】" in out and "主人私密身份段落" in out
    assert "【近期动态】" in out
    assert "你们在私聊里聊过：周五面试" in out
    assert "她在频道发过：猫咪照片" in out


async def test_owner_ids_failure_means_not_owner(store, clock):
    _seed_private(store)

    def boom():
        raise RuntimeError("x")

    a = ContextAssembler(store, boom, config={}, clock=clock)
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "主人私密身份段落" not in out
    assert "猫咪照片" not in out


# ---------------------------------------------------------------- attribution


async def test_source_attribution_phrasing(store, clock):
    _item(store, topic="旧日记", summary="以前写过的东西", kind="note", source_kind="history_import")
    _item(
        store,
        topic="同事八卦",
        summary="同事要离职",
        kind="note",
        source_kind="third_party",
        source_refs=[{"source": "她同事"}],
    )
    a = _assembler(store, clock, config={"inject_max_items": 10, "inject_max_chars": 4000})
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "（旧记录，非共同经历）：旧日记" in out
    assert "你们在私聊里聊过：旧日记" not in out
    assert "据她同事说：同事八卦" in out


async def test_self_expression_and_promise_phrasing(store, clock):
    _item(
        store,
        subject_type="self",
        topic="帮她查展览",
        summary="你说会帮她查展览时间",
        kind="bot_promise",
        source_kind="self_expression",
    )
    _item(store, subject_type="self", topic="喜欢雨天", kind="note", source_kind="self_expression")
    a = _assembler(store, clock, config={"inject_max_items": 10})
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "你答应过（未完成）：帮她查展览" in out
    assert "你说过：喜欢雨天" in out


async def test_inactive_items_excluded(store, clock):
    it = _item(store, topic="已完成的事")
    store.update_item(it["item_id"], expected_revision=1, reason="t", source_msg_id=None, status="done")
    a = _assembler(store, clock)
    assert "已完成的事" not in await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)


# ---------------------------------------------------------------- outbound


async def test_only_sent_bot_outbound_included(store, clock):
    _sent(store, OWNER, "这是草稿", status="draft")
    _sent(store, OWNER, "这是失败的", status="failed")
    _sent(store, OWNER, "这是不确定的", status="uncertain")
    _sent(store, OWNER, "手动发的", author="manual", tg=(9,))
    _sent(store, OWNER, "真实发出的一句" + "很长" * 60, tg=(10,))
    a = _assembler(store, clock)
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "草稿" not in out and "失败" not in out and "不确定" not in out
    assert "手动发的" not in out
    line = next(ln for ln in out.splitlines() if ln.startswith("- 你说过：真实发出的一句"))
    assert len(line[len("- 你说过：") :]) <= ctx_mod.OUTBOUND_MAX_CHARS


async def test_outbound_at_most_three(store, clock):
    for i in range(5):
        clock.t += 1
        _sent(store, OWNER, f"消息{i}", tg=(100 + i,))
    a = _assembler(store, clock)
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert out.count("- 你说过：消息") == 3
    assert "消息4" in out and "消息0" not in out


async def test_outbound_defensive_filter_when_store_misbehaves(store, clock):
    class LeakyStore:
        def __getattr__(self, name):
            return getattr(store, name)

        def recent_outbound(self, *a, **k):
            return [{"status": "draft", "author": "bot", "text": "泄漏草稿"}]

    a = _assembler(LeakyStore(), clock)
    assert "泄漏草稿" not in await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)


# ---------------------------------------------------------------- closure


async def test_closure_endpoint_only_when_ended(store, clock):
    store.put_closure(OWNER, state="continue", summary="还在聊", next_opening="x")
    a = _assembler(store, clock)
    assert "上次结束点" not in await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    store.put_closure(OWNER, state="ended", ended_at=T0, summary="聊到晚饭", next_opening="周五面试")
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "【上次结束点】" in out and "聊到晚饭" in out and "周五面试" in out


# ---------------------------------------------------------------- budget


async def test_budget_max_items(store, clock):
    for i in range(6):
        _item(store, topic=f"事项{i}", summary=f"摘要{i}", importance=1 + (i % 3))
    a = _assembler(store, clock, config={"inject_max_items": 2, "inject_max_chars": 4000})
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    item_lines = [ln for ln in out.splitlines() if ln.startswith("- [")]
    assert len(item_lines) == 2
    # importance 3 的排前
    assert all("事项2" in ln or "事项5" in ln for ln in item_lines)


async def test_budget_max_chars(store, clock):
    for i in range(3):
        _item(store, topic=f"长事项{i}", summary="很" * 150)
    a = _assembler(store, clock, config={"inject_max_items": 3, "inject_max_chars": 400})
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    item_lines = [ln for ln in out.splitlines() if ln.startswith("- [")]
    assert 1 <= len(item_lines) < 3
    assert sum(len(ln) for ln in item_lines) <= 400


async def test_budget_first_item_clipped_when_oversized(store, clock):
    _item(store, topic="超长", summary="长" * 160)
    a = _assembler(store, clock, config={"inject_max_items": 3, "inject_max_chars": 100})
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    item_lines = [ln for ln in out.splitlines() if ln.startswith("- [")]
    assert len(item_lines) == 1 and len(item_lines[0]) <= 100


async def test_config_clamped():
    a = ContextAssembler(None, set, config={"inject_max_items": 99, "inject_max_chars": "bad"})
    assert a.max_items == 10 and a.max_chars == 600


# ---------------------------------------------------------------- proactive


async def test_proactive_focuses_one_item(store, clock):
    focus = _item(store, topic="焦点事项", summary="今天要问的", importance=1)
    for i in range(4):
        _item(store, topic=f"背景{i}", summary=f"背景摘要{i}", importance=3)
    a = _assembler(store, clock, config={"inject_max_items": 5, "inject_max_chars": 4000})
    out = await a.assemble(OWNER, purpose="proactive", sender_id=None, focus_item_id=focus["item_id"])
    item_lines = [ln for ln in out.splitlines() if ln.startswith("- [")]
    assert len(item_lines) == 2
    assert "焦点事项" in item_lines[0] and "本次焦点" in item_lines[0]
    assert "本次焦点" not in item_lines[1]
    assert "只围绕一个话题" in out


async def test_proactive_focus_not_permitted_is_ignored(store, clock):
    secret = _item(store, topic="主人私事")
    _item(store, scope_key=GROUP, subject_id="5", topic="群事", allowed_audiences=[GROUP])
    a = _assembler(store, clock)
    out = await a.assemble(GROUP, purpose="proactive", focus_item_id=secret["item_id"])
    assert "主人私事" not in out
    assert "群事" in out and "本次焦点" not in out


async def test_proactive_profile_and_l1(store, clock):
    calls = []

    async def profile(scope):
        calls.append(scope)
        return "兴趣：猫"

    def l1(scope, n):
        assert n == 12
        return [("user", f"u{i}") for i in range(20)] + [("assistant", "好呀")]

    a = _assembler(store, clock, profile_text=profile, l1_recent=l1)
    out = await a.assemble(OWNER, purpose="proactive")
    assert "【画像摘要】\n兴趣：猫" in out
    assert "你：好呀" in out and "她：u19" in out
    assert out.count("\n她：") + out.count("\n你：") == 12
    # reply 模式不注入画像 / L1；群聊不注入画像
    out_r = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "画像摘要" not in out_r and "最近对话" not in out_r
    out_g = await a.assemble(GROUP, purpose="proactive")
    assert "画像摘要" not in out_g
    assert "对方：u19" in out_g
    assert calls == [OWNER]


# ---------------------------------------------------------------- isolation / empty


async def test_empty_returns_blank(store, clock):
    a = _assembler(store, clock)
    assert await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID) == ""
    assert await a.assemble("", purpose="reply") == ""


async def test_sub_source_failure_isolated(store, clock):
    _item(store, topic="周五面试")
    _sent(store, OWNER, "我在呢", tg=(1,))

    class Broken:
        def __getattr__(self, name):
            return getattr(store, name)

        def get_private_context(self, *a, **k):
            raise RuntimeError("db gone")

        def get_closure(self, *a, **k):
            raise RuntimeError("db gone")

    async def bad_profile(scope):
        raise RuntimeError("profile")

    def bad_l1(scope, n):
        raise RuntimeError("l1")

    a = _assembler(Broken(), clock, profile_text=bad_profile, l1_recent=bad_l1)
    out = await a.assemble(OWNER, purpose="proactive")
    assert "周五面试" in out and "你说过：我在呢" in out


async def test_not_before_hint(store, clock):
    _item(store, topic="周五面试", not_before=T0 + 3600, time_basis="周五")
    a = _assembler(store, clock)
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "时间：周五" in out and "先别问结果" in out
    clock.t = T0 + 7200
    out = await a.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "先别问结果" not in out


# ---------------------------------------------------------------- proactive eligibility (coverage F1)


@pytest.mark.parametrize(
    "over",
    [
        {"proactive_allowed": 0},  # “不用提醒”
        {"ask_count": 2},
        {"last_mentioned_at": T0 - 3600},  # 6h 冷却内
        {"not_before": T0 + 3600},
        {"expires_at": T0 - 1},
    ],
)
async def test_proactive_background_requires_candidate_eligibility(store, clock, over):
    _item(store, topic="周五交报告", summary="她周五交报告", **over)
    text = await _assembler(store, clock).assemble(OWNER, purpose="proactive", focus_item_id=None)
    assert "周五交报告" not in text
    # 普通回复上下文不变
    reply = await _assembler(store, clock).assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "周五交报告" in reply


async def test_proactive_background_eligible_item_kept(store, clock):
    _item(store, topic="周五交报告", summary="她周五交报告")
    text = await _assembler(store, clock).assemble(OWNER, purpose="proactive")
    assert "周五交报告" in text


async def test_proactive_closure_opening_only_for_focus(store, clock):
    focus = _item(store, topic="搬家", summary="她要搬家")
    store.put_closure(OWNER, state="ended", ended_at=T0, summary="聊了面试", next_opening="周五面试")
    asm = _assembler(store, clock)
    text = await asm.assemble(OWNER, purpose="proactive", focus_item_id=focus["item_id"])
    assert "下次可以从" not in text and "上次聊到：聊了面试" in text
    store.put_closure(OWNER, next_opening="搬家")
    text = await asm.assemble(OWNER, purpose="proactive", focus_item_id=focus["item_id"])
    assert "下次可以从“搬家”接上" in text
    reply = await asm.assemble(OWNER, purpose="reply", sender_id=OWNER_UID)
    assert "下次可以从“搬家”接上" in reply
