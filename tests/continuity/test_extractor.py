"""ItemExtractor / ExtractionWorker / record_bot_promise 测试（tmp_path DB + 可控时钟 + 假 LLM）"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity import models as m
from astrbot_plugin_iris_memory.iris_memory.continuity.extractor import (
    ExtractionWorker,
    ItemExtractor,
    parse_llm_json,
    record_bot_promise,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.store import ContinuityStore

OWNER = "100000001"
SCOPE = f"private:{OWNER}"
HOUR = 3600.0
DAY = 86400.0
# 2026-11-02（周一）10:00 Asia/Shanghai
BASE = datetime(2026, 11, 2, 10, 0, tzinfo=m.TZ_SHANGHAI).timestamp()


def local(ts):
    return datetime.fromtimestamp(ts, m.TZ_SHANGHAI)


class Clock:
    def __init__(self, t: float = BASE):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class FakeLLM:
    """按顺序返回脚本：str / dict（转 JSON）/ 异常实例 / callable(prompt)->上述之一"""

    def __init__(self, *script):
        self.script = list(script)
        self.prompts = []
        self.timeouts = []

    async def __call__(self, prompt: str, timeout: float) -> str:
        self.prompts.append(prompt)
        self.timeouts.append(timeout)
        if not self.script:
            raise AssertionError("unexpected LLM call")
        out = self.script.pop(0)
        if callable(out) and not isinstance(out, Exception):
            out = out(prompt)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, dict):
            return json.dumps(out, ensure_ascii=False)
        return out

    @property
    def calls(self) -> int:
        return len(self.prompts)


def ops(*items):
    return {"ops": list(items)}


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


CFG = {"extract_timeout_s": 5, "extract_debounce_s": 0, "extract_queue_max": 50}


def make(store, clock, *script):
    llm = FakeLLM(*script)
    return ItemExtractor(store, llm, CFG, clock), llm


def inbound(store, msg_id, text, ts=BASE, reply_to=None, sender=OWNER):
    return store.record_inbound(SCOPE, msg_id, sender, text, ts, reply_to)


def new_item(store, topic, kind=m.KIND_PLAN, **kw):
    return store.create_item(
        subject_id=OWNER,
        subject_type="user",
        scope_key=SCOPE,
        topic=topic,
        summary=topic,
        kind=kind,
        source_kind=m.SOURCE_SHARED_DIALOGUE,
        **kw,
    )


def snapshot(store):
    return [
        (i["item_id"], i["revision"], i["status"])
        for i in store.list_items(statuses=None, limit=100)
    ]


# ---------------------------------------------------------------------------
# T01 创建 + 不用提醒覆盖 + 落盘
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t01_create_no_remind_persisted(store, clock, db_path):
    text = "周五交报告，先记着，不用提醒"
    ext, llm = make(
        store,
        clock,
        ops(
            {
                "op": "create",
                "kind": "plan",
                "topic": "周五交报告",
                "summary": "她周五要交报告",
                "when_text": "周五",
                "remind_requested": True,  # LLM 说要提醒，被确定性覆盖
                "evidence": "周五交报告",
            }
        ),
    )
    assert inbound(store, 101, text)
    clock.advance(10)
    res = await ext.process_scope(SCOPE, OWNER)

    assert res.status == "done" and llm.calls == 1 and len(res.created) == 1
    item = store.get_item(res.created[0])
    assert item["kind"] == "plan"
    assert item["remind_requested"] == 0
    assert item["proactive_allowed"] == 0
    assert item["source_kind"] == "shared_dialogue"
    assert item["allowed_audiences"] == [SCOPE]
    assert item["source_refs"][0]["msg_id"] == 101
    assert item["time_basis"] == "周五" and item["time_precision"] == "day"
    start = local(item["event_at_start"])
    assert (start.year, start.month, start.day) == (2026, 11, 6)
    assert item["not_before"] == item["event_at_end"]
    assert item["expires_at"] == pytest.approx(item["event_at_end"] + 3 * DAY)
    assert store.get_inbound(SCOPE, 101)["extract_status"] == "done"

    store.close()
    reopened = ContinuityStore(db_path, clock=clock)
    reopened.open()
    try:
        again = reopened.get_item(res.created[0])
        assert again is not None and again["remind_requested"] == 0 and again["proactive_allowed"] == 0
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_remind_phrase_sets_remind_requested(store, clock):
    ext, _ = make(
        store,
        clock,
        ops(
            {
                "op": "create",
                "kind": "plan",
                "topic": "体检",
                "summary": "她明天体检",
                "when_text": "明天上午",
                "remind_requested": False,
                "evidence": "明天上午体检",
            }
        ),
    )
    inbound(store, 1, "明天上午体检，到时候提醒我")
    res = await ext.process_scope(SCOPE, OWNER)
    item = store.get_item(res.created[0])
    assert item["remind_requested"] == 1 and item["proactive_allowed"] == 1


@pytest.mark.asyncio
async def test_time_uses_original_msg_ts(store, clock):
    """处理时刻晚于消息时间：'明天' 仍按原始消息时间解析"""
    ext, _ = make(
        store,
        clock,
        ops({"op": "create", "kind": "plan", "topic": "面试", "summary": "", "when_text": "明天",
             "evidence": "明天面试"}),
    )
    inbound(store, 1, "明天面试", ts=BASE)
    clock.advance(20 * HOUR)  # 已到 11-03 06:00
    res = await ext.process_scope(SCOPE, OWNER)
    item = store.get_item(res.created[0])
    assert local(item["event_at_start"]).day == 3
    assert item["summary"] == "面试"  # 空 summary 回退为 topic


@pytest.mark.asyncio
async def test_unknown_time_defaults(store, clock):
    ext, _ = make(
        store,
        clock,
        ops(
            {"op": "create", "kind": "pending_result", "topic": "面试结果", "summary": "等面试结果",
             "when_text": "", "evidence": "等面试结果"},
            {"op": "create", "kind": "unfinished_topic", "topic": "猫的事", "summary": "猫的事没聊完",
             "evidence": "猫的事"},
            {"op": "create", "kind": "note", "topic": "喜欢抹茶", "summary": "她喜欢抹茶",
             "evidence": "喜欢抹茶"},
        ),
    )
    inbound(store, 1, "在等面试结果，猫的事回头再说，记一下我喜欢抹茶")
    res = await ext.process_scope(SCOPE, OWNER)
    assert len(res.created) == 3
    by_kind = {store.get_item(i)["kind"]: store.get_item(i) for i in res.created}
    pr = by_kind["pending_result"]
    assert pr["event_at_start"] is None and pr["time_precision"] == "none"
    assert pr["not_before"] == pytest.approx(BASE + 12 * HOUR)
    assert pr["expires_at"] == pytest.approx(BASE + 7 * DAY)
    assert by_kind["unfinished_topic"]["expires_at"] == pytest.approx(BASE + 2 * DAY)
    assert by_kind["note"]["expires_at"] is None


# ---------------------------------------------------------------------------
# T02 改期 → 完成
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t02_reschedule_then_done(store, clock):
    created = {}

    def reschedule(prompt):
        assert created["id"] in prompt
        return ops({"op": "reschedule", "item_id": created["id"], "when_text": "下周一",
                    "evidence": "改到下周一"})

    def complete(prompt):
        return ops({"op": "complete", "item_id": created["id"], "evidence": "交了"})

    ext, llm = make(
        store,
        clock,
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "她周五交报告",
             "when_text": "周五", "evidence": "周五交报告"}),
        reschedule,
        complete,
    )
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    created["id"] = item_id = res.created[0]
    first = store.get_item(item_id)
    assert first["revision"] == 1 and local(first["event_at_start"]).day == 6

    clock.advance(HOUR)
    inbound(store, 2, "报告改到下周一了", ts=clock())
    res2 = await ext.process_scope(SCOPE, OWNER)
    assert res2.updated == [item_id]
    second = store.get_item(item_id)
    assert second["revision"] == 2
    assert local(second["event_at_start"]).day == 9 and second["time_basis"] == "下周一"
    assert second["not_before"] == second["event_at_end"]
    hist = store.item_history(item_id)
    assert [h["revision"] for h in hist] == [1, 2]
    assert hist[0]["snapshot"]["time_basis"] == "周五"
    assert hist[1]["reason"] == "extract_reschedule" and hist[1]["source_msg_id"] == "2"

    clock.advance(HOUR)
    inbound(store, 3, "交了", ts=clock())
    res3 = await ext.process_scope(SCOPE, OWNER)
    assert res3.closed == [item_id]
    third = store.get_item(item_id)
    assert third["revision"] == 3 and third["status"] == "done"
    assert third["closed_reason"] == "completed"
    assert llm.calls == 3
    assert len(store.list_items(statuses=None)) == 1


# ---------------------------------------------------------------------------
# 校验失败 → 不写库
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "好的我记下了",  # 非 JSON
        '{"ops": [',  # 截断 JSON
        {"items": []},  # 缺 ops
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
             "when_text": "周五", "evidence": "周六交报告"}),  # evidence 非子串
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
             "when_text": "下周五", "evidence": "交报告"}),  # when_text 非子串
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
             "evidence": "交"}),  # evidence 过短
        ops({"op": "create", "kind": "bot_promise", "topic": "交报告", "summary": "x",
             "evidence": "交报告"}),  # kind 不允许
        ops({"op": "create", "kind": "plan", "topic": "交" * 41, "summary": "x",
             "evidence": "交报告"}),  # topic 超长
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
             "remind_requested": "yes", "evidence": "交报告"}),  # 类型不符
        ops({"op": "complete", "item_id": "deadbeef", "evidence": "交报告"}),  # 未知 item_id
        ops({"op": "explode", "evidence": "交报告"}),
    ],
)
async def test_invalid_llm_output_changes_nothing(store, clock, reply):
    existing = new_item(store, "别的事")
    before = snapshot(store)
    ext, llm = make(store, clock, reply)
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 1
    assert res.status == "failed" and not res.created and not res.updated and not res.closed
    assert snapshot(store) == before
    assert store.get_item(existing["item_id"])["revision"] == 1
    row = store.get_inbound(SCOPE, 1)
    assert row["extract_status"] == "failed" and row["last_error"]


@pytest.mark.asyncio
async def test_unknown_item_id_does_not_touch_other_items(store, clock):
    a = new_item(store, "交报告")
    ext, _ = make(
        store,
        clock,
        ops(
            {"op": "complete", "item_id": "0" * 32, "evidence": "交了"},
            {"op": "cancel", "item_id": a["item_id"] + "x", "evidence": "交了"},
        ),
    )
    inbound(store, 1, "交了")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "failed" and res.invalid_ops == 2
    assert store.get_item(a["item_id"])["revision"] == 1
    assert store.get_item(a["item_id"])["status"] == "active"


@pytest.mark.asyncio
async def test_partial_invalid_ops_apply_valid_only(store, clock):
    ext, _ = make(
        store,
        clock,
        ops(
            {"op": "create", "kind": "plan", "topic": "交报告", "summary": "x", "evidence": "周五交报告",
             "when_text": "周五"},
            {"op": "create", "kind": "plan", "topic": "编造", "summary": "x", "evidence": "不存在的话"},
        ),
    )
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "done" and len(res.created) == 1 and res.invalid_ops == 1
    assert [i["topic"] for i in store.list_items()] == ["交报告"]


@pytest.mark.asyncio
async def test_empty_ops_marks_done(store, clock):
    ext, llm = make(store, clock, '```json\n{"ops": []}\n```')
    inbound(store, 1, "最近好累啊")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 1 and res.status == "done"
    assert store.list_items(statuses=None) == []
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "done"


def test_parse_llm_json_variants():
    assert parse_llm_json('{"ops": []}') == []
    assert parse_llm_json('结果如下 {"ops": [{"op": "create"}]} 完') == [{"op": "create"}]
    assert parse_llm_json("[1,2]") is None
    assert parse_llm_json('{"ops": {}}') is None
    assert parse_llm_json(None) is None


# ---------------------------------------------------------------------------
# 去重 / 预筛
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_inbound_no_duplicate_item(store, clock):
    reply = ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
                 "when_text": "周五", "evidence": "周五交报告"})
    ext, llm = make(store, clock, reply)
    assert inbound(store, 7, "周五交报告") is True
    assert inbound(store, 7, "周五交报告") is False  # 重放
    res = await ext.process_scope(SCOPE, OWNER)
    assert len(res.created) == 1 and llm.calls == 1
    # 重放在处理后再次到达 → 不再入队，不再调 LLM
    assert inbound(store, 7, "周五交报告") is False
    res2 = await ext.process_scope(SCOPE, OWNER)
    assert res2.status == "empty" and llm.calls == 1
    assert len(store.list_items(statuses=None)) == 1


@pytest.mark.asyncio
async def test_same_topic_twice_dedupes_to_one_item(store, clock):
    reply = ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
                 "when_text": "周五", "evidence": "周五交报告"})
    ext, llm = make(store, clock, reply, reply)
    inbound(store, 1, "周五交报告")
    r1 = await ext.process_scope(SCOPE, OWNER)
    inbound(store, 2, "周五交报告哦")
    r2 = await ext.process_scope(SCOPE, OWNER)
    assert r1.created == r2.created
    assert len(store.list_items(statuses=None)) == 1


@pytest.mark.asyncio
async def test_ack_only_skips_llm(store, clock):
    a = new_item(store, "交报告")
    ext, llm = make(store, clock)
    inbound(store, 1, "哈哈")
    inbound(store, 2, "嗯嗯")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "skipped" and llm.calls == 0
    assert store.get_item(a["item_id"])["revision"] == 1
    assert {store.get_inbound(SCOPE, i)["extract_status"] for i in (1, 2)} == {"skipped"}


@pytest.mark.asyncio
async def test_no_cue_skips_llm(store, clock):
    ext, llm = make(store, clock)
    inbound(store, 1, "今天的云好好看")  # 含“今天” → 有线索，会调 LLM
    inbound(store, 2, "猫咪在睡觉")
    store.mark_inbound(SCOPE, [1], "done")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "skipped" and llm.calls == 0


@pytest.mark.asyncio
async def test_named_active_item_passes_prefilter(store, clock):
    a = new_item(store, "雅思")
    ext, llm = make(store, clock, ops({"op": "update", "item_id": a["item_id"],
                                       "summary": "她觉得雅思口语很难", "evidence": "雅思口语好难"}))
    inbound(store, 1, "雅思口语好难")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 1 and res.updated == [a["item_id"]]
    assert store.get_item(a["item_id"])["summary"] == "她觉得雅思口语很难"


@pytest.mark.asyncio
async def test_ack_rows_skipped_in_mixed_batch(store, clock):
    ext, _ = make(store, clock, ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
                                     "evidence": "周五交报告", "when_text": "周五"}))
    inbound(store, 1, "哈哈")
    inbound(store, 2, "周五交报告")
    await ext.process_scope(SCOPE, OWNER)
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "skipped"
    assert store.get_inbound(SCOPE, 2)["extract_status"] == "done"


@pytest.mark.asyncio
async def test_stale_rows_not_processed(store, clock):
    ext, llm = make(store, clock)
    inbound(store, 1, "周五交报告", ts=BASE - 2 * DAY)
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "empty" and llm.calls == 0
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "stale"


# ---------------------------------------------------------------------------
# decline / reply_to / 最近主动焦点（§5.12）
# ---------------------------------------------------------------------------


def linked_outbound(store, tg_id, item_ids=()):
    ob = store.add_outbound(SCOPE, "proactive", "报告写得怎么样了", "sending", "bot", item_ids=item_ids)
    store.mark_outbound(ob, "sent", tg_msg_ids=[tg_id])
    return ob


@pytest.mark.asyncio
async def test_decline_only_affects_reply_linked_item(store, clock):
    a = new_item(store, "交报告")
    b = new_item(store, "搬家")
    linked_outbound(store, 500, [a["item_id"]])
    seen = {}

    def reply(prompt):
        seen["prompt"] = prompt
        return ops(
            {"op": "decline", "item_id": a["item_id"], "evidence": "不想聊这个"},
            {"op": "decline", "item_id": b["item_id"], "evidence": "不想聊这个"},
        )

    ext, _ = make(store, clock, reply)
    inbound(store, 1, "不想聊这个了", reply_to=500)
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [a["item_id"]] and res.invalid_ops == 1
    da = store.get_item(a["item_id"])
    assert da["status"] == "declined" and da["proactive_allowed"] == 0
    assert store.get_item(b["item_id"])["status"] == "active"
    assert store.get_item(b["item_id"])["revision"] == 1
    assert "reply_linked" in seen["prompt"]


@pytest.mark.asyncio
async def test_decline_without_link_at_most_one_and_needs_phrase(store, clock):
    a = new_item(store, "交报告")
    b = new_item(store, "搬家")
    ext, _ = make(
        store,
        clock,
        ops(
            {"op": "decline", "item_id": a["item_id"], "evidence": "别再问了"},
            {"op": "decline", "item_id": b["item_id"], "evidence": "别再问了"},
        ),
        ops({"op": "decline", "item_id": b["item_id"], "evidence": "交报告"}),
    )
    inbound(store, 1, "交报告的事别再问了")
    r1 = await ext.process_scope(SCOPE, OWNER)
    assert r1.closed == [a["item_id"]]
    assert store.get_item(b["item_id"])["status"] == "active"

    inbound(store, 2, "交报告好烦")  # 无拒绝措辞 → decline 被拦
    r2 = await ext.process_scope(SCOPE, OWNER)
    assert r2.status == "failed" and store.get_item(b["item_id"])["status"] == "active"


@pytest.mark.asyncio
async def test_t08_ack_reply_to_unrelated_outbound(store, clock):
    a = new_item(store, "交报告")
    linked_outbound(store, 600, [])  # 与事项无关的出站
    ext, llm = make(store, clock)
    inbound(store, 1, "哈哈", reply_to=600)
    inbound(store, 2, "哈哈哈哈", reply_to=999)  # 未知出站
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "skipped" and llm.calls == 0
    assert snapshot(store) == [(a["item_id"], 1, "active")]


@pytest.mark.asyncio
async def test_ack_reply_to_linked_outbound_changes_nothing(store, clock):
    a = new_item(store, "交报告")
    linked_outbound(store, 700, [a["item_id"]])
    ext, llm = make(store, clock)
    inbound(store, 1, "好", reply_to=700)
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "skipped" and llm.calls == 0
    assert store.get_item(a["item_id"])["revision"] == 1


@pytest.mark.asyncio
async def test_reply_link_makes_message_relevant(store, clock):
    a = new_item(store, "交报告")
    linked_outbound(store, 800, [a["item_id"]])
    ext, llm = make(store, clock, ops({"op": "complete", "item_id": a["item_id"], "evidence": "弄好啦"}))
    inbound(store, 1, "弄好啦", reply_to=800)  # 无规则线索，但 reply_to 关联事项
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 1 and res.closed == [a["item_id"]]


@pytest.mark.asyncio
async def test_recent_proactive_focus_candidate(store, clock):
    a = new_item(store, "交报告")
    turn = store.create_turn(SCOPE, input_msg_id=0, session_version=0, forced=False, item_id=a["item_id"])
    store.update_turn(turn["turn_id"], state="sent")
    seen = {}

    def reply(prompt):
        seen["prompt"] = prompt
        return ops({"op": "complete", "item_id": a["item_id"], "evidence": "交了"})

    ext, _ = make(store, clock, reply)
    clock.advance(10 * 60)
    inbound(store, 1, "交了", ts=clock())
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [a["item_id"]]
    assert "recent_proactive_focus" in seen["prompt"]


@pytest.mark.asyncio
async def test_old_proactive_focus_not_tagged(store, clock):
    a = new_item(store, "交报告")
    turn = store.create_turn(SCOPE, input_msg_id=0, session_version=0, forced=False, item_id=a["item_id"])
    store.update_turn(turn["turn_id"], state="sent")
    seen = {}

    def reply(prompt):
        seen["prompt"] = prompt
        return ops()

    ext, _ = make(store, clock, reply)
    clock.advance(31 * 60)
    inbound(store, 1, "交了", ts=clock())
    await ext.process_scope(SCOPE, OWNER)
    assert "recent_proactive_focus" not in seen["prompt"]


@pytest.mark.asyncio
async def test_revision_conflict_reapplies_on_fresh_revision(store, clock):
    """F8：LLM 期间事项被别处修改但仍 active → 在新 revision 上重放一次"""
    a = new_item(store, "交报告")

    def reply(prompt):
        # LLM 期间事项被别处修改（如主动记账）
        store.update_item(a["item_id"], expected_revision=1, reason="mention", ask_count=1)
        return ops({"op": "complete", "item_id": a["item_id"], "evidence": "交了"})

    ext, _ = make(store, clock, reply)
    inbound(store, 1, "交了")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "done" and res.closed == [a["item_id"]] and res.invalid_ops == 0
    item = store.get_item(a["item_id"])
    assert item["status"] == "done" and item["revision"] == 3 and item["ask_count"] == 1
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "done"


@pytest.mark.asyncio
async def test_revision_conflict_closed_item_leaves_inbound_pending(store, clock):
    """F8：LLM 期间事项已被关闭 → 不标 done，入站行回到 pending 重新抽取"""
    a = new_item(store, "交报告")

    def reply(prompt):
        store.update_item(a["item_id"], expected_revision=1, reason="other", status="cancelled")
        return ops({"op": "complete", "item_id": a["item_id"], "evidence": "交了"})

    ext, _ = make(store, clock, reply)
    inbound(store, 1, "交了")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [] and res.retry is True
    row = store.get_inbound(SCOPE, 1)
    assert row["extract_status"] == "pending" and row["last_error"] == "revision_conflict"
    assert store.get_item(a["item_id"])["status"] == "cancelled"


# ---------------------------------------------------------------------------
# LLM 超时 / 重试
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_timeout_retries_once_then_back_to_pending(store, clock):
    """F7：暂时性失败 → attempts<3 时回到 pending（带 last_error），第 3 次才 failed"""
    ext, llm = make(store, clock, *([asyncio.TimeoutError()] * 6))
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 2 and res.llm_attempts == 2
    assert res.status == "failed" and res.error == "llm_timeout" and res.retry is True
    row = store.get_inbound(SCOPE, 1)
    assert row["extract_status"] == "pending" and row["last_error"] == "llm_timeout"
    assert row["attempts"] == 1
    assert store.list_items(statuses=None) == []
    assert llm.timeouts == [5.0, 5.0]
    res = await ext.process_scope(SCOPE, OWNER)
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "pending" and res.retry
    res = await ext.process_scope(SCOPE, OWNER)
    row = store.get_inbound(SCOPE, 1)
    assert row["attempts"] == 3 and row["extract_status"] == "failed" and res.retry is False


@pytest.mark.asyncio
async def test_llm_network_error_back_to_pending(store, clock):
    ext, _ = make(store, clock, ConnectionError(), ConnectionError())
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.error == "llm_network" and res.retry
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "pending"


@pytest.mark.asyncio
async def test_validation_failure_stays_failed(store, clock):
    ext, _ = make(store, clock, ops({"op": "create", "kind": "plan", "topic": "x", "evidence": "不存在的片段"}))
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.status == "failed" and not res.retry
    assert store.get_inbound(SCOPE, 1)["extract_status"] == "failed"


@pytest.mark.asyncio
async def test_claim_error_is_transient(clock):
    class Broken:
        def claim_pending_inbound(self, *a, **kw):
            raise RuntimeError("locked")

    ext = ItemExtractor(Broken(), FakeLLM(), CFG, clock)
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.error == "claim_error" and res.retry is True


@pytest.mark.asyncio
async def test_worker_backs_off_on_transient_failure(store, clock):
    """F7：worker 对暂时性失败按退避重新登记，最多 MAX_ATTEMPTS-1 次"""
    from astrbot_plugin_iris_memory.iris_memory.continuity import extractor as ex_mod
    from astrbot_plugin_iris_memory.iris_memory.continuity.extractor import ExtractResult

    class Fake:
        batch_limit = 20
        _config = CFG
        store = None

        async def process_scope(self, scope, owner):
            return ExtractResult(scope_key=scope, status="failed", error="llm_timeout", retry=True)

    worker = ExtractionWorker(Fake(), config=CFG)
    loop = asyncio.get_running_loop()
    await worker._process(SCOPE)
    assert SCOPE in worker._due
    assert worker._due[SCOPE] - loop.time() == pytest.approx(ex_mod.RETRY_BACKOFF_S, abs=1.0)
    worker._due.clear(); worker._first.clear()
    await worker._process(SCOPE)
    assert worker._due[SCOPE] - loop.time() == pytest.approx(2 * ex_mod.RETRY_BACKOFF_S, abs=1.0)
    worker._due.clear(); worker._first.clear()
    await worker._process(SCOPE)  # 用尽
    assert SCOPE not in worker._due


@pytest.mark.asyncio
async def test_llm_timeout_then_success(store, clock):
    ext, llm = make(
        store,
        clock,
        asyncio.TimeoutError(),
        ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
             "when_text": "周五", "evidence": "周五交报告"}),
    )
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 2 and res.status == "done" and len(res.created) == 1


@pytest.mark.asyncio
async def test_llm_non_retryable_error_fails_once(store, clock):
    ext, llm = make(store, clock, RuntimeError("no provider"))
    inbound(store, 1, "周五交报告")
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 1 and res.status == "failed"
    assert store.get_inbound(SCOPE, 1)["last_error"] == "llm_error:RuntimeError"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def wait_until(pred, timeout=3.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return pred()


@pytest.mark.asyncio
async def test_worker_recovers_pending_and_marks_stale(store, clock):
    reply = ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
                 "when_text": "周五", "evidence": "周五交报告"})
    ext, llm = make(store, clock, reply)
    inbound(store, 1, "周五交报告")
    store.record_inbound(SCOPE, 0, OWNER, "明天考试", BASE - 2 * DAY)  # >24h
    worker = ExtractionWorker(ext, config={"extract_debounce_s": 0.01, "extract_queue_max": 5})
    await worker.start()
    try:
        assert store.get_inbound(SCOPE, 0)["extract_status"] == "stale"
        assert await wait_until(lambda: store.get_inbound(SCOPE, 1)["extract_status"] == "done")
        assert await wait_until(worker.is_idle)
    finally:
        await worker.stop()
    assert llm.calls == 1 and len(store.list_items()) == 1
    assert not worker.running


@pytest.mark.asyncio
async def test_worker_debounces_notifications_into_one_batch(store, clock):
    reply = ops({"op": "create", "kind": "plan", "topic": "交报告", "summary": "x",
                 "when_text": "周五", "evidence": "周五交报告"})
    ext, llm = make(store, clock, reply)
    worker = ExtractionWorker(ext, config={"extract_debounce_s": 0.1, "extract_queue_max": 5})
    await worker.start()
    try:
        for i, t in enumerate(["周五", "交报告", "周五交报告"], start=1):
            inbound(store, i, t)
            worker.notify(SCOPE)
            await asyncio.sleep(0.02)
        assert await wait_until(
            lambda: all(store.get_inbound(SCOPE, i)["extract_status"] == "done" for i in (1, 2, 3))
        )
    finally:
        await worker.stop()
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_worker_queue_max_drops_oldest_signal(store, clock):
    ext, llm = make(store, clock)
    worker = ExtractionWorker(ext, config={"extract_debounce_s": 10, "extract_queue_max": 2})
    worker.notify("private:1")
    worker.notify("private:2")
    worker.notify("private:3")
    assert worker.dropped == 1
    assert set(worker._due) == {"private:2", "private:3"}


@pytest.mark.asyncio
async def test_worker_survives_extractor_error(store, clock):
    class Boom(ItemExtractor):
        async def process_scope(self, scope_key, owner_id):
            raise RuntimeError("boom")

    worker = ExtractionWorker(Boom(store, FakeLLM(), CFG, clock), config={"extract_debounce_s": 0})
    await worker.start()
    try:
        worker.notify(SCOPE)
        assert await wait_until(worker.is_idle)
        assert worker.running
    finally:
        await worker.stop()


# ---------------------------------------------------------------------------
# Frisk 承诺
# ---------------------------------------------------------------------------


def test_record_bot_promise(store, clock):
    """无可挂靠事项：只作上下文（proactive_allowed=0），永远不会“到期”"""
    now = clock()
    item = record_bot_promise(store, SCOPE, OWNER, "好呀。到时候我问问你结果！加油", [77, 78], now)
    assert item is not None
    assert item["kind"] == "bot_promise" and item["subject_type"] == "self"
    assert item["source_kind"] == "self_expression"
    assert item["topic"] == "到时候我问问你结果"
    assert item["last_outbound_msg_ids"] == [77, 78]
    assert item["time_precision"] == "none" and item["event_at_start"] is None
    assert item["proactive_allowed"] == 0 and item["not_before"] is None
    assert item["expires_at"] == pytest.approx(now + 7 * DAY)
    assert item["allowed_audiences"] == [SCOPE]
    for dt in (13 * HOUR, 3 * DAY):
        assert not m.proactive_eligible(item, SCOPE, now + dt)
    # 同一承诺重复记录 → 去重
    again = record_bot_promise(store, SCOPE, OWNER, "好呀。到时候我问问你结果！加油", [79], now)
    assert again["item_id"] == item["item_id"]


def test_record_bot_promise_requires_promise_and_receipt(store, clock):
    assert record_bot_promise(store, SCOPE, OWNER, "以后再聊吧", [1], clock()) is None
    assert record_bot_promise(store, SCOPE, OWNER, "我会提醒你的", [], clock()) is None
    assert store.list_items(statuses=None) == []


def test_record_bot_promise_store_error_returns_none(clock):
    class Broken:
        def create_item(self, **kw):
            raise RuntimeError("db down")

    assert record_bot_promise(Broken(), SCOPE, OWNER, "我会提醒你的", [1], clock()) is None


# ---------------------------------------------------------------------------
# Frisk 承诺挂靠（state F2 / coverage F4）
# ---------------------------------------------------------------------------


def _friday_plan(store):
    from astrbot_plugin_iris_memory.iris_memory.continuity.extractor import compute_time_fields

    return new_item(store, "周五面试", **compute_time_fields("plan", "周五", BASE))


def test_bot_promise_links_to_plan_and_inherits_window(store, clock):
    plan = _friday_plan(store)
    clock.advance(HOUR)
    promise = record_bot_promise(store, SCOPE, OWNER, "好呀，周五面试加油，到时候我问你结果", [501], clock())
    assert promise["proactive_allowed"] == 0
    assert promise["not_before"] == plan["not_before"]
    assert promise["event_at_start"] == plan["event_at_start"]
    assert promise["expires_at"] == plan["expires_at"]
    refs = [r for r in promise["source_refs"] if r.get("linked_item_id")]
    assert refs == [{"linked_item_id": plan["item_id"], "link": "topic"}]
    carrier = store.get_item(plan["item_id"])
    marks = [r for r in carrier["source_refs"] if r.get("kind") == "bot_promise"]
    assert marks and marks[0]["tg_ids"] == [501] and len(marks[0]["text"]) <= 80
    # 周二 09:00：事件前，承诺绝不是候选
    tue = BASE + 23 * HOUR
    assert not m.proactive_eligible(store.get_item(promise["item_id"]), SCOPE, tue)


def test_closing_user_item_closes_linked_promise(store, clock):
    plan = _friday_plan(store)
    promise = record_bot_promise(store, SCOPE, OWNER, "到时候我问你结果", [501], clock())
    cur = store.get_item(plan["item_id"])
    store.update_item(plan["item_id"], expected_revision=cur["revision"], reason="extract_complete",
                      status="done", closed_reason="completed")
    p = store.get_item(promise["item_id"])
    assert p["status"] == "done" and p["closed_reason"] == "linked_completed"
    assert store.item_history(promise["item_id"])[-1]["reason"] == "linked_close"


@pytest.mark.asyncio
async def test_extractor_complete_closes_linked_promise(store, clock):
    plan = _friday_plan(store)
    promise = record_bot_promise(store, SCOPE, OWNER, "到时候我问你结果", [501], clock())
    ext, _ = make(store, clock, ops({"op": "complete", "item_id": plan["item_id"], "evidence": "面完了，搞定"}))
    clock.advance(4 * DAY)
    inbound(store, 1, "面完了，搞定", ts=clock())
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [plan["item_id"]]
    assert store.get_item(promise["item_id"])["status"] == "done"


def test_bot_promise_prefers_reply_linked_item(store, clock):
    a = new_item(store, "交报告")
    _friday_plan(store)  # 更近、话题重叠，但不是 reply 关联的
    linked_outbound(store, 600, [a["item_id"]])
    inbound(store, 9, "还没写完", ts=clock(), reply_to=600)
    promise = record_bot_promise(store, SCOPE, OWNER, "周五面试的事到时候我问你", [601], clock())
    link = [r for r in promise["source_refs"] if r.get("linked_item_id")][0]
    assert link == {"linked_item_id": a["item_id"], "link": "reply"}


def test_bot_promise_falls_back_to_most_recent_item(store, clock):
    a = new_item(store, "交报告")
    clock.advance(3 * DAY)
    promise = record_bot_promise(store, SCOPE, OWNER, "我会提醒你的", [700], clock())
    link = [r for r in promise["source_refs"] if r.get("linked_item_id")][0]
    assert link == {"linked_item_id": a["item_id"], "link": "recent"}
    assert promise["proactive_allowed"] == 0


@pytest.mark.asyncio
async def test_plan_created_after_promise_adopts_it(store, clock):
    inbound(store, 1, "周五面试", ts=clock())
    promise = record_bot_promise(store, SCOPE, OWNER, "好呀，到时候我问你结果", [801], clock() + 5)
    assert not [r for r in promise["source_refs"] if r.get("linked_item_id")]
    ext, _ = make(store, clock, ops({"op": "create", "kind": "plan", "topic": "周五面试",
                                     "when_text": "周五", "evidence": "周五面试"}))
    clock.advance(10)
    res = await ext.process_scope(SCOPE, OWNER)
    plan = store.get_item(res.created[0])
    p = store.get_item(promise["item_id"])
    assert [r for r in p["source_refs"] if r.get("linked_item_id")] == [
        {"linked_item_id": plan["item_id"], "link": "adopted"}
    ]
    assert p["not_before"] == plan["not_before"] and p["proactive_allowed"] == 0


# ---------------------------------------------------------------------------
# 关闭类 op 限制（state F4）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_linked_complete_only_closes_linked_item(store, clock):
    a = new_item(store, "交报告")
    b = new_item(store, "体检")
    linked_outbound(store, 500, [a["item_id"]])
    ext, _ = make(store, clock, ops(
        {"op": "complete", "item_id": a["item_id"], "evidence": "搞定了"},
        {"op": "complete", "item_id": b["item_id"], "evidence": "搞定了"},
    ))
    inbound(store, 1, "搞定了", reply_to=500)
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [a["item_id"]] and res.invalid_ops == 1
    assert store.get_item(b["item_id"])["status"] == "active"


@pytest.mark.asyncio
async def test_unlinked_one_closing_op_per_evidence_message(store, clock):
    a = new_item(store, "交报告")
    b = new_item(store, "体检")
    ext, _ = make(store, clock, ops(
        {"op": "complete", "item_id": a["item_id"], "evidence": "都搞定了"},
        {"op": "cancel", "item_id": b["item_id"], "evidence": "都搞定了"},
    ))
    inbound(store, 1, "交报告和体检都搞定了")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [a["item_id"]]
    assert store.get_item(b["item_id"])["status"] == "active"


@pytest.mark.asyncio
async def test_closing_op_requires_cue(store, clock):
    a = new_item(store, "交报告")
    ext, _ = make(store, clock, ops({"op": "complete", "item_id": a["item_id"], "evidence": "交报告"}))
    inbound(store, 1, "交报告好烦啊")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [] and store.get_item(a["item_id"])["status"] == "active"


@pytest.mark.asyncio
async def test_reschedule_requires_cue(store, clock):
    a = new_item(store, "交报告")
    ext, _ = make(store, clock, ops({"op": "reschedule", "item_id": a["item_id"],
                                     "when_text": "周六", "evidence": "周六"}))
    inbound(store, 1, "交报告周六好累")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.updated == [] and store.get_item(a["item_id"])["revision"] == 1


@pytest.mark.asyncio
async def test_pending_result_completes_on_result_cue(store, clock):
    a = new_item(store, "面试结果", kind=m.KIND_PENDING_RESULT)
    ext, _ = make(store, clock, ops({"op": "complete", "item_id": a["item_id"], "evidence": "面试过了"}))
    inbound(store, 1, "面试过了！")
    res = await ext.process_scope(SCOPE, OWNER)
    assert res.closed == [a["item_id"]]


@pytest.mark.asyncio
async def test_haha_never_closes(store, clock):
    a = new_item(store, "交报告")
    linked_outbound(store, 500, [a["item_id"]])
    ext, llm = make(store, clock)
    inbound(store, 1, "哈哈", reply_to=500)
    res = await ext.process_scope(SCOPE, OWNER)
    assert llm.calls == 0 and res.closed == []
    assert store.get_item(a["item_id"])["status"] == "active"


# ---------------------------------------------------------------------------
# 过去日期（state F5）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_result_past_weekday_resolves_to_past(store, clock):
    clock.advance(2 * DAY)  # 周三
    ext, _ = make(store, clock, ops({"op": "create", "kind": "pending_result", "topic": "面试结果",
                                     "when_text": "周一", "evidence": "周一面试"}))
    inbound(store, 1, "周一面试，在等结果", ts=clock())
    res = await ext.process_scope(SCOPE, OWNER)
    item = store.get_item(res.created[0])
    assert local(item["event_at_start"]).strftime("%Y-%m-%d") == "2026-11-02"
    assert item["not_before"] <= clock() < item["expires_at"]


@pytest.mark.asyncio
async def test_past_marker_absolute_date_resolves_to_past(store, clock):
    ext, _ = make(store, clock, ops({"op": "create", "kind": "plan", "topic": "考试",
                                     "when_text": "1号", "evidence": "1号考的"}))
    inbound(store, 1, "1号考的，感觉一般")
    res = await ext.process_scope(SCOPE, OWNER)
    item = store.get_item(res.created[0])
    assert local(item["event_at_start"]).strftime("%Y-%m-%d") == "2026-11-01"
    assert item["time_precision"] == "fuzzy"
