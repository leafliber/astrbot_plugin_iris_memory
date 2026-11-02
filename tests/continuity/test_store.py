"""ContinuityStore 单元测试（tmp_path DB + 可控时钟）"""

import sqlite3
from pathlib import Path

import pytest

from astrbot_plugin_iris_memory.iris_memory.continuity import models as m
from astrbot_plugin_iris_memory.iris_memory.continuity import store as store_mod
from astrbot_plugin_iris_memory.iris_memory.continuity.store import (
    ContinuityStore,
    ItemNotFound,
    RevisionConflict,
)

OWNER = "private:100000001"
DAY = 86400.0


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


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


def _item(**over):
    base = dict(
        subject_id="100000001",
        subject_type="user",
        scope_key=OWNER,
        topic="周五面试",
        summary="她周五下午有个面试",
        kind="plan",
        source_kind="shared_dialogue",
        source_refs=[{"adapter": "telegram", "scope_key": OWNER, "msg_id": 10, "at": 1.0}],
    )
    base.update(over)
    return base


# ---------------------------------------------------------------- schema / lifecycle


def test_open_sets_pragmas_and_schema(store, db_path):
    assert store.get_schema_version() == store_mod.SCHEMA_VERSION == 1
    con = sqlite3.connect(str(db_path))
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    for t in (
        "items", "item_history", "inbound", "watermarks", "outbound", "outbound_ids",
        "proactive_turns", "closures", "windows", "private_context", "observed_posts",
        "cursors", "counters",
    ):
        assert t in tables
    assert not list(db_path.parent.glob("*.backup_v*.db"))  # 新库不备份


def test_not_open_raises(db_path, clock):
    s = ContinuityStore(db_path, clock=clock)
    with pytest.raises(RuntimeError):
        s.get_item("x")


def test_migration_idempotent_when_opened_twice(db_path, clock):
    s1 = ContinuityStore(db_path, clock=clock)
    s1.open()
    s1.open()  # 再次 open 无副作用
    s1.set_cursor("c", "1")
    s1.close()
    s2 = ContinuityStore(db_path, clock=clock)
    s2.open()
    assert s2.get_schema_version() == 1
    assert s2.get_cursor("c") == "1"
    s2.close()
    assert not list(db_path.parent.glob("*.backup_v*.db"))


def test_backup_before_migration_for_existing_version(db_path, clock, monkeypatch):
    s = ContinuityStore(db_path, clock=clock)
    s.open()
    s.set_cursor("keep", "yes")
    s.close()
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(store_mod._MIGRATIONS, 1, "ALTER TABLE cursors ADD COLUMN extra TEXT")
    s2 = ContinuityStore(db_path, clock=clock)
    s2.open()
    assert s2.get_schema_version() == 2
    assert s2.get_cursor("keep") == "yes"
    s2.close()
    backup = db_path.with_name("continuity.backup_v1.db")
    assert backup.exists()
    con = sqlite3.connect(str(backup))
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1
        assert con.execute("SELECT value FROM cursors WHERE name='keep'").fetchone()[0] == "yes"
    finally:
        con.close()


def test_failed_migration_rolls_back(db_path, clock, monkeypatch):
    s = ContinuityStore(db_path, clock=clock)
    s.open()
    s.close()
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(
        store_mod._MIGRATIONS, 1, "ALTER TABLE cursors ADD COLUMN ok TEXT; THIS IS NOT SQL"
    )
    s2 = ContinuityStore(db_path, clock=clock)
    with pytest.raises(sqlite3.Error):
        s2.open()
    assert not s2.is_open
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 1)
    s3 = ContinuityStore(db_path, clock=clock)
    s3.open()
    assert s3.get_schema_version() == 1
    cols = {r[1] for r in s3._conn().execute("PRAGMA table_info(cursors)")}
    assert "ok" not in cols
    s3.close()


# ---------------------------------------------------------------- items


def test_create_item_defaults_and_json_decoded(store, clock):
    it = store.create_item(**_item())
    assert len(it["item_id"]) == 32
    assert it["revision"] == 1 and it["status"] == "active"
    assert it["allowed_audiences"] == [OWNER]
    assert isinstance(it["source_refs"], list) and it["source_refs"][0]["msg_id"] == 10
    assert it["last_outbound_msg_ids"] == []
    assert it["dedupe_key"] == f"{OWNER}|plan|{m.norm_topic('周五面试')}"
    assert it["created_at"] == clock.t
    hist = store.item_history(it["item_id"])
    assert [h["revision"] for h in hist] == [1]
    assert hist[0]["snapshot"]["topic"] == "周五面试"


def test_create_item_dedupe_returns_existing(store, clock):
    a = store.create_item(**_item())
    clock.advance(10)
    b = store.create_item(**_item(topic=" 周五 面试！", summary="另一种说法"))
    assert b["item_id"] == a["item_id"]
    assert b["summary"] == a["summary"]
    assert b["revision"] == 1
    n = store._conn().execute("SELECT COUNT(*) FROM items").fetchone()[0]
    assert n == 1
    assert len(store.item_history(a["item_id"])) == 1
    assert store.find_active_by_dedupe(a["dedupe_key"])["item_id"] == a["item_id"]


def test_create_item_dedupe_only_for_active(store):
    a = store.create_item(**_item())
    store.update_item(a["item_id"], expected_revision=1, reason="done", status="done")
    b = store.create_item(**_item())
    assert b["item_id"] != a["item_id"]
    # 不同 kind 不去重
    c = store.create_item(**_item(kind="pending_result"))
    assert c["item_id"] not in (a["item_id"], b["item_id"])


def test_create_item_validation_and_clipping(store):
    with pytest.raises(ValueError):
        store.create_item(**_item(kind="bogus"))
    with pytest.raises(ValueError):
        store.create_item(**_item(source_kind="rumor"))
    with pytest.raises(ValueError):
        store.create_item(**_item(topic=""))
    with pytest.raises(ValueError):
        store.create_item(**_item(unknown_field=1))
    it = store.create_item(**_item(topic="长" * 100, summary="短" * 300, importance=9))
    assert len(it["topic"]) == m.TOPIC_MAX_CHARS
    assert len(it["summary"]) == m.SUMMARY_MAX_CHARS
    assert it["importance"] == 3


def test_update_item_revision_history_and_conflict(store, clock):
    it = store.create_item(**_item(not_before=100.0))
    clock.advance(5)
    u = store.update_item(
        it["item_id"], expected_revision=1, reason="reschedule", source_msg_id="42",
        not_before=200.0, time_basis="下周一",
    )
    assert u["revision"] == 2 and u["not_before"] == 200.0 and u["updated_at"] == clock.t
    hist = store.item_history(it["item_id"])
    assert [h["revision"] for h in hist] == [1, 2]
    assert hist[0]["snapshot"]["not_before"] == 100.0  # 旧时间保留在历史
    assert hist[1]["reason"] == "reschedule" and hist[1]["source_msg_id"] == "42"
    with pytest.raises(RevisionConflict) as ei:
        store.update_item(it["item_id"], expected_revision=1, reason="stale", status="done")
    assert ei.value.actual == 2
    assert store.get_item(it["item_id"])["status"] == "active"
    assert len(store.item_history(it["item_id"])) == 2
    # expected_revision=None 不检查
    u3 = store.update_item(it["item_id"], expected_revision=None, reason="x", importance=2)
    assert u3["revision"] == 3


def test_update_item_errors(store):
    it = store.create_item(**_item())
    with pytest.raises(ItemNotFound):
        store.update_item("nope", expected_revision=None, reason="x", status="done")
    with pytest.raises(ValueError):
        store.update_item(it["item_id"], expected_revision=1, reason="x", revision=9)
    with pytest.raises(ValueError):
        store.update_item(it["item_id"], expected_revision=1, reason="x", status="weird")
    assert store.get_item(it["item_id"])["revision"] == 1


def test_update_item_topic_recomputes_dedupe(store):
    it = store.create_item(**_item())
    u = store.update_item(it["item_id"], expected_revision=1, reason="x", topic="周六面试")
    assert u["dedupe_key"] == m.make_dedupe_key(OWNER, "plan", "周六面试")


def test_list_items_audience_filter(store, clock):
    a = store.create_item(**_item(topic="私聊事项"))
    g = store.create_item(**_item(topic="群事项", scope_key="-100123", allowed_audiences=["-100123"]))
    both = store.create_item(**_item(topic="共享", allowed_audiences=[OWNER, "-100123"], importance=3))
    extra = store.create_item(**_item(topic="另一个"))
    ids_owner = {i["item_id"] for i in store.list_items(audience=OWNER)}
    assert ids_owner == {a["item_id"], both["item_id"], extra["item_id"]}
    ids_group = {i["item_id"] for i in store.list_items(audience="-100123")}
    assert ids_group == {g["item_id"], both["item_id"]}
    # 子串不匹配（精确 audience）
    assert store.list_items(audience="private:94337951") == []
    # 排序：importance 高者先
    assert store.list_items(audience=OWNER)[0]["item_id"] == both["item_id"]
    # scope / statuses
    assert {i["item_id"] for i in store.list_items(scope_key="-100123")} == {g["item_id"]}
    store.update_item(a["item_id"], expected_revision=1, reason="d", status="done")
    assert a["item_id"] not in {i["item_id"] for i in store.list_items(audience=OWNER)}
    assert [i["item_id"] for i in store.list_items(statuses=("done",))] == [a["item_id"]]
    assert len(store.list_items(limit=1)) == 1


def test_expire_items_rules(store, clock):
    now = clock.t
    past = now - 10
    topic_tmp = store.create_item(**_item(topic="t1", kind="unfinished_topic", expires_at=past))
    note_tmp = store.create_item(**_item(topic="t2", kind="interest_question", expires_at=past))
    future_tmp = store.create_item(**_item(topic="t3", kind="unfinished_topic", expires_at=now + 100))
    tmp_remind = store.create_item(
        **_item(topic="t4", kind="note", expires_at=past, remind_requested=True)
    )
    plan = store.create_item(**_item(topic="p1", kind="plan", expires_at=past))
    pend_remind_recent = store.create_item(
        **_item(topic="p2", kind="pending_result", expires_at=past, remind_requested=True)
    )
    promise_remind_old = store.create_item(
        **_item(topic="p3", kind="bot_promise", subject_type="self",
                source_kind="self_expression", expires_at=now - 8 * DAY, remind_requested=True)
    )
    no_exp = store.create_item(**_item(topic="n", kind="note"))

    changed = store.expire_items(now)
    get = lambda it: store.get_item(it["item_id"])  # noqa: E731

    assert get(topic_tmp)["status"] == "expired"
    assert get(topic_tmp)["revision"] == 2
    assert store.item_history(topic_tmp["item_id"])[-1]["reason"] == "expire"
    assert get(note_tmp)["status"] == "expired"
    assert get(future_tmp)["status"] == "active"
    assert get(no_exp)["status"] == "active"
    # remind_requested 临时项：不过期，宽限内仍可主动
    assert get(tmp_remind)["status"] == "active" and get(tmp_remind)["proactive_allowed"] == 1
    # 受保护 kind：不改状态，只退出主动候选
    assert get(plan)["status"] == "active" and get(plan)["proactive_allowed"] == 0
    assert get(pend_remind_recent)["status"] == "active"
    assert get(pend_remind_recent)["proactive_allowed"] == 1  # 未超过 +7d
    assert get(promise_remind_old)["status"] == "active"
    assert get(promise_remind_old)["proactive_allowed"] == 0
    assert changed == 4

    # 幂等：再次执行无变化，且从不删除行
    assert store.expire_items(now) == 0
    assert store._conn().execute("SELECT COUNT(*) FROM items").fetchone()[0] == 8


# ---------------------------------------------------------------- inbound / watermark


def test_record_inbound_idempotent(store, clock):
    assert store.record_inbound(OWNER, 100, "100000001", "周五面试", clock.t, None) is True
    store.mark_inbound(OWNER, [100], "done")
    clock.advance(5)
    assert store.record_inbound(OWNER, 100, "100000001", "改过的文本", clock.t, 7) is False
    row = store.get_inbound(OWNER, 100)
    assert row["text"] == "周五面试" and row["extract_status"] == "done"
    assert row["reply_to_msg_id"] is None
    # 不同 scope 同 msg_id 是新消息
    assert store.record_inbound("-100123", 100, "1", "x", clock.t) is True


def test_claim_and_mark_inbound(store, clock):
    store.record_inbound(OWNER, 1, "u", "old", clock.t - 2 * DAY)
    store.record_inbound(OWNER, 3, "u", "b", clock.t)
    store.record_inbound(OWNER, 2, "u", "a", clock.t)
    store.record_inbound("other", 5, "u", "c", clock.t)
    assert store.pending_inbound_scopes() == sorted([OWNER, "other"])
    rows = store.claim_pending_inbound(OWNER)
    assert [r["msg_id"] for r in rows] == [2, 3]
    assert all(r["attempts"] == 1 for r in rows)
    assert store.get_inbound(OWNER, 1)["extract_status"] == "stale"
    assert store.mark_inbound(OWNER, [2, 3], "failed", error="bad json") == 2
    assert store.get_inbound(OWNER, 2)["last_error"] == "bad json"
    assert store.claim_pending_inbound(OWNER) == []
    with pytest.raises(ValueError):
        store.mark_inbound(OWNER, [2], "weird")
    assert [r["msg_id"] for r in store.claim_pending_inbound(limit=5)] == [5]


def test_advance_watermark(store):
    assert store.get_watermark(OWNER) is None
    assert store.advance_watermark(OWNER, 10, 1.0) == (True, 1)
    assert store.advance_watermark(OWNER, 10, 1.0) == (False, 1)  # 重放
    assert store.advance_watermark(OWNER, 9, 0.5) == (False, 1)  # 旧消息
    assert store.advance_watermark(OWNER, 11, 2.0) == (True, 2)
    wm = store.get_watermark(OWNER)
    assert wm["last_msg_id"] == 11 and wm["last_msg_ts"] == 2.0 and wm["session_version"] == 2
    assert store.advance_watermark("-100", 1, 1.0) == (True, 1)  # 会话独立


# ---------------------------------------------------------------- outbound


def test_outbound_receipt_replay_dedupe(store, clock):
    oid = store.add_outbound(OWNER, "proactive", "在吗", "sending", "bot", turn_id="t1",
                             item_ids=["i1"])
    o = store.get_outbound(oid)
    assert o["status"] == "sending" and o["sent_at"] is None and o["item_ids"] == ["i1"]
    clock.advance(1)
    o = store.mark_outbound(oid, "sent", tg_msg_ids=[555])
    assert o["status"] == "sent" and o["tg_msg_ids"] == [555] and o["sent_at"] == clock.t
    sent_at = o["sent_at"]
    clock.advance(1)
    o = store.mark_outbound(oid, "sent", tg_msg_ids=[555, 555])  # 回执重放
    assert o["tg_msg_ids"] == [555] and o["sent_at"] == sent_at
    n = store._conn().execute("SELECT COUNT(*) FROM outbound_ids").fetchone()[0]
    assert n == 1
    assert store.outbound_by_tg_id(OWNER, 555)["id"] == oid
    assert store.outbound_by_tg_id("-100", 555) is None
    # 已 sent 不被降级
    o = store.mark_outbound(oid, "failed", error="late error")
    assert o["status"] == "sent"
    # 同一 tg id 不能被另一行占用
    oid2 = store.add_outbound(OWNER, "reply", "x", "sending", "bot")
    o2 = store.mark_outbound(oid2, "sent", tg_msg_ids=[555, 556])
    assert o2["tg_msg_ids"] == [556]
    assert store.outbound_by_tg_id(OWNER, 555)["id"] == oid
    with pytest.raises(KeyError):
        store.mark_outbound(9999, "sent")
    with pytest.raises(ValueError):
        store.add_outbound(OWNER, "proactive", "x", "bogus", "bot")


def test_outbound_queries(store, clock):
    a = store.add_outbound(OWNER, "reply", "a", "sent", "bot")
    clock.advance(1)
    store.add_outbound(OWNER, "manual", "m", "sent", "manual")
    clock.advance(1)
    c = store.add_outbound(OWNER, "proactive", "c", "sending", "bot", turn_id="t", part_index=1)
    clock.advance(1)
    d = store.add_outbound(OWNER, "reply", "d", "sent", "bot")
    store.add_outbound(OWNER, "proactive", "draft", "draft", "bot", turn_id="t", part_index=0)
    assert store.get_outbound(a)["sent_at"] is not None
    assert [o["id"] for o in store.recent_outbound(OWNER)] == [d, a]
    assert [o["id"] for o in store.recent_outbound(OWNER, limit=1)] == [d]
    assert len(store.recent_outbound(OWNER, authors=("bot", "manual"))) == 3
    assert [o["id"] for o in store.outbound_in_state(["sending"])] == [c]
    assert [o["part_index"] for o in store.outbound_for_turn("t")] == [0, 1]
    store.mark_outbound(c, "uncertain", error="timeout")
    assert store.get_outbound(c)["error"] == "timeout"
    store.mark_outbound(c, "sent", tg_msg_ids=[9])  # 对账补写
    assert store.get_outbound(c)["error"] == "timeout"
    assert store.get_outbound(c)["status"] == "sent"


# ---------------------------------------------------------------- turns


def test_turns_crud(store, clock):
    t = store.create_turn(OWNER, input_msg_id=11, session_version=2, forced=True)
    assert t["state"] == "deciding" and t["forced"] == 1 and t["quota_state"] == "none"
    assert t["input_msg_id"] == 11 and t["session_version"] == 2
    clock.advance(3)
    u = store.update_turn(t["turn_id"], state="uncertain", quota_state="reserved",
                          item_id="i1", llm_calls=2, parts_total=2, parts_sent=1, reason="timeout")
    assert u["state"] == "uncertain" and u["item_id"] == "i1" and u["updated_at"] == clock.t
    t2 = store.create_turn("-100", input_msg_id=0, session_version=0, forced=False)
    assert [x["turn_id"] for x in store.turns_in_state(["uncertain", "sending"])] == [t["turn_id"]]
    assert [x["turn_id"] for x in store.turns_in_state(["deciding"], scope_key="-100")] == [t2["turn_id"]]
    with pytest.raises(ValueError):
        store.update_turn(t["turn_id"], state="bogus")
    with pytest.raises(ValueError):
        store.update_turn(t["turn_id"], scope_key="x")
    with pytest.raises(KeyError):
        store.update_turn("missing", state="sent")


# ---------------------------------------------------------------- misc tables


def test_closures(store, clock):
    assert store.get_closure(OWNER) is None
    c = store.put_closure(OWNER, state="waiting", explicit_wait=True, wait_since=clock.t)
    assert c["state"] == "waiting" and c["explicit_wait"] == 1 and c["open_questions"] == []
    c = store.put_closure(OWNER, state="ended", ended_at=clock.t, summary="晚安",
                          open_questions=["周五面试"], next_opening="面试")
    assert c["explicit_wait"] == 1  # 未给出字段保留
    assert c["open_questions"] == ["周五面试"] and c["next_opening"] == "面试"
    with pytest.raises(ValueError):
        store.put_closure(OWNER, state="sleeping")
    assert store.get_closure(OWNER)["state"] == "ended"


def test_windows(store):
    store.save_window(OWNER, [{"role": "user", "ts": 1.0}])
    store.save_window(OWNER, [{"role": "user", "ts": 2.0}])
    store.save_window("-100", [])
    assert store.load_windows() == {OWNER: [{"role": "user", "ts": 2.0}], "-100": []}
    store.delete_window("-100")
    assert "-100" not in store.load_windows()
    with pytest.raises(ValueError):
        store.save_window(OWNER, {"not": "list"})


def test_private_context(store):
    store.put_private_context("recent_dynamics", "v1", "digest", OWNER)
    store.put_private_context("recent_dynamics", "v2", "digest", OWNER)
    store.put_private_context("identity", "id", "persona", OWNER)
    rows = store.get_private_context(OWNER)
    assert [r["section"] for r in rows] == ["identity", "recent_dynamics"]
    assert rows[1]["content"] == "v2"
    assert store.get_private_context("-100") == []


def test_observed_posts(store, clock):
    post = dict(chat_id=-1001000000001, msg_id=1, posted_at=100.0, sender_id="1",
                author_kind="owner", text="hello", via="realtime")
    assert store.add_observed_post(**post) is True
    assert store.add_observed_post(**{**post, "text": "dup", "via": "backfill"}) is False
    assert store.add_observed_post(**{**post, "msg_id": 2, "posted_at": 200.0,
                                      "is_forward": True, "fwd_from": "x"}) is True
    store.add_observed_post(chat_id=5, msg_id=1, posted_at=300.0, author_kind="other",
                            text="g", via="realtime")
    rows = store.posts_since([-1001000000001], 50.0, 10)
    assert [r["msg_id"] for r in rows] == [1, 2] and rows[0]["text"] == "hello"
    assert [r["msg_id"] for r in store.posts_since([-1001000000001], 100.0, 10)] == [2]
    assert len(store.posts_since([-1001000000001, 5], 0, 10)) == 3
    assert store.posts_since([], 0, 10) == []
    with pytest.raises(ValueError):
        store.add_observed_post(**{**post, "msg_id": 3, "author_kind": "alien"})


def test_cursors_and_counters(store, clock):
    assert store.get_cursor("digest") is None
    store.set_cursor("digest", 123.5)
    store.set_cursor("digest", "200")
    assert store.get_cursor("digest") == "200"
    assert store.bump_counter("extract_llm") == 1
    assert store.bump_counter("extract_llm", 2) == 3
    assert store.bump_counter("x", day="2026-01-01") == 1
    today = m.local_day(clock.t)
    assert store.get_counters(today) == {"extract_llm": 3}
    assert store.get_counters() == {"extract_llm": 3}
    assert store.get_counters("2026-01-01") == {"x": 1}


def test_local_day_uses_shanghai():
    # 2026-01-01 16:30 UTC = 2026-01-02 00:30 Asia/Shanghai
    assert m.local_day(1767285000.0) == "2026-01-02"


# ---------------------------------------------------------------- restart


def test_reopen_preserves_everything(db_path, clock):
    s = ContinuityStore(db_path, clock=clock)
    s.open()
    it = s.create_item(**_item())
    s.update_item(it["item_id"], expected_revision=1, reason="decline", status="declined",
                  proactive_allowed=False)
    s.record_inbound(OWNER, 7, "u", "hi", clock.t)
    s.advance_watermark(OWNER, 7, clock.t)
    oid = s.add_outbound(OWNER, "proactive", "x", "sending", "bot")
    s.mark_outbound(oid, "sent", tg_msg_ids=[77])
    t = s.create_turn(OWNER, input_msg_id=7, session_version=1, forced=False)
    s.update_turn(t["turn_id"], state="uncertain", quota_state="reserved")
    s.put_closure(OWNER, state="ended", ended_at=clock.t)
    s.save_window(OWNER, [{"a": 1}])
    s.put_private_context("recent_dynamics", "c", "digest", OWNER)
    s.add_observed_post(chat_id=1, msg_id=1, posted_at=1.0, author_kind="owner", text="t", via="v")
    s.set_cursor("cur", "9")
    s.bump_counter("n", 4)
    s.close()

    s2 = ContinuityStore(db_path, clock=clock)
    s2.open()
    got = s2.get_item(it["item_id"])
    assert got["status"] == "declined" and got["proactive_allowed"] == 0 and got["revision"] == 2
    assert len(s2.item_history(it["item_id"])) == 2
    assert s2.record_inbound(OWNER, 7, "u", "hi", clock.t) is False
    assert s2.advance_watermark(OWNER, 7, clock.t) == (False, 1)
    assert s2.outbound_by_tg_id(OWNER, 77)["id"] == oid
    assert [x["turn_id"] for x in s2.turns_in_state(["uncertain"])] == [t["turn_id"]]
    assert s2.get_closure(OWNER)["state"] == "ended"
    assert s2.load_windows() == {OWNER: [{"a": 1}]}
    assert s2.get_private_context(OWNER)[0]["content"] == "c"
    assert s2.add_observed_post(chat_id=1, msg_id=1, posted_at=1.0, author_kind="owner",
                                text="t", via="v") is False
    assert s2.get_cursor("cur") == "9"
    assert s2.get_counters() == {"n": 4}
    # 重启后被拒绝的事项不被同 key 新建“复活”的 active 去重命中
    assert s2.create_item(**_item())["item_id"] != it["item_id"]
    s2.close()


# ---------------------------------------------------------------- linked bot promises (state F2)


def _linked_promise(store, user_item, **over):
    fields = _item(
        subject_type="self",
        topic="到时候我问你结果",
        summary="到时候我问你结果",
        kind="bot_promise",
        source_kind="self_expression",
        source_refs=[{"adapter": "telegram", "msg_id": 501}, {"linked_item_id": user_item["item_id"], "link": "topic"}],
        proactive_allowed=0,
        not_before=user_item.get("not_before"),
    )
    fields.update(over)
    return store.create_item(**fields)


@pytest.mark.parametrize("status", ["done", "cancelled", "declined"])
def test_closing_user_item_closes_linked_promise_same_tx(store, status):
    plan = store.create_item(**_item(not_before=1_800_100_000.0))
    other = store.create_item(**_item(topic="搬家"))
    p = _linked_promise(store, plan)
    unrelated = _linked_promise(store, other, topic="回头跟你说")
    assert [x["item_id"] for x in store.linked_promises(plan["item_id"])] == [p["item_id"]]
    store.update_item(plan["item_id"], expected_revision=1, reason="x", status=status, closed_reason="c")
    got = store.get_item(p["item_id"])
    assert got["status"] == status and got["closed_reason"] == "linked_c" and got["proactive_allowed"] == 0
    assert store.get_item(unrelated["item_id"])["status"] == "active"


def test_linked_promise_close_rolls_back_with_user_item(store, monkeypatch):
    plan = store.create_item(**_item())
    p = _linked_promise(store, plan)
    real = store.update_item

    def flaky(item_id, **kw):
        if item_id == p["item_id"]:
            raise RuntimeError("boom")
        return real(item_id, **kw)

    monkeypatch.setattr(store, "update_item", flaky)
    with pytest.raises(RuntimeError):
        store.update_item(plan["item_id"], expected_revision=1, reason="x", status="done")
    monkeypatch.undo()
    assert store.get_item(plan["item_id"])["status"] == "active"
    assert store.get_item(p["item_id"])["status"] == "active"


def test_rescheduling_user_item_moves_linked_promise_window(store):
    plan = store.create_item(**_item(not_before=1_800_100_000.0, expires_at=1_800_400_000.0))
    p = _linked_promise(store, plan)
    store.update_item(plan["item_id"], expected_revision=1, reason="extract_reschedule",
                      not_before=1_800_200_000.0, expires_at=1_800_500_000.0, time_basis="周六")
    got = store.get_item(p["item_id"])
    assert got["not_before"] == 1_800_200_000.0 and got["expires_at"] == 1_800_500_000.0
    assert got["time_basis"] == "周六" and got["status"] == "active"


def test_latest_inbound(store):
    store.record_inbound(OWNER, 1, "100000001", "a", 100.0)
    store.record_inbound(OWNER, 2, "100000001", "b", 200.0)
    store.record_inbound(OWNER, 3, "other", "c", 300.0)
    assert store.latest_inbound(OWNER)["msg_id"] == 3
    assert store.latest_inbound(OWNER, "100000001")["msg_id"] == 2
    assert store.latest_inbound(OWNER, "100000001", since_ts=250.0) is None
