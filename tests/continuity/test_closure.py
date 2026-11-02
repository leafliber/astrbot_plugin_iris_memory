"""ClosureTracker 测试（tmp_path DB + 可控时钟）。"""

from pathlib import Path

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity.closure import (
    ENDING_HINT,
    ClosureTracker,
)
from astrbot_plugin_iris_memory.iris_memory.continuity.store import ContinuityStore

OWNER = "private:100000001"


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
def store(tmp_path: Path, clock):
    s = ContinuityStore(tmp_path / "continuity" / "continuity.db", clock=clock)
    s.open()
    yield s
    s.close()


@pytest.fixture
def tracker(store, clock):
    return ClosureTracker(store, clock)


def _item(store, topic, importance=2, **over):
    base = dict(
        subject_id="100000001",
        subject_type="user",
        scope_key=OWNER,
        topic=topic,
        summary=f"关于{topic}",
        kind="plan",
        source_kind="shared_dialogue",
        importance=importance,
    )
    base.update(over)
    return store.create_item(**base)


def _snapshot(store):
    return [(i["item_id"], i["status"], i["revision"]) for i in store.list_items(statuses=None, limit=100)]


def test_default_state_is_continue(tracker):
    cur = tracker.current(OWNER)
    assert cur["state"] == "continue" and cur["open_questions"] == []
    assert tracker.hint_for_reply(OWNER) is None


def test_t06_goodnight_ends_then_question_reopens(tracker, store, clock):
    _item(store, "周五面试", importance=3)
    _item(store, "搬家", importance=1)
    before = _snapshot(store)

    assert tracker.on_owner_message(OWNER, "今天面试完了，有点累但还挺开心的") == "continue"
    tracker.on_bot_sent(OWNER, "辛苦啦，开心就好")
    assert tracker.on_owner_message(OWNER, "晚安") == "ended"

    cur = tracker.current(OWNER)
    assert cur["state"] == "ended" and cur["ended_at"] == clock.t
    assert cur["summary"] == "今天面试完了，有点累但还挺开心的"
    assert cur["open_questions"] == ["周五面试", "搬家"]
    assert cur["next_opening"] == "周五面试"
    assert tracker.hint_for_reply(OWNER) == ENDING_HINT

    clock.advance(600)
    assert tracker.on_owner_message(OWNER, "对了你明天有空吗？") == "continue"
    assert tracker.hint_for_reply(OWNER) is None
    # 收尾/重开不创建、不完成、不修改任何事项
    assert _snapshot(store) == before


def test_summary_clipped_to_80(tracker):
    tracker.on_owner_message(OWNER, "长" * 200)
    tracker.on_owner_message(OWNER, "晚安")
    assert tracker.current(OWNER)["summary"] == "长" * 80


def test_ending_with_unanswered_question_does_not_end(tracker):
    tracker.on_owner_message(OWNER, "你觉得我穿哪件好？")
    assert tracker.on_owner_message(OWNER, "晚安") == "continue"
    assert tracker.hint_for_reply(OWNER) is None
    # 回答之后再道别 → ended
    tracker.on_bot_sent(OWNER, "蓝色那件")
    tracker.on_owner_message(OWNER, "好")
    tracker.on_bot_sent(OWNER, "好梦")
    assert tracker.current(OWNER)["state"] == "ended"


def test_bot_ending_after_owner_ack_ends(tracker):
    tracker.on_owner_message(OWNER, "我今天去爬山了")
    tracker.on_bot_sent(OWNER, "哇，累不累")
    tracker.on_owner_message(OWNER, "嗯嗯")
    assert tracker.on_bot_sent(OWNER, "晚安") == "ended"
    cur = tracker.current(OWNER)
    assert cur["summary"] == "我今天去爬山了"
    # 主人最新一条不是收尾语 → 不给回复提示
    assert tracker.hint_for_reply(OWNER) is None


def test_bot_ending_after_substantive_does_not_end(tracker):
    tracker.on_owner_message(OWNER, "我今天去爬山了")
    assert tracker.on_bot_sent(OWNER, "晚安") == "continue"


def test_explicit_wait_states(tracker, clock):
    assert tracker.on_owner_message(OWNER, "等我说完") == "waiting"
    assert tracker.current(OWNER)["explicit_wait"] == 1
    assert tracker.on_owner_message(OWNER, "今天发生了一件事") == "waiting"
    assert tracker.on_owner_message(OWNER, "说完了") == "continue"
    assert tracker.current(OWNER)["explicit_wait"] == 0


def test_explicit_wait_times_out(store, clock):
    t = ClosureTracker(store, clock, explicit_wait_timeout_s=180)
    t.on_owner_message(OWNER, "等我说完")
    clock.advance(200)
    assert t.on_owner_message(OWNER, "算了不说了") == "continue"


def test_media_reopens(tracker):
    tracker.on_owner_message(OWNER, "晚安")
    assert tracker.current(OWNER)["state"] == "ended"
    assert tracker.on_owner_message(OWNER, "", has_media=True) == "continue"


def test_empty_message_is_noop(tracker):
    tracker.on_owner_message(OWNER, "晚安")
    assert tracker.on_owner_message(OWNER, "") == "ended"


def test_suppress_proactive(tracker, clock):
    assert tracker.suppress_proactive(OWNER, clock.t, 120, False) is False
    tracker.on_owner_message(OWNER, "晚安")
    ended = clock.t
    assert tracker.suppress_proactive(OWNER, ended + 60, 120, False) is True
    assert tracker.suppress_proactive(OWNER, ended + 60, 120, True) is False
    assert tracker.suppress_proactive(OWNER, ended + 120 * 60 + 1, 120, False) is False
    tracker.on_owner_message(OWNER, "我醒了")
    assert tracker.suppress_proactive(OWNER, ended + 60, 120, False) is False


def test_state_persists_across_trackers(store, clock):
    ClosureTracker(store, clock).on_owner_message(OWNER, "晚安")
    assert ClosureTracker(store, clock).current(OWNER)["state"] == "ended"


def test_store_failure_degrades(clock):
    class Broken:
        def get_closure(self, *_):
            raise RuntimeError("db down")

        def put_closure(self, *_, **__):
            raise RuntimeError("db down")

        def list_items(self, *_, **__):
            raise RuntimeError("db down")

    t = ClosureTracker(Broken(), clock)
    assert t.on_owner_message(OWNER, "晚安") == "continue"
    assert t.on_bot_sent(OWNER, "晚安") == "continue"
    assert t.current(OWNER)["state"] == "continue"
    assert t.hint_for_reply(OWNER) is None
    assert t.suppress_proactive(OWNER, clock.t, 120, False) is False


def test_next_opening_only_from_proactive_allowed_items(tracker, store):
    """coverage F1：“不用提醒”的事项不能成为下次开场"""
    _item(store, "周五交报告", importance=3, proactive_allowed=0, remind_requested=0)
    _item(store, "搬家", importance=1)
    tracker.on_owner_message(OWNER, "今天好累")
    tracker.on_bot_sent(OWNER, "早点休息")
    assert tracker.on_owner_message(OWNER, "晚安") == "ended"
    cur = tracker.current(OWNER)
    assert cur["next_opening"] == "搬家"
    assert "周五交报告" in cur["open_questions"]


def test_next_opening_empty_when_nothing_openable(tracker, store):
    _item(store, "周五交报告", proactive_allowed=0)
    tracker.on_owner_message(OWNER, "晚安")
    assert tracker.current(OWNER)["next_opening"] == ""
