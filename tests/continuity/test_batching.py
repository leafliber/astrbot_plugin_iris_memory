"""MessageBatcher 测试：短合批、0 等待放行、显式等待与超时、媒体直通。"""

import asyncio
import time

import pytest
from astrbot_plugin_iris_memory.iris_memory.continuity.batching import (
    PASS,
    STOP,
    MessageBatcher,
)

SCOPE = "private:100000001"
CFG = {"merge_quiet_ms": 80, "merge_cap_ms": 150, "explicit_wait_timeout_s": 180}


class Ev:
    def __init__(self, text: str, has_media: bool = False):
        self.message_str = text
        self.has_media = has_media
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class OffsetClock:
    """真实单调时钟 + 可手动快进的偏移（真实小睡眠仍然生效）。"""

    def __init__(self):
        self.offset = 0.0

    def __call__(self) -> float:
        return time.monotonic() + self.offset

    def advance(self, s: float) -> None:
        self.offset += s


async def _send_burst(b, events, gap):
    tasks = []
    for ev in events:
        tasks.append(asyncio.create_task(b.gate(SCOPE, ev)))
        await asyncio.sleep(gap)
    return await asyncio.gather(*tasks)


def _no_leaked_tasks():
    cur = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not cur and not t.done()]


@pytest.mark.asyncio
async def test_burst_of_three_fragments_merges_into_last():
    b = MessageBatcher(CFG)
    evs = [Ev("我今天"), Ev("去面试了"), Ev("感觉还行")]
    t0 = time.monotonic()
    res = await _send_burst(b, evs, 0.02)
    elapsed = time.monotonic() - t0
    assert res == [STOP, STOP, PASS]
    assert evs[0].stopped and evs[1].stopped and not evs[2].stopped
    assert evs[2].message_str == "我今天\n去面试了\n感觉还行"
    # 总附加等待不超过硬上限（留调度余量）
    assert elapsed <= 0.15 + 0.1
    assert _no_leaked_tasks() == []


@pytest.mark.asyncio
async def test_cap_bounds_wait_for_long_burst():
    b = MessageBatcher(CFG)
    evs = [Ev(f"片段{i}") for i in range(6)]
    t0 = time.monotonic()
    res = await _send_burst(b, evs, 0.05)  # 每条都落在静默窗口内
    first_pass = time.monotonic() - t0
    assert res.count(PASS) >= 1
    # 第一个放行的事件出现在硬上限附近，而不是无限延长
    passed = [e for e, r in zip(evs, res) if r == PASS]
    assert passed[0].message_str.startswith("片段0")
    assert first_pass <= 0.05 * 6 + 0.15 + 0.1


@pytest.mark.asyncio
async def test_single_question_passes_without_wait():
    b = MessageBatcher(CFG)
    ev = Ev("你今天吃了什么？")
    t0 = time.monotonic()
    assert await b.gate(SCOPE, ev) == PASS
    assert time.monotonic() - t0 < 0.02
    assert ev.message_str == "你今天吃了什么？" and not ev.stopped


@pytest.mark.asyncio
async def test_question_flushes_pending_fragments_immediately():
    b = MessageBatcher(CFG)
    e1 = Ev("我在想一件事")
    t1 = asyncio.create_task(b.gate(SCOPE, e1))
    await asyncio.sleep(0.01)
    e2 = Ev("你觉得我该去吗？")
    t0 = time.monotonic()
    assert await b.gate(SCOPE, e2) == PASS
    assert time.monotonic() - t0 < 0.02
    assert await t1 == STOP
    assert e2.message_str == "我在想一件事\n你觉得我该去吗？"


@pytest.mark.asyncio
async def test_single_fragment_passes_after_quiet_window():
    b = MessageBatcher(CFG)
    ev = Ev("今天好累")
    t0 = time.monotonic()
    assert await b.gate(SCOPE, ev) == PASS
    el = time.monotonic() - t0
    assert 0.07 <= el <= 0.15 + 0.05
    assert ev.message_str == "今天好累"


@pytest.mark.asyncio
async def test_explicit_wait_then_done_passes_all_text():
    b = MessageBatcher(CFG)
    seq = [Ev("等我说完"), Ev("今天面试"), Ev("问了好多算法题"), Ev("说完了")]
    res = []
    for ev in seq:
        res.append(await b.gate(SCOPE, ev))
        await asyncio.sleep(0.2)  # 远超普通合批窗口：显式等待不得被窗口覆盖
    assert res == [STOP, STOP, STOP, PASS]
    assert seq[-1].message_str == "等我说完\n今天面试\n问了好多算法题\n说完了"
    assert not b.in_explicit_wait(SCOPE)


@pytest.mark.asyncio
async def test_explicit_wait_question_ends_wait():
    b = MessageBatcher(CFG)
    assert await b.gate(SCOPE, Ev("先别回")) == STOP
    assert await b.gate(SCOPE, Ev("我有个事")) == STOP
    q = Ev("你怎么想的？")
    assert await b.gate(SCOPE, q) == PASS
    assert q.message_str == "先别回\n我有个事\n你怎么想的？"


@pytest.mark.asyncio
async def test_explicit_wait_timeout_all_stop_and_buffer_dropped():
    clock = OffsetClock()
    b = MessageBatcher(CFG, clock=clock)
    evs = [Ev("等我说完"), Ev("第一段"), Ev("第二段")]
    for ev in evs:
        assert await b.gate(SCOPE, ev) == STOP
    assert all(e.stopped for e in evs)
    assert b.in_explicit_wait(SCOPE)
    clock.advance(181)
    assert not b.in_explicit_wait(SCOPE)
    # 超时后的下一条消息按普通规则处理，旧缓冲不会混入
    q = Ev("在吗？")
    assert await b.gate(SCOPE, q) == PASS
    assert q.message_str == "在吗？"


@pytest.mark.asyncio
async def test_media_passes_and_does_not_break_explicit_wait():
    b = MessageBatcher(CFG)
    assert await b.gate(SCOPE, Ev("等我说完")) == STOP
    pic = Ev("", has_media=True)
    assert await b.gate(SCOPE, pic) == PASS and not pic.stopped
    assert b.in_explicit_wait(SCOPE)
    assert await b.gate(SCOPE, Ev("就是这张图")) == STOP
    done = Ev("说完了")
    assert await b.gate(SCOPE, done) == PASS
    assert done.message_str == "等我说完\n就是这张图\n说完了"


@pytest.mark.asyncio
async def test_media_does_not_join_merge():
    b = MessageBatcher(CFG)
    e1 = Ev("看这个")
    t1 = asyncio.create_task(b.gate(SCOPE, e1))
    await asyncio.sleep(0.01)
    pic = Ev("配文", has_media=True)
    assert await b.gate(SCOPE, pic) == PASS
    assert pic.message_str == "配文"
    assert await t1 == PASS
    assert e1.message_str == "看这个"


@pytest.mark.asyncio
async def test_scopes_are_independent():
    b = MessageBatcher(CFG)
    a, c = Ev("嗯嗯"), Ev("好的")
    ra, rc = await asyncio.gather(b.gate("private:1", a), b.gate("private:2", c))
    assert ra == PASS and rc == PASS
    assert a.message_str == "嗯嗯" and c.message_str == "好的"


@pytest.mark.asyncio
async def test_injected_sleep_is_used():
    calls = []

    async def fake_sleep(d):
        calls.append(d)
        await asyncio.sleep(0)

    clock = OffsetClock()
    b = MessageBatcher(CFG, sleep=fake_sleep, clock=clock)

    async def run():
        ev = Ev("随便说说")
        task = asyncio.create_task(b.gate(SCOPE, ev))
        while not calls:
            await asyncio.sleep(0)
        clock.advance(1.0)  # 快进越过窗口
        return await task, ev

    res, ev = await asyncio.wait_for(run(), 1.0)
    assert res == PASS and calls and calls[0] <= 0.08 + 1e-6


@pytest.mark.asyncio
async def test_cancelled_gate_leaves_no_tasks():
    b = MessageBatcher(CFG)
    task = asyncio.create_task(b.gate(SCOPE, Ev("一半")))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert _no_leaked_tasks() == []


@pytest.mark.asyncio
async def test_empty_text_passes_untouched():
    b = MessageBatcher(CFG)
    ev = Ev("   ")
    assert await b.gate(SCOPE, ev) == PASS and not ev.stopped


@pytest.mark.asyncio
async def test_gate_failure_degrades_to_pass():
    b = MessageBatcher(CFG)

    class Bad(Ev):
        def stop(self):
            raise RuntimeError("boom")

    assert await b.gate(SCOPE, Bad("等我说完")) == PASS


def test_config_object_and_defaults():
    class C:
        merge_quiet_ms = 500
        merge_cap_ms = 100  # 小于 quiet → 抬到 quiet
        explicit_wait_timeout_s = "bad"

    b = MessageBatcher(C())
    assert b.quiet_s == 0.5 and b.cap_s == 0.5 and b.wait_timeout_s == 180
    d = MessageBatcher()
    assert (d.quiet_s, d.cap_s, d.wait_timeout_s) == (1.2, 2.0, 180)


# ---------------------------------------------------------------- restart restore (F10)


class Wall:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.mark.asyncio
async def test_explicit_wait_restored_from_persisted_closure():
    wall = Wall()
    rows = {SCOPE: {"state": "waiting", "explicit_wait": 1, "wait_since": wall.t - 30}}
    b = MessageBatcher(CFG, closure_getter=rows.get, wall_clock=wall)
    ev = Ev("然后那个面试官")
    assert await b.gate(SCOPE, ev) == STOP and ev.stopped
    assert b.in_explicit_wait(SCOPE)
    done = Ev("说完了")
    assert await b.gate(SCOPE, done) == PASS
    assert done.message_str == "然后那个面试官\n说完了"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        {"state": "waiting", "explicit_wait": 1, "wait_since": 1_800_000_000.0 - 181},  # 已超时
        {"state": "waiting", "explicit_wait": 0, "wait_since": 1_800_000_000.0 - 10},
        {"state": "continue", "explicit_wait": 0, "wait_since": None},
        None,
    ],
)
async def test_explicit_wait_not_restored_when_stale_or_absent(row):
    b = MessageBatcher(CFG, closure_getter=lambda sk: row, wall_clock=Wall())
    assert await b.gate(SCOPE, Ev("你好呀？")) == PASS
    assert not b.in_explicit_wait(SCOPE)


@pytest.mark.asyncio
async def test_closure_seed_once_per_scope_and_errors_degrade():
    calls = []

    def getter(sk):
        calls.append(sk)
        raise RuntimeError("db")

    b = MessageBatcher(CFG, closure_getter=getter, wall_clock=Wall())
    assert await b.gate(SCOPE, Ev("你好？")) == PASS
    assert await b.gate(SCOPE, Ev("在吗？")) == PASS
    assert calls == [SCOPE]


@pytest.mark.asyncio
async def test_set_closure_getter_backward_compatible():
    b = MessageBatcher(CFG)  # 旧构造方式不变
    b.set_closure_getter(lambda sk: {"state": "waiting", "explicit_wait": 1, "wait_since": time.time() - 5})
    assert await b.gate(SCOPE, Ev("嗯然后")) == STOP
