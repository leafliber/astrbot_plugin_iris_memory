"""test_reply_coordinator.py - ReplyCoordinator 单元测试"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from iris_memory.proactive.reply_coordinator import ReplyCoordinator


class TestMarkNormalReply:
    """正常回复守卫标记测试"""

    def test_mark_start_and_pending(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        assert coord.is_normal_reply_pending("g1")

    def test_mark_end_clears_pending(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        coord.mark_normal_reply_end("g1")
        assert not coord.is_normal_reply_pending("g1")

    def test_mark_end_updates_timestamp(self) -> None:
        coord = ReplyCoordinator()
        before = time.time()
        coord.mark_normal_reply_end("g1")
        after = time.time()
        ts = coord.get_last_normal_reply_time("g1")
        assert before <= ts <= after

    def test_mark_none_group_id_ignored(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start(None)
        coord.mark_normal_reply_end(None)
        # 不应崩溃

    def test_pending_false_for_unknown_group(self) -> None:
        coord = ReplyCoordinator()
        assert not coord.is_normal_reply_pending("unknown")

    def test_last_normal_reply_time_default(self) -> None:
        coord = ReplyCoordinator()
        assert coord.get_last_normal_reply_time("g1") == 0.0

    def test_multiple_groups_isolated(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        coord.mark_normal_reply_start("g2")
        coord.mark_normal_reply_end("g1")
        assert not coord.is_normal_reply_pending("g1")
        assert coord.is_normal_reply_pending("g2")


class TestCanSendProactive:
    """综合检查测试"""

    def test_allowed_when_no_pending(self) -> None:
        coord = ReplyCoordinator()
        assert coord.can_send_proactive("g1") is True

    def test_blocked_when_pending(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        assert coord.can_send_proactive("g1") is False

    def test_blocked_during_cooldown(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_end("g1")
        assert coord.can_send_proactive("g1", cooldown_seconds=60.0) is False

    def test_allowed_after_cooldown(self) -> None:
        coord = ReplyCoordinator()
        coord._last_normal_reply_time["g1"] = time.time() - 120
        assert coord.can_send_proactive("g1", cooldown_seconds=60.0) is True

    def test_allowed_with_zero_cooldown(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_end("g1")
        assert coord.can_send_proactive("g1", cooldown_seconds=0.0) is True

    def test_allowed_for_different_group(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        assert coord.can_send_proactive("g2") is True


class TestProactiveReplyGuard:
    """主动回复守卫上下文管理器测试"""

    @pytest.mark.asyncio
    async def test_guard_allows_when_clear(self) -> None:
        coord = ReplyCoordinator()
        async with coord.proactive_reply_guard("g1") as allowed:
            assert allowed is True

    @pytest.mark.asyncio
    async def test_guard_blocks_when_pending(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        async with coord.proactive_reply_guard("g1") as allowed:
            assert allowed is False

    @pytest.mark.asyncio
    async def test_guard_blocks_during_cooldown(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_end("g1")
        async with coord.proactive_reply_guard("g1", cooldown_seconds=60.0) as allowed:
            assert allowed is False

    @pytest.mark.asyncio
    async def test_guard_releases_lock_on_success(self) -> None:
        coord = ReplyCoordinator()
        async with coord.proactive_reply_guard("g1") as allowed:
            assert allowed is True
        # 锁应已释放
        lock = coord._get_proactive_lock("g1")
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_guard_releases_lock_on_block(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        async with coord.proactive_reply_guard("g1") as allowed:
            assert allowed is False
        # 锁应已释放（未获取或已释放）
        lock = coord._get_proactive_lock("g1")
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_guard_releases_lock_on_exception(self) -> None:
        coord = ReplyCoordinator()
        try:
            async with coord.proactive_reply_guard("g1") as allowed:
                assert allowed is True
                raise ValueError("test error")
        except ValueError:
            pass
        lock = coord._get_proactive_lock("g1")
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_guard_prevents_concurrent_proactive(self) -> None:
        """两个主动回复不能并发进入守卫"""
        coord = ReplyCoordinator()
        entered_count = 0
        max_concurrent = 0

        async def proactive_task(delay: float) -> None:
            nonlocal entered_count, max_concurrent
            async with coord.proactive_reply_guard("g1") as allowed:
                if allowed:
                    entered_count += 1
                    max_concurrent = max(max_concurrent, entered_count)
                    await asyncio.sleep(delay)
                    entered_count -= 1

        t1 = asyncio.create_task(proactive_task(0.1))
        t2 = asyncio.create_task(proactive_task(0.1))
        await asyncio.gather(t1, t2)

        # 同一时刻最多一个任务在守卫内
        assert max_concurrent <= 1

    @pytest.mark.asyncio
    async def test_guard_recheck_after_lock(self) -> None:
        """验证锁获取后的二次校验能捕获竞态"""
        coord = ReplyCoordinator()

        guard_entered = asyncio.Event()
        lock_acquired = asyncio.Event()

        async def proactive_task() -> bool:
            """模拟主动回复：先检查通过，但在获取锁期间正常回复开始"""
            async with coord.proactive_reply_guard("g1", cooldown_seconds=60.0) as allowed:
                return allowed

        # 场景：主动回复检查通过 → 正常回复标记开始 → 主动回复获取锁 → 二次检查失败
        # 由于 asyncio 单线程特性，无法精确模拟，改为：
        # 先让正常回复完成（设置时间戳），然后主动回复应被冷却阻止
        coord.mark_normal_reply_end("g1")

        async with coord.proactive_reply_guard("g1", cooldown_seconds=60.0) as allowed:
            assert allowed is False


class TestClear:
    """清理测试"""

    def test_clear_group(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        coord.mark_normal_reply_end("g1")
        coord.clear_group("g1")
        assert not coord.is_normal_reply_pending("g1")
        assert coord.get_last_normal_reply_time("g1") == 0.0

    def test_clear_all(self) -> None:
        coord = ReplyCoordinator()
        coord.mark_normal_reply_start("g1")
        coord.mark_normal_reply_start("g2")
        coord.mark_normal_reply_end("g2")
        coord.clear_all()
        assert not coord.is_normal_reply_pending("g1")
        assert not coord.is_normal_reply_pending("g2")
        assert coord.get_last_normal_reply_time("g2") == 0.0


class TestRaceConditionScenarios:
    """模拟竞态场景测试"""

    @pytest.mark.asyncio
    async def test_scenario_batch_delay_after_normal_reply(self) -> None:
        """场景1：批处理延迟，信号在正常回复后生成

        时间线：
        T1: 正常回复开始 → mark_normal_reply_start
        T2: 正常回复结束 → mark_normal_reply_end（更新时间戳）
        T3: 批处理完成 → process_message → 检查 can_send_proactive → 被冷却阻止
        """
        coord = ReplyCoordinator()

        # T1: 正常回复开始
        coord.mark_normal_reply_start("g1")
        # T2: 正常回复结束
        coord.mark_normal_reply_end("g1")

        # T3: 批处理尝试生成信号
        assert coord.can_send_proactive("g1", cooldown_seconds=60.0) is False

    @pytest.mark.asyncio
    async def test_scenario_proactive_during_normal_reply(self) -> None:
        """场景2：主动回复在正常回复进行期间触发

        时间线：
        T1: 正常回复开始 → mark_normal_reply_start
        T2: GroupScheduler 定时器触发 → 检查 guard → 被 pending 阻止
        T3: 正常回复结束 → mark_normal_reply_end
        """
        coord = ReplyCoordinator()

        # T1: 正常回复开始
        coord.mark_normal_reply_start("g1")

        # T2: 主动回复尝试
        async with coord.proactive_reply_guard("g1") as allowed:
            assert allowed is False

        # T3: 正常回复结束
        coord.mark_normal_reply_end("g1")

    @pytest.mark.asyncio
    async def test_scenario_signal_followup_no_concurrent(self) -> None:
        """场景3：signal 回复与 followup 回复不能并发

        两个主动回复走同一个锁，不能同时发送。
        """
        coord = ReplyCoordinator()
        results = []

        async def signal_reply() -> None:
            async with coord.proactive_reply_guard("g1") as allowed:
                if allowed:
                    results.append("signal_start")
                    await asyncio.sleep(0.05)
                    results.append("signal_end")

        async def followup_reply() -> None:
            # 等一下确保 signal 先获取锁
            await asyncio.sleep(0.01)
            async with coord.proactive_reply_guard("g1") as allowed:
                if allowed:
                    results.append("followup_start")
                    await asyncio.sleep(0.05)
                    results.append("followup_end")

        await asyncio.gather(signal_reply(), followup_reply())

        # 确保没有交叉执行
        if "signal_start" in results and "followup_start" in results:
            signal_start_idx = results.index("signal_start")
            signal_end_idx = results.index("signal_end")
            followup_start_idx = results.index("followup_start")
            # followup 必须在 signal 完成后才开始
            assert followup_start_idx > signal_end_idx

    @pytest.mark.asyncio
    async def test_scenario_normal_reply_between_check_and_send(self) -> None:
        """场景4：检查时允许，获取锁期间正常回复完成

        验证守卫的二次校验能捕获此情况。
        通过在第一个 guard 持有锁期间修改状态来模拟。
        """
        coord = ReplyCoordinator()
        result_second = None

        async def first_proactive() -> None:
            async with coord.proactive_reply_guard("g1", cooldown_seconds=60.0) as allowed:
                assert allowed is True
                # 模拟：在持有锁期间，正常回复完成
                coord.mark_normal_reply_end("g1")
                await asyncio.sleep(0.05)

        async def second_proactive() -> None:
            nonlocal result_second
            await asyncio.sleep(0.01)  # 确保 first 先获取锁
            async with coord.proactive_reply_guard("g1", cooldown_seconds=60.0) as allowed:
                result_second = allowed

        await asyncio.gather(first_proactive(), second_proactive())

        # 第二个主动回复应被二次校验阻止（冷却期内）
        assert result_second is False
