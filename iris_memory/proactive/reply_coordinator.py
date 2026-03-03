"""
回复协调器

集中式群级别回复协调，解决正常回复与主动回复之间的竞态条件：

1. 「正常回复进行中」守卫：process_normal_message 返回 prompt 时标记开始，
   handle_llm_response 完成时标记结束。主动回复在发送前检查此标记。

2. 群级别回复锁：防止多个主动回复（signal + followup）并发发送。

3. 发送前二次校验：在实际发送主动回复前（而非仅信号入队前）再次检查
   正常回复时间戳，消除 TOCTOU 窗口。

设计要点：
- 所有状态按 group_id 隔离
- 使用 asyncio.Lock（单线程事件循环中无死锁风险）
- 守卫标记为 Set[str]，查询 O(1)
- 锁按需创建，避免预分配
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional, Set

from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.reply_coordinator")


class ReplyCoordinator:
    """群级别回复协调器

    解决三种竞态条件：
    - 正常回复与主动回复并发
    - 信号回复与 FollowUp 回复并发
    - 批处理延迟导致信号在清除后重新入队

    使用方式：
        coordinator = ReplyCoordinator()

        # 正常回复流程（MessageProcessor）
        coordinator.mark_normal_reply_start(group_id)
        try:
            ... LLM 回复 ...
        finally:
            coordinator.mark_normal_reply_end(group_id, cooldown_seconds)

        # 主动回复流程（ProactiveManager）
        async with coordinator.proactive_reply_guard(group_id, cooldown_seconds) as allowed:
            if allowed:
                await send_reply(...)
    """

    def __init__(self) -> None:
        # 正在进行正常 LLM 回复的群组集合
        self._normal_reply_pending: Set[str] = set()

        # 群级别主动回复锁（防止 signal + followup 并发发送）
        self._proactive_locks: Dict[str, asyncio.Lock] = {}

        # 正常回复完成时间戳（group_id -> timestamp）
        self._last_normal_reply_time: Dict[str, float] = {}

    # ── 正常回复守卫 ──────────────────────────────────────────

    def mark_normal_reply_start(self, group_id: Optional[str]) -> None:
        """标记群组开始正常 LLM 回复流程

        在 process_normal_message 返回非 None prompt 时调用。

        Args:
            group_id: 群组 ID，私聊传 None 则忽略
        """
        if not group_id:
            return
        self._normal_reply_pending.add(group_id)
        logger.debug(f"Normal reply started for group {group_id}")

    def mark_normal_reply_end(
        self,
        group_id: Optional[str],
        cooldown_seconds: float = 0.0,
    ) -> None:
        """标记群组正常 LLM 回复流程结束

        在 handle_llm_response 完成后调用。同时更新正常回复时间戳。

        Args:
            group_id: 群组 ID，私聊传 None 则忽略
            cooldown_seconds: 冷却时间（仅用于日志），实际冷却通过时间戳检查
        """
        if not group_id:
            return
        self._normal_reply_pending.discard(group_id)
        self._last_normal_reply_time[group_id] = time.time()
        logger.debug(f"Normal reply ended for group {group_id}")

    def is_normal_reply_pending(self, group_id: str) -> bool:
        """检查群组是否有正在进行的正常回复

        Args:
            group_id: 群组 ID

        Returns:
            True 表示有正常回复正在进行，主动回复应等待/跳过
        """
        return group_id in self._normal_reply_pending

    def get_last_normal_reply_time(self, group_id: str) -> float:
        """获取群组最后一次正常回复的时间戳

        Args:
            group_id: 群组 ID

        Returns:
            时间戳（秒），无记录返回 0.0
        """
        return self._last_normal_reply_time.get(group_id, 0.0)

    # ── 主动回复协调 ──────────────────────────────────────────

    def can_send_proactive(
        self,
        group_id: str,
        cooldown_seconds: float = 0.0,
    ) -> bool:
        """综合检查是否可以发送主动回复

        合并所有前置校验到一个方法，在实际发送前调用（而非仅入队前）：
        1. 正常回复是否正在进行
        2. 正常回复冷却是否已过

        Args:
            group_id: 群组 ID
            cooldown_seconds: 正常回复后的冷却时间（秒）

        Returns:
            True 表示可以发送
        """
        # 检查 1：正常回复进行中
        if group_id in self._normal_reply_pending:
            logger.debug(
                f"Cannot send proactive reply: normal reply pending "
                f"for group {group_id}"
            )
            return False

        # 检查 2：正常回复冷却
        if cooldown_seconds > 0:
            last_normal = self._last_normal_reply_time.get(group_id, 0.0)
            elapsed = time.time() - last_normal
            if last_normal > 0 and elapsed < cooldown_seconds:
                logger.debug(
                    f"Cannot send proactive reply: normal reply "
                    f"{elapsed:.1f}s ago (< {cooldown_seconds}s) "
                    f"for group {group_id}"
                )
                return False

        return True

    def _get_proactive_lock(self, group_id: str) -> asyncio.Lock:
        """获取群级别主动回复锁（按需创建）

        Args:
            group_id: 群组 ID

        Returns:
            asyncio.Lock 实例
        """
        if group_id not in self._proactive_locks:
            self._proactive_locks[group_id] = asyncio.Lock()
        return self._proactive_locks[group_id]

    def proactive_reply_guard(
        self,
        group_id: str,
        cooldown_seconds: float = 0.0,
    ) -> "_ProactiveReplyGuard":
        """获取主动回复守卫（上下文管理器）

        用法：
            async with coordinator.proactive_reply_guard(group_id, cooldown) as allowed:
                if allowed:
                    await send_reply(...)

        守卫功能：
        1. 获取群级别锁（防止并发主动回复）
        2. 锁获取后二次校验 can_send_proactive（消除 TOCTOU）
        3. 退出时自动释放锁

        Args:
            group_id: 群组 ID
            cooldown_seconds: 正常回复后的冷却时间

        Returns:
            异步上下文管理器，yield bool 表示是否允许发送
        """
        return _ProactiveReplyGuard(self, group_id, cooldown_seconds)

    # ── 清理 ──────────────────────────────────────────────────

    def clear_group(self, group_id: str) -> None:
        """清除某群的所有协调状态

        Args:
            group_id: 群组 ID
        """
        self._normal_reply_pending.discard(group_id)
        self._last_normal_reply_time.pop(group_id, None)
        self._proactive_locks.pop(group_id, None)

    def clear_all(self) -> None:
        """清除所有协调状态"""
        self._normal_reply_pending.clear()
        self._last_normal_reply_time.clear()
        self._proactive_locks.clear()


class _ProactiveReplyGuard:
    """主动回复守卫（异步上下文管理器）

    在获取锁后、实际发送前进行二次校验，
    消除「检查时可以发送，发送时正常回复已开始」的 TOCTOU 窗口。
    """

    def __init__(
        self,
        coordinator: ReplyCoordinator,
        group_id: str,
        cooldown_seconds: float,
    ) -> None:
        self._coordinator = coordinator
        self._group_id = group_id
        self._cooldown_seconds = cooldown_seconds
        self._lock: Optional[asyncio.Lock] = None
        self._allowed = False

    async def __aenter__(self) -> bool:
        self._lock = self._coordinator._get_proactive_lock(self._group_id)

        # 锁获取前快速检查（避免不必要的等待）
        if not self._coordinator.can_send_proactive(
            self._group_id, self._cooldown_seconds
        ):
            self._allowed = False
            return False

        # 获取锁（防止并发主动回复）
        await self._lock.acquire()

        # 锁获取后二次校验（消除 TOCTOU 窗口）
        if not self._coordinator.can_send_proactive(
            self._group_id, self._cooldown_seconds
        ):
            self._lock.release()
            self._allowed = False
            logger.debug(
                f"Proactive reply guard: re-check failed after lock "
                f"acquisition for group {self._group_id}"
            )
            return False

        self._allowed = True
        return True

    async def __aexit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        if self._lock is not None and self._lock.locked():
            self._lock.release()
