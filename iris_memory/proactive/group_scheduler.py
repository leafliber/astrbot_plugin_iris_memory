"""
群定时器调度器

每个群维护一个独立的 asyncio.Task，定期检查该群的信号队列。
当群沉默超时后销毁定时器，节省资源。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from iris_memory.config import get_store
from iris_memory.proactive.models import AggregatedDecision, Signal
from iris_memory.proactive.signal_queue import SignalQueue
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.group_scheduler")

# 回调类型：接收 AggregatedDecision，执行回复
ReplyCallback = Callable[[AggregatedDecision], Coroutine[Any, Any, None]]

# LLM 确认回调类型：接收群信号和上下文，返回是否应回复
LLMConfirmCallback = Callable[
    [str, List[Signal], List[Dict[str, Any]]],
    Coroutine[Any, Any, bool],
]


class GroupScheduler:
    """群定时器调度器

    生命周期：
    - 群收到第一条消息时创建定时器
    - 群收到新消息时重置沉默计时
    - 群沉默超过 silence_timeout 后销毁定时器

    检查逻辑（每 check_interval 秒执行一次）：
    1. 检查群沉默时间是否达到 min_silence
    2. 获取该群有效信号
    3. 聚合信号权重
    4. 根据阈值决定直接回复 / LLM 确认 / 跳过
    """

    def __init__(
        self,
        signal_queue: SignalQueue,
        on_reply: Optional[ReplyCallback] = None,
        on_llm_confirm: Optional[LLMConfirmCallback] = None,
    ) -> None:
        self._cfg = get_store()
        self._signal_queue = signal_queue
        self._on_reply = on_reply
        self._on_llm_confirm = on_llm_confirm

        # group_id -> asyncio.Task
        self._timers: Dict[str, asyncio.Task] = {}
        # group_id -> 最后一次有消息活动的事件循环时间
        self._active_groups: Set[str] = set()
        self._closed = False

    def set_reply_callback(self, callback: ReplyCallback) -> None:
        """设置回复回调"""
        self._on_reply = callback

    def set_llm_confirm_callback(self, callback: LLMConfirmCallback) -> None:
        """设置 LLM 确认回调"""
        self._on_llm_confirm = callback

    def ensure_timer(self, group_id: str) -> None:
        """确保某群有定时器运行

        如果定时器不存在或已完成，创建新的定时器。
        """
        if self._closed:
            return

        if group_id in self._timers:
            task = self._timers[group_id]
            if not task.done():
                return  # 定时器仍在运行

        # 创建新定时器
        self._timers[group_id] = asyncio.create_task(
            self._timer_loop(group_id),
            name=f"proactive-timer-{group_id}",
        )
        self._active_groups.add(group_id)
        logger.debug(f"Timer created for group {group_id}")

    async def _timer_loop(self, group_id: str) -> None:
        """群定时器主循环

        Args:
            group_id: 群组 ID
        """
        check_interval = self._cfg.get("proactive_reply.signal_check_interval_seconds", 30)
        silence_timeout = self._cfg.get("proactive_reply.signal_silence_timeout_seconds", 600)

        try:
            while not self._closed:
                await asyncio.sleep(check_interval)

                if self._closed:
                    break

                # 检查沉默超时 → 销毁定时器
                silence = self._signal_queue.get_silence_duration(group_id)
                if silence >= silence_timeout:
                    logger.debug(
                        f"Group {group_id} silence timeout "
                        f"({silence:.0f}s >= {silence_timeout}s), "
                        f"destroying timer"
                    )
                    # 清除该群所有信号
                    self._signal_queue.clear_group(group_id)
                    break

                # 检查是否达到最小沉默时间
                min_silence = self._cfg.get("proactive_reply.signal_min_silence_seconds", 60)
                if silence < min_silence:
                    continue  # 群还在活跃，跳过

                # 获取该群有效信号
                signals = self._signal_queue.get_signals(group_id)
                if not signals:
                    continue  # 无信号，跳过

                # 聚合决策
                await self._aggregate_and_decide(group_id, signals)

        except asyncio.CancelledError:
            logger.debug(f"Timer cancelled for group {group_id}")
        except Exception as e:
            logger.error(f"Timer error for group {group_id}: {e}")
        finally:
            self._active_groups.discard(group_id)
            self._timers.pop(group_id, None)

    async def _aggregate_and_decide(
        self, group_id: str, signals: List[Signal]
    ) -> None:
        """聚合信号并做出决策

        Args:
            group_id: 群组 ID
            signals: 有效信号列表
        """
        aggregated_weight = self._signal_queue.aggregate_weight(group_id)

        # 找到权重最高的信号对应的用户
        best_signal = max(signals, key=lambda s: s.weight)
        target_user_id = best_signal.user_id
        session_key = best_signal.session_key

        # 收集近期消息上下文（从信号 metadata 中提取）
        recent_messages: List[Dict[str, Any]] = []
        for s in signals:
            preview = s.metadata.get("text_preview", "")
            if preview:
                recent_messages.append({
                    "sender_id": s.user_id,
                    "content": preview,
                    "timestamp": s.created_at.isoformat(),
                })

        weight_direct = self._cfg.get("proactive_reply.signal_weight_direct_reply", 0.8)
        weight_llm = self._cfg.get("proactive_reply.signal_weight_llm_confirm", 0.5)
        proactive_mode = self._cfg.get("proactive_reply.proactive_mode", "rule")

        if aggregated_weight >= weight_direct:
            # 高权重 → 直接回复
            decision = AggregatedDecision(
                should_reply=True,
                session_key=session_key,
                group_id=group_id,
                target_user_id=target_user_id,
                aggregated_weight=aggregated_weight,
                signals=list(signals),
                reason=f"聚合权重 {aggregated_weight:.2f} >= {weight_direct}",
                recent_messages=recent_messages,
                llm_confirmed=False,
            )
            await self._execute_reply(decision)

        elif (
            aggregated_weight >= weight_llm
            and proactive_mode == "hybrid"
        ):
            # 中等权重 + hybrid 模式 → LLM 确认
            should_reply = await self._try_llm_confirm(
                group_id, signals, recent_messages
            )
            if should_reply:
                decision = AggregatedDecision(
                    should_reply=True,
                    session_key=session_key,
                    group_id=group_id,
                    target_user_id=target_user_id,
                    aggregated_weight=aggregated_weight,
                    signals=list(signals),
                    reason=(
                        f"聚合权重 {aggregated_weight:.2f} 经 LLM 确认后回复"
                    ),
                    recent_messages=recent_messages,
                    llm_confirmed=True,
                )
                await self._execute_reply(decision)
            else:
                logger.debug(
                    f"Group {group_id}: LLM declined reply "
                    f"(weight={aggregated_weight:.2f})"
                )
                # 清除已处理的信号
                self._signal_queue.clear_group(group_id)
        else:
            # 权重不足，清除信号
            logger.debug(
                f"Group {group_id}: weight {aggregated_weight:.2f} "
                f"below threshold, skipping"
            )
            self._signal_queue.clear_group(group_id)

    async def _try_llm_confirm(
        self,
        group_id: str,
        signals: List[Signal],
        recent_messages: List[Dict[str, Any]],
    ) -> bool:
        """尝试 LLM 确认

        Args:
            group_id: 群组 ID
            signals: 信号列表
            recent_messages: 近期消息

        Returns:
            是否应回复
        """
        if not self._on_llm_confirm:
            logger.debug("No LLM confirm callback, falling back to rule decision")
            return False

        try:
            return await self._on_llm_confirm(group_id, signals, recent_messages)
        except Exception as e:
            logger.warning(f"LLM confirm failed for group {group_id}: {e}")
            return False

    async def _execute_reply(self, decision: AggregatedDecision) -> None:
        """执行回复

        Args:
            decision: 聚合决策结果
        """
        if not self._on_reply:
            logger.warning("No reply callback set, cannot execute reply")
            return

        try:
            # 清除该群已处理的信号
            self._signal_queue.clear_group(decision.group_id)
            # 执行回复回调
            await self._on_reply(decision)
            logger.info(
                f"Proactive reply triggered for group {decision.group_id}, "
                f"user {decision.target_user_id}, "
                f"weight={decision.aggregated_weight:.2f}, "
                f"reason={decision.reason}"
            )
        except Exception as e:
            logger.error(
                f"Failed to execute proactive reply for "
                f"group {decision.group_id}: {e}"
            )

    def has_active_timer(self, group_id: str) -> bool:
        """检查某群是否有活跃的定时器"""
        return group_id in self._active_groups

    @property
    def active_group_count(self) -> int:
        """活跃的群定时器数量"""
        return len(self._active_groups)

    async def close(self) -> None:
        """关闭所有定时器"""
        self._closed = True

        for group_id, task in list(self._timers.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._timers.clear()
        self._active_groups.clear()
        logger.info("GroupScheduler closed, all timers destroyed")
