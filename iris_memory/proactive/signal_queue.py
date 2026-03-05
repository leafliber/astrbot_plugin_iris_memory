"""
信号队列

按群隔离的信号存储与管理，支持 TTL 过期、容量限制、按群清除。
使用内存中的 dict 存储，无需持久化。
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from iris_memory.config import get_store
from iris_memory.proactive.models import Signal, SignalType
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.signal_queue")


class SignalQueue:
    """信号队列

    按 group_id 隔离信号。每个群维护一个独立的信号列表。
    定时器检查时仅读取对应群的信号进行聚合决策。

    主要操作：
    - enqueue：入队信号
    - get_signals：获取某群的有效信号（自动过滤过期）
    - clear_session：清除某会话的所有信号
    - clear_group：清除某群的所有信号
    """

    def __init__(self) -> None:
        self._cfg = get_store()
        # group_id -> List[Signal]
        self._queues: Dict[str, List[Signal]] = {}
        # group_id -> 最近一条消息的时间戳
        self._last_message_time: Dict[str, datetime] = {}

    def enqueue(self, signal: Signal) -> bool:
        """信号入队

        Args:
            signal: 信号对象

        Returns:
            True 入队成功，False 超过容量限制
        """
        group_id = signal.group_id
        if group_id not in self._queues:
            self._queues[group_id] = []

        # 容量检查
        max_signals = self._cfg.get("proactive_reply.signal_max_signals_per_group", 50)
        if len(self._queues[group_id]) >= max_signals:
            # 移除权重最低的信号（保留高权重信号）
            min_idx = min(
                range(len(self._queues[group_id])),
                key=lambda i: self._queues[group_id][i].weight,
            )
            removed = self._queues[group_id].pop(min_idx)
            logger.debug(
                f"Signal queue overflow for group {group_id}, "
                f"removed lowest weight signal (w={removed.weight:.2f})"
            )

        self._queues[group_id].append(signal)
        logger.debug(
            f"Signal enqueued: type={signal.signal_type.value}, "
            f"group={group_id}, weight={signal.weight:.2f}"
        )
        return True

    def get_signals(self, group_id: str) -> List[Signal]:
        """获取某群的有效信号（自动过滤过期信号）

        Args:
            group_id: 群组 ID

        Returns:
            有效信号列表
        """
        if group_id not in self._queues:
            return []

        now = datetime.now()
        valid = []
        expired_count = 0

        for signal in self._queues[group_id]:
            if signal.expires_at and now >= signal.expires_at:
                expired_count += 1
            else:
                valid.append(signal)

        # 更新队列，移除过期信号
        if expired_count > 0:
            self._queues[group_id] = valid
            logger.debug(
                f"Removed {expired_count} expired signals for group {group_id}"
            )

        return valid

    def clear_session(self, session_key: str) -> int:
        """清除某会话的所有信号

        Args:
            session_key: 会话键（user_id:group_id）

        Returns:
            清除的信号数量
        """
        removed = 0
        for group_id in list(self._queues.keys()):
            before = len(self._queues[group_id])
            self._queues[group_id] = [
                s for s in self._queues[group_id]
                if s.session_key != session_key
            ]
            removed += before - len(self._queues[group_id])
            if not self._queues[group_id]:
                del self._queues[group_id]

        if removed > 0:
            logger.debug(f"Cleared {removed} signals for session {session_key}")
        return removed

    def clear_group(self, group_id: str) -> int:
        """清除某群的所有信号

        Args:
            group_id: 群组 ID

        Returns:
            清除的信号数量
        """
        if group_id not in self._queues:
            return 0
        count = len(self._queues[group_id])
        del self._queues[group_id]
        if count > 0:
            logger.debug(f"Cleared {count} signals for group {group_id}")
        return count

    def update_last_message_time(self, group_id: str) -> None:
        """更新群的最后消息时间"""
        self._last_message_time[group_id] = datetime.now()

    def get_last_message_time(self, group_id: str) -> Optional[datetime]:
        """获取群的最后消息时间"""
        return self._last_message_time.get(group_id)

    def get_silence_duration(self, group_id: str) -> float:
        """获取群的沉默时长（秒）

        Args:
            group_id: 群组 ID

        Returns:
            沉默时长（秒），如果没有记录则返回 float('inf')
        """
        last_time = self._last_message_time.get(group_id)
        if last_time is None:
            return float("inf")
        return (datetime.now() - last_time).total_seconds()

    def get_active_groups(self) -> List[str]:
        """获取所有有信号的群组 ID"""
        return list(self._queues.keys())

    def aggregate_weight(self, group_id: str) -> float:
        """计算某群所有有效信号的聚合权重

        使用最大值 + 衰减叠加策略：
        - 取最高权重作为基础
        - 其余信号权重按 0.3 系数叠加

        Args:
            group_id: 群组 ID

        Returns:
            聚合权重（0.0 - 1.0）
        """
        signals = self.get_signals(group_id)
        if not signals:
            return 0.0

        weights = sorted([s.weight for s in signals], reverse=True)
        base = weights[0]
        bonus = sum(w * 0.5 for w in weights[1:])
        return min(1.0, base + bonus)

    @property
    def total_signals(self) -> int:
        """所有群的总信号数"""
        return sum(len(signals) for signals in self._queues.values())

    @property
    def group_count(self) -> int:
        """有信号的群数"""
        return len(self._queues)
