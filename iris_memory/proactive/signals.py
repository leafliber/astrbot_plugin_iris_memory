from __future__ import annotations

import time

from astrbot.api import logger

from .config import ConfigManager
from .perception import WindowMessage
from .state import GroupState, StateManager


class SignalGate:
    """本地信号门控：零 LLM 成本地评估两种唤醒源是否值得进入决策层。

    - 消息唤醒：锚点用户/关键词命中 → follow_up；采样阈值达标 → chime_in
    - 定时器唤醒：冷场静默 / 话题刚结束 → initiate
    """

    def __init__(self, config: ConfigManager, state: StateManager) -> None:
        self._config = config
        self._state = state

    def evaluate_message(self, group_id: str, sender_id: str, text: str) -> str | None:
        """消息唤醒门控，返回候选动机 "follow_up" | "chime_in" | None。"""
        if not self._config.enabled:
            return None
        data = self._state.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return None
        if self._state.is_muted():
            return None

        if sender_id and self._state.match_anchor_user(group_id, sender_id):
            logger.debug("Iris Reply: anchor user trigger for group %s", group_id)
            self._state.reset_sampling(group_id)
            return "follow_up"

        matched = self._state.match_anchor_keyword(group_id, text)
        if matched:
            logger.debug("Iris Reply: anchor keyword trigger for group %s, keywords=%s", group_id, matched)
            self._state.reset_sampling(group_id)
            return "follow_up"

        self._state.increment_msg_count(group_id)
        if self._state.should_trigger_sampling(group_id):
            logger.debug("Iris Reply: sampling trigger for group %s", group_id)
            self._state.reset_sampling(group_id)
            return "chime_in"
        return None

    def evaluate_timer(self, group_id: str, messages: list[WindowMessage]) -> str | None:
        """定时器唤醒门控，返回 "initiate" | None。

        不再使用硬静默阈值，改为回复意愿值概率门控：意愿随静默时间积累、
        随对话活跃消退，积累速率由回复意愿等级与话题状态（drift / 锚点）
        共同调节，达到随机采样的阈值时点火。
        """
        if not self._config.enabled or not self._config.proactive_enabled:
            return None
        data = self._state.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return None
        if not messages:
            return None
        if data.initiate_pending_since > 0:
            return None

        now = time.time()
        if (
            data.initiate_daily_count >= self._config.proactive_max_per_day
            or data.initiate_no_reply_streak >= self._config.proactive_max_streak
        ):
            # 触达上限期间清零意愿，防止次日/解禁瞬间点火
            self._state.reset_initiate_drive(group_id)
            return None
        if now - data.last_initiate_time < self._config.proactive_min_interval * 60:
            return None

        quiet = now - messages[-1].timestamp
        fired = self._state.update_initiate_drive(
            group_id,
            now=now,
            quiet_seconds=quiet,
            freeze=self._state.is_muted(),
        )
        return "initiate" if fired else None
