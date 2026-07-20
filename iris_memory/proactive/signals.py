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
        """定时器唤醒门控，返回 "initiate" | None。"""
        if not self._config.enabled or not self._config.proactive_enabled:
            return None
        data = self._state.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return None
        if self._state.is_muted():
            return None
        if not messages:
            return None
        if data.initiate_pending_since > 0:
            return None
        if data.initiate_daily_count >= self._config.proactive_max_per_day:
            return None
        if data.initiate_no_reply_streak >= self._config.proactive_max_streak:
            return None

        now = time.time()
        if now - data.last_initiate_time < self._config.proactive_min_interval * 60:
            return None

        quiet_threshold = self._config.proactive_quiet_minutes * 60
        # 话题刚结束（drifted）不久时，用更短的静默阈值提前发起
        if 0 < data.last_drift_time and now - data.last_drift_time < quiet_threshold:
            quiet_threshold = min(quiet_threshold, self._config.proactive_drift_delay * 60)

        quiet = now - messages[-1].timestamp
        if quiet < quiet_threshold:
            return None
        return "initiate"
