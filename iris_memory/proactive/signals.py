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
        # 内存态即可：重启后保守地全部失效。
        self._owner_last: dict[str, float] = {}
        self._engaged: dict[str, dict[str, float]] = {}

    # ---- Frisk 群聊门控 ----

    def mark_engaged(self, group_id: str, sender_id: str, now: float | None = None) -> None:
        """非主人群友明确找她（@ / 回复 / 叫名字）：从此刻起开放有限时长的跟进。"""
        if not group_id or not sender_id:
            return
        self._engaged.setdefault(group_id, {})[str(sender_id)] = (
            now if now is not None else time.time()
        )

    def is_engaged(self, group_id: str, sender_id: str, now: float | None = None) -> bool:
        since = self._engaged.get(group_id, {}).get(str(sender_id))
        if since is None:
            return False
        now = now if now is not None else time.time()
        if now - since < self._config.engaged_window * 60:
            return True
        self._engaged[group_id].pop(str(sender_id), None)
        return False

    def mentions_bot_name(self, text: str) -> bool:
        low = (text or "").lower()
        return any(k in low for k in self._config.bot_name_keywords)

    def evaluate_group_message(
        self,
        group_id: str,
        sender_id: str,
        text: str,
        owner_ids: set[str],
        now: float | None = None,
    ) -> tuple[str | None, str]:
        """群聊消息门控，返回 (motive, trigger)。

        - 主人发言：每条（去抖后）都交给决策层，由她自己决定接不接；
        - 其他人：不再做计数采样插话。只有叫她名字时进入一次决策，
          或在明确找过她之后 engaged_window 内命中锚点时跟进。
        trigger: "owner" | "name" | "follow" | ""
        """
        if not self._config.enabled:
            return None, ""
        now = now if now is not None else time.time()
        sender_id = str(sender_id or "")
        is_owner = bool(sender_id) and sender_id in owner_ids
        self._state.increment_msg_count(group_id)
        data = self._state.get_state(group_id)
        # 冷却/静音只约束她的“不请自来”：叫她名字不受影响；
        # 主人发言不受冷却影响（静音时段仍然遵守）。
        cooling = data.state == GroupState.COOLDOWN
        muted = self._state.is_muted()

        if self.mentions_bot_name(text):
            if not is_owner:
                self.mark_engaged(group_id, sender_id, now)
            else:
                self._owner_last[group_id] = now
            self._state.reset_sampling(group_id)
            return "chime_in", "name"

        if muted or (cooling and not is_owner):
            return None, ""

        engaged = is_owner or self.is_engaged(group_id, sender_id, now)
        if engaged:
            if sender_id and self._state.match_anchor_user(group_id, sender_id):
                self._state.reset_sampling(group_id)
                return "follow_up", "follow"
            if self._state.match_anchor_keyword(group_id, text):
                self._state.reset_sampling(group_id)
                return "follow_up", "follow"

        if is_owner:
            last = self._owner_last.get(group_id, 0.0)
            if now - last >= self._config.owner_chime_debounce:
                self._owner_last[group_id] = now
                self._state.reset_sampling(group_id)
                return "chime_in", "owner"
        return None, ""

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

        达到最小静默底线后按风险率进行一次独立概率试验；风险率综合静默
        时长、回复意愿、话题状态和无人接话疲劳，不存在必然点火阈值。
        """
        if not self._config.enabled or not self._config.proactive_enabled:
            return None
        # Frisk：群里默认不定时主动开话题（只对主人私聊保留定时发起）。
        if (
            not str(group_id).startswith("private:")
            and not self._config.group_initiate_enabled
        ):
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
            # 滚动额度/无人接话退避期间推进概率时钟，防止解禁瞬间点火。
            self._state.reset_initiate_drive(group_id)
            return None
        if now - data.last_initiate_time < self._config.proactive_min_interval * 60:
            return None

        last_activity = max(messages[-1].timestamp, data.last_human_message_at)
        quiet = max(0.0, now - last_activity)
        fired = self._state.evaluate_initiate_hazard(
            group_id,
            now=now,
            quiet_seconds=quiet,
            freeze=self._state.is_muted(),
        )
        return "initiate" if fired else None
