from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from astrbot.api import logger

from .config import ConfigManager
from .prompts import (
    BACKOFF_BASE,
    DEFAULT_LEVEL,
    MAX_BACKOFF_LEVEL,
    VALID_LEVELS,
    WILLINGNESS_THRESHOLD_ADJUST,
    display_level,
)


class GroupState(Enum):
    IDLE = "idle"
    COOLDOWN = "cooldown"


@dataclass
class ThreadAnchor:
    """统一对话锚点：记录"我上次以什么动机说了什么、在关注谁/什么"。

    三种发言模式（chime_in / follow_up / initiate）以及被动回复共用同一锚点结构。
    participants / keywords 带 TTL，过期后不再触发跟进匹配；
    topic / bot_message / reason 随锚点保留，用于决策 prompt 中的身份与话题连续性。
    """

    kind: str = ""  # "chime_in" | "follow_up" | "initiate" | "passive" | "reply"
    topic: str = ""
    bot_message: str = ""
    participants: set[str] = field(default_factory=set)
    participant_ttls: dict[str, float] = field(default_factory=dict)
    keywords: set[str] = field(default_factory=set)
    keyword_ttls: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    created_at: float = 0.0

    @property
    def active(self) -> bool:
        return bool(self.participants or self.keywords)

    @property
    def has_context(self) -> bool:
        return bool(self.bot_message or self.participants or self.keywords)

    def clear(self) -> None:
        self.kind = ""
        self.topic = ""
        self.bot_message = ""
        self.participants.clear()
        self.participant_ttls.clear()
        self.keywords.clear()
        self.keyword_ttls.clear()
        self.reason = ""
        self.created_at = 0.0


@dataclass
class GroupStateData:
    state: GroupState = GroupState.IDLE
    cooldown_until: float = 0.0
    msg_count: int = 0
    last_sample_time: float = 0.0
    backoff_level: int = 0
    last_backoff_time: float = 0.0
    consecutive_replies: int = 0
    willingness: str = DEFAULT_LEVEL
    boost_initial: float = 1.0
    boost_set_at: float = 0.0
    boost_until: float = 0.0
    last_detect_time: float = 0.0
    anchor: ThreadAnchor = field(default_factory=ThreadAnchor)
    last_observation: str = ""
    last_drift_time: float = 0.0
    last_initiate_time: float = 0.0
    initiate_daily_count: int = 0
    initiate_count_date: str = ""
    initiate_pending_since: float = 0.0
    initiate_no_reply_streak: int = 0
    dirty: bool = False


class StateManager:
    N_MAX = 120
    T_MAX = 300
    MAX_ANCHOR_USERS = 20
    MAX_ANCHOR_KEYWORDS = 10
    MAX_COOLDOWN_MINUTES = 120
    BACKOFF_DECAY_INTERVAL = 300.0

    _GROUP_IDS_KEY = "iris_reply:group_ids"
    _GROUP_KEY_PREFIX = "state:"

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._groups: dict[str, GroupStateData] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._dirty_groups: set[str] = set()
        self._global_lock = asyncio.Lock()
        self._whitelist: set[str] = set()
        self._whitelist_dirty: bool = False
        self._group_ids_dirty: bool = False

    def get_lock(self, group_id: str) -> asyncio.Lock:
        return self._locks.setdefault(group_id, asyncio.Lock())

    def remove_group_lock(self, group_id: str) -> None:
        self._locks.pop(group_id, None)

    def _mark_dirty(self, group_id: str, data: GroupStateData) -> None:
        data.dirty = True
        self._dirty_groups.add(group_id)

    def _ensure_group(self, group_id: str) -> GroupStateData:
        if group_id not in self._groups:
            self._groups[group_id] = GroupStateData(
                last_sample_time=time.time(),
            )
            self._group_ids_dirty = True
        return self._groups[group_id]

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def get_state(self, group_id: str) -> GroupStateData:
        data = self._ensure_group(group_id)
        now = time.time()
        if data.state == GroupState.COOLDOWN and now >= data.cooldown_until:
            data.state = GroupState.IDLE
            self._mark_dirty(group_id, data)
        if data.anchor.participants or data.anchor.keywords:
            self._cleanup_expired_anchor(group_id, data, now)
        today = self._today()
        if data.initiate_count_date != today:
            data.initiate_count_date = today
            data.initiate_daily_count = 0
            data.initiate_no_reply_streak = 0
            self._mark_dirty(group_id, data)
        return data

    def is_muted(self) -> bool:
        now = time.localtime()
        current_mins = now.tm_hour * 60 + now.tm_min
        start_hour, start_minute, end_hour, end_minute = self._config.mute_period
        start_mins = start_hour * 60 + start_minute
        end_mins = end_hour * 60 + end_minute
        if start_mins <= end_mins:
            return start_mins <= current_mins < end_mins
        return current_mins >= start_mins or current_mins < end_mins

    def set_cooldown(self, group_id: str, minutes: int) -> int:
        data = self._ensure_group(group_id)
        minutes = max(1, min(self.MAX_COOLDOWN_MINUTES, minutes))
        data.state = GroupState.COOLDOWN
        data.cooldown_until = time.time() + minutes * 60
        self._mark_dirty(group_id, data)
        return minutes

    # ---- 对话锚点 ----

    def write_anchor(
        self,
        group_id: str,
        *,
        kind: str,
        topic: str = "",
        bot_message: str = "",
        users: list[str] | None = None,
        keywords: list[str] | None = None,
        reason: str = "",
        ttl_minutes: float | None = None,
    ) -> None:
        """发言后整体写入新锚点（替换旧锚点）。"""
        data = self._ensure_group(group_id)
        anchor = ThreadAnchor(
            kind=kind,
            topic=topic,
            bot_message=bot_message,
            reason=reason,
            created_at=time.time(),
        )
        data.anchor = anchor
        self._fill_anchor_watch(group_id, data, users, keywords, ttl_minutes)
        self._mark_dirty(group_id, data)

    def add_anchor_watch(
        self,
        group_id: str,
        users: list[str] | None = None,
        keywords: list[str] | None = None,
        reason: str = "",
        ttl_minutes: float | None = None,
    ) -> None:
        """不发言时向现有锚点合并关注对象。"""
        data = self._ensure_group(group_id)
        if not data.anchor.created_at:
            data.anchor.created_at = time.time()
        self._fill_anchor_watch(group_id, data, users, keywords, ttl_minutes)
        if reason:
            data.anchor.reason = reason
        self._mark_dirty(group_id, data)

    def _fill_anchor_watch(
        self,
        group_id: str,
        data: GroupStateData,
        users: list[str] | None,
        keywords: list[str] | None,
        ttl_minutes: float | None,
    ) -> None:
        now = time.time()
        ttl = (ttl_minutes if ttl_minutes is not None else self._config.follow_up_ttl) * 60
        anchor = data.anchor
        if users:
            for uid in users:
                if uid in anchor.participants or len(anchor.participants) < self.MAX_ANCHOR_USERS:
                    anchor.participants.add(uid)
                    anchor.participant_ttls[uid] = now + ttl
        if keywords:
            for kw in keywords:
                if kw in anchor.keywords or len(anchor.keywords) < self.MAX_ANCHOR_KEYWORDS:
                    anchor.keywords.add(kw)
                    anchor.keyword_ttls[kw] = now + ttl

    def remove_anchor_watch(
        self,
        group_id: str,
        user_ids: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        data = self._ensure_group(group_id)
        anchor = data.anchor
        if user_ids:
            for uid in user_ids:
                anchor.participants.discard(uid)
                anchor.participant_ttls.pop(uid, None)
        if keywords:
            for kw in keywords:
                anchor.keywords.discard(kw)
                anchor.keyword_ttls.pop(kw, None)
        if not user_ids and not keywords:
            anchor.clear()
        self._mark_dirty(group_id, data)

    def close_anchor(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.anchor.clear()
        self._mark_dirty(group_id, data)

    def _cleanup_expired_anchor(self, group_id: str, data: GroupStateData, now: float) -> None:
        anchor = data.anchor
        expired_users = [uid for uid, ttl in anchor.participant_ttls.items() if now >= ttl]
        for uid in expired_users:
            anchor.participants.discard(uid)
            anchor.participant_ttls.pop(uid, None)
        expired_keywords = [kw for kw, ttl in anchor.keyword_ttls.items() if now >= ttl]
        for kw in expired_keywords:
            anchor.keywords.discard(kw)
            anchor.keyword_ttls.pop(kw, None)
        if expired_users or expired_keywords:
            self._mark_dirty(group_id, data)

    def match_anchor_user(self, group_id: str, sender_id: str) -> bool:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        return sender_id in data.anchor.participants

    def match_anchor_keyword(self, group_id: str, text: str) -> list[str]:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return []
        matched = []
        text_lower = text.lower()
        for kw in data.anchor.keywords:
            if kw.lower() in text_lower:
                matched.append(kw)
        return matched

    def get_anchor(self, group_id: str) -> ThreadAnchor:
        return self.get_state(group_id).anchor

    def set_observation(self, group_id: str, observation: str) -> None:
        if not observation:
            return
        data = self._ensure_group(group_id)
        data.last_observation = observation
        self._mark_dirty(group_id, data)

    def get_observation(self, group_id: str) -> str:
        return self._ensure_group(group_id).last_observation

    def record_drift(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.last_drift_time = time.time()
        self._mark_dirty(group_id, data)

    # ---- 主动发起 ----

    def is_initiate_pending(self, group_id: str) -> bool:
        return self._ensure_group(group_id).initiate_pending_since > 0

    def consume_initiate_pending(self, group_id: str) -> bool:
        """群友在发起后发言（接话）：清除 pending 并重置无人接话计数。"""
        data = self._ensure_group(group_id)
        if data.initiate_pending_since <= 0:
            return False
        data.initiate_pending_since = 0.0
        data.initiate_no_reply_streak = 0
        self._mark_dirty(group_id, data)
        return True

    def record_initiate(
        self,
        group_id: str,
        *,
        topic: str = "",
        bot_message: str = "",
        users: list[str] | None = None,
        keywords: list[str] | None = None,
        reason: str = "",
    ) -> None:
        """主动发起成功后记账：次数、时间、接话 pending，并写入发起锚点。"""
        data = self.get_state(group_id)  # get_state 处理跨天重置
        now = time.time()
        data.initiate_daily_count += 1
        data.last_initiate_time = now
        data.initiate_pending_since = now
        self.write_anchor(
            group_id,
            kind="initiate",
            topic=topic,
            bot_message=bot_message,
            users=users,
            keywords=keywords,
            reason=reason,
        )
        self._mark_dirty(group_id, data)

    def record_initiate_unanswered(self, group_id: str) -> None:
        """发起后超时无人接话：清除 pending 并累计 streak。"""
        data = self.get_state(group_id)
        if data.initiate_pending_since <= 0:
            return
        data.initiate_pending_since = 0.0
        data.initiate_no_reply_streak += 1
        self._mark_dirty(group_id, data)

    # ---- 采样与自适应阈值 ----

    def increment_msg_count(self, group_id: str) -> int:
        data = self._ensure_group(group_id)
        data.msg_count += 1
        self._mark_dirty(group_id, data)
        return data.msg_count

    def _current_boost(self, data: GroupStateData) -> float:
        now = time.time()
        if now >= data.boost_until or data.boost_initial >= 1.0:
            return 1.0
        elapsed = now - data.boost_set_at
        total = data.boost_until - data.boost_set_at
        if total <= 0:
            return 1.0
        progress = min(1.0, elapsed / total)
        return data.boost_initial + (1.0 - data.boost_initial) * progress

    def _decay_backoff(self, group_id: str, data: GroupStateData, now: float) -> None:
        if data.backoff_level <= 0 or data.last_backoff_time <= 0:
            return
        elapsed = now - data.last_backoff_time
        levels_to_decay = int(elapsed / self.BACKOFF_DECAY_INTERVAL)
        if levels_to_decay > 0:
            data.backoff_level = max(0, data.backoff_level - levels_to_decay)
            data.last_backoff_time = now
            self._mark_dirty(group_id, data)

    def get_effective_thresholds(self, group_id: str) -> tuple[int, int]:
        data = self._ensure_group(group_id)
        now = time.time()
        self._decay_backoff(group_id, data, now)
        backoff_factor = BACKOFF_BASE ** data.backoff_level
        boost = self._current_boost(data)
        combined = backoff_factor * boost
        w_adj = WILLINGNESS_THRESHOLD_ADJUST.get(
            data.willingness, WILLINGNESS_THRESHOLD_ADJUST[DEFAULT_LEVEL]
        )
        effective_n = min(
            max(5, int(self._config.default_n * combined * w_adj["n_factor"])),
            self.N_MAX,
        )
        effective_t = min(
            max(5, int(self._config.default_t * combined * w_adj["t_factor"])),
            self.T_MAX,
        )
        return effective_n, effective_t

    def should_trigger_sampling(self, group_id: str) -> bool:
        data = self.get_state(group_id)
        if data.state == GroupState.COOLDOWN:
            return False
        if self.is_muted():
            return False
        effective_n, effective_t = self.get_effective_thresholds(group_id)
        if data.msg_count >= effective_n:
            return True
        elapsed = time.time() - data.last_sample_time
        if elapsed >= effective_t * 60 and data.msg_count > 0:
            return True
        return False

    def reset_sampling(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.msg_count = 0
        data.last_sample_time = time.time()
        self._mark_dirty(group_id, data)

    def record_skip_reply(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        if data.backoff_level < MAX_BACKOFF_LEVEL:
            data.backoff_level += 1
        if data.last_backoff_time <= 0:
            data.last_backoff_time = time.time()
        data.consecutive_replies = 0
        self._mark_dirty(group_id, data)

    def record_actual_reply(self, group_id: str, *, count_consecutive: bool = True) -> None:
        data = self._ensure_group(group_id)
        if data.consecutive_replies == 0:
            data.backoff_level = max(0, data.backoff_level - 1)
        if count_consecutive:
            data.consecutive_replies += 1

        max_br = self._config.max_boosted_replies
        now = time.time()

        if data.consecutive_replies <= max_br:
            data.boost_initial = self._config.boost_factor
            data.boost_set_at = now
            data.boost_until = now + self._config.boost_duration * 60
        else:
            data.boost_initial = 1.0
            fatigue = min(data.consecutive_replies - max_br, MAX_BACKOFF_LEVEL)
            data.backoff_level = max(data.backoff_level, fatigue)
            data.last_backoff_time = now

        self._mark_dirty(group_id, data)

    def can_detect(self, group_id: str, *, follow_up: bool = False) -> bool:
        data = self._ensure_group(group_id)
        now = time.time()
        min_interval = float(self._config.trigger_min_interval)
        if follow_up:
            min_interval *= 0.5
        return (now - data.last_detect_time) >= min_interval

    def record_detect_time(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.last_detect_time = time.time()

    def reset_group(self, group_id: str) -> None:
        data = self._ensure_group(group_id)
        data.msg_count = 0
        data.last_sample_time = time.time()
        data.backoff_level = 0
        data.last_backoff_time = 0.0
        data.consecutive_replies = 0
        data.boost_initial = 1.0
        data.boost_set_at = 0.0
        data.boost_until = 0.0
        data.last_detect_time = 0.0
        data.anchor.clear()
        data.last_observation = ""
        data.last_drift_time = 0.0
        data.last_initiate_time = 0.0
        data.initiate_daily_count = 0
        data.initiate_count_date = self._today()
        data.initiate_pending_since = 0.0
        data.initiate_no_reply_streak = 0
        data.state = GroupState.IDLE
        self._mark_dirty(group_id, data)

    # ---- 持久化 ----

    async def save_dirty(self, save_fn) -> None:
        async with self._global_lock:
            if self._whitelist_dirty:
                self._whitelist_dirty = False
                try:
                    await save_fn("iris_reply:whitelist", list(self._whitelist))
                except Exception as e:
                    self._whitelist_dirty = True
                    logger.warning("Iris Reply: whitelist KV save failed: %s", e)
            if self._group_ids_dirty:
                self._group_ids_dirty = False
                try:
                    await save_fn(self._GROUP_IDS_KEY, list(self._groups.keys()))
                except Exception as e:
                    self._group_ids_dirty = True
                    logger.warning("Iris Reply: group manifest KV save failed: %s", e)
            dirty_snapshot = list(self._dirty_groups)
            self._dirty_groups.clear()
            for gid in dirty_snapshot:
                data = self._groups.get(gid)
                if data:
                    data.dirty = False
            for gid in dirty_snapshot:
                data = self._groups.get(gid)
                if not data:
                    continue
                snapshot = self._serialize_group(data)
                try:
                    await save_fn(f"{self._GROUP_KEY_PREFIX}{gid}", snapshot)
                except Exception as e:
                    data.dirty = True
                    self._dirty_groups.add(gid)
                    logger.warning("Iris Reply: KV save failed for group %s: %s", gid, e)

    def get_willingness(self, group_id: str) -> str:
        data = self._ensure_group(group_id)
        if data.willingness not in VALID_LEVELS:
            data.willingness = DEFAULT_LEVEL
        return data.willingness

    def set_willingness(self, group_id: str, level: str) -> None:
        if level not in VALID_LEVELS:
            return
        data = self._ensure_group(group_id)
        data.willingness = level
        self._mark_dirty(group_id, data)

    def is_whitelisted(self, group_id: str) -> bool:
        return group_id in self._whitelist

    def add_to_whitelist(self, group_id: str) -> None:
        self._whitelist.add(group_id)
        self._whitelist_dirty = True

    def remove_from_whitelist(self, group_id: str) -> None:
        self._whitelist.discard(group_id)
        self._whitelist_dirty = True

    def get_whitelist(self) -> set[str]:
        return set(self._whitelist)

    async def load_all(self, load_fn) -> None:
        try:
            wl_data = await load_fn("iris_reply:whitelist")
            if wl_data and isinstance(wl_data, list):
                self._whitelist = set(str(g) for g in wl_data)
        except Exception as e:
            logger.warning("Iris Reply: whitelist KV load failed: %s", e)

        try:
            group_ids = await load_fn(self._GROUP_IDS_KEY)
            if group_ids and isinstance(group_ids, list):
                for gid in group_ids:
                    gdata = await load_fn(f"{self._GROUP_KEY_PREFIX}{str(gid)}")
                    if gdata and isinstance(gdata, dict):
                        self._groups[str(gid)] = self._deserialize_group(gdata)
        except Exception as e:
            logger.warning("Iris Reply: per-group KV load failed, running in memory-only mode: %s", e)

    async def save_all(self, save_fn) -> None:
        async with self._global_lock:
            try:
                await save_fn("iris_reply:whitelist", list(self._whitelist))
                self._whitelist_dirty = False
            except Exception as e:
                logger.warning("Iris Reply: whitelist KV save failed: %s", e)
            all_gids = list(self._groups.keys())
            self._dirty_groups.clear()
            for gid, data in self._groups.items():
                data.dirty = False
            for gid in all_gids:
                data = self._groups.get(gid)
                if not data:
                    continue
                snapshot = self._serialize_group(data)
                try:
                    await save_fn(f"{self._GROUP_KEY_PREFIX}{gid}", snapshot)
                except Exception as e:
                    data.dirty = True
                    self._dirty_groups.add(gid)
                    logger.warning("Iris Reply: KV save failed for group %s: %s", gid, e)
            try:
                await save_fn(self._GROUP_IDS_KEY, list(self._groups.keys()))
                self._group_ids_dirty = False
            except Exception as e:
                self._group_ids_dirty = True
                logger.warning("Iris Reply: group manifest KV save failed: %s", e)

    def _serialize_group(self, data: GroupStateData) -> dict[str, Any]:
        anchor = data.anchor
        return {
            "state": data.state.value,
            "cooldown_until": data.cooldown_until,
            "msg_count": data.msg_count,
            "last_sample_time": data.last_sample_time,
            "backoff_level": data.backoff_level,
            "last_backoff_time": data.last_backoff_time,
            "consecutive_replies": data.consecutive_replies,
            "willingness": data.willingness,
            "boost_initial": data.boost_initial,
            "boost_set_at": data.boost_set_at,
            "boost_until": data.boost_until,
            "last_detect_time": data.last_detect_time,
            "anchor": {
                "kind": anchor.kind,
                "topic": anchor.topic,
                "bot_message": anchor.bot_message,
                "participants": list(anchor.participants),
                "participant_ttls": anchor.participant_ttls,
                "keywords": list(anchor.keywords),
                "keyword_ttls": anchor.keyword_ttls,
                "reason": anchor.reason,
                "created_at": anchor.created_at,
            },
            "last_observation": data.last_observation,
            "last_drift_time": data.last_drift_time,
            "last_initiate_time": data.last_initiate_time,
            "initiate_daily_count": data.initiate_daily_count,
            "initiate_count_date": data.initiate_count_date,
            "initiate_pending_since": data.initiate_pending_since,
            "initiate_no_reply_streak": data.initiate_no_reply_streak,
        }

    def _deserialize_group(self, d: dict[str, Any]) -> GroupStateData:
        a = d.get("anchor", {})
        anchor = ThreadAnchor(
            kind=a.get("kind", ""),
            topic=a.get("topic", ""),
            bot_message=a.get("bot_message", ""),
            participants=set(a.get("participants", [])),
            participant_ttls=a.get("participant_ttls", {}),
            keywords=set(a.get("keywords", [])),
            keyword_ttls=a.get("keyword_ttls", {}),
            reason=a.get("reason", ""),
            created_at=a.get("created_at", 0.0),
        )
        willingness = d.get("willingness", DEFAULT_LEVEL)
        if willingness not in VALID_LEVELS:
            willingness = DEFAULT_LEVEL
        backoff_level = min(d.get("backoff_level", 0), MAX_BACKOFF_LEVEL)
        last_backoff_time = d.get("last_backoff_time", 0.0)
        if backoff_level > 0 and last_backoff_time <= 0:
            last_backoff_time = time.time()
        state = GroupState.IDLE
        if d.get("state") == GroupState.COOLDOWN.value:
            state = GroupState.COOLDOWN
        return GroupStateData(
            state=state,
            cooldown_until=d.get("cooldown_until", 0.0),
            msg_count=d.get("msg_count", 0),
            last_sample_time=d.get("last_sample_time", time.time()),
            backoff_level=backoff_level,
            last_backoff_time=last_backoff_time,
            consecutive_replies=d.get("consecutive_replies", 0),
            willingness=willingness,
            boost_initial=d.get("boost_initial", 1.0),
            boost_set_at=d.get("boost_set_at", 0.0),
            boost_until=d.get("boost_until", 0.0),
            last_detect_time=d.get("last_detect_time", 0.0),
            anchor=anchor,
            last_observation=d.get("last_observation", ""),
            last_drift_time=d.get("last_drift_time", 0.0),
            last_initiate_time=d.get("last_initiate_time", 0.0),
            initiate_daily_count=d.get("initiate_daily_count", 0),
            initiate_count_date=d.get("initiate_count_date", ""),
            initiate_pending_since=d.get("initiate_pending_since", 0.0),
            initiate_no_reply_streak=d.get("initiate_no_reply_streak", 0),
        )

    def get_status_text(self, group_id: str) -> str:
        data = self.get_state(group_id)
        effective_n, effective_t = self.get_effective_thresholds(group_id)
        backoff_factor = BACKOFF_BASE ** data.backoff_level
        current_boost = self._current_boost(data)
        lines = [
            f"群 {group_id} 状态:",
            f"  状态机: {data.state.value}",
            f"  回复意愿: {display_level(data.willingness)}",
            f"  消息计数: {data.msg_count}/{effective_n}",
            f"  退避等级: {data.backoff_level} (×{backoff_factor:.2f})",
            f"  频率提升: ×{current_boost:.2f}" + (f" (初始×{data.boost_initial:.2f})" if current_boost < 1.0 else ""),
            f"  连续回复: {data.consecutive_replies}",
            f"  有效阈值: N={effective_n}, T={effective_t}分钟",
        ]
        if data.state == GroupState.COOLDOWN:
            remaining = max(0, data.cooldown_until - time.time())
            lines.append(f"  冷却剩余: {remaining / 60:.1f} 分钟")
        if current_boost < 1.0:
            remaining = max(0, data.boost_until - time.time())
            lines.append(f"  Boost 剩余: {remaining / 60:.1f} 分钟")
        anchor = data.anchor
        if anchor.has_context:
            kind_text = anchor.kind or "-"
            lines.append(f"  锚点类型: {kind_text}")
        if anchor.topic:
            lines.append(f"  锚点话题: {anchor.topic}")
        if anchor.participants:
            lines.append(f"  锚点用户: {', '.join(sorted(anchor.participants))}")
        if anchor.keywords:
            lines.append(f"  锚点关键词: {', '.join(sorted(anchor.keywords))}")
        if anchor.reason:
            lines.append(f"  锚点原因: {anchor.reason}")
        lines.append(f"  今日发起: {data.initiate_daily_count} 次")
        if data.initiate_pending_since > 0:
            waiting = (time.time() - data.initiate_pending_since) / 60
            lines.append(f"  发起接话等待中: {waiting:.1f} 分钟")
        if data.initiate_no_reply_streak > 0:
            lines.append(f"  无人接话 streak: {data.initiate_no_reply_streak}")
        return "\n".join(lines)
