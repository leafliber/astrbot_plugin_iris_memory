from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .parser import Decision


@dataclass
class LLMCallLog:
    group_id: str
    motive: str
    system_prompt: str
    user_prompt: str
    response_text: str
    action: str
    message: str
    observation: str
    watch_users: list[str]
    watch_keywords: list[str]
    watch_reason: str
    drifted: bool
    timestamp: float
    duration_ms: float = 0.0


@dataclass
class GroupStats:
    group_id: str
    total_decisions: int = 0
    total_replies: int = 0
    total_skips: int = 0
    total_errors: int = 0
    total_drifts: int = 0
    total_initiates: int = 0
    total_passive_replies: int = 0
    last_decision_time: float = 0.0
    last_motive: str = ""
    last_reply_time: float = 0.0
    current_state: str = "idle"
    willingness: str = "medium"
    msg_count: int = 0
    effective_n: int = 0
    effective_t: int = 0
    backoff_level: int = 0
    consecutive_replies: int = 0
    initiate_daily_count: int = 0


MAX_LOG_ENTRIES = 500


class StatsCollector:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._llm_logs: deque[LLMCallLog] = deque(maxlen=MAX_LOG_ENTRIES)
        self._group_stats: dict[str, GroupStats] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _ensure_group(self, group_id: str) -> GroupStats:
        if group_id not in self._group_stats:
            self._group_stats[group_id] = GroupStats(group_id=group_id)
        return self._group_stats[group_id]

    def record_decision(
        self,
        group_id: str,
        motive: str,
        *,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        decision: Decision,
        duration_ms: float,
    ) -> None:
        if not self._enabled:
            return
        log = LLMCallLog(
            group_id=group_id,
            motive=motive,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            action=decision.action,
            message=decision.message,
            observation=decision.observation,
            watch_users=decision.watch,
            watch_keywords=decision.watch_keywords,
            watch_reason=decision.why,
            drifted=decision.drifted,
            timestamp=time.time(),
            duration_ms=duration_ms,
        )
        self._llm_logs.append(log)

        gs = self._ensure_group(group_id)
        gs.total_decisions += 1
        gs.last_decision_time = time.time()
        gs.last_motive = motive
        if decision.drifted:
            gs.total_drifts += 1
        elif decision.should_speak:
            gs.total_replies += 1
            gs.last_reply_time = time.time()
            if motive == "initiate":
                gs.total_initiates += 1
        else:
            gs.total_skips += 1

    def record_decision_error(self, group_id: str, motive: str) -> None:
        if not self._enabled:
            return
        gs = self._ensure_group(group_id)
        gs.total_decisions += 1
        gs.total_errors += 1
        gs.last_decision_time = time.time()
        gs.last_motive = motive

    def record_passive_reply(self, group_id: str) -> None:
        if not self._enabled:
            return
        gs = self._ensure_group(group_id)
        gs.total_passive_replies += 1
        gs.last_reply_time = time.time()

    def update_group_state(
        self,
        group_id: str,
        state: str,
        willingness: str,
        msg_count: int,
        effective_n: int,
        effective_t: int,
        backoff_level: int,
        consecutive_replies: int,
        initiate_daily_count: int,
    ) -> None:
        if not self._enabled:
            return
        gs = self._ensure_group(group_id)
        gs.current_state = state
        gs.willingness = willingness
        gs.msg_count = msg_count
        gs.effective_n = effective_n
        gs.effective_t = effective_t
        gs.backoff_level = backoff_level
        gs.consecutive_replies = consecutive_replies
        gs.initiate_daily_count = initiate_daily_count

    @staticmethod
    def _summary(gs: GroupStats) -> dict[str, Any]:
        return {
            "group_id": gs.group_id,
            "total_decisions": gs.total_decisions,
            "total_replies": gs.total_replies,
            "total_skips": gs.total_skips,
            "total_errors": gs.total_errors,
            "total_drifts": gs.total_drifts,
            "total_initiates": gs.total_initiates,
            "total_passive_replies": gs.total_passive_replies,
            "last_decision_time": gs.last_decision_time,
            "last_motive": gs.last_motive,
            "last_reply_time": gs.last_reply_time,
            "current_state": gs.current_state,
            "willingness": gs.willingness,
            "msg_count": gs.msg_count,
            "effective_n": gs.effective_n,
            "effective_t": gs.effective_t,
            "backoff_level": gs.backoff_level,
            "consecutive_replies": gs.consecutive_replies,
            "initiate_daily_count": gs.initiate_daily_count,
        }

    def get_group_summaries(self) -> list[dict[str, Any]]:
        result = [self._summary(gs) for gs in self._group_stats.values()]
        result.sort(key=lambda x: x["last_decision_time"], reverse=True)
        return result

    def get_llm_logs(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        logs = list(self._llm_logs)
        if group_id:
            logs = [l for l in logs if l.group_id == group_id]
        logs = logs[::-1]
        sliced = logs[offset:offset + limit]
        result = []
        for log in sliced:
            result.append({
                "group_id": log.group_id,
                "motive": log.motive,
                "system_prompt": log.system_prompt,
                "user_prompt": log.user_prompt,
                "response_text": log.response_text,
                "action": log.action,
                "message": log.message,
                "observation": log.observation,
                "watch_users": log.watch_users,
                "watch_keywords": log.watch_keywords,
                "watch_reason": log.watch_reason,
                "drifted": log.drifted,
                "timestamp": log.timestamp,
                "duration_ms": round(log.duration_ms, 1),
            })
        return result

    def get_group_detail(self, group_id: str) -> dict[str, Any] | None:
        gs = self._group_stats.get(group_id)
        if not gs:
            return None
        return self._summary(gs)

    def clear_logs(self) -> None:
        self._llm_logs.clear()
        self._group_stats.clear()
