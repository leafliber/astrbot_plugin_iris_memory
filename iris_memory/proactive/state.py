"""Session settings and bounded counters; never store credentials or model traces."""

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .policy import MOTIVES


@dataclass
class Session:
    origin: str
    entry_id: str
    kind: str
    enabled: bool = False
    willingness: str = "medium"
    quiet_start: str = "01:00"
    quiet_end: str = "07:00"
    cooldown_until: float = 0
    last_human: float = 0
    last_bot: float = 0
    last_decision: float = 0
    last_watch: float = 0
    day: str = ""
    calls_today: int = 0
    calls: int = 0
    sends: int = 0
    calls_by_motive: dict = field(default_factory=dict)
    sends_by_motive: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    initiate_times: list[float] = field(default_factory=list)
    no_reply_streak: int = 0
    # Generation invalidates a prepared candidate on ANY intervening activity/control.
    generation: int = 0
    anchors: list[dict] = field(default_factory=list)
    pending_at: float = 0
    next_check: float = 0
    was_muted: bool = False
    checked_at: float = 0
    last_result: str = "idle"

    def roll(self, now: float) -> None:
        day = datetime.fromtimestamp(now).date().isoformat()
        if day != self.day:
            self.day, self.calls_today = day, 0
        self.initiate_times = [t for t in self.initiate_times if now - 86400 < t <= now]
        self.anchors = [a for a in self.anchors if a["expires_at"] > now]

    def outcome(self, code: str) -> str:
        self.last_result = code
        self.outcomes[code] = self.outcomes.get(code, 0) + 1
        return code

    def reserve_call(self, motive: str) -> None:
        self.calls_today += 1
        self.calls += 1
        self.calls_by_motive[motive] = self.calls_by_motive.get(motive, 0) + 1

    def sent(self, motive: str, now: float) -> None:
        self.sends += 1
        self.sends_by_motive[motive] = self.sends_by_motive.get(motive, 0) + 1
        self.last_bot = now
        self.generation += 1
        self.pending_at = 0
        if motive == "initiate":
            self.initiate_times.append(now)
            self.no_reply_streak += 1

    def reset(self, now: float) -> None:
        # Explicit administrator operation, including daily and rolling quotas.
        self.calls_today = self.calls = self.sends = self.no_reply_streak = 0
        self.calls_by_motive.clear()
        self.sends_by_motive.clear()
        self.outcomes.clear()
        self.anchors.clear()
        self.initiate_times.clear()
        self.cooldown_until = self.last_decision = self.last_watch = self.pending_at = 0
        self.next_check = now
        self.generation += 1
        self.day = datetime.fromtimestamp(now).date().isoformat()
        self.last_result = "reset"

    def view(self, now: float, config, live: bool) -> dict:
        self.roll(now)
        return {
            "origin": self.origin,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "enabled": self.enabled,
            "live": live,
            "willingness": self.willingness,
            "quiet_start": self.quiet_start,
            "quiet_end": self.quiet_end,
            "cooldown_minutes": max(0, int((self.cooldown_until - now + 59) // 60)),
            "pending_at": self.pending_at or None,
            "last_result": self.last_result,
            "anchor_count": len(self.anchors),
            "day": self.day,
            "calls_today": self.calls_today,
            "daily_limit": config.proactive_daily_calls,
            "calls": self.calls,
            "sends": self.sends,
            "calls_by_motive": {m: self.calls_by_motive.get(m, 0) for m in MOTIVES},
            "sends_by_motive": {m: self.sends_by_motive.get(m, 0) for m in MOTIVES},
            "outcomes": dict(self.outcomes),
            "initiates_24h": len(self.initiate_times),
            "initiate_limit": config.proactive_initiate_limit,
        }


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS sessions (origin TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )

    def load(self) -> dict[str, Session]:
        return {
            origin: Session(**json.loads(payload))
            for origin, payload in self.db.execute("SELECT origin,payload FROM sessions")
        }

    def save(self, state: Session) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?)",
                (state.origin, json.dumps(asdict(state), ensure_ascii=False)),
            )

    def close(self) -> None:
        self.db.close()
