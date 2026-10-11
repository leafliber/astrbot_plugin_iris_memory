"""Local policy → one Iris prepare → one AstrBot decision → confirmed output.

The five AstrBot lifecycle hooks live here; no alternate memory store or model loop.
"""

import asyncio
import random
import time
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from ..config import DEFAULT_TEMPLATE
from ..extras import strip_markdown
from ..mapping import component_kind, map_entry, stable_key
from ..prompt import encode, render
from .decision import SYSTEM, parse
from .policy import (
    DISPLAY,
    LEVELS,
    crossed_quiet,
    message_motive,
    minute_of_day,
    muted,
    related_goals,
)
from .state import Session, Store


class Proactive:
    def __init__(self, plugin, *, clock=time.time, rng=None):
        self.plugin = plugin
        self.bridge = plugin.bridge
        self.config = plugin.settings
        self.clock = clock
        self.rng = rng or random.Random()
        namespace = stable_key(self.config.base_url.rstrip("/"), self.config.token)
        self.store = Store(self.bridge.data_dir / f"proactive-{namespace}.sqlite3")
        self.sessions = self.store.load()
        self.live = {}  # Entry + latest receive receipt; rebuilt after restart.
        self.goals = {}
        self.goal_expiry = {}
        self.locks = {}
        self.tasks = set()
        self.inflight = set()
        self.running = set()
        self.normal_until = {}
        self.scheduler = None
        self.closed = False
        for state in self.sessions.values():
            state.pending_at = 0  # Old appointments never survive a restart.
            state.generation += 1

    def bind(self, event, *, replace_receipt=False) -> Session:
        entry = map_entry(event, self.config)
        state = self.sessions.get(entry.origin)
        if state is None:
            state = Session(
                entry.origin,
                entry.id,
                entry.kind,
                willingness=self.config.proactive_willingness,
                quiet_start=self.config.proactive_quiet_start,
                quiet_end=self.config.proactive_quiet_end,
            )
            self.sessions[entry.origin] = state
        elif state.entry_id != entry.id:
            state.entry_id = entry.id
            state.anchors.clear()
            state.generation += 1
            self.goals.pop(entry.origin, None)
            self.goal_expiry.pop(entry.origin, None)
        if (
            replace_receipt
            or entry.origin not in self.live
            or self.live[entry.origin][0].id != entry.id
        ):
            self.live[entry.origin] = (entry, event.get_extra("iris_received"))
        self.locks.setdefault(entry.origin, asyncio.Lock())
        return state

    async def start(self) -> None:
        if self.scheduler is None and not self.closed:
            self.scheduler = asyncio.create_task(self._schedule())

    def blocked(self, state, now, *, motive=None, check_interval=True, check_calls=True) -> str:
        state.roll(now)
        if self.closed or not self.config.proactive_enabled or not state.enabled:
            return "disabled"
        if state.origin not in self.live:
            return "awaiting_message"
        if self.normal_until.get(state.origin, 0) > now:
            return "normal_reply"
        if muted(state, now):
            return "quiet_hours"
        if state.cooldown_until > now:
            return "cooldown"
        if check_calls and state.calls_today >= self.config.proactive_daily_calls:
            return "daily_limit"
        if (
            check_interval
            and state.last_decision
            and now - state.last_decision < self.config.proactive_min_interval_seconds
        ):
            return "decision_interval"
        if motive == "initiate":
            if state.kind == "private" and not self.config.proactive_private_care_enabled:
                return "private_care_disabled"
            if len(state.initiate_times) >= self.config.proactive_initiate_limit:
                return "rolling_limit"
            if (
                state.initiate_times
                and now - state.initiate_times[-1]
                < self.config.proactive_initiate_interval_minutes * 60
            ):
                return "initiate_interval"
            if state.no_reply_streak >= 2:
                return "unanswered_limit"
        return ""

    def invalidate_goals(self, event) -> None:
        origin = event.unified_msg_origin
        self.goal_expiry.pop(origin, None)
        self.goals.pop(origin, None)
        state = self.sessions.get(origin)
        if state:
            state.anchors.clear()
            state.generation += 1
            self.store.save(state)

    async def refresh_goals(self, state) -> None:
        now = self.clock()
        if self.goal_expiry.get(state.origin, 0) > now:
            return
        goals = []
        try:
            async with asyncio.timeout(self.config.tool_timeout):
                for offset in range(0, 1000, 100):
                    page = await self.bridge.client.list_goals(
                        {
                            "entry_id": state.entry_id,
                            "state": "open",
                            "limit": 100,
                            "offset": offset,
                        }
                    )
                    goals.extend(
                        g
                        for g in page.get("items", [])
                        if g.get("entry_id") == state.entry_id and g.get("state", "open") == "open"
                    )
                    if offset + 100 >= page.get("total", 0):
                        break
        except Exception:
            # An expired cache must not resurrect a completed or unauthorized goal.
            goals = []
            self.bridge.record("proactive_goals_unavailable")
        self.goals[state.origin] = goals
        self.goal_expiry[state.origin] = now + (300 if goals else 30)
        ids = {g["id"] for g in goals}
        state.anchors = [a for a in state.anchors if a["goal_id"] in ids]

    def note(self, state, stage, code) -> None:
        self.bridge.diagnostics.proactive(state.origin, state.entry_id, stage, code)

    async def on_message(self, event) -> None:
        if self.closed or event.get_extra("iris_proactive_observed"):
            return
        event.set_extra("iris_proactive_observed", True)
        if str(event.get_sender_id()) == str(event.get_self_id()):
            return
        state = self.bind(event, replace_receipt=True)
        now = self.clock()
        state.last_human = now
        state.generation += 1
        state.pending_at = 0
        state.no_reply_streak = 0
        state.next_check = now + self.config.proactive_silence_minutes * 60
        self.store.save(state)
        reason = self.blocked(state, now)
        if reason:
            self.note(state, "message", reason)
            return
        text = event.get_message_str().strip()
        # Explicit replies, private incoming messages and commands stay in AstrBot's pipeline.
        if (
            state.kind == "private"
            or getattr(event, "is_at_or_wake_command", False)
            or getattr(event, "is_wake", False)
            or event.get_extra("provider_request")
            or text.startswith(("/", "iris_reply", "iris_mem"))
        ):
            self.note(state, "message", "explicit_reply")
            return
        generation = state.generation
        await self.refresh_goals(state)
        reason = self.blocked(state, self.clock())
        if generation != state.generation or reason:
            self.note(
                state, "message", "new_activity" if generation != state.generation else reason
            )
            return
        sender = str(event.get_sender_id())
        name = str(event.get_sender_name() or "")
        matches = related_goals(
            self.goals.get(state.origin, []), text, sender, name, state.anchors, now
        )
        named = any(
            alias.strip().casefold() in text.casefold()
            for alias in self.config.proactive_bot_aliases
        )
        named |= any(
            component_kind(c) == "at" and str(getattr(c, "qq", "")) == str(event.get_self_id())
            for c in event.get_messages()
        )
        motive = message_motive(
            self.config, state, text=text, named=named, matches=matches, now=now
        )
        self.note(state, "message", motive or "no_signal")
        if motive:
            self.launch(
                state,
                motive,
                {
                    "text": text[:3000],
                    "sender": sender,
                    "name": name[:100],
                    "matched_goals": matches,
                },
            )

    async def before_llm(self, event, request) -> None:
        state = self.bind(event)
        # Any ordinary provider request supersedes a pending proactive candidate.
        state.generation += 1
        state.pending_at = 0
        state.last_bot = self.clock()  # Reserve a quiet window while the normal reply runs.
        self.normal_until[state.origin] = self.clock() + self.config.proactive_decision_timeout
        self.store.save(state)

    async def after_sent(self, event) -> None:
        if not event.get_extra("iris_confirmed_send") or event.get_extra(
            "iris_proactive_after_sent"
        ):
            return
        event.set_extra("iris_proactive_after_sent", True)
        state = self.bind(event)
        self.normal_until.pop(state.origin, None)
        state.last_bot = self.clock()
        result = event.get_result()
        if result is not None and result.is_llm_result():
            state.cooldown_until = max(
                state.cooldown_until, self.clock() + self.config.proactive_cooldown_minutes * 60
            )
        state.generation += 1
        state.pending_at = 0
        self.store.save(state)

    def launch(self, state, motive, signal=None) -> None:
        if state.origin in self.running or self.closed:
            return
        self.running.add(state.origin)
        generation = state.generation

        async def run():
            try:
                await self.decide(state, motive, signal or {}, generation=generation)
            finally:
                self.running.discard(state.origin)

        task = asyncio.create_task(run())
        self.tasks.add(task)

        def done(task):
            self.tasks.discard(task)
            if not task.cancelled() and task.exception():
                self.bridge.record("proactive_task_failed")

        task.add_done_callback(done)

    async def decide(self, state, motive, signal=None, *, generation=None) -> str:
        lock = self.locks[state.origin]
        if lock.locked():
            self.note(state, "decision_gate", "busy")
            self.note(state, "outcome", "busy")
            return "busy"
        async with lock:
            task = asyncio.current_task()
            self.inflight.add(task)
            result = "failed"
            try:
                result = await self._decide(state, motive, signal or {}, generation)
                return result
            except asyncio.CancelledError:
                result = "cancelled"
                state.outcome("cancelled")
                raise
            except Exception:
                self.bridge.record("proactive_decision_failed")
                return state.outcome("failed")
            finally:
                self.note(state, "outcome", result)
                self.inflight.discard(task)
                self.store.save(state)

    async def _decide(self, state, motive, signal, generation) -> str:
        now = self.clock()
        generation = state.generation if generation is None else generation
        reason = self.blocked(state, now, motive=motive)
        self.note(
            state,
            "decision_gate",
            reason or ("new_activity" if generation != state.generation else "accepted"),
        )
        if reason or generation != state.generation:
            return state.outcome(reason or "new_activity")
        state.pending_at = 0
        state.next_check = (
            now
            + self.rng.uniform(
                self.config.proactive_delay_min_minutes, self.config.proactive_delay_max_minutes
            )
            * 60
        )
        await self.refresh_goals(state)
        entry, receipt = self.live[state.origin]
        async with asyncio.timeout(self.config.prepare_timeout):
            if receipt is None or not await asyncio.shield(receipt):
                return state.outcome("input_unavailable")
            prepared = await self.bridge.prepare_material(
                entry.id, self.bridge.prepare_options(), purpose="proactive"
            )
        # Use the full connector projection even when the passive template is customized.
        material = render(prepared, [], replace(self.config, prompt_template=DEFAULT_TEMPLATE))
        prompt = encode(
            {
                "motive": motive,
                "willingness": state.willingness,
                "kind": state.kind,
                "local_time": datetime.fromtimestamp(self.clock()).astimezone().isoformat(),
                "silent_minutes": max(0, int((now - max(state.last_human, state.last_bot)) / 60)),
                "signal": signal,
            }
        )
        async with asyncio.timeout(self.config.proactive_decision_timeout):
            provider = (
                self.config.proactive_provider_id
                or await self.plugin.context.get_current_chat_provider_id(state.origin)
            )
        if not provider:
            return state.outcome("provider_unavailable")
        reason = self.blocked(state, self.clock(), motive=motive)
        if reason or state.generation != generation:
            return state.outcome(reason or "new_activity")
        state.last_decision = self.clock()
        state.reserve_call(motive)  # Failed / timed out / malformed calls count too.
        self.note(state, "calls", motive)
        if motive == "watch":
            state.last_watch = self.clock()
        self.store.save(state)
        async with asyncio.timeout(self.config.proactive_decision_timeout):
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider,
                prompt=prompt,
                system_prompt=SYSTEM + "\n\n" + material.text,
                contexts=[],
                tools=None,
            )
        try:
            decision = parse(response.completion_text, self.config.proactive_max_message_chars)
        except (ValueError, TypeError, AttributeError):
            return state.outcome("invalid_decision")
        # Recheck immediately before applying observations or sending. No stale anchors either.
        now = self.clock()
        reason = self.blocked(state, now, motive=motive, check_interval=False, check_calls=False)
        if reason or generation != state.generation:
            return state.outcome(reason or "new_activity")
        own_ids = {g["id"] for g in self.goals.get(state.origin, [])}
        available = {g["id"] for g in prepared.get("goals", [])} & own_ids
        if motive == "follow_up" and not available.intersection(signal.get("matched_goals", [])):
            return state.outcome("goal_unavailable")
        state.anchors = [
            {
                "goal_id": goal_id,
                "users": [signal["sender"]] if signal.get("sender") else [],
                "expires_at": now + 600,
            }
            for goal_id in decision.watch_goal_ids
            if goal_id in available
        ][:10]
        if motive == "watch" or not decision.speak:
            state.last_watch = now
            if decision.cooldown_minutes:
                state.cooldown_until = max(
                    state.cooldown_until, now + decision.cooldown_minutes * 60
                )
            return state.outcome("watch")
        text = strip_markdown(decision.content) if self.config.strip_markdown else decision.content
        if not text.strip():
            return state.outcome("empty_content")
        # No await between the final guards above and invoking the platform send.
        async with asyncio.timeout(self.config.proactive_decision_timeout):
            accepted = await self.plugin._send_to_session(state.origin, text)
        if accepted is not True:
            return state.outcome("send_unconfirmed")
        self.bridge.submit_output(entry, text, "proactive:" + uuid4().hex)
        # Same opt-in approximation as ordinary PL-1 replies; never trust arbitrary model IDs.
        used = material.memory_ids
        if self.config.assume_injected_used and used and prepared.get("recall_id"):
            self.bridge.enqueue(lambda: self.bridge.client.feedback(prepared["recall_id"], used))
        now = self.clock()
        state.sent(motive, now)
        self.note(state, "sends", motive)
        state.cooldown_until = max(
            state.cooldown_until,
            now + max(self.config.proactive_cooldown_minutes, decision.cooldown_minutes) * 60,
        )
        return state.outcome("sent")

    async def tick(self) -> None:
        now = self.clock()
        for origin in list(self.live):
            state = self.sessions[origin]
            state.roll(now)
            if not state.enabled or not self.config.proactive_enabled:
                self.note(state, "scheduler", "disabled")
                continue
            crossed = crossed_quiet(state, state.checked_at, now)
            state.checked_at = now
            if muted(state, now):
                self.note(state, "scheduler", "quiet_hours")
                state.was_muted = True
                state.pending_at = 0
                self.store.save(state)
                continue
            if state.was_muted or crossed:
                self.note(state, "scheduler", "quiet_resume")
                state.was_muted = False
                state.pending_at = 0
                state.next_check = (
                    now
                    + self.rng.uniform(
                        self.config.proactive_delay_min_minutes,
                        self.config.proactive_delay_max_minutes,
                    )
                    * 60
                )
                self.store.save(state)
                continue
            automatic = (
                self.config.proactive_private_care_enabled
                if state.kind == "private"
                else self.config.proactive_initiate_enabled
            )
            reason = (
                "automatic_disabled"
                if not automatic
                else self.blocked(state, now, motive="initiate")
            )
            if reason:
                self.note(state, "scheduler", reason)
                continue
            if (
                now - max(state.last_human, state.last_bot)
                < self.config.proactive_silence_minutes * 60
            ):
                self.note(state, "scheduler", "not_silent")
                continue
            if origin in self.running or self.locks[origin].locked():
                self.note(state, "scheduler", "busy")
                continue
            if state.pending_at:
                self.note(state, "scheduler", "due" if now >= state.pending_at else "scheduled")
                if now >= state.pending_at:
                    self.launch(state, "initiate")
                continue
            if now < state.next_check:
                self.note(state, "scheduler", "waiting")
                continue
            state.next_check = now + self.config.proactive_check_interval_seconds
            factor = {"low": 0.5, "medium": 1.0, "high": 1.5}[state.willingness]
            probability = 1 - (1 - self.config.proactive_initiate_probability) ** factor
            reserved = self.rng.random() < probability
            self.note(state, "scheduler", "reserved" if reserved else "probability")
            if reserved:
                state.pending_at = (
                    now
                    + self.rng.uniform(
                        self.config.proactive_delay_min_minutes,
                        self.config.proactive_delay_max_minutes,
                    )
                    * 60
                )
            self.store.save(state)

    async def _schedule(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                self.bridge.record("proactive_scheduler_failed")
            # Lightweight local checks; probability evaluations still use configured intervals.
            await asyncio.sleep(min(30, self.config.proactive_check_interval_seconds))

    async def send_reminder(self, origin, text) -> bool:
        state = self.sessions.get(origin)
        if state is None:
            return False
        reason = self.blocked(state, self.clock(), check_calls=False, check_interval=False)
        if reason:
            self.note(state, "reminder", reason)
            return False
        lock = self.locks[origin]
        if lock.locked():
            self.note(state, "reminder", "busy")
            return False
        async with lock:
            accepted = await self.plugin._send_to_session(origin, text)
            if accepted is True:
                now = self.clock()
                state.sent("reminder", now)
                self.note(state, "sends", "reminder")
                self.note(state, "reminder", "sent")
                state.cooldown_until = max(
                    state.cooldown_until, now + self.config.proactive_cooldown_minutes * 60
                )
                self.store.save(state)
                return True
        self.note(state, "reminder", "send_unconfirmed")
        return False

    def update(self, origin, changes) -> dict:
        if origin not in self.sessions:
            raise ValueError("unknown session")
        if (
            not isinstance(changes, dict)
            or not changes
            or not set(changes)
            <= {"enabled", "willingness", "cooldown_minutes", "quiet_start", "quiet_end", "reset"}
        ):
            raise ValueError("invalid settings")
        # Validate the complete patch before any changes (including reset).
        for key, value in changes.items():
            if key in {"enabled", "reset"} and type(value) is not bool:
                raise ValueError("invalid switch")
            if key == "willingness" and (not isinstance(value, str) or value not in LEVELS):
                raise ValueError("invalid willingness")
            if key == "cooldown_minutes" and (type(value) is not int or not 0 <= value <= 120):
                raise ValueError("invalid cooldown")
            if key in {"quiet_start", "quiet_end"}:
                minute_of_day(value)
        state = self.sessions[origin]
        now = self.clock()
        if changes.get("reset"):
            state.reset(now)
            self.bridge.diagnostics.reset_session(origin)
        for key, value in changes.items():
            if key == "cooldown_minutes":
                state.cooldown_until = now + value * 60
            elif key == "willingness":
                state.willingness = LEVELS[value]
            elif key != "reset":
                setattr(state, key, value)
        state.pending_at = 0
        state.generation += 1
        state.checked_at = now
        state.next_check = (
            now
            + self.rng.uniform(
                self.config.proactive_delay_min_minutes, self.config.proactive_delay_max_minutes
            )
            * 60
        )
        self.store.save(state)
        return state.view(now, self.config, origin in self.live)

    def status_text(self, state) -> str:
        data = state.view(self.clock(), self.config, state.origin in self.live)
        return (
            f"主动回复：{'开启' if state.enabled else '关闭'}；总开关：{'开启' if self.config.proactive_enabled else '关闭'}\n"
            f"意愿：{DISPLAY[state.willingness]}；冷却：{data['cooldown_minutes']} 分钟；静音：{state.quiet_start}–{state.quiet_end}\n"
            f"今日决策：{data['calls_today']}/{data['daily_limit']}；累计调用：{state.calls}；主动发言：{state.sends}\n"
            f"滚动 24 小时发起：{data['initiates_24h']}/{data['initiate_limit']}；最近结果：{state.last_result}\n"
            f"调用分布：{data['calls_by_motive']}；发言分布：{data['sends_by_motive']}"
        )

    async def command(self, event) -> str:
        if not event.is_admin():
            return "仅 AstrBot 管理员可以管理主动回复。"
        await self.bridge.receive(event)
        state = self.bind(event)
        parts = event.get_message_str().strip().lstrip("/").split()
        args = parts[1:]
        action = args[0] if args else "status"
        value = args[1] if len(args) == 2 else None
        if len(args) > 2 or (value is not None and action not in {"cooldown", "willingness"}):
            return "参数不正确。"
        try:
            if action in {"enable", "disable", "reset"}:
                self.update(
                    state.origin,
                    {"reset": True} if action == "reset" else {"enabled": action == "enable"},
                )
            elif action == "cooldown":
                self.update(state.origin, {"cooldown_minutes": 5 if value is None else int(value)})
            elif action == "willingness" and value is not None:
                self.update(state.origin, {"willingness": value})
            elif action == "initiate":
                # Manual request bypasses probability/silence only, never safety gates or quotas.
                result = await self.decide(state, "initiate")
                return "主动发起：" + result + "（sent 表示平台接受；其他结果未发送）"
            elif action not in {"status", "willingness"}:
                return "/iris_reply enable|disable|status|reset|cooldown [0–120 分钟]|willingness [低|中|高]|initiate"
        except (ValueError, TypeError):
            return "参数无效：冷却需 0–120 整数分钟，意愿为低／中／高。"
        return self.status_text(state)

    async def set_cooldown(self, event, minutes=5) -> str:
        if not event.is_admin():
            return "仅 AstrBot 管理员可以设置主动回复冷却。"
        if type(minutes) is not int or not 1 <= minutes <= 120:
            return "冷却需 1–120 整数分钟。"
        state = self.bind(event)
        self.update(state.origin, {"cooldown_minutes": minutes})
        return f"当前会话主动回复冷却已设置为 {minutes} 分钟。"

    def snapshot(self) -> dict:
        return {
            "enabled": self.config.proactive_enabled,
            "initiate_enabled": self.config.proactive_initiate_enabled,
            "private_care_enabled": self.config.proactive_private_care_enabled,
            "sessions": [
                s.view(self.clock(), self.config, s.origin in self.live)
                for s in self.sessions.values()
            ],
        }

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        tasks = list(self.tasks | self.inflight) + ([self.scheduler] if self.scheduler else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for state in self.sessions.values():
            state.pending_at = 0
            self.store.save(state)
        self.store.close()
