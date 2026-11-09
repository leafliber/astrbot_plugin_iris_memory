from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.star import Context

from .config import ConfigManager
from .decision import (
    INPUT_SAFETY_COOLDOWN_MINUTES,
    DecisionCore,
    DecisionRequest,
)
from .parser import Decision
from .perception import ContextPackager, SlidingWindow, WindowMessage
from .prompts import PRIVATE_SPEAK_HINT_INITIATE, SPEAK_HINTS, SPEAK_SPLIT_NOTE
from .signals import SignalGate
from .state import GroupState, StateManager
from .stats import StatsCollector
from .time_hint import resolve_datetime_reminder, wrap_system_reminder
from ..llm_modules import PROACTIVE_REPLY_INITIATE

# 发起评估被 LLM 否决后的重试间隔（秒），仅内存记录
_SKIP_RETRY_SECONDS = 30 * 60
# 接入连续性桥接时：分段上限与重启后过期候选的重新抽样区间（秒）
_MAX_SPEECH_PARTS = 3
_RESTART_RESAMPLE_SECONDS = (30.0, 180.0)
# 未接入桥接时的独立禁发频道（与 continuity.proactive_bridge.DEFAULT_FORBIDDEN_CHATS 一致）
_LEGACY_FORBIDDEN_CHATS: frozenset = frozenset()


def _legacy_chat_variants(chat_id) -> set[str]:
    s = str(chat_id or "").strip()
    if not s:
        return set()
    out = {s}
    if s.startswith("-100") and len(s) > 4:
        out.add(s[4:])
    elif s.startswith("-"):
        out.add(s[1:])
    else:
        out.add("-100" + s)
        out.add("-" + s)
    return out


def _is_silent_or_forbidden(chat_id, silent_sessions) -> bool:
    try:
        from ..continuity.service import is_silent_or_forbidden

        return bool(is_silent_or_forbidden(chat_id, silent_sessions))
    except Exception:
        pass
    ids = _legacy_chat_variants(chat_id)
    if not ids:
        return False
    blocked = {str(x).strip() for x in (silent_sessions or ()) if str(x).strip()}
    if str(chat_id).strip() in blocked:
        return True
    return any(ids & _legacy_chat_variants(c) for c in _LEGACY_FORBIDDEN_CHATS | blocked)


def _topic_matches(topic: str, avoided: list[str]) -> bool:
    """决策话题与「不要主动提起」话题规范化后相同或互相包含（≥2 字）"""
    from ..continuity.models import norm_topic

    t = norm_topic(topic)
    if len(t) < 2:
        return False
    for a in avoided or ():
        n = norm_topic(a)
        if len(n) >= 2 and (n == t or n in t or t in n):
            return True
    return False


def _avoid_block(topics: list[str]) -> str:
    if not topics:
        return ""
    lines = "\n".join(f"- {t}" for t in topics)
    return f"【不要主动提起】\n以下话题对方已拒绝、已完成或已取消，不要主动提起，也不要换个说法绕回去：\n{lines}"


def split_speech_parts(text: str, max_parts: int = _MAX_SPEECH_PARTS) -> list[str]:
    """按空行切分为不超过 max_parts 段；多出的段并入最后一段。"""
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text or "") if c.strip()]
    if len(chunks) <= max_parts:
        return chunks
    return chunks[: max_parts - 1] + ["\n\n".join(chunks[max_parts - 1:])]


def _record_initiate_send(group_id: str, text: str, success: bool, error: str = "") -> None:
    """写入统一运行日志（proactive 类型，主动发起直发结果）。"""
    try:
        from ..core.run_log import get_run_log_manager

        get_run_log_manager().record(
            "proactive",
            f"主动发起{'已发送' if success else '发送失败'}",
            success=success,
            group_id=group_id,
            wake="timer",
            motive="initiate",
            stage="send",
            message=text,
            error=error,
        )
    except Exception:
        pass


class ProactiveEngine:
    """主动发起引擎：按群独立预约，在自然空档中概率评估并直发新话题。

    直发通路不经过 AstrBot 事件管线（after_message_sent 等钩子不会触发），
    因此所有记账（入窗 / 锚点 / pending / 统计）都在此手动完成。
    """

    def __init__(
        self,
        context: Context,
        config: ConfigManager,
        state: StateManager,
        window: SlidingWindow,
        signals: SignalGate,
        decision_core: DecisionCore,
        stats: StatsCollector,
        *,
        llm_manager,
        packager: ContextPackager,
        umo_get: Callable[[str], str | None],
        is_busy: Callable[[str], bool],
        self_id_get: Callable[[], str],
        save_fn: Callable[[], Awaitable[None]],
        on_initiate_sent: Callable[[str, str], Awaitable[None]] | None = None,
        text_transform: Callable[[str], str] | None = None,
        profile_get: Callable[[str, str], Awaitable[str]] | None = None,
        bridge=None,
    ) -> None:
        self._context = context
        self._config = config
        self._state = state
        self._window = window
        self._signals = signals
        self._core = decision_core
        self._llm_manager = llm_manager
        self._stats = stats
        self._packager = packager
        self._umo_get = umo_get
        self._is_busy = is_busy
        self._self_id_get = self_id_get
        self._save_fn = save_fn
        self._on_initiate_sent = on_initiate_sent
        self._text_transform = text_transform
        self._profile_get = profile_get
        # 连续性桥接（iris_memory.continuity.proactive_bridge.ProactiveBridge）；
        # None 时完全保持原有行为（context.send_message 直发）。
        self._bridge = bridge
        if bridge is not None:
            # 私聊暂停只能由本人新私聊消息或管理员重置解除，不按 24h 自动清零
            state.time_reset_private = False
        self._task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._initiating: set[str] = set()
        self._skip_retry_after: dict[str, float] = {}
        self._avoid_cache: dict[str, list[str]] = {}

    def _legacy_target_refusal(self, group_id: str) -> str:
        """未接入桥接时的发送目标硬拦截（禁发频道 / silent_sessions / 私聊白名单）。"""
        scope = str(group_id or "").strip()
        if not scope:
            return "目标会话无效"
        try:
            silent = list(getattr(self._config, "silent_sessions", None) or [])
        except Exception:
            silent = []
        if scope.startswith("private:"):
            uid = scope.split(":", 1)[1].strip()
            try:
                allowed = {str(x).strip() for x in self._config.private_whitelist}
            except Exception:
                allowed = set()
            if not uid or uid not in allowed:
                return "私聊对象不在白名单"
            if _is_silent_or_forbidden(uid, silent):
                return "目标会话禁止发言"
            return ""
        peer = scope.partition("#")[0].strip()
        if scope in {str(x).strip() for x in silent} or _is_silent_or_forbidden(peer, silent):
            return "目标会话禁止发言"
        return ""

    def is_initiating(self, group_id: str) -> bool:
        return group_id in self._initiating

    async def start(self) -> None:
        if not self._config.proactive_enabled:
            logger.info("Iris Reply: proactive engine disabled by config")
            return
        if self._bridge is not None:
            await self._bridge_startup()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Iris Reply: proactive engine started (interval=%dmin)",
            self._config.proactive_check_interval,
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _bridge_startup(self) -> None:
        """启动时：对账遗留发送、恢复持久化窗口、过期候选重新抽样（不集中补发）。"""
        try:
            await self._bridge.reconcile(force_query=True)
        except Exception as e:
            logger.warning("Iris Reply: bridge reconcile at startup failed: %s", type(e).__name__)
        try:
            await self._bridge.restore_windows(self._window)
        except Exception as e:
            logger.warning("Iris Reply: bridge restore_windows failed: %s", type(e).__name__)
        now = time.time()
        for group_id in self._state.get_whitelist():
            when = self._state.get_next_initiate_check(group_id)
            if 0 < when <= now:
                self._state.set_next_initiate_check(
                    group_id,
                    max(now + random.uniform(*_RESTART_RESAMPLE_SECONDS), when),
                )

    async def _bridge_reconcile(self) -> None:
        if self._bridge is None:
            return
        try:
            await self._bridge.reconcile()
        except Exception as e:
            logger.warning("Iris Reply: bridge reconcile failed: %s", type(e).__name__)

    def _is_paused(self, group_id: str, force: bool) -> bool:
        if force:
            return False
        data = self._state.get_state(group_id)
        return data.initiate_no_reply_streak >= self._config.proactive_max_streak

    def notify_human_message(self, group_id: str, timestamp: float | None = None) -> None:
        """让旧候选失效，并从这条真实消息之后为单群独立预约。"""
        timestamp = time.time() if timestamp is None else timestamp
        self._state.record_human_message(group_id, timestamp)
        self._schedule_group(group_id, now=timestamp, after_activity=True)
        self._wake_event.set()

    def _schedule_group(
        self,
        group_id: str,
        *,
        now: float | None = None,
        after_activity: bool = False,
        not_before: float = 0.0,
    ) -> float:
        """生成带秒级抖动的每群独立候选时间。"""
        now = time.time() if now is None else now
        if self._state.is_muted():
            # 静音结束后重新抽 30–90 分钟，避免所有群在开闸时集中发言。
            when = now + self._state.seconds_until_unmuted(now)
            when += random.uniform(30 * 60.0, 90 * 60.0)
        elif after_activity:
            earliest = now + self._state.minimum_quiet_seconds()
            jitter = random.uniform(
                0.0,
                max(60.0, self._config.proactive_check_interval * 120.0),
            )
            when = earliest + jitter
        else:
            base = max(60.0, self._config.proactive_check_interval * 60.0)
            when = now + random.uniform(base * 0.6, base * 2.8)
        when = max(when, not_before)
        self._state.set_next_initiate_check(group_id, when)
        return when

    async def _loop(self) -> None:
        while True:
            if not self._config.proactive_enabled:
                await asyncio.sleep(60)
                continue
            try:
                await self._ensure_schedules()
                deadline = self._next_deadline()
                timeout = 60.0 if deadline <= 0 else max(
                    0.05, min(60.0, deadline - time.time())
                )
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
                    self._wake_event.clear()
                except asyncio.TimeoutError:
                    pass
                await self._bridge_reconcile()
                await self._scan_due()
            except Exception as e:
                logger.warning("Iris Reply: proactive scan error: %s", e)

    async def _ensure_schedules(self) -> None:
        now = time.time()
        for group_id in self._state.get_whitelist():
            await self._check_pending_timeout(group_id)
            if not self._window.get_messages(group_id):
                continue
            if self._state.get_next_initiate_check(group_id) <= 0:
                self._schedule_group(group_id, now=now)

    def _next_deadline(self) -> float:
        deadlines = [
            self._state.get_next_initiate_check(gid)
            for gid in self._state.get_whitelist()
            if self._window.get_messages(gid)
            and self._state.get_next_initiate_check(gid) > 0
        ]
        return min(deadlines) if deadlines else 0.0

    async def _scan_due(self) -> None:
        now = time.time()
        due = [
            gid
            for gid in self._state.get_whitelist()
            if self._window.get_messages(gid)
            and 0 < self._state.get_next_initiate_check(gid) <= now
        ]
        random.shuffle(due)
        for group_id in due:
            await self._check_pending_timeout(group_id)
            if self._is_busy(group_id) or group_id in self._initiating:
                self._schedule_group(group_id, now=now)
                continue
            retry_after = self._skip_retry_after.get(group_id, 0)
            if retry_after > now:
                self._schedule_group(
                    group_id,
                    now=now,
                    not_before=retry_after + random.uniform(30.0, 180.0),
                )
                continue
            if not self._state.can_detect(group_id):
                self._schedule_group(group_id, now=now)
                continue
            messages = self._window.get_messages(group_id)
            async with self._state.get_lock(group_id):
                motive = self._signals.evaluate_timer(group_id, messages)
            self._schedule_group(group_id, now=now)
            if motive:
                result = await self.attempt_initiate(group_id)
                logger.info("Iris Reply: proactive attempt for group %s: %s", group_id, result)

    async def _check_pending_timeout(self, group_id: str) -> None:
        data = self._state.get_state(group_id)
        if data.initiate_pending_since <= 0:
            return
        timeout = self._config.proactive_pending_timeout * 60
        if time.time() - data.initiate_pending_since >= timeout:
            async with self._state.get_lock(group_id):
                self._state.record_initiate_unanswered(group_id)
            await self._save_fn()
            logger.info("Iris Reply: initiate unanswered in group %s, streak recorded", group_id)

    async def attempt_initiate(self, group_id: str, force: bool = False) -> str:
        """执行一次主动发起。force 跳过门控（管理命令调试用），但保留互斥护栏。"""
        if group_id in self._initiating:
            return "该群正在发起中"
        if self._is_busy(group_id):
            return "该群有回复进行中，稍后再试"
        if not force and not self._state.is_whitelisted(group_id):
            return "该群未启用"

        if self._bridge is None:
            refusal = self._legacy_target_refusal(group_id)
            if refusal:
                return f"已拒绝发起：{refusal}"

        umo = self._umo_get(group_id)
        if not umo:
            return "暂无该群的会话标识（等群里有过消息后再试）"

        provider_id = self._config.provider_id
        if not provider_id:
            try:
                provider_id = await self._context.get_current_chat_provider_id(umo)
            except Exception:
                provider_id = None
        if not provider_id:
            return "无法获取 LLM 提供商"

        messages = self._window.get_messages(group_id)
        quiet_minutes = int((time.time() - messages[-1].timestamp) / 60) if messages else 0
        context_cutoff = max(
            [m.timestamp for m in messages]
            + [self._state.get_state(group_id).last_human_message_at]
        )

        self._initiating.add(group_id)
        bridge = self._bridge
        turn_id = ""
        turn_reason = "aborted"
        try:
            self._state.record_detect_time(group_id)
            candidates = None
            avoid: list[str] = []
            private = bridge is not None and group_id.startswith("private:")
            self._avoid_cache.pop(group_id, None)
            if bridge is not None:
                start = await bridge.begin_turn(group_id, force)
                if start.turn is None:
                    return f"已跳过发起：{start.reason}"
                turn_id = start.turn["turn_id"]
                candidates = start.candidates or None
                if private:
                    try:
                        avoid = list(await bridge.avoid_topics(group_id) or [])
                    except Exception as e:
                        logger.warning(
                            "Iris Reply: avoid_topics failed for %s: %s", group_id, type(e).__name__,
                        )
                        avoid = []
                    self._avoid_cache[group_id] = avoid
            req = DecisionRequest(
                group_id=group_id,
                wake="timer",
                motive="initiate",
                quiet_minutes=quiet_minutes,
                private=private,
                candidates=candidates,
                avoid_topics=avoid or None,
            )
            outcome = await self._core.decide(req, self._llm_manager, provider_id)

            if outcome.error or outcome.decision is None:
                turn_reason = "decision_error"
                self._stats.record_decision_error(group_id, "initiate")
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                if outcome.error_kind == "input_content_safety_1026":
                    async with self._state.get_lock(group_id):
                        cleanup = self._core.clear_rejected_dynamic_context(
                            group_id,
                            outcome.dynamic_context_sources,
                        )
                        cooldown = self._state.set_cooldown(
                            group_id,
                            INPUT_SAFETY_COOLDOWN_MINUTES,
                        )
                    await self._save_fn()
                    logger.warning(
                        "Iris Reply: initiate input rejected by provider safety filter "
                        "for group %s (1026, retryable=false, dynamic_sources=%d, "
                        "window_removed=%d, observation_cleared=%s, "
                        "anchor_cleared=%s, cooldown=%dmin)",
                        group_id,
                        cleanup.dynamic_source_count,
                        cleanup.window_removed,
                        cleanup.observation_cleared,
                        cleanup.anchor_cleared,
                        cooldown,
                    )
                return f"决策调用失败: {outcome.error}"

            decision = outcome.decision
            self._stats.record_decision(
                group_id, "initiate",
                system_prompt=outcome.system_prompt,
                user_prompt=outcome.user_prompt,
                response_text=outcome.raw_text,
                decision=decision,
                duration_ms=outcome.duration_ms,
            )
            logger.info(
                "Iris Reply: initiate decision for group %s: speak=%s, drifted=%s, "
                "cooldown=%d, source=%s, why_now=%.100s, topic=%.100s",
                group_id, decision.should_speak, decision.drifted,
                decision.cooldown_minutes, decision.topic_source,
                decision.why_now, decision.topic,
            )

            async with self._state.get_lock(group_id):
                if decision.observation:
                    self._state.set_observation(group_id, decision.observation)
                if decision.cooldown_minutes:
                    self._state.set_cooldown(group_id, decision.cooldown_minutes)
                if decision.drifted:
                    self._state.close_anchor(group_id)
                    self._state.record_drift(group_id)

            if decision.parse_failed:
                turn_reason = "decision_parse_failed"
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "决策结果解析失败"

            if not decision.should_speak:
                turn_reason = "llm_declined"
                # LLM 否决：重置概率时钟，避免短时间连续询问模型。
                async with self._state.get_lock(group_id):
                    self._state.reset_initiate_drive(group_id)
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "LLM 决定暂不发起"

            # 决策明确选中了本回合的有效候选事项时，不做话题回避判定：
            # 例如“面试”已完成、“面试结果”仍是待续候选，跟进候选是合法的。
            chosen_candidate = bool(decision.item_id) and any(
                str(c.get("item_id")) == str(decision.item_id) for c in (candidates or [])
            )
            if avoid and not chosen_candidate and _topic_matches(decision.topic, avoid):
                turn_reason = "avoided_topic"
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "已跳过发起：avoided_topic"

            guard = self._initiate_guard_reason(
                group_id, context_cutoff, require_quiet=not force
            )
            if guard:
                turn_reason = "guard"
                await self._save_fn()
                return f"已取消发起：{guard}"

            if bridge is not None:
                await bridge.mark_turn(
                    turn_id, "generating", item_id=decision.item_id, llm_calls=1
                )

            # 发言文本走主管线人格生成（决策只给切入角度，不直出消息）
            text = await self._generate_speech(group_id, umo, provider_id, decision)
            if not text:
                text = decision.message.strip()
            if not text:
                turn_reason = "speech_failed"
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "发言文本生成失败"

            text = text[: self._config.proactive_max_message_len]
            if bridge is not None:
                turn_reason = "send_aborted"
                return await self._send_via_bridge(
                    group_id, umo, turn_id, text, decision, context_cutoff, force
                )
            if self._text_transform is not None:
                # 直发通路不触发 on_decorating_result，消息始终以纯文本发送，
                # 由调用方补齐与管线一致的文本处理（如 Markdown 去除）
                try:
                    text = self._text_transform(text) or text
                except Exception as e:
                    logger.warning(
                        "Iris Reply: text_transform error for group %s: %s",
                        group_id, e,
                    )
            chain = MessageChain().message(text)
            # 紧贴发送动作做最后一次 TOCTOU 校验，尽量缩小竞态窗口。
            guard = self._initiate_guard_reason(
                group_id, context_cutoff, require_quiet=not force
            )
            if guard:
                await self._save_fn()
                return f"已取消发起：{guard}"
            refusal = self._legacy_target_refusal(group_id)
            if refusal:
                return f"已拒绝发起：{refusal}"
            ok = await self._context.send_message(umo, chain)
            if not ok:
                _record_initiate_send(group_id, text, False, "未找到匹配的消息平台")
                return "发送失败：未找到匹配的消息平台"

            _record_initiate_send(group_id, text, True)

            self._window.append(group_id, WindowMessage(
                sender_id=self._self_id_get() or "iris",
                sender_name="我",
                content=text,
                timestamp=time.time(),
            ))
            async with self._state.get_lock(group_id):
                self._state.record_initiate(
                    group_id,
                    topic=decision.topic or decision.observation,
                    bot_message=text,
                    users=decision.watch or None,
                    keywords=decision.watch_keywords or None,
                    reason=decision.why_now or decision.why,
                )
            now = time.time()
            self._schedule_group(
                group_id,
                now=now,
                not_before=now + self._config.proactive_min_interval * 60.0,
            )
            await self._save_fn()
            if self._on_initiate_sent is not None:
                try:
                    await self._on_initiate_sent(group_id, text)
                except Exception as e:
                    logger.warning(
                        "Iris Reply: on_initiate_sent callback error for group %s: %s",
                        group_id, e,
                    )
            return f"已发起: {text[:50]}"
        except Exception as e:
            logger.error("Iris Reply: initiate failed for group %s: %s", group_id, e)
            return f"发起异常: {e}"
        finally:
            if bridge is not None and turn_id:
                # 未走到发送记账的回合（否决/失败/取消/异常）在此收尾；已终结则无操作
                try:
                    await bridge.close_turn(turn_id, "skipped", turn_reason)
                except Exception as e:
                    logger.warning(
                        "Iris Reply: close_turn failed for group %s: %s",
                        group_id, type(e).__name__,
                    )
            self._initiating.discard(group_id)

    def _transform(self, group_id: str, text: str) -> str:
        if self._text_transform is None:
            return text
        try:
            return self._text_transform(text) or text
        except Exception as e:
            logger.warning(
                "Iris Reply: text_transform error for group %s: %s", group_id, e,
            )
            return text

    async def _send_via_bridge(
        self,
        group_id: str,
        umo: str,
        turn_id: str,
        text: str,
        decision: Decision,
        context_cutoff: float,
        force: bool,
    ) -> str:
        """连续性桥接发送：分段（≤3）、每段前复查、带回执发送、按 §5.9 记账。"""
        bridge = self._bridge
        parts = [
            p for p in (self._transform(group_id, raw).strip() for raw in split_speech_parts(text))
            if p
        ]
        if not parts:
            self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
            await self._save_fn()
            return "发言文本生成失败"
        await bridge.mark_turn(turn_id, "sending", parts_total=len(parts), llm_calls=2)
        stop_reason = ""
        try:
            for idx, part in enumerate(parts):
                guard = self._initiate_guard_reason(
                    group_id, context_cutoff, require_quiet=not force
                )
                if guard:
                    stop_reason = "guard"
                    break
                check = await bridge.presend_check(
                    turn_id, umo, paused=self._is_paused(group_id, force)
                )
                if not check.ok:
                    stop_reason = check.reason
                    break
                outcome = await bridge.send_part(turn_id, umo, part, idx)
                if not outcome.sent:
                    stop_reason = outcome.error or f"send_{outcome.status}"
                    break
        except Exception as e:
            # 已发出的分段仍要记账；异常只终止剩余分段
            stop_reason = stop_reason or "send_exception"
            logger.warning(
                "Iris Reply: bridge send error for group %s: %s", group_id, type(e).__name__,
            )
        result = await bridge.finish_turn(turn_id, stop_reason=stop_reason)
        return await self._account_bridge_result(group_id, decision, result)

    async def _owner_answered(self, result) -> bool:
        """首段发出后本人已回复（因新消息中止，或水位已越过回合起点）"""
        reason = str(getattr(result, "reason", "") or "")
        if reason == "new_inbound" or reason.startswith("stale"):
            return True
        checker = getattr(self._bridge, "owner_replied_since", None)
        if checker is None:
            return False
        try:
            return bool(await checker(result.turn_id))
        except Exception:
            return False

    async def _account_bridge_result(self, group_id: str, decision: Decision, result) -> str:
        now = time.time()
        if result.parts_sent >= 1:
            # 至少一段真实发出：额度只计 1 次；pending=now，
            # 除非本人已在首段之后回复（已接话，不再等待、不计未回复）
            answered = await self._owner_answered(result)
            sent_text = "\n\n".join(result.sent_texts)
            _record_initiate_send(group_id, sent_text, True)
            self._window.append(group_id, WindowMessage(
                sender_id=self._self_id_get() or "iris",
                sender_name="我",
                content=sent_text,
                timestamp=now,
            ))
            async with self._state.get_lock(group_id):
                self._state.record_initiate(
                    group_id,
                    topic=decision.topic or decision.observation,
                    bot_message=sent_text,
                    users=decision.watch or None,
                    keywords=decision.watch_keywords or None,
                    reason=decision.why_now or decision.why,
                    set_pending=not answered,
                )
            self._schedule_group(
                group_id,
                now=now,
                not_before=now + self._config.proactive_min_interval * 60.0,
            )
            await self._save_fn()
            if self._on_initiate_sent is not None:
                try:
                    await self._on_initiate_sent(group_id, sent_text)
                except Exception as e:
                    logger.warning(
                        "Iris Reply: on_initiate_sent callback error for group %s: %s",
                        group_id, e,
                    )
            return f"已发起: {sent_text[:50]}"
        if result.state == "uncertain":
            # 结果不确定：bridge 已预占额度（不设 pending）；等对账，min_interval 内不再发起
            _record_initiate_send(group_id, "", False, "发送结果不确定，等待对账")
            self._schedule_group(
                group_id,
                now=now,
                not_before=now + self._config.proactive_min_interval * 60.0,
            )
            await self._save_fn()
            return "发送结果不确定，等待对账"
        reason = result.reason or result.state
        if result.state == "failed":
            # 确认失败：不扣额度、不计未回复；短期内不重试
            _record_initiate_send(group_id, "", False, reason)
            self._skip_retry_after[group_id] = now + _SKIP_RETRY_SECONDS
            await self._save_fn()
            return f"发送失败：{reason}"
        await self._save_fn()
        return f"已取消发起：{reason}"

    def _initiate_guard_reason(
        self,
        group_id: str,
        context_cutoff: float,
        *,
        require_quiet: bool,
    ) -> str:
        """发送前二次确认现场，防止生成期间群聊已经恢复。"""
        if self._is_busy(group_id):
            return "群内已有回复进行中"
        data = self._state.get_state(group_id)
        messages = self._window.get_messages(group_id)
        latest = max(
            [m.timestamp for m in messages] + [data.last_human_message_at]
        )
        if latest > context_cutoff + 1e-6:
            return "决策期间出现了新消息"
        if not require_quiet:
            return ""
        if self._state.is_muted():
            return "当前进入静音时段"
        if data.state == GroupState.COOLDOWN:
            return "当前处于冷却状态"
        if not messages:
            return "缺少可用群聊上下文"
        quiet = time.time() - max(data.last_human_message_at, messages[-1].timestamp)
        if quiet < self._state.minimum_quiet_seconds():
            return "群聊尚未达到最小静默时间"
        return ""

    async def _generate_speech(
        self,
        group_id: str,
        umo: str,
        provider_id: str,
        decision: Decision,
    ) -> str:
        """用主管线人格生成发起文本。

        决策层只给出切入角度（topic），具体措辞由 bot 当前人格结合群聊
        上下文产出，保证主动消息与被动回复的语气、风格一致。
        失败时返回空字符串，由调用方回退或重试。
        """
        persona_prompt = ""
        try:
            personality = await self._context.persona_manager.get_default_persona_v3(umo)
            if personality:
                persona_prompt = personality.get("prompt", "") or ""
        except Exception as e:
            logger.warning(
                "Iris Reply: persona resolve failed for group %s: %s", group_id, e,
            )

        messages = self._window.get_messages(group_id)
        context_block = self._packager.package(group_id, messages, "initiate")
        bridged = self._bridge is not None
        private = bridged and group_id.startswith("private:")
        if private:
            hint = PRIVATE_SPEAK_HINT_INITIATE.format(
                topic=decision.topic.strip() or "顺着你们最近的私聊自然地开口",
                why_now=decision.why_now or "你们的私聊出现了自然的空档",
                topic_source=decision.topic_source or "recent_context",
            )
        else:
            topic = decision.topic.strip() or "自由选择一个轻松、大家能接上话的角度"
            hint = SPEAK_HINTS["initiate"].format(
                topic=topic,
                why_now=decision.why_now or "当前群聊出现了自然的交流空档",
                topic_source=decision.topic_source or "recent_context",
            )
        if bridged:
            hint = f"{hint}\n{SPEAK_SPLIT_NOTE}"
        prompt = f"{context_block}\n\n{hint}"

        # 直连 llm_generate 不经过主管线，需自行注入当前时间，
        # 否则 LLM 无时间锚点，会从窗口里的旧消息推断时间（如早上说晚上好）。
        system_prompt = persona_prompt or ""
        time_hint = wrap_system_reminder(resolve_datetime_reminder(self._context, umo))
        if time_hint:
            system_prompt = f"{time_hint}\n\n{system_prompt}".strip()

        # 接入桥接时：连续性上下文（与普通回复同一组装逻辑，已含私聊画像）
        continuity = ""
        if bridged:
            continuity = await self._bridge.speech_context(group_id, decision.item_id)
            if continuity:
                system_prompt = f"{system_prompt}\n\n{continuity}".strip()
            avoid_text = _avoid_block(self._avoid_cache.get(group_id) or []) if private else ""
            if avoid_text:
                system_prompt = f"{system_prompt}\n\n{avoid_text}".strip()

        # 直连 generate_direct 不会触发 on_llm_request，画像需要在此手动注入，
        # 否则主动消息只能看到人格，看不到长期记忆。
        if self._profile_get and not continuity:
            try:
                profile_text = await self._profile_get(group_id, umo)
            except Exception as e:
                profile_text = ""
                logger.warning("Iris Reply: profile resolve failed for %s: %s", group_id, e)
            if profile_text:
                system_prompt = (
                    f"{system_prompt}\n\n【你记得的关于对方的信息】\n{profile_text}"
                ).strip()

        try:
            text = await self._llm_manager.generate_direct(
                prompt=prompt,
                system_prompt=system_prompt or None,
                provider_id=provider_id,
                module=PROACTIVE_REPLY_INITIATE,
            )
        except Exception as e:
            logger.warning(
                "Iris Reply: speech generation failed for group %s: %s", group_id, e,
            )
            return ""
        return (text or "").strip()
