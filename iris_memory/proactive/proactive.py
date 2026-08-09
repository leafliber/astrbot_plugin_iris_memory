from __future__ import annotations

import asyncio
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
from .prompts import SPEAK_HINTS
from .signals import SignalGate
from .state import StateManager
from .stats import StatsCollector
from .time_hint import resolve_datetime_reminder, wrap_system_reminder
from iris_memory.llm_modules import PROACTIVE_REPLY_INITIATE

# 发起评估被 LLM 否决后的重试间隔（秒），仅内存记录
_SKIP_RETRY_SECONDS = 30 * 60


def _record_initiate_send(group_id: str, text: str, success: bool, error: str = "") -> None:
    """写入统一运行日志（proactive 类型，主动发起直发结果）。"""
    try:
        from iris_memory.core.run_log import get_run_log_manager

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
    """主动发起引擎：定时扫描白名单群，在冷场或话题结束时评估并直发新话题。

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
        self._task: asyncio.Task | None = None
        self._initiating: set[str] = set()
        self._skip_retry_after: dict[str, float] = {}

    def is_initiating(self, group_id: str) -> bool:
        return group_id in self._initiating

    async def start(self) -> None:
        if not self._config.proactive_enabled:
            logger.info("Iris Reply: proactive engine disabled by config")
            return
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

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.proactive_check_interval * 60)
            if not self._config.proactive_enabled:
                continue
            try:
                await self._scan()
            except Exception as e:
                logger.warning("Iris Reply: proactive scan error: %s", e)

    async def _scan(self) -> None:
        for group_id in self._state.get_whitelist():
            await self._check_pending_timeout(group_id)
            if self._is_busy(group_id) or group_id in self._initiating:
                continue
            if self._skip_retry_after.get(group_id, 0) > time.time():
                continue
            if not self._state.can_detect(group_id):
                continue
            messages = self._window.get_messages(group_id)
            async with self._state.get_lock(group_id):
                motive = self._signals.evaluate_timer(group_id, messages)
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

        self._initiating.add(group_id)
        try:
            self._state.record_detect_time(group_id)
            req = DecisionRequest(
                group_id=group_id,
                wake="timer",
                motive="initiate",
                quiet_minutes=quiet_minutes,
            )
            outcome = await self._core.decide(req, self._llm_manager, provider_id)

            if outcome.error or outcome.decision is None:
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
                "Iris Reply: initiate decision for group %s: speak=%s, drifted=%s, cooldown=%d, topic=%.100s",
                group_id, decision.should_speak, decision.drifted,
                decision.cooldown_minutes, decision.topic,
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
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "决策结果解析失败"

            if not decision.should_speak:
                # LLM 否决：清零意愿值并重新采样阈值，下一轮从零积累
                async with self._state.get_lock(group_id):
                    self._state.reset_initiate_drive(group_id)
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "LLM 决定暂不发起"

            # 发言文本走主管线人格生成（决策只给切入角度，不直出消息）
            text = await self._generate_speech(group_id, umo, provider_id, decision)
            if not text:
                text = decision.message.strip()
            if not text:
                self._skip_retry_after[group_id] = time.time() + _SKIP_RETRY_SECONDS
                await self._save_fn()
                return "发言文本生成失败"

            text = text[: self._config.proactive_max_message_len]
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
                    reason=decision.why,
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
            self._initiating.discard(group_id)

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
        topic = decision.topic.strip() or "自由选择一个轻松、大家能接上话的角度"
        hint = SPEAK_HINTS["initiate"].format(topic=topic)
        prompt = f"{context_block}\n\n{hint}"

        # 直连 llm_generate 不经过主管线，需自行注入当前时间，
        # 否则 LLM 无时间锚点，会从窗口里的旧消息推断时间（如早上说晚上好）。
        system_prompt = persona_prompt or ""
        time_hint = wrap_system_reminder(resolve_datetime_reminder(self._context, umo))
        if time_hint:
            system_prompt = f"{time_hint}\n\n{system_prompt}".strip()

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
