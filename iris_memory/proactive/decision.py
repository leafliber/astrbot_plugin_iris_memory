from __future__ import annotations

import time
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .config import ConfigManager
from .parser import Decision, parse_decision
from .perception import ContextPackager, SlidingWindow
from .prompts import MOTIVE_INSTRUCTIONS, VALID_MOTIVES, WILLINGNESS_PROMPTS
from .state import StateManager, ThreadAnchor
from .time_hint import wrap_system_reminder

# llm_generate(chat_provider_id=..., prompt=..., system_prompt=...) -> response
LlmGenerateFn = Callable[..., Awaitable[Any]]

# group_id -> 当前时间提示（<system_reminder> 块），无则返回空串
TimeHintGet = Callable[[str], str]

_MOTIVE_LABELS = {
    "chime_in": "插话",
    "follow_up": "跟进",
    "initiate": "主动发起",
    "watch": "跟进评估",
}

_INPUT_SENSITIVE_1026 = re.compile(
    r"(?:input\s+new_sensitive\s*\(\s*1026\s*\)|"
    r"['\"]?message['\"]?\s*:\s*['\"]input\s+new_sensitive|"
    r"['\"]?code['\"]?\s*:\s*1026)",
    re.IGNORECASE,
)

INPUT_SAFETY_COOLDOWN_MINUTES = 30


def _redact_dynamic_context_sources(
    sources: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """为安全拒绝日志移除消息正文、昵称、用户 ID 与关键词值。"""
    redacted: list[dict[str, Any]] = []
    for source in sources or []:
        item = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "content",
                "values",
                "sender_id",
                "sender_name",
            }
        }
        item["redacted"] = True
        redacted.append(item)
    return redacted


def _record_decision_log(
    req: "DecisionRequest",
    provider_id: str,
    outcome: "DecisionOutcome",
) -> None:
    """写入统一运行日志（proactive 类型），失败不影响主流程。"""
    try:
        from iris_memory.core.run_log import get_run_log_manager

        motive_label = _MOTIVE_LABELS.get(req.motive, req.motive)
        wake_label = "定时" if req.wake == "timer" else "消息"

        if outcome.error or outcome.decision is None:
            title = f"{motive_label}决策失败（{wake_label}触发）"
            safety_rejected = outcome.error_kind == "input_content_safety_1026"
            get_run_log_manager().record(
                "proactive",
                title,
                success=False,
                group_id=req.group_id,
                wake=req.wake,
                motive=req.motive,
                quiet_minutes=req.quiet_minutes,
                provider_id=provider_id,
                system_prompt=outcome.system_prompt,
                user_prompt=(
                    "[Provider 输入安全过滤拒绝，动态 prompt 已脱敏]"
                    if safety_rejected
                    else outcome.user_prompt
                ),
                raw_response=outcome.raw_text,
                duration_ms=round(outcome.duration_ms, 1),
                error=outcome.error,
                error_kind=outcome.error_kind,
                retryable=outcome.retryable,
                dynamic_context_sources=(
                    _redact_dynamic_context_sources(outcome.dynamic_context_sources)
                    if safety_rejected
                    else outcome.dynamic_context_sources
                ),
            )
            return

        d = outcome.decision
        # 标签优先级与 main.py _handle_reply_decision 的实际消费顺序保持一致：
        # parse_failed → cooldown（命中即跳过）→ drifted → should_speak → skip，
        # 避免 cooldown 与 speak 同现时日志误标「决定发言」。
        if d.parse_failed:
            result_label = "解析失败"
        elif d.cooldown_minutes:
            result_label = f"请求冷却 {d.cooldown_minutes} 分钟"
        elif d.drifted:
            result_label = "话题漂移"
        elif d.should_speak:
            result_label = "决定发言"
        else:
            result_label = "决定跳过"

        get_run_log_manager().record(
            "proactive",
            f"{motive_label}决策：{result_label}（{wake_label}触发）",
            success=not d.parse_failed,
            group_id=req.group_id,
            wake=req.wake,
            motive=req.motive,
            quiet_minutes=req.quiet_minutes,
            provider_id=provider_id,
            result=result_label,
            should_speak=d.should_speak,
            message=d.message,
            observation=d.observation,
            watch_users=d.watch,
            watch_keywords=d.watch_keywords,
            watch_reason=d.why,
            drifted=d.drifted,
            cooldown_minutes=d.cooldown_minutes,
            parse_failed=d.parse_failed,
            system_prompt=outcome.system_prompt,
            user_prompt=outcome.user_prompt,
            raw_response=outcome.raw_text,
            duration_ms=round(outcome.duration_ms, 1),
            error="",
        )
    except Exception:
        pass


@dataclass
class DecisionRequest:
    """一次统一决策请求。wake 为唤醒源，motive 为候选动机（LLM 可否决）。"""

    group_id: str
    wake: str  # "message" | "timer"
    motive: str  # "chime_in" | "follow_up" | "initiate" | "watch"
    quiet_minutes: int = 0  # 仅 initiate 使用


@dataclass
class DecisionOutcome:
    """决策调用结果，附带 prompt 与原始响应（供统计与日志）。"""

    decision: Decision | None
    system_prompt: str
    user_prompt: str
    raw_text: str = ""
    error: str = ""
    duration_ms: float = 0.0
    error_kind: str = ""
    retryable: bool = True
    dynamic_context_sources: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SafetyCleanupResult:
    """一次 Provider 输入安全拒绝后的主动回复上下文清理结果。"""

    window_removed: int
    observation_cleared: bool
    anchor_cleared: bool
    dynamic_source_count: int


def classify_decision_error(exc: BaseException | str) -> tuple[str, bool]:
    """将已知 Provider 错误分类为（类型，是否可原样重试）。"""
    message = str(exc)
    if _INPUT_SENSITIVE_1026.search(message):
        return "input_content_safety_1026", False
    return "provider_error", True


def build_anchor_block(anchor: ThreadAnchor, motive: str) -> str:
    """构建 <thread> 锚点块，无锚点信息时返回空字符串。"""
    if not anchor.has_context:
        return ""
    parts = []
    if anchor.bot_message:
        parts.append(f'你之前在群里说："{anchor.bot_message}"')
    if anchor.participants:
        parts.append(f"你关注这些用户：{', '.join(sorted(anchor.participants))}")
    if anchor.keywords:
        parts.append(f"你关注这些关键词：{', '.join(sorted(anchor.keywords))}")
    if anchor.reason:
        parts.append(f"原因：{anchor.reason}")
    text = "\n\n<thread>" + "；".join(parts)
    if motive == "follow_up":
        text += "。现在相关对话有了新进展，请综合评估所有新消息后决定是否回应。"
    text += "</thread>"
    return text


class DecisionCore:
    """统一决策核心：三种发言动机 + 跟进评估共用同一 prompt 骨架与同一次 LLM 调用。"""

    def __init__(
        self,
        config: ConfigManager,
        state: StateManager,
        window: SlidingWindow,
        packager: ContextPackager,
        time_hint_get: TimeHintGet | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._window = window
        self._packager = packager
        self._time_hint_get = time_hint_get

    def build_prompt(self, req: DecisionRequest) -> tuple[str, str]:
        """组装 (user_prompt, system_prompt)。"""
        willingness = self._state.get_willingness(req.group_id)
        prompts = WILLINGNESS_PROMPTS[willingness]

        user_prompt = prompts["persona"]

        observation = self._state.get_observation(req.group_id)
        if observation:
            user_prompt += f"\n\n<recent_observation>之前的观察：{observation}</recent_observation>"

        user_prompt += build_anchor_block(self._state.get_anchor(req.group_id), req.motive)

        instruction = MOTIVE_INSTRUCTIONS[req.motive]
        if req.motive == "initiate":
            instruction = instruction.format(quiet_minutes=max(0, req.quiet_minutes))
            instruction += "\n" + prompts["initiate_style"]
            custom = self._config.proactive_instruction
            if custom:
                instruction += f"\n话题倾向：{custom}"
        user_prompt += f"\n\n<instruction>{instruction}</instruction>"

        messages = self._window.get_messages(req.group_id)
        user_prompt += "\n\n" + self._packager.package(req.group_id, messages, req.motive)

        system_prompt = prompts["decision_system"]
        time_hint = wrap_system_reminder(
            self._time_hint_get(req.group_id) if self._time_hint_get else ""
        )
        if time_hint:
            system_prompt = f"{time_hint}\n\n{system_prompt}"
        return user_prompt, system_prompt

    def collect_dynamic_context_sources(self, req: DecisionRequest) -> list[dict[str, Any]]:
        """返回决策 prompt 中所有动态内容的来源，供安全拦截定位。"""
        sources: list[dict[str, Any]] = []

        observation = self._state.get_observation(req.group_id)
        if observation:
            sources.append({"source": "recent_observation", "content": observation})

        anchor = self._state.get_anchor(req.group_id)
        if anchor.bot_message:
            sources.append({"source": "thread.bot_message", "content": anchor.bot_message})
        if anchor.participants:
            sources.append(
                {
                    "source": "thread.participants",
                    "values": sorted(anchor.participants),
                }
            )
        if anchor.keywords:
            sources.append(
                {"source": "thread.keywords", "values": sorted(anchor.keywords)}
            )
        if anchor.reason:
            sources.append({"source": "thread.reason", "content": anchor.reason})

        if req.motive == "initiate" and self._config.proactive_instruction:
            sources.append(
                {
                    "source": "proactive_instruction",
                    "content": self._config.proactive_instruction,
                }
            )

        messages = self._window.get_messages(req.group_id)
        total = len(messages)
        for index, msg in enumerate(messages):
            sources.append(
                {
                    "source": "window_message",
                    "window_index": index,
                    "from_latest": total - index - 1,
                    "sender_id": msg.sender_id,
                    "sender_name": msg.sender_name,
                    "timestamp": msg.timestamp,
                    "chars": len(msg.content),
                    "content": msg.content,
                }
            )
        return sources

    def clear_rejected_dynamic_context(
        self,
        group_id: str,
        dynamic_context_sources: list[dict[str, Any]] | None,
    ) -> SafetyCleanupResult:
        """清除被 Provider 拒绝的动态上下文，并保留请求期间的新消息。

        调用方应持有该群的 StateManager 锁，并在返回后持久化 dirty state。
        """
        sources = dynamic_context_sources or []
        timestamps = [
            item.get("timestamp")
            for item in sources
            if item.get("source") == "window_message"
            and isinstance(item.get("timestamp"), (int, float))
            and not isinstance(item.get("timestamp"), bool)
        ]
        window_removed = 0
        if timestamps:
            window_removed = self._window.discard_through(
                group_id,
                max(timestamps),
            )

        cleared = self._state.clear_decision_context(group_id)
        return SafetyCleanupResult(
            window_removed=window_removed,
            observation_cleared=cleared["observation"],
            anchor_cleared=cleared["anchor"],
            dynamic_source_count=len(sources),
        )

    async def decide(
        self,
        req: DecisionRequest,
        llm_generate: LlmGenerateFn,
        provider_id: str,
    ) -> DecisionOutcome:
        """执行一次决策调用。LLM 异常不抛出，以 error 字段返回。"""
        assert req.motive in VALID_MOTIVES, f"unknown motive: {req.motive}"
        user_prompt, system_prompt = self.build_prompt(req)
        dynamic_context_sources = self.collect_dynamic_context_sources(req)
        start = time.time()
        try:
            response = await llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            error_kind, retryable = classify_decision_error(e)
            outcome = DecisionOutcome(
                decision=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
                error_kind=error_kind,
                retryable=retryable,
                dynamic_context_sources=dynamic_context_sources,
            )
            _record_decision_log(req, provider_id, outcome)
            return outcome
        raw = response.completion_text or ""
        outcome = DecisionOutcome(
            decision=parse_decision(raw, mode=req.motive),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_text=raw,
            duration_ms=(time.time() - start) * 1000,
            dynamic_context_sources=dynamic_context_sources,
        )
        _record_decision_log(req, provider_id, outcome)
        return outcome
