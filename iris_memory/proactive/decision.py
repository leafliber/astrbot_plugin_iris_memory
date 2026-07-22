from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .config import ConfigManager
from .parser import Decision, parse_decision
from .perception import ContextPackager, SlidingWindow
from .prompts import MOTIVE_INSTRUCTIONS, VALID_MOTIVES, WILLINGNESS_PROMPTS
from .state import StateManager, ThreadAnchor

# llm_generate(chat_provider_id=..., prompt=..., system_prompt=...) -> response
LlmGenerateFn = Callable[..., Awaitable[Any]]

_MOTIVE_LABELS = {
    "chime_in": "插话",
    "follow_up": "跟进",
    "initiate": "主动发起",
    "watch": "跟进评估",
}


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
                user_prompt=outcome.user_prompt,
                raw_response=outcome.raw_text,
                duration_ms=round(outcome.duration_ms, 1),
                error=outcome.error,
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
    ) -> None:
        self._config = config
        self._state = state
        self._window = window
        self._packager = packager

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
        return user_prompt, prompts["decision_system"]

    async def decide(
        self,
        req: DecisionRequest,
        llm_generate: LlmGenerateFn,
        provider_id: str,
    ) -> DecisionOutcome:
        """执行一次决策调用。LLM 异常不抛出，以 error 字段返回。"""
        assert req.motive in VALID_MOTIVES, f"unknown motive: {req.motive}"
        user_prompt, system_prompt = self.build_prompt(req)
        start = time.time()
        try:
            response = await llm_generate(
                chat_provider_id=provider_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            outcome = DecisionOutcome(
                decision=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
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
        )
        _record_decision_log(req, provider_id, outcome)
        return outcome
