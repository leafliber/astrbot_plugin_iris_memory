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
            return DecisionOutcome(
                decision=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        raw = response.completion_text or ""
        return DecisionOutcome(
            decision=parse_decision(raw, mode=req.motive),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_text=raw,
            duration_ms=(time.time() - start) * 1000,
        )
