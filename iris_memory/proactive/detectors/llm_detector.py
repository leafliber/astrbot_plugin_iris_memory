"""
L3 LLM 确认检测器

处理边缘复杂场景，仅用于中置信度案例。
使用极简 prompt，控制 token < 500。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from iris_memory.proactive.core.models import (
    LLMResult,
    ProactiveContext,
    ReplyType,
    UrgencyLevel,
    VectorResult,
)
from iris_memory.proactive.detectors.base import BaseDetector
from iris_memory.utils.logger import get_logger

logger = get_logger("proactive.detector.llm")

# LLM 确认 Prompt 模板（极简，< 500 tokens）
LLM_CONFIRM_PROMPT = """你是一个智能助手。根据以下对话上下文，判断是否应该主动回复。

对话摘要：
{recent_text}

当前情绪：{emotion}
会话类型：{session_type}
沉默时长：{silence}秒
最佳匹配场景：{best_scene}

请判断是否应该主动回复。保守策略：宁可漏过，不要骚扰用户。

输出JSON格式：
{{"should_reply": true/false, "urgency": "high/medium/low", "reason": "简短原因", "reply_type": "question/emotion/chat/followup"}}"""


class LLMDetector(BaseDetector):
    """L3 LLM 确认检测器

    触发条件：
    - VectorDetector 中置信区间（0.6-0.85）
    - 多轮跟进边缘案例

    通过配额管理器限流，避免过多 LLM 调用。
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        quota_manager: Optional[Any] = None,
        max_prompt_tokens: int = 500,
    ) -> None:
        super().__init__(name="llm")
        self._llm_provider = llm_provider
        self._quota_manager = quota_manager
        self._max_prompt_tokens = max_prompt_tokens

    async def detect(
        self,
        context: ProactiveContext,
        vector_result: Optional[VectorResult] = None,
    ) -> LLMResult:
        """执行 LLM 确认检测

        Args:
            context: 主动回复上下文
            vector_result: L2 向量检测结果（用于上下文参考）

        Returns:
            LLMResult 检测结果
        """
        if not self._llm_provider:
            logger.warning("LLM provider not available, skipping L3")
            return LLMResult(should_reply=False, reason="no_llm_provider")

        # 配额检查
        if self._quota_manager:
            can_use = await self._quota_manager.acquire(
                context.session_key,
                max_per_hour=5,
            )
            if not can_use:
                logger.debug(f"LLM quota exceeded for {context.session_key}")
                return LLMResult(
                    should_reply=False,
                    reason="llm_quota_exceeded",
                )

        # 构建 prompt
        prompt = self._build_prompt(context, vector_result)

        try:
            response = await self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return LLMResult(
                should_reply=False,
                reason=f"llm_error: {str(e)[:100]}",
            )

    def _build_prompt(
        self,
        context: ProactiveContext,
        vector_result: Optional[VectorResult],
    ) -> str:
        """构建极简 prompt"""
        # 截断 recent_text
        recent_text = context.conversation.recent_text[:300]

        emotion = "unknown"
        if context.user.emotional_state:
            emotion = getattr(
                context.user.emotional_state,
                "primary",
                getattr(context.user.emotional_state, "emotion", "unknown"),
            )

        best_scene = "none"
        if vector_result and vector_result.best_match:
            best_scene = vector_result.best_match.trigger_pattern[:100]

        return LLM_CONFIRM_PROMPT.format(
            recent_text=recent_text,
            emotion=emotion,
            session_type=context.session_type,
            silence=int(context.conversation.silence_duration),
            best_scene=best_scene,
        )

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if hasattr(self._llm_provider, "text_chat"):
            # AstrBot LLM provider 接口
            result = await self._llm_provider.text_chat(
                prompt=prompt,
                session_id="proactive_detector",
            )
            if hasattr(result, "completion_text"):
                return result.completion_text
            return str(result)
        elif callable(self._llm_provider):
            return await self._llm_provider(prompt)
        else:
            raise RuntimeError("Unsupported LLM provider type")

    @staticmethod
    def _parse_response(response: str) -> LLMResult:
        """解析 LLM 响应"""
        # 尝试从响应中提取 JSON
        try:
            # 查找 JSON 块
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                # 解析 urgency
                urgency_str = data.get("urgency", "low")
                try:
                    urgency = UrgencyLevel(urgency_str)
                except ValueError:
                    urgency = UrgencyLevel.LOW

                # 解析 reply_type
                reply_type_str = data.get("reply_type", "chat")
                try:
                    reply_type = ReplyType(reply_type_str)
                except ValueError:
                    reply_type = ReplyType.CHAT

                return LLMResult(
                    should_reply=bool(data.get("should_reply", False)),
                    urgency=urgency,
                    reason=data.get("reason", ""),
                    confidence=0.7 if data.get("should_reply") else 0.3,
                    reply_type=reply_type,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # 回退：保守策略
        return LLMResult(
            should_reply=False,
            reason="parse_failed",
            confidence=0.0,
        )
