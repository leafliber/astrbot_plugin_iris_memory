"""
LLM 提取器 - 基于 LLM 的画像提取
"""

import json
from datetime import date
from typing import Dict, Any, Optional

from iris_memory.utils.logger import get_logger
from iris_memory.utils.provider_utils import (
    extract_provider_id,
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)
from iris_memory.analysis.persona.keyword_maps import ExtractionResult, PERSONA_EXTRACTION_PROMPT

logger = get_logger("persona_extractor")


class LLMExtractor:
    """基于 LLM 的画像提取"""

    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        max_tokens: int = 300,
        daily_limit: int = 50,
    ):
        self._astrbot_context = astrbot_context
        self._provider_id = normalize_provider_id(provider_id)  # "default" 或具体 provider_id
        self._max_tokens = max_tokens
        self._daily_limit = daily_limit

        # 每日调用计数
        self._call_date: Optional[date] = None
        self._call_count: int = 0

        # 缓存 provider
        self._resolved_provider = None
        self._resolved_provider_id: Optional[str] = None

    def _reset_daily_counter(self) -> None:
        """日期翻转时重置计数器"""
        today = date.today()
        if self._call_date != today:
            self._call_date = today
            self._call_count = 0

    def _is_within_limit(self) -> bool:
        """检查是否在每日限制内"""
        self._reset_daily_counter()
        return self._call_count < self._daily_limit

    async def _resolve_provider(self) -> bool:
        """解析 LLM 提供者

        支持：
        - provider_id == "default" 或 None → 使用 AstrBot 默认提供者
        - provider_id == 具体 ID → 查找并使用该提供者
        """
        if self._resolved_provider is not None:
            return True
        if not self._astrbot_context:
            return False

        try:
            # 指定了具体 provider_id → 尝试匹配
            if self._provider_id and self._provider_id != "default":
                provider, resolved_id = get_provider_by_id(self._astrbot_context, self._provider_id)
                if provider:
                    self._resolved_provider = provider
                    self._resolved_provider_id = resolved_id
                    logger.info(f"Persona LLM provider resolved: {resolved_id}")
                    return True
                logger.warning(
                    f"Persona LLM provider '{self._provider_id}' not found, "
                    f"falling back to default"
                )

            # 默认提供者
            provider, provider_id = get_default_provider(self._astrbot_context)
            if provider:
                self._resolved_provider = provider
                self._resolved_provider_id = provider_id or extract_provider_id(provider)
                logger.info(f"Persona LLM provider (default): {self._resolved_provider_id}")
                return True
        except Exception as e:
            logger.debug(f"Failed to resolve persona LLM provider: {e}")
        return False

    async def extract(
        self,
        content: str,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """使用 LLM 从文本中提取画像信息"""
        result = ExtractionResult(source="llm")

        if not self._is_within_limit():
            logger.debug("Persona LLM daily limit reached, skipping")
            return result

        if not await self._resolve_provider():
            logger.debug("Persona LLM provider not available")
            return result

        try:
            prompt = PERSONA_EXTRACTION_PROMPT.format(content=content[:1000])
            response = await self._call_llm(prompt)
            if not response:
                return result

            self._call_count += 1
            parsed = self._parse_response(response)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Persona LLM extraction failed: {e}")

        return result

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM"""
        ctx = self._astrbot_context
        pid = self._resolved_provider_id

        # 优先使用 llm_generate
        if ctx and hasattr(ctx, "llm_generate") and pid:
            try:
                resp = await ctx.llm_generate(
                    chat_provider_id=pid,
                    prompt=prompt,
                )
                if resp and hasattr(resp, "completion_text"):
                    return (resp.completion_text or "").strip()
            except Exception as e:
                logger.debug(f"llm_generate failed for persona: {e}")

        # 回退: text_chat
        provider = self._resolved_provider
        if provider and hasattr(provider, "text_chat"):
            try:
                resp = await provider.text_chat(prompt=prompt, context=[])
                if hasattr(resp, "completion_text"):
                    return (resp.completion_text or "").strip()
                if isinstance(resp, dict):
                    return (resp.get("text", "") or resp.get("content", "")).strip()
                return str(resp).strip() if resp else None
            except Exception as e:
                logger.debug(f"text_chat failed for persona: {e}")

        return None

    @staticmethod
    def _parse_response(response: str) -> Optional[ExtractionResult]:
        """解析 LLM JSON 响应为 ExtractionResult"""
        import re

        raw: Optional[Dict] = None
        # 直接 JSON
        try:
            raw = json.loads(response)
        except json.JSONDecodeError:
            pass

        if raw is None:
            # code-fenced JSON
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if m:
                try:
                    raw = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if raw is None:
            # 裸 {...}
            m = re.search(r"(\{[\s\S]*\})", response)
            if m:
                try:
                    raw = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if not raw or not isinstance(raw, dict):
            return None

        result = ExtractionResult(source="llm")

        # interests
        interests = raw.get("interests")
        if isinstance(interests, dict):
            result.interests = {
                k: max(0.0, min(1.0, float(v)))
                for k, v in interests.items()
                if isinstance(v, (int, float))
            }

        # social_style
        ss = raw.get("social_style")
        if ss and isinstance(ss, str) and ss.lower() != "null":
            result.social_style = ss

        # reply_preference
        rp = raw.get("reply_preference")
        if rp and isinstance(rp, str) and rp.lower() in ("brief", "detailed"):
            result.reply_style_preference = rp.lower()

        # formality
        fm = raw.get("formality")
        if fm is not None and isinstance(fm, (int, float)):
            result.formality_adjustment = max(-1.0, min(1.0, float(fm)))

        # topic_blacklist
        tb = raw.get("topic_blacklist")
        if isinstance(tb, list):
            result.topic_blacklist = [str(t) for t in tb if t]

        # work / life info
        wi = raw.get("work_info")
        if wi and isinstance(wi, str) and wi.lower() != "null":
            result.work_info = wi
        li = raw.get("life_info")
        if li and isinstance(li, str) and li.lower() != "null":
            result.life_info = li

        # confidence
        conf = raw.get("confidence")
        if conf is not None and isinstance(conf, (int, float)):
            result.confidence = max(0.0, min(1.0, float(conf)))

        return result

    @property
    def remaining_daily_calls(self) -> int:
        self._reset_daily_counter()
        return max(0, self._daily_limit - self._call_count)
