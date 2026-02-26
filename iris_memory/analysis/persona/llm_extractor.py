"""
LLM 提取器 - 基于 LLM 的画像提取
"""

import json
from typing import Dict, Any, Optional

from iris_memory.utils.logger import get_logger
from iris_memory.utils.llm_helper import resolve_llm_provider, call_llm, parse_llm_json
from iris_memory.utils.rate_limiter import DailyCallLimiter
from iris_memory.core.provider_utils import normalize_provider_id
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

        self._limiter = DailyCallLimiter(daily_limit)

        # 缓存 provider
        self._resolved_provider = None
        self._resolved_provider_id: Optional[str] = None

    def _is_within_limit(self) -> bool:
        """检查是否在每日限制内"""
        return self._limiter.is_within_limit()

    async def _resolve_provider(self) -> bool:
        """解析 LLM 提供者（委托给 llm_helper）"""
        if self._resolved_provider is not None:
            return True

        provider, resolved_id = resolve_llm_provider(
            self._astrbot_context,
            self._provider_id or "",
            label="Persona",
        )
        if provider:
            self._resolved_provider = provider
            self._resolved_provider_id = resolved_id
            return True
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

            self._limiter.increment()
            parsed = self._parse_response(response)
            if parsed:
                return parsed
        except Exception as e:
            logger.warning(f"Persona LLM extraction failed: {e}")

        return result

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM（委托给 llm_helper）"""
        result = await call_llm(
            self._astrbot_context,
            self._resolved_provider,
            self._resolved_provider_id,
            prompt,
        )
        return result.content if result.success and result.content else None

    @staticmethod
    def _parse_response(response: str) -> Optional[ExtractionResult]:
        """解析 LLM JSON 响应为 ExtractionResult"""
        raw = parse_llm_json(response)
        if not raw:
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

        # ── v2 新增维度解析 ──

        # directness
        dn = raw.get("directness")
        if dn is not None and isinstance(dn, (int, float)):
            result.directness_adjustment = max(-1.0, min(1.0, float(dn)))

        # humor
        hm = raw.get("humor")
        if hm is not None and isinstance(hm, (int, float)):
            result.humor_adjustment = max(-1.0, min(1.0, float(hm)))

        # empathy
        em = raw.get("empathy")
        if em is not None and isinstance(em, (int, float)):
            result.empathy_adjustment = max(-1.0, min(1.0, float(em)))

        # work_style
        ws = raw.get("work_style")
        if ws and isinstance(ws, str) and ws.lower() != "null":
            result.work_style = ws

        # work_challenge
        wc = raw.get("work_challenge")
        if wc and isinstance(wc, str) and wc.lower() != "null":
            result.work_challenge = wc

        # lifestyle
        ls = raw.get("lifestyle")
        if ls and isinstance(ls, str) and ls.lower() != "null":
            result.lifestyle = ls

        # emotional_triggers
        et = raw.get("emotional_triggers")
        if isinstance(et, list):
            result.emotional_triggers = [str(t) for t in et if t]

        # emotional_soothers
        es = raw.get("emotional_soothers")
        if isinstance(es, dict):
            result.emotional_soothers = {
                str(k): str(v) for k, v in es.items() if k and v
            }

        # social_boundaries
        sb = raw.get("social_boundaries")
        if isinstance(sb, dict):
            result.social_boundaries = {
                str(k): str(v) for k, v in sb.items() if k and v
            }

        # personality (Big Five)
        personality = raw.get("personality")
        if isinstance(personality, dict):
            for trait in ("openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"):
                val = personality.get(trait)
                if val is not None and isinstance(val, (int, float)):
                    clamped = max(-0.1, min(0.1, float(val)))
                    setattr(result, f"personality_{trait}_delta", clamped)

        # proactive_reply
        pr = raw.get("proactive_reply")
        if pr is not None and isinstance(pr, (int, float)):
            result.proactive_reply_delta = max(-0.1, min(0.1, float(pr)))

        # trust_delta - 信任度增量
        td = raw.get("trust_delta")
        if td is not None and isinstance(td, (int, float)):
            result.trust_delta = max(0.0, min(0.1, float(td)))

        # intimacy_delta - 亲密度增量
        id_ = raw.get("intimacy_delta")
        if id_ is not None and isinstance(id_, (int, float)):
            result.intimacy_delta = max(0.0, min(0.1, float(id_)))

        return result

    @property
    def remaining_daily_calls(self) -> int:
        return self._limiter.remaining
