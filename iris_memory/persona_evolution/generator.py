"""
Iris Chat Memory - 人格自迭代阶段 B：候选生成

输入当前 system_prompt、编辑模式、目标快照、阶段 A 结构化风格画像、
区块/全文规则与 protected_fragments，输出文档 §9.2 的候选 JSON。
本阶段不接触原始语料，切断不可信语料到人格发布的直接指令通路。
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core import get_logger
from .analyzer import extract_json_object
from .models import ErrorCode
from .prompts import build_generation_prompt
from iris_memory.llm_modules import PERSONA_EVOLUTION_GENERATE

logger = get_logger("persona_evolution.generator")


def parse_generation(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验并规范化阶段 B 输出（文档 §9.2 schema）

    Returns:
        {candidate_prompt, change_summary, rationale, confidence}；
        schema 不合法返回 None
    """
    candidate = data.get("candidate_prompt")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    change_summary = data.get("change_summary")
    if not isinstance(change_summary, list):
        return None
    change_summary = [str(v) for v in change_summary if isinstance(v, str)]
    rationale = data.get("rationale")
    if not isinstance(rationale, str):
        return None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return {
        "candidate_prompt": candidate,
        "change_summary": change_summary,
        "rationale": rationale.strip(),
        "confidence": confidence,
    }


class CandidateGenerator:
    """阶段 B：候选人格生成（无状态，LLM 调用方注入）"""

    async def generate(
        self,
        llm_manager: Any,
        *,
        current_prompt: str,
        edit_mode: str,
        goal_snapshot: Dict[str, Any],
        style_profile: Dict[str, Any],
        protected_fragments: List[str],
        block_max_chars: int = 1500,
        max_change_ratio: float = 0.20,
        full_max_growth_ratio: float = 1.25,
        full_max_length: int = 20000,
        provider_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ErrorCode]]:
        """生成候选人格

        Returns:
            (候选字典, None)：成功；
            (None, ErrorCode)：解析失败 / Provider 异常
        """
        prompt = build_generation_prompt(
            current_prompt=current_prompt,
            edit_mode=edit_mode,
            goal_snapshot=goal_snapshot,
            style_profile=style_profile,
            protected_fragments=protected_fragments,
            block_max_chars=block_max_chars,
            max_change_ratio=max_change_ratio,
            full_max_growth_ratio=full_max_growth_ratio,
            full_max_length=full_max_length,
        )
        try:
            raw = await llm_manager.generate_direct(
                prompt,
                module=PERSONA_EVOLUTION_GENERATE,
                provider_id=provider_id or None,
            )
        except asyncio.TimeoutError:
            logger.warning("阶段 B 候选生成调用超时")
            return None, ErrorCode.PROVIDER_TIMEOUT
        except Exception as e:
            logger.warning(f"阶段 B 候选生成调用失败：{e}")
            return None, ErrorCode.PROVIDER_ERROR

        data = extract_json_object(raw)
        if data is None:
            logger.warning("阶段 B 输出不是合法 JSON")
            return None, ErrorCode.GENERATION_PARSE_FAILED
        result = parse_generation(data)
        if result is None:
            logger.warning("阶段 B 输出 schema 校验失败")
            return None, ErrorCode.GENERATION_PARSE_FAILED
        return result, None
