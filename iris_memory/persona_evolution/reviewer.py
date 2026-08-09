"""
Iris Chat Memory - 人格自迭代阶段 C：完整人格独立审查

仅 full_prompt 模式强制执行。审查调用不复用生成调用的上下文，
输入修改前后人格、目标与结构化风格画像，不接触原始语料。
默认阈值（文档 §9.3）：身份≥0.80 / 约束≥0.90 / 目标≥0.70 /
隐私≥0.90，且 prompt_injection_suspected=false。
阶段 C 是辅助闸门，不替代代码校验。
"""

import asyncio
from typing import Any, Dict, Optional, Tuple

from iris_memory.core import get_logger
from .analyzer import extract_json_object
from .models import ErrorCode
from .prompts import build_review_prompt

logger = get_logger("persona_evolution.reviewer")

# 文档 §9.3 默认阈值
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "identity_consistency": 0.80,
    "constraint_preservation": 0.90,
    "goal_alignment": 0.70,
    "privacy_safety": 0.90,
}

_SCORE_FIELDS = tuple(DEFAULT_THRESHOLDS)


def parse_review(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验并规范化阶段 C 输出

    Returns:
        审查结果字典（含 pass / reasons）；schema 不合法返回 None
    """
    scores: Dict[str, float] = {}
    for field in _SCORE_FIELDS:
        try:
            score = float(data.get(field))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= score <= 1.0:
            return None
        scores[field] = score
    injection = data.get("prompt_injection_suspected")
    passed = data.get("pass")
    if not isinstance(injection, bool) or not isinstance(passed, bool):
        return None
    reasons = data.get("reasons")
    if not isinstance(reasons, list):
        return None
    return {
        **scores,
        "prompt_injection_suspected": injection,
        "pass": passed,
        "reasons": [str(r) for r in reasons if isinstance(r, str)],
    }


def review_passed(
    review: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> bool:
    """按阈值判定审查是否通过（含 pass 字段与注入标记）"""
    limits = thresholds or DEFAULT_THRESHOLDS
    if review.get("prompt_injection_suspected"):
        return False
    if not review.get("pass"):
        return False
    for field, minimum in limits.items():
        if float(review.get(field, 0.0)) < minimum:
            return False
    return True


class PromptReviewer:
    """阶段 C：完整人格独立审查（无状态）"""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self._thresholds = thresholds or DEFAULT_THRESHOLDS

    async def review(
        self,
        llm_manager: Any,
        *,
        base_prompt: str,
        candidate_prompt: str,
        goal_snapshot: Dict[str, Any],
        style_profile: Dict[str, Any],
        provider_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ErrorCode]]:
        """执行独立审查

        Returns:
            (审查结果, None)：审查通过；
            (审查结果, REVIEW_FAILED)：达到阈值判定但未通过；
            (None, REVIEW_PARSE_FAILED / PROVIDER_ERROR)：解析或调用失败
        """
        prompt = build_review_prompt(
            base_prompt=base_prompt,
            candidate_prompt=candidate_prompt,
            goal_snapshot=goal_snapshot,
            style_profile=style_profile,
        )
        try:
            raw = await llm_manager.generate_direct(
                prompt,
                module="persona_evolution_review",
                provider_id=provider_id or None,
            )
        except asyncio.TimeoutError:
            logger.warning("阶段 C 独立审查调用超时")
            return None, ErrorCode.PROVIDER_TIMEOUT
        except Exception as e:
            logger.warning(f"阶段 C 独立审查调用失败：{e}")
            return None, ErrorCode.PROVIDER_ERROR

        data = extract_json_object(raw)
        if data is None:
            logger.warning("阶段 C 输出不是合法 JSON")
            return None, ErrorCode.REVIEW_PARSE_FAILED
        result = parse_review(data)
        if result is None:
            logger.warning("阶段 C 输出 schema 校验失败")
            return None, ErrorCode.REVIEW_PARSE_FAILED

        if not review_passed(result, self._thresholds):
            logger.info(f"阶段 C 审查未通过：{result.get('reasons')}")
            return result, ErrorCode.REVIEW_FAILED
        return result, None
