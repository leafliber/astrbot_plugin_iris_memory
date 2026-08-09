"""
Iris Chat Memory - 人格自迭代阶段 A：风格归纳

输入脱敏语料与目标方向，输出文档 §9.1 的结构化风格画像（严格 JSON）。
原始语料只进入本阶段，且与指令明确分隔；JSON 容错解析；
confidence 低于配置阈值（默认 0.65）时停止本轮，返回低置信度标记。
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.core import get_logger
from .models import ErrorCode
from .prompts import build_analysis_prompt

logger = get_logger("persona_evolution.analyzer")

# verbosity / emoji_style 合法枚举（文档 §9.1）
_VERBOSITY_ENUM = {"short", "medium", "long"}
_EMOJI_STYLE_ENUM = {"none", "low", "medium", "high"}


def extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """容错提取 LLM 输出中的第一个 JSON 对象

    与 learning.reviewer._parse_verdicts 同一思路：
    正则截取最外层 {...} 块后 json.loads。

    Returns:
        解析出的 dict；失败或不是对象返回 None
    """
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _str_list(value: Any) -> List[str]:
    """宽松解析字符串列表：非列表/非字符串元素降级为空"""
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, str) and v.strip()]


def parse_style_profile(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """校验并规范化阶段 A 输出（文档 §9.1 schema）

    Returns:
        规范化后的风格画像；schema 不合法返回 None
    """
    verbosity = data.get("verbosity")
    emoji_style = data.get("emoji_style")
    if verbosity not in _VERBOSITY_ENUM or emoji_style not in _EMOJI_STYLE_ENUM:
        return None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    sentence_rhythm = data.get("sentence_rhythm")
    evidence_summary = data.get("evidence_summary")
    if not isinstance(sentence_rhythm, str) or not isinstance(evidence_summary, str):
        return None
    return {
        "tone": _str_list(data.get("tone")),
        "verbosity": verbosity,
        "sentence_rhythm": sentence_rhythm.strip(),
        "punctuation": _str_list(data.get("punctuation")),
        "emoji_style": emoji_style,
        "interaction_patterns": _str_list(data.get("interaction_patterns")),
        "humor_style": _str_list(data.get("humor_style")),
        "avoid_patterns": _str_list(data.get("avoid_patterns")),
        "confidence": confidence,
        "evidence_summary": evidence_summary.strip(),
    }


class StyleAnalyzer:
    """阶段 A：风格归纳（无状态，LLM 调用方注入）"""

    def __init__(self, min_confidence: float = 0.65):
        self._min_confidence = min_confidence

    async def analyze(
        self,
        llm_manager: Any,
        corpus_texts: List[str],
        goal_text: str,
        provider_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ErrorCode]]:
        """执行风格归纳

        Args:
            llm_manager: LLMManager 实例（generate_direct）
            corpus_texts: 脱敏语料文本（已按送模长度截断）
            goal_text: 目标方向文本
            provider_id: Job 指定 Provider，None 走模块配置

        Returns:
            (风格画像, None)：成功且置信度达标；
            (None, ErrorCode)：解析失败 / 低置信度 / Provider 异常
        """
        prompt = build_analysis_prompt(corpus_texts, goal_text)
        try:
            raw = await llm_manager.generate_direct(
                prompt,
                module="persona_evolution_analysis",
                provider_id=provider_id or None,
            )
        except asyncio.TimeoutError:
            logger.warning("阶段 A 风格归纳调用超时")
            return None, ErrorCode.PROVIDER_TIMEOUT
        except Exception as e:
            logger.warning(f"阶段 A 风格归纳调用失败：{e}")
            return None, ErrorCode.PROVIDER_ERROR

        data = extract_json_object(raw)
        if data is None:
            logger.warning("阶段 A 输出不是合法 JSON")
            return None, ErrorCode.ANALYSIS_PARSE_FAILED
        profile = parse_style_profile(data)
        if profile is None:
            logger.warning("阶段 A 输出 schema 校验失败")
            return None, ErrorCode.ANALYSIS_PARSE_FAILED

        if profile["confidence"] < self._min_confidence:
            logger.info(
                f"阶段 A 置信度 {profile['confidence']:.2f} "
                f"低于阈值 {self._min_confidence:.2f}，停止本轮"
            )
            return None, ErrorCode.ANALYSIS_LOW_CONFIDENCE
        return profile, None
