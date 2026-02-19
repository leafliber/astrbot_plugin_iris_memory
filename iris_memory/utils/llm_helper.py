"""
统一 LLM 调用工具

将散布在 llm_processor / llm_extractor / llm_enhanced_base 中的
``llm_generate → text_chat`` fallback 模式和 JSON 解析逻辑
集中到一处，避免散弹修改。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from iris_memory.utils.logger import get_logger
from iris_memory.core.provider_utils import (
    normalize_provider_id,
    extract_provider_id,
    get_provider_by_id,
    get_default_provider,
)

logger = get_logger("llm_helper")


# ── Provider / Context 协议 ────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """LLM 提供者协议"""
    async def text_chat(self, *, prompt: str, context: list[Any]) -> Any: ...


@runtime_checkable
class AstrBotContext(Protocol):
    """AstrBot 上下文协议（仅 LLM 相关部分）"""
    async def llm_generate(self, *, chat_provider_id: str, prompt: str) -> Any: ...

# 延迟加载 token 估算（避免循环导入）
_token_estimator = None


def _estimate_tokens(text: str) -> int:
    """使用 token_manager 的精确估算（tiktoken 优先，加权启发式兜底）"""
    global _token_estimator
    if _token_estimator is None:
        try:
            from iris_memory.utils.token_manager import TokenBudget
            _token_estimator = TokenBudget()
        except Exception:
            _token_estimator = "fallback"
    if _token_estimator == "fallback":
        # 简单加权估算：中文 ~1.5 char/token, 英文 ~4 char/token
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other = len(text) - chinese
        return int(chinese / 1.5 + other / 4) or 1
    return _token_estimator.estimate_tokens(text)


# ── 数据类型 ────────────────────────────────────────────

@dataclass
class LLMCallResult:
    """LLM 调用结果"""
    success: bool = False
    content: str = ""
    parsed_json: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    error: str = ""


# ── JSON 解析 ────────────────────────────────────────────

def parse_llm_json(response: Optional[str]) -> Optional[Dict[str, Any]]:
    """从 LLM 响应文本中提取 JSON 字典。

    三级尝试：
    1. 直接 ``json.loads``
    2. 从 markdown 代码块 (````json ... ````) 提取
    3. 匹配第一个 ``{...}`` 子串

    Returns:
        解析成功返回 dict，否则 None。
    """
    if not response:
        return None

    # 1) 直接尝试
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) code-fenced JSON
    try:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if m:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 3) 裸 {...}（非贪婪）
    try:
        m = re.search(r"(\{[\s\S]*?\})", response)
        if m:
            result = json.loads(m.group(1))
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    return None


# ── Provider 解析 ────────────────────────────────────────

def resolve_llm_provider(
    context: Optional[AstrBotContext],
    provider_id: str = "",
    *,
    label: str = "LLM",
) -> tuple[Optional[LLMProvider], Optional[str]]:
    """解析 LLM 提供者。

    Args:
        context: AstrBot 上下文
        provider_id: 期望的 provider ID（"" / "default" 表示使用默认）
        label: 用于日志的标签

    Returns:
        (provider, resolved_id) 或 (None, None)
    """
    if not context:
        return None, None

    pid = normalize_provider_id(provider_id)

    if pid and pid not in ("", "default"):
        try:
            provider, resolved_id = get_provider_by_id(context, pid)
            if provider:
                logger.debug(f"[{label}] Provider resolved: {resolved_id or pid}")
                return provider, resolved_id or pid
            logger.warning(f"[{label}] Provider '{pid}' not found, falling back to default")
        except Exception as e:
            logger.warning(f"[{label}] Error resolving provider '{pid}': {e}")

    provider, resolved_id = get_default_provider(context)
    if provider:
        resolved_id = resolved_id or extract_provider_id(provider)
        logger.debug(f"[{label}] Using default provider: {resolved_id}")
    return provider, resolved_id


# ── 单次 LLM 调用 ────────────────────────────────────────


async def call_llm(
    context: Optional[AstrBotContext],
    provider: Optional[LLMProvider],
    provider_id: Optional[str],
    prompt: str,
    *,
    parse_json: bool = False,
) -> LLMCallResult:
    """统一的单次 LLM 调用（``llm_generate`` → ``text_chat`` fallback）。

    Args:
        context: AstrBot 上下文（可能为 None）
        provider: 已解析的 provider 实例（可能为 None）
        provider_id: provider ID（用于 ``llm_generate``）
        prompt: 提示词
        parse_json: 是否尝试从响应中解析 JSON

    Returns:
        LLMCallResult
    """
    # ① 尝试 context.llm_generate
    if context and hasattr(context, "llm_generate") and provider_id:
        try:
            resp = await context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            if resp and hasattr(resp, "completion_text"):
                text = (resp.completion_text or "").strip()
                tokens = _estimate_tokens(prompt + text)
                return LLMCallResult(
                    success=True,
                    content=text,
                    parsed_json=parse_llm_json(text) if parse_json else None,
                    tokens_used=tokens,
                )
        except Exception as e:
            logger.debug(f"llm_generate failed: {e}")

    # ② 回退：provider.text_chat
    if provider and hasattr(provider, "text_chat"):
        try:
            resp = await provider.text_chat(prompt=prompt, context=[])
            text = _extract_text(resp)
            tokens = _estimate_tokens(prompt + text)
            return LLMCallResult(
                success=True,
                content=text,
                parsed_json=parse_llm_json(text) if parse_json else None,
                tokens_used=tokens,
            )
        except Exception as e:
            logger.warning(f"text_chat failed: {e}")

    return LLMCallResult(success=False, error="No suitable LLM method found")


def _extract_text(resp: Any) -> str:
    """从各种 LLM 响应形态中提取文本"""
    if hasattr(resp, "completion_text"):
        return (resp.completion_text or "").strip()
    if isinstance(resp, dict):
        return (resp.get("text", "") or resp.get("content", "")).strip()
    return str(resp).strip() if resp else ""
