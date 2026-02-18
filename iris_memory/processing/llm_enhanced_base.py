"""
LLM增强基类
提供统一的LLM调用封装，支持多种检测模式的基类
"""
import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.utils.logger import get_logger
from iris_memory.utils.provider_utils import (
    extract_provider_id,
    get_default_provider,
    get_provider_by_id,
    normalize_provider_id,
)

logger = get_logger("llm_enhanced_base")

MAX_RETRIES = 2
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 5.0
BACKOFF_MULTIPLIER = 2.0


class DetectionMode(str, Enum):
    """检测模式"""
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class LLMCallResult:
    """LLM调用结果"""
    success: bool
    content: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0


class LLMEnhancedBase(ABC):
    """LLM增强基类
    
    提供统一的LLM调用封装，支持：
    - 延迟初始化provider
    - 每日调用限制
    - 重试机制
    - JSON响应解析
    - 多种检测模式（rule/llm/hybrid）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        daily_limit: int = 0,
        max_tokens: int = 300,
        temperature: float = 0.3,
    ):
        self._astrbot_context = astrbot_context
        self._configured_provider_id = normalize_provider_id(provider_id)
        self._mode = DetectionMode(mode) if isinstance(mode, str) else mode
        self._daily_limit = daily_limit
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        self._resolved_provider = None
        self._resolved_provider_id: Optional[str] = None
        self._provider_initialized = False
        
        self._call_date: Optional[date] = None
        self._call_count: int = 0
        
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0,
        }
    
    @property
    def mode(self) -> DetectionMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: DetectionMode):
        self._mode = DetectionMode(value) if isinstance(value, str) else value
    
    @property
    def is_llm_enabled(self) -> bool:
        return self._mode in (DetectionMode.LLM, DetectionMode.HYBRID)
    
    def _reset_daily_counter(self) -> None:
        today = date.today()
        if self._call_date != today:
            self._call_date = today
            self._call_count = 0
    
    def _is_within_limit(self) -> bool:
        if self._daily_limit <= 0:
            return True
        self._reset_daily_counter()
        return self._call_count < self._daily_limit
    
    @property
    def remaining_daily_calls(self) -> int:
        self._reset_daily_counter()
        if self._daily_limit <= 0:
            return -1
        return max(0, self._daily_limit - self._call_count)
    
    async def _resolve_provider(self) -> bool:
        if self._provider_initialized:
            return self._resolved_provider is not None
        
        self._provider_initialized = True
        
        if not self._astrbot_context:
            logger.debug("No AstrBot context available")
            return False
        
        try:
            if self._configured_provider_id and self._configured_provider_id not in ("", "default"):
                provider, resolved_id = get_provider_by_id(
                    self._astrbot_context, self._configured_provider_id
                )
                if provider:
                    self._resolved_provider = provider
                    self._resolved_provider_id = resolved_id
                    logger.info(f"LLM Enhanced provider resolved: {resolved_id}")
                    return True
                logger.warning(
                    f"Provider '{self._configured_provider_id}' not found, "
                    f"falling back to default"
                )
            
            provider, provider_id = get_default_provider(self._astrbot_context)
            if provider:
                self._resolved_provider = provider
                self._resolved_provider_id = provider_id or extract_provider_id(provider)
                logger.info(f"LLM Enhanced provider (default): {self._resolved_provider_id}")
                return True
        except Exception as e:
            logger.debug(f"Failed to resolve provider: {e}")
        
        return False
    
    async def _call_llm(self, prompt: str) -> LLMCallResult:
        if not await self._resolve_provider():
            return LLMCallResult(success=False, error="Provider not available")
        
        if not self._is_within_limit():
            return LLMCallResult(success=False, error="Daily limit reached")
        
        backoff = INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                result = await self._call_llm_once(prompt)
                if result.success:
                    self._call_count += 1
                    self._stats["successful_calls"] += 1
                    self._stats["total_tokens"] += result.tokens_used
                    return result
            except Exception as e:
                last_error = e
                logger.debug(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        
        self._stats["failed_calls"] += 1
        return LLMCallResult(success=False, error=str(last_error))
    
    async def _call_llm_once(self, prompt: str) -> LLMCallResult:
        self._stats["total_calls"] += 1
        
        ctx = self._astrbot_context
        pid = self._resolved_provider_id
        
        if ctx and hasattr(ctx, "llm_generate") and pid:
            try:
                resp = await ctx.llm_generate(
                    chat_provider_id=pid,
                    prompt=prompt,
                )
                if resp and hasattr(resp, "completion_text"):
                    text = (resp.completion_text or "").strip()
                    tokens = len(prompt) // 4 + len(text) // 4
                    return LLMCallResult(
                        success=True,
                        content=text,
                        parsed_json=self._parse_json_response(text),
                        tokens_used=tokens
                    )
            except Exception as e:
                logger.debug(f"llm_generate failed: {e}")
        
        provider = self._resolved_provider
        if provider and hasattr(provider, "text_chat"):
            try:
                resp = await provider.text_chat(prompt=prompt, context=[])
                if hasattr(resp, "completion_text"):
                    text = (resp.completion_text or "").strip()
                elif isinstance(resp, dict):
                    text = (resp.get("text", "") or resp.get("content", "")).strip()
                else:
                    text = str(resp).strip() if resp else ""
                
                tokens = len(prompt) // 4 + len(text) // 4
                return LLMCallResult(
                    success=True,
                    content=text,
                    parsed_json=self._parse_json_response(text),
                    tokens_used=tokens
                )
            except Exception as e:
                logger.debug(f"text_chat failed: {e}")
        
        return LLMCallResult(success=False, error="No suitable LLM method found")
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if json_match:
                return json.loads(json_match.group(1))
            
            json_match = re.search(r"(\{[\s\S]*?\})", response)
            if json_match:
                return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "daily_limit": self._daily_limit,
            "remaining_calls": self.remaining_daily_calls,
            "mode": self._mode.value,
        }
    
    @abstractmethod
    async def detect(self, *args, **kwargs) -> Any:
        """子类实现的检测方法"""
        pass
    
    @abstractmethod
    def _rule_detect(self, *args, **kwargs) -> Any:
        """子类实现的规则检测方法"""
        pass
