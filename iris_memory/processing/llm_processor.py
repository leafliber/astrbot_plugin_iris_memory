"""
LLM消息处理器
使用AstrBot默认LLM进行消息分类和摘要生成
"""
import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger
from iris_memory.utils.llm_helper import (
    resolve_llm_provider,
    call_llm,
    parse_llm_json,
)
from iris_memory.core.provider_utils import (
    extract_provider_id,
    normalize_provider_id,
)
from iris_memory.core.constants import LLMRetryConfig, CircuitBreakerConfig, LLMRateLimitConfig
from iris_memory.utils.rate_limiter import DailyCallLimiter

logger = get_logger("llm_processor")


# 从统一常量获取
MAX_RETRIES = LLMRetryConfig.MAX_RETRIES
INITIAL_BACKOFF = LLMRetryConfig.INITIAL_BACKOFF
MAX_BACKOFF = LLMRetryConfig.MAX_BACKOFF
BACKOFF_MULTIPLIER = LLMRetryConfig.BACKOFF_MULTIPLIER
LLM_CALL_TIMEOUT = LLMRetryConfig.CALL_TIMEOUT

CIRCUIT_BREAKER_FAILURE_THRESHOLD = CircuitBreakerConfig.FAILURE_THRESHOLD
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = CircuitBreakerConfig.RECOVERY_TIMEOUT
CIRCUIT_BREAKER_HALF_OPEN_MAX = CircuitBreakerConfig.HALF_OPEN_MAX


@dataclass
class LLMClassificationResult:
    """LLM分类结果"""
    layer: str  # "immediate", "batch", "discard"
    confidence: float
    reason: str
    metadata: Dict[str, Any]


@dataclass
class LLMSummaryResult:
    """LLM摘要结果"""
    summary: str
    key_points: List[str]
    user_preferences: List[str]
    token_used: int


class LLMMessageProcessor:
    """LLM消息处理器
    
    注意：AstrBot的provider在插件加载后才初始化，因此采用延迟初始化策略。
    initialize()方法不立即检查provider可用性，而是在实际使用时按需获取。
    """
    
    def __init__(
        self,
        astrbot_context=None,
        classification_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        max_tokens: int = 200,
        provider_id: Optional[str] = None
    ):
        self.astrbot_context = astrbot_context
        self._configured_provider_id = normalize_provider_id(provider_id)  # 配置指定的 provider_id
        self.classification_prompt = classification_prompt or (
            "分析以下用户消息，判断其记忆价值。\n"
            "考虑因素：是否包含用户偏好、情感、重要事实、个人信息等。\n"
            "layer字段选择：\n"
            "- immediate: 需要立即处理的高价值消息（用户明确表达偏好、重要情感、关键信息）\n"
            "- batch: 普通消息，可以批量处理\n"
            "- discard: 无价值消息（闲聊、问候、确认等）\n"
            "回复严格的JSON格式：\n"
            '{"layer": "immediate|batch|discard", "confidence": 0.8, "reason": "原因说明"}'
        )
        self.summary_prompt = summary_prompt or (
            "总结以下对话内容，提取关键信息和用户偏好。\n"
            "要求：\n"
            "1. 简洁明了，不超过100字\n"
            "2. 突出用户的观点和偏好\n"
            "3. 忽略无意义的寒暄\n\n"
            "回复严格的JSON格式：\n"
            '{"summary": "摘要内容", "key_points": ["要点1", "要点2"], "user_preferences": ["偏好1"]}'
        )
        self.max_tokens = max_tokens
        self.llm_api = None
        self.default_provider_id = None  # 默认提供商ID，用于新API调用
        
        # 延迟初始化相关
        self._init_attempted = False  # 是否已尝试初始化
        self._init_retry_count = 0
        self._max_init_retries = 3
        
        # 速率限制
        self._rate_limiter = DailyCallLimiter(
            daily_limit=LLMRateLimitConfig.DAILY_CALL_LIMIT
        )
        
        # 统计信息
        self.stats = {
            "classification_calls": 0,
            "summary_calls": 0,
            "failed_calls": 0,
            "total_tokens_used": 0,
            "circuit_breaker_rejections": 0,
            "rate_limit_rejections": 0,
        }
        
        # 熔断器状态
        self._cb_failure_count = 0
        self._cb_last_failure_time = 0.0
        self._cb_state = "closed"  # "closed" | "open" | "half_open"
    
    async def initialize(self) -> bool:
        """初始化LLM API（延迟初始化策略）
        
        由于 AstrBot 的 provider 在插件加载后才初始化，
        此方法仅标记处理器已准备好，实际 provider 获取延迟到第一次使用时。
        
        Returns:
            bool: 始终返回 True（处理器已准备好，provider 将在使用时按需获取）
        """
        if not self.astrbot_context:
            logger.info("AstrBot context not available, LLM features disabled")
            return False
        
        logger.debug("LLM processor initialized (provider will be loaded on first use)")
        return True
    
    async def _try_init_provider(self) -> bool:
        """尝试初始化 provider（按需调用）
        
        支持通过配置指定 provider_id，优先级：
        1. 初始化时传入的 provider_id 参数
        2. 未指定或为空则使用 AstrBot 默认 provider
        
        Returns:
            bool: 是否成功获取 provider
        """
        # 如果已经初始化成功，直接返回
        if self.llm_api is not None:
            return True
        
        # 如果已尝试太多次，不再重试
        if self._init_retry_count >= self._max_init_retries:
            return False
        
        self._init_retry_count += 1
        
        try:
            # 解析 provider
            provider = await self._resolve_provider()
            
            if provider:
                self.llm_api = provider
                self.default_provider_id = extract_provider_id(self.llm_api)
                logger.debug(
                    f"LLM provider loaded on demand: {self.default_provider_id} "
                    f"(attempt {self._init_retry_count})"
                )
                return True
            
            logger.debug(
                f"No LLM providers available yet "
                f"(attempt {self._init_retry_count}/{self._max_init_retries})"
            )
            return False
        except Exception as e:
            logger.warning(
                f"Failed to load LLM provider "
                f"(attempt {self._init_retry_count}/{self._max_init_retries}): {e}"
            )
            return False
    
    async def _resolve_provider(self):
        """解析 LLM 提供者（委托给 llm_helper）"""
        provider, _ = resolve_llm_provider(
            self.astrbot_context,
            self._configured_provider_id,
            label="LLMProcessor",
        )
        return provider
    
    async def classify_message(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> Optional[LLMClassificationResult]:
        """使用LLM分类消息"""
        # 尝试按需初始化 provider
        if not await self._try_init_provider():
            return None
        
        try:
            prompt = self._build_classification_prompt(message, context)
            response = await self._call_llm(prompt, max_tokens=150)
            
            if not response:
                return None
            
            self.stats["classification_calls"] += 1
            result = self._parse_json_response(response)
            
            if not result:
                return None
            
            layer = result.get("layer", "batch")
            if layer not in ["immediate", "batch", "discard"]:
                layer = "batch"
            
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return LLMClassificationResult(
                layer=layer,
                confidence=confidence,
                reason=result.get("reason", "LLM classified"),
                metadata={"raw_response": response}
            )
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None
    
    async def generate_summary(
        self,
        messages: List[str],
        user_id: str,
        context: Optional[Dict] = None
    ) -> Optional[LLMSummaryResult]:
        """使用LLM生成批量消息摘要"""
        # 尝试按需初始化 provider
        if not await self._try_init_provider():
            return None
        
        if not messages:
            return None
        
        try:
            prompt = self._build_summary_prompt(messages, context)
            response = await self._call_llm(prompt, max_tokens=self.max_tokens)
            
            if not response:
                return None
            
            self.stats["summary_calls"] += 1
            result = self._parse_json_response(response)
            
            if not result:
                return LLMSummaryResult(
                    summary=response[:500],
                    key_points=[],
                    user_preferences=[],
                    token_used=len(response) // 4
                )
            
            return LLMSummaryResult(
                summary=result.get("summary", response[:500]),
                key_points=result.get("key_points", []),
                user_preferences=result.get("user_preferences", []),
                token_used=len(response) // 4
            )
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return None
    
    def _build_classification_prompt(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> str:
        """构建分类提示词"""
        ctx_str = ""
        if context:
            session_count = context.get("session_message_count", 0)
            last_topic = context.get("last_topic", "")
            if last_topic:
                ctx_str = f"\n上下文：当前会话第{session_count}条消息，上一话题：{last_topic}"
        
        return f"""{self.classification_prompt}{ctx_str}

用户消息：
```
{message}
```

分析："""
    
    def _build_summary_prompt(
        self,
        messages: List[str],
        context: Optional[Dict] = None
    ) -> str:
        """构建摘要提示词"""
        formatted_messages = "\n".join([
            f"{i+1}. {msg}" for i, msg in enumerate(messages[-10:])
        ])
        
        ctx_str = ""
        if context:
            user_persona = context.get("user_persona", {})
            if user_persona:
                ctx_str = f"\n用户画像：{user_persona}"
        
        return f"""{self.summary_prompt}{ctx_str}

对话内容：
{formatted_messages}

总结："""
    
    async def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3
    ) -> Optional[str]:
        """调用LLM API（带指数退避重试 + 熔断器机制）

        熔断器状态流转：
        - closed: 正常工作，失败计数 >= 阈值时进入 open
        - open: 拒绝所有请求，超过恢复超时后进入 half_open
        - half_open: 允许少量请求试探，成功则 closed，失败则 open

        Args:
            prompt: 提示词
            max_tokens: 最大返回token数
            temperature: 温度参数

        Returns:
            Optional[str]: LLM响应文本，失败或熔断器拒绝则返回None
        """
        # 尝试按需初始化 provider
        if not await self._try_init_provider():
            return None
        
        # 速率限制检查
        if not self._rate_limiter.is_within_limit():
            self.stats["rate_limit_rejections"] += 1
            remaining = self._rate_limiter.remaining
            logger.warning(
                f"LLM daily rate limit reached "
                f"(limit={LLMRateLimitConfig.DAILY_CALL_LIMIT}, remaining={remaining})"
            )
            return None
        
        # 熔断器检查
        if not self._circuit_breaker_allow():
            self.stats["circuit_breaker_rejections"] += 1
            logger.warning("LLM circuit breaker OPEN — request rejected")
            return None
        
        backoff = INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                result = await self._call_llm_once(prompt, max_tokens, temperature)
                if result is not None:
                    self._circuit_breaker_on_success()
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        
        # 所有重试都失败
        self._circuit_breaker_on_failure()
        logger.error(f"LLM API call failed after {MAX_RETRIES} attempts: {last_error}")
        self.stats["failed_calls"] += 1
        return None
    
    def _circuit_breaker_allow(self) -> bool:
        """检查熔断器是否允许请求通过"""
        if self._cb_state == "closed":
            return True
        
        if self._cb_state == "open":
            elapsed = time.time() - self._cb_last_failure_time
            if elapsed >= CIRCUIT_BREAKER_RECOVERY_TIMEOUT:
                self._cb_state = "half_open"
                logger.debug(
                    f"LLM circuit breaker transitioning to HALF_OPEN "
                    f"after {elapsed:.0f}s recovery timeout"
                )
                return True
            return False
        
        # half_open: 允许通过
        return True
    
    def _circuit_breaker_on_success(self) -> None:
        """请求成功时重置熔断器"""
        if self._cb_state != "closed":
            logger.debug(
                f"LLM circuit breaker CLOSED (recovered from {self._cb_state})"
            )
        self._cb_failure_count = 0
        self._cb_state = "closed"
    
    def _circuit_breaker_on_failure(self) -> None:
        """请求失败时更新熔断器"""
        self._cb_failure_count += 1
        self._cb_last_failure_time = time.time()
        
        if self._cb_state == "half_open":
            self._cb_state = "open"
            logger.warning(
                "LLM circuit breaker OPEN (half_open probe failed)"
            )
        elif self._cb_failure_count >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            self._cb_state = "open"
            logger.warning(
                f"LLM circuit breaker OPEN after {self._cb_failure_count} "
                f"consecutive failures"
            )
    
    async def _call_llm_once(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """单次调用LLM API（委托给 llm_helper.call_llm，带超时保护）
        
        Raises:
            asyncio.TimeoutError: 超过 LLM_CALL_TIMEOUT 秒未响应
        """
        async with asyncio.timeout(LLM_CALL_TIMEOUT):
            result = await call_llm(
                self.astrbot_context,
                self.llm_api,
                self.default_provider_id,
                prompt,
            )
        if result.success:
            self.stats["total_tokens_used"] += result.tokens_used
            self._rate_limiter.increment()
            return result.content
        return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析JSON响应（委托给 llm_helper.parse_llm_json）"""
        return parse_llm_json(response)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def is_available(self) -> bool:
        """检查LLM是否可用"""
        return self.llm_api is not None

    @property
    def provider_id(self) -> Optional[str]:
        """当前提供者 ID"""
        return self.default_provider_id
