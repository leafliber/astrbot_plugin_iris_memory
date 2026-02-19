"""
LLM增强基类
提供统一的LLM调用封装，支持多种检测模式的基类

包含两层抽象：
- LLMEnhancedBase: 底层LLM调用封装（provider管理、重试、JSON解析）
- LLMEnhancedDetector: 上层模板方法模式（统一 detect/rule/llm/hybrid 流程）
"""
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.utils.logger import get_logger
from iris_memory.utils.llm_helper import (
    LLMCallResult,
    resolve_llm_provider,
    call_llm,
    parse_llm_json,
)
from iris_memory.utils.rate_limiter import DailyCallLimiter
from iris_memory.core.provider_utils import normalize_provider_id

logger = get_logger("llm_enhanced_base")

# 泛型结果类型，子类指定具体的检测结果类型
T = TypeVar('T')

MAX_RETRIES = 2
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 5.0
BACKOFF_MULTIPLIER = 2.0


class DetectionMode(str, Enum):
    """检测模式"""
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


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
        
        self._limiter = DailyCallLimiter(daily_limit)
        
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
    
    def _is_within_limit(self) -> bool:
        return self._limiter.is_within_limit()
    
    @property
    def remaining_daily_calls(self) -> int:
        return self._limiter.remaining
    
    async def _resolve_provider(self) -> bool:
        """LLM 提供者解析（委托给 llm_helper）"""
        if self._provider_initialized:
            return self._resolved_provider is not None
        
        self._provider_initialized = True
        
        provider, resolved_id = resolve_llm_provider(
            self._astrbot_context,
            self._configured_provider_id or "",
            label=f"LLMEnhanced({self.__class__.__name__})",
        )
        if provider:
            self._resolved_provider = provider
            self._resolved_provider_id = resolved_id
            return True
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
                    self._limiter.increment()
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
        """LLM 单次调用（委托给 llm_helper）"""
        self._stats["total_calls"] += 1
        result = await call_llm(
            self._astrbot_context,
            self._resolved_provider,
            self._resolved_provider_id,
            prompt,
            parse_json=True,
        )
        return result
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """JSON 解析（委托给 llm_helper）"""
        return parse_llm_json(response)
    
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


class LLMEnhancedDetector(LLMEnhancedBase, Generic[T]):
    """增强的LLM检测器基类 - 支持泛型结果类型和模板方法模式
    
    将 detect/rule/llm/hybrid 的公共流程提取到基类中，
    子类只需实现以下抽象方法：
    
    - _build_prompt(*args, **kwargs) -> str: 构建LLM提示词
    - _parse_llm_result(data: Dict) -> T: 解析LLM返回的JSON
    - _rule_detect(*args, **kwargs) -> T: 规则检测实现
    
    可选覆写：
    - _get_empty_result() -> T: 空输入时的默认结果
    - _should_skip_input(*args, **kwargs) -> bool: 是否跳过输入
    - _hybrid_detect(*args, **kwargs) -> T: 自定义混合检测逻辑
    - _llm_detect(*args, **kwargs) -> T: 自定义LLM检测逻辑
    """
    
    async def detect(self, *args, **kwargs) -> T:
        """模板方法 - 统一检测流程
        
        根据当前模式分派到对应的检测方法。
        子类通常不需要覆写此方法。
        """
        if self._should_skip_input(*args, **kwargs):
            return self._get_empty_result()
        
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(*args, **kwargs)
        elif self._mode == DetectionMode.LLM:
            return await self._llm_detect(*args, **kwargs)
        else:
            return await self._hybrid_detect(*args, **kwargs)
    
    async def _llm_detect(self, *args, **kwargs) -> T:
        """统一LLM检测流程
        
        构建提示词 → 调用LLM → 解析结果 → 失败时回退到规则。
        子类可覆写以定制LLM检测逻辑。
        """
        prompt = self._build_prompt(*args, **kwargs)
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(
                f"LLM detection failed for {self.__class__.__name__}, "
                f"falling back to rule"
            )
            return self._rule_detect(*args, **kwargs)
        
        parsed = self._parse_llm_result(result.parsed_json)
        return parsed
    
    async def _hybrid_detect(self, *args, **kwargs) -> T:
        """默认混合检测逻辑
        
        规则预筛 → LLM确认。
        子类应覆写此方法以提供定制化的混合逻辑。
        """
        rule_result = self._rule_detect(*args, **kwargs)
        
        llm_result = await self._llm_detect(*args, **kwargs)
        if hasattr(llm_result, 'confidence') and llm_result.confidence >= 0.6:
            if hasattr(llm_result, 'source'):
                llm_result.source = "hybrid"
            return llm_result
        
        if hasattr(rule_result, 'source'):
            rule_result.source = "hybrid"
        return rule_result
    
    def _should_skip_input(self, *args, **kwargs) -> bool:
        """判断是否跳过输入（空值、过短等）
        
        子类可覆写以提供定制的跳过逻辑。
        默认行为：检查第一个位置参数是否为空字符串。
        """
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str) and not first_arg.strip():
                return True
            if isinstance(first_arg, list) and not first_arg:
                return True
        return False
    
    def _get_empty_result(self) -> T:
        """返回空输入时的默认结果
        
        子类必须覆写此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_empty_result()"
        )
    
    @abstractmethod
    def _build_prompt(self, *args, **kwargs) -> str:
        """构建LLM提示词
        
        子类实现。将输入转换为LLM可理解的提示词。
        """
        pass
    
    @abstractmethod
    def _parse_llm_result(self, data: Dict[str, Any]) -> T:
        """解析LLM结果
        
        子类实现。将LLM返回的JSON字典转换为结果对象。
        应使用 BaseDetectionResult 的工具方法处理通用转换。
        """
        pass
    
    @abstractmethod
    def _rule_detect(self, *args, **kwargs) -> T:
        """规则检测实现
        
        子类实现。纯规则逻辑，不涉及LLM调用。
        """
        pass
