"""
LLM消息处理器
使用AstrBot默认LLM进行消息分类和摘要生成
"""
import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger

logger = get_logger("llm_processor")


# 重试配置
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # 初始退避时间（秒）
MAX_BACKOFF = 30.0  # 最大退避时间（秒）
BACKOFF_MULTIPLIER = 2.0  # 退避乘数


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
        max_tokens: int = 200
    ):
        self.astrbot_context = astrbot_context
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
        
        # 统计信息
        self.stats = {
            "classification_calls": 0,
            "summary_calls": 0,
            "failed_calls": 0,
            "total_tokens_used": 0
        }
    
    async def initialize(self) -> bool:
        """初始化LLM API（延迟初始化策略）
        
        由于 AstrBot 的 provider 在插件加载后才初始化，
        此方法仅标记处理器已准备好，实际 provider 获取延迟到第一次使用时。
        
        Returns:
            bool: 始终返回 True（处理器已准备好，provider 将在使用时按需获取）
        """
        if not self.astrbot_context:
            logger.warning("AstrBot context not available, LLM features disabled")
            return False
        
        logger.info("LLM processor initialized (provider will be loaded on first use)")
        return True
    
    async def _try_init_provider(self) -> bool:
        """尝试初始化 provider（按需调用）
        
        使用 get_using_provider() 获取默认聊天模型，而不是 get_all_providers()[0]
        确保使用用户配置的默认模型，而非配置列表中的第一个
        
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
            # 优先使用 get_using_provider() 获取默认模型
            # 这是用户配置的默认聊天模型
            provider = self.astrbot_context.get_using_provider()
            if provider:
                self.llm_api = provider
                self.default_provider_id = getattr(
                    self.llm_api, 'id', None
                ) or getattr(self.llm_api, 'provider_id', None)
                logger.info(
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
            logger.debug(
                f"Failed to load LLM provider "
                f"(attempt {self._init_retry_count}/{self._max_init_retries}): {e}"
            )
            return False
    
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
        """调用LLM API（带指数退避重试机制）

        Args:
            prompt: 提示词
            max_tokens: 最大返回token数
            temperature: 温度参数

        Returns:
            Optional[str]: LLM响应文本，失败则返回None
        """
        # 尝试按需初始化 provider
        if not await self._try_init_provider():
            return None
        
        backoff = INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                result = await self._call_llm_once(prompt, max_tokens, temperature)
                if result is not None:
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                
                if attempt < MAX_RETRIES - 1:
                    # 等待退避时间后重试
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        
        # 所有重试都失败
        logger.error(f"LLM API call failed after {MAX_RETRIES} attempts: {last_error}")
        self.stats["failed_calls"] += 1
        return None
    
    async def _call_llm_once(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Optional[str]:
        """单次调用LLM API（不带重试）
        
        使用 AstrBot v4.5.7+ 新API:
        - self.context.llm_generate() 直接调用
        - Provider.text_chat() 旧方式作为回退
        
        Args:
            prompt: 提示词
            max_tokens: 最大返回token数
            temperature: 温度参数
            
        Returns:
            Optional[str]: LLM响应文本
            
        Raises:
            Exception: API调用异常
        """
        # 使用 AstrBot API - llm_generate
        if self.astrbot_context and hasattr(self.astrbot_context, 'llm_generate') and self.default_provider_id:
            llm_resp = await self.astrbot_context.llm_generate(
                chat_provider_id=self.default_provider_id,
                prompt=prompt
            )
            if llm_resp and hasattr(llm_resp, 'completion_text'):
                text = llm_resp.completion_text or ""
                self.stats["total_tokens_used"] += len(prompt) // 4 + len(text) // 4
                return text.strip()
        
        # 回退: 使用 Provider.text_chat()
        if self.llm_api and hasattr(self.llm_api, 'text_chat'):
            response = await self.llm_api.text_chat(
                prompt=prompt,
                context=[]
            )
            
            if hasattr(response, 'completion_text'):
                text = response.completion_text or ""
            elif isinstance(response, dict):
                text = response.get("text", "") or response.get("content", "")
            else:
                text = str(response) if response else ""
            
            self.stats["total_tokens_used"] += len(prompt) // 4 + len(text) // 4
            return text.strip()
        
        logger.warning("No suitable LLM method found")
        return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析JSON响应"""
        if not response:
            return None
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        try:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            json_match = re.search(r'(\{[\s\S]*?\})', response)
            if json_match:
                return json.loads(json_match.group(1))
                
        except json.JSONDecodeError:
            pass
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def is_available(self) -> bool:
        """检查LLM是否可用"""
        return self.llm_api is not None
