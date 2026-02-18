"""
LLM增强触发器检测器
使用LLM进行语义层面的记忆触发器检测
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_memory.core.types import TriggerType
from iris_memory.capture.trigger_detector import TriggerDetector
from iris_memory.processing.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedBase,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_trigger_detector")


TRIGGER_DETECTION_PROMPT = """分析以下文本是否值得作为记忆保存。

## 判断标准
文本应该被记忆，如果包含：
1. **个人偏好**：喜欢、讨厌、爱好、习惯等
2. **重要事实**：姓名、生日、职业、住址、联系方式等
3. **情感体验**：重要的情感经历或感受
4. **关系信息**：家人、朋友、同事等人际关系
5. **用户请求**：明确表示"记住这个"、"别忘了"等

文本不应该被记忆，如果是：
1. **简单问候**：你好、在吗、哈哈等
2. **确认回复**：嗯、好的、OK等
3. **查询请求**：我是谁、你喜欢什么等（这是查询，不是新记忆）
4. **无关闲聊**：天气、新闻等不涉及用户个人的内容
5. **已解决的求助**：有人吗？我自己找到答案了

## 待分析文本
{text}

## 输出格式
请以JSON格式返回：
```json
{{
  "should_remember": true/false,
  "trigger_type": "explicit|preference|emotion|relationship|fact|boundary|null",
  "confidence": 0.0-1.0,
  "reason": "简短的判断理由",
  "memory_value": "high|medium|low",
  "key_info": ["提取的关键信息"]
}}
```

仅返回JSON，不要有其他文字。"""


@dataclass
class TriggerDetectionResult:
    """触发器检测结果"""
    should_remember: bool
    trigger_type: Optional[TriggerType]
    confidence: float
    reason: str
    memory_value: str  # "high" | "medium" | "low"
    key_info: List[str]
    source: str  # "rule" | "llm" | "hybrid"


class LLMTriggerDetector(LLMEnhancedBase):
    """LLM增强触发器检测器
    
    支持三种模式：
    - rule: 仅使用规则检测（快速）
    - llm: 仅使用LLM检测（语义理解）
    - hybrid: 规则预筛 + LLM确认（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        daily_limit: int = 200,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=200,
        )
        self._rule_detector = TriggerDetector()
    
    async def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """检测文本是否值得记忆
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            TriggerDetectionResult: 检测结果
        """
        if not text or len(text.strip()) < 3:
            return TriggerDetectionResult(
                should_remember=False,
                trigger_type=None,
                confidence=1.0,
                reason="文本过短",
                memory_value="low",
                key_info=[],
                source="rule"
            )
        
        if self._mode == DetectionMode.RULE:
            return self._rule_detect(text, context)
        elif self._mode == DetectionMode.LLM:
            return await self._llm_detect(text, context)
        else:
            return await self._hybrid_detect(text, context)
    
    def _rule_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """规则检测"""
        if self._rule_detector.is_query(text):
            return TriggerDetectionResult(
                should_remember=False,
                trigger_type=None,
                confidence=0.9,
                reason="查询类消息",
                memory_value="low",
                key_info=[],
                source="rule"
            )
        
        triggers = self._rule_detector.detect_triggers(text)
        
        if not triggers:
            return TriggerDetectionResult(
                should_remember=False,
                trigger_type=None,
                confidence=0.8,
                reason="无触发器",
                memory_value="low",
                key_info=[],
                source="rule"
            )
        
        highest = self._rule_detector.get_highest_confidence_trigger(text)
        if highest:
            return TriggerDetectionResult(
                should_remember=True,
                trigger_type=highest["type"],
                confidence=highest["confidence"],
                reason=f"规则匹配: {highest['pattern']}",
                memory_value="medium",
                key_info=[],
                source="rule"
            )
        
        return TriggerDetectionResult(
            should_remember=False,
            trigger_type=None,
            confidence=0.5,
            reason="触发器置信度低",
            memory_value="low",
            key_info=[],
            source="rule"
        )
    
    async def _llm_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """LLM检测"""
        prompt = TRIGGER_DETECTION_PROMPT.format(text=text[:500])
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(f"LLM trigger detection failed, falling back to rule")
            return self._rule_detect(text, context)
        
        return self._parse_llm_result(result.parsed_json)
    
    async def _hybrid_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = self._rule_detect(text, context)
        
        if rule_result.should_remember and rule_result.confidence >= 0.8:
            if rule_result.trigger_type == TriggerType.EXPLICIT:
                return rule_result
            if rule_result.trigger_type == TriggerType.BOUNDARY:
                return rule_result
        
        if not rule_result.should_remember and rule_result.confidence >= 0.8:
            if self._might_have_hidden_value(text):
                llm_result = await self._llm_detect(text, context)
                if llm_result.should_remember:
                    llm_result.source = "hybrid"
                    return llm_result
            return rule_result
        
        llm_result = await self._llm_detect(text, context)
        if llm_result.confidence >= 0.6:
            llm_result.source = "hybrid"
            return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_have_hidden_value(self, text: str) -> bool:
        """检查是否有隐藏价值"""
        value_indicators = [
            "今天", "昨天", "最近", "发现", "终于", "终于",
            "觉得", "感觉", "认为", "希望", "打算", "计划",
            "第一次", "好久", "突然", "意外"
        ]
        return any(indicator in text for indicator in value_indicators)
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> TriggerDetectionResult:
        """解析LLM结果"""
        should_remember = data.get("should_remember", False)
        
        trigger_type_str = data.get("trigger_type", "").lower()
        trigger_type_map = {
            "explicit": TriggerType.EXPLICIT,
            "preference": TriggerType.PREFERENCE,
            "emotion": TriggerType.EMOTION,
            "relationship": TriggerType.RELATIONSHIP,
            "fact": TriggerType.FACT,
            "boundary": TriggerType.BOUNDARY,
        }
        trigger_type = trigger_type_map.get(trigger_type_str)
        
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        key_info = data.get("key_info", [])
        if isinstance(key_info, str):
            key_info = [key_info]
        
        return TriggerDetectionResult(
            should_remember=should_remember,
            trigger_type=trigger_type,
            confidence=confidence,
            reason=data.get("reason", "LLM判断"),
            memory_value=data.get("memory_value", "medium"),
            key_info=key_info,
            source="llm"
        )
    
    def has_trigger(self, text: str) -> bool:
        """判断文本是否包含触发器（同步版本）"""
        result = self._rule_detect(text)
        return result.should_remember
    
    def is_query(self, text: str) -> bool:
        """判断是否为查询类消息"""
        return self._rule_detector.is_query(text)
    
    def get_trigger_types(self, text: str) -> List[TriggerType]:
        """获取触发器类型列表"""
        return self._rule_detector.get_trigger_types(text)
