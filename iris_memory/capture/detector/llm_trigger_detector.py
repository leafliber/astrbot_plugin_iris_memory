"""
LLM增强触发器检测器
使用LLM进行语义层面的记忆触发器检测

基于 LLMEnhancedDetector 模板方法模式
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iris_memory.core.types import TriggerType
from iris_memory.capture.detector.trigger_detector import TriggerDetector
from iris_memory.core.detection.base_result import BaseDetectionResult
from iris_memory.core.detection.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedDetector,
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
class TriggerDetectionResult(BaseDetectionResult):
    """触发器检测结果"""
    should_remember: bool = False
    trigger_type: Optional[TriggerType] = None
    memory_value: str = "low"  # "high" | "medium" | "low"
    key_info: List[str] = field(default_factory=list)


# 触发器类型枚举映射
_TRIGGER_TYPE_MAP = {
    "explicit": TriggerType.EXPLICIT,
    "preference": TriggerType.PREFERENCE,
    "emotion": TriggerType.EMOTION,
    "relationship": TriggerType.RELATIONSHIP,
    "fact": TriggerType.FACT,
    "boundary": TriggerType.BOUNDARY,
}

# 隐藏价值指示词
_VALUE_INDICATORS = [
    "今天", "昨天", "最近", "发现", "终于",
    "觉得", "感觉", "认为", "希望", "打算", "计划",
    "第一次", "好久", "突然", "意外"
]


class LLMTriggerDetector(LLMEnhancedDetector[TriggerDetectionResult]):
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
    
    def _should_skip_input(self, text: str = "", **kwargs) -> bool:
        """文本过短时跳过"""
        return not text or len(text.strip()) < 3
    
    def _get_empty_result(self) -> TriggerDetectionResult:
        """空输入默认结果"""
        return TriggerDetectionResult(
            confidence=1.0,
            source="rule",
            reason="文本过短",
            should_remember=False,
        )
    
    def _rule_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """规则检测"""
        if self._rule_detector.is_query(text):
            return TriggerDetectionResult(
                should_remember=False,
                confidence=0.9,
                reason="查询类消息",
                source="rule",
            )
        
        triggers = self._rule_detector.detect_triggers(text)
        
        if not triggers:
            return TriggerDetectionResult(
                should_remember=False,
                confidence=0.8,
                reason="无触发器",
                source="rule",
            )
        
        highest = self._rule_detector.get_highest_confidence_trigger(text)
        if highest:
            return TriggerDetectionResult(
                should_remember=True,
                trigger_type=highest["type"],
                confidence=highest["confidence"],
                reason=f"规则匹配: {highest['pattern']}",
                memory_value="medium",
                source="rule",
            )
        
        return TriggerDetectionResult(
            should_remember=False,
            confidence=0.5,
            reason="触发器置信度低",
            source="rule",
        )
    
    def _build_prompt(self, text: str, **kwargs) -> str:
        """构建LLM提示词"""
        return TRIGGER_DETECTION_PROMPT.format(text=text[:500])
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> TriggerDetectionResult:
        """解析LLM结果"""
        return TriggerDetectionResult(
            should_remember=data.get("should_remember", False),
            trigger_type=BaseDetectionResult.map_enum(
                data.get("trigger_type", ""), _TRIGGER_TYPE_MAP
            ),
            confidence=BaseDetectionResult.parse_confidence(data),
            reason=data.get("reason", "LLM判断"),
            memory_value=data.get("memory_value", "medium"),
            key_info=BaseDetectionResult.ensure_list(data.get("key_info", [])),
            source="llm",
        )
    
    async def _hybrid_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDetectionResult:
        """混合检测：规则预筛 + LLM确认"""
        rule_result = self._rule_detect(text, context)
        
        # 高置信度的显式/边界触发 → 直接返回
        if rule_result.should_remember and rule_result.confidence >= 0.8:
            if rule_result.trigger_type in (TriggerType.EXPLICIT, TriggerType.BOUNDARY):
                return rule_result
        
        # 高置信度无触发 → 检查隐藏价值后返回
        if not rule_result.should_remember and rule_result.confidence >= 0.8:
            if self._might_have_hidden_value(text):
                llm_result = await self._llm_detect(text, context)
                if llm_result.should_remember:
                    llm_result.source = "hybrid"
                    return llm_result
            return rule_result
        
        # 不确定区间 → LLM确认
        llm_result = await self._llm_detect(text, context)
        if llm_result.confidence >= 0.6:
            llm_result.source = "hybrid"
            return llm_result
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_have_hidden_value(self, text: str) -> bool:
        """检查是否有隐藏价值"""
        return any(indicator in text for indicator in _VALUE_INDICATORS)
    
