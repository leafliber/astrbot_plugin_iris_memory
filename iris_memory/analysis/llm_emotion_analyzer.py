"""
LLM增强情感分析器
使用LLM进行深度情感分析，识别复杂情感表达

重构版本：继承 LLMEnhancedDetector 模板方法模式
修复了 sync/async 阻抗不匹配问题
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from iris_memory.core.types import EmotionType
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.processing.detection_result import BaseDetectionResult
from iris_memory.processing.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedDetector,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("llm_emotion_analyzer")


EMOTION_ANALYSIS_PROMPT = """分析以下文本的情感。

## 情感类型
- joy: 开心、快乐、高兴、喜悦
- sadness: 难过、伤心、悲伤、痛苦
- anger: 生气、愤怒、恼火
- fear: 害怕、恐惧、担心
- anxiety: 焦虑、不安、紧张
- excitement: 兴奋、激动、期待
- calm: 平静、淡定、冷静
- neutral: 中性、无明显情感

## 深度分析要求
请识别：
1. **讽刺/反讽**：如"真是太棒了"（实际是抱怨）
2. **隐喻情感**：如"心里像压了块石头"（压抑、沉重）
3. **复杂情感**：如"苦中作乐"、"又爱又恨"
4. **文化语境**：如"呵呵"可能是嘲讽而非开心

## 待分析文本
{text}

## 输出格式
请以JSON格式返回：
```json
{{
  "primary": "主要情感类型",
  "secondary": ["次要情感类型"],
  "intensity": 0.0-1.0,
  "confidence": 0.0-1.0,
  "is_sarcastic": true/false,
  "is_complex": true/false,
  "context_note": "上下文说明（如讽刺、隐喻等）"
}}
```

仅返回JSON，不要有其他文字。"""


# 情感类型枚举映射
_EMOTION_TYPE_MAP = {
    "joy": EmotionType.JOY,
    "sadness": EmotionType.SADNESS,
    "anger": EmotionType.ANGER,
    "fear": EmotionType.FEAR,
    "anxiety": EmotionType.ANXIETY,
    "excitement": EmotionType.EXCITEMENT,
    "calm": EmotionType.CALM,
    "neutral": EmotionType.NEUTRAL,
}

# 讽刺/复杂情感指示词
_SARCASM_INDICATORS = [
    "真是", "太棒了", "呵呵", "哼", "切",
    "你说呢", "是吧", "对吧", "好呀"
]
_COMPLEX_INDICATORS = [
    "又", "但", "不过", "虽然", "尽管",
    "矛盾", "纠结", "复杂"
]


@dataclass
class EmotionAnalysisResult(BaseDetectionResult):
    """情感分析结果"""
    primary: EmotionType = EmotionType.NEUTRAL
    secondary: List[EmotionType] = field(default_factory=list)
    intensity: float = 0.5
    is_sarcastic: bool = False
    is_complex: bool = False
    context_note: str = ""


class LLMEmotionAnalyzer(LLMEnhancedDetector[EmotionAnalysisResult]):
    """LLM增强情感分析器
    
    支持三种模式：
    - rule: 仅使用词典+规则（快速）
    - llm: 仅使用LLM（深度理解）
    - hybrid: 混合权重（推荐）
    """
    
    def __init__(
        self,
        astrbot_context=None,
        provider_id: Optional[str] = None,
        mode: DetectionMode = DetectionMode.HYBRID,
        llm_weight: float = 0.4,
        enable_context_aware: bool = True,
        daily_limit: int = 0,
    ):
        super().__init__(
            astrbot_context=astrbot_context,
            provider_id=provider_id,
            mode=mode,
            daily_limit=daily_limit,
            max_tokens=250,
        )
        self._llm_weight = BaseDetectionResult.clamp(llm_weight)
        self._rule_weight = 1.0 - self._llm_weight
        self._enable_context_aware = enable_context_aware
        self._rule_analyzer = EmotionAnalyzer()
    
    def _should_skip_input(self, text: str = "", **kwargs) -> bool:
        """空文本时跳过"""
        return not text
    
    def _get_empty_result(self) -> EmotionAnalysisResult:
        """空输入默认结果"""
        return EmotionAnalysisResult(
            confidence=0.5,
            source="rule",
            reason="空文本",
        )
    
    async def _rule_detect_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """规则检测（异步版本）
        
        解决 sync/async 阻抗不匹配问题：统一使用异步版本
        """
        try:
            result = await self._rule_analyzer.analyze_emotion(text, context)
        except Exception:
            result = {
                "primary": EmotionType.NEUTRAL,
                "secondary": [],
                "intensity": 0.5,
                "confidence": 0.3,
                "contextual_correction": False,
            }
        
        return EmotionAnalysisResult(
            primary=result.get("primary", EmotionType.NEUTRAL),
            secondary=result.get("secondary", []),
            intensity=result.get("intensity", 0.5),
            confidence=result.get("confidence", 0.5),
            is_sarcastic=result.get("contextual_correction", False),
            source="rule",
            reason="规则分析",
        )
    
    def _rule_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """规则检测（同步回退版本）
        
        仅在无法使用异步版本时使用。
        """
        return EmotionAnalysisResult(
            confidence=0.3,
            source="rule",
            reason="同步回退",
        )
    
    def _build_prompt(self, text: str, **kwargs) -> str:
        """构建LLM提示词"""
        return EMOTION_ANALYSIS_PROMPT.format(text=text[:500])
    
    async def _llm_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """LLM检测（覆写以使用异步规则回退）"""
        prompt = self._build_prompt(text)
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug("LLM emotion analysis failed, falling back to rule")
            return await self._rule_detect_async(text, context)
        
        return self._parse_llm_result(result.parsed_json)
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> EmotionAnalysisResult:
        """解析LLM结果"""
        primary = BaseDetectionResult.map_enum(
            data.get("primary", "neutral"), _EMOTION_TYPE_MAP, EmotionType.NEUTRAL
        )
        
        # 解析次要情感，去重 + 排除主情感 + 限制数量
        secondary = []
        for s in data.get("secondary", []):
            if isinstance(s, str):
                sec = _EMOTION_TYPE_MAP.get(s.lower())
                if sec and sec != primary:
                    secondary.append(sec)
        
        return EmotionAnalysisResult(
            primary=primary,
            secondary=secondary[:2],
            intensity=BaseDetectionResult.clamp(
                float(data.get("intensity", 0.5))
            ),
            confidence=BaseDetectionResult.parse_confidence(data),
            is_sarcastic=data.get("is_sarcastic", False),
            is_complex=data.get("is_complex", False),
            context_note=data.get("context_note", ""),
            source="llm",
            reason="LLM分析",
        )
    
    async def _hybrid_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """混合检测：规则 + LLM加权"""
        rule_result = await self._rule_detect_async(text, context)
        
        # 可能需要上下文感知 → LLM合并
        if self._enable_context_aware and self._might_need_context_awareness(text):
            llm_result = await self._llm_detect(text, context)
            return self._merge_results(rule_result, llm_result)
        
        # 高强度+高置信度 → 直接返回规则结果
        if rule_result.intensity > 0.7 and rule_result.confidence > 0.7:
            rule_result.source = "hybrid"
            return rule_result
        
        # 低强度+低置信度 → LLM合并
        if rule_result.intensity < 0.3 and rule_result.confidence < 0.5:
            llm_result = await self._llm_detect(text, context)
            return self._merge_results(rule_result, llm_result)
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_need_context_awareness(self, text: str) -> bool:
        """检查是否可能需要上下文感知"""
        text_lower = text.lower()
        return any(
            ind in text_lower
            for ind in _SARCASM_INDICATORS + _COMPLEX_INDICATORS
        )
    
    def _merge_results(
        self,
        rule_result: EmotionAnalysisResult,
        llm_result: EmotionAnalysisResult
    ) -> EmotionAnalysisResult:
        """合并规则和LLM结果"""
        # 讽刺/复杂情感 → 直接使用LLM结果
        if llm_result.is_sarcastic:
            llm_result.source = "hybrid"
            llm_result.context_note = llm_result.context_note or "检测到讽刺/反讽"
            return llm_result
        
        if llm_result.is_complex:
            llm_result.source = "hybrid"
            llm_result.context_note = llm_result.context_note or "复杂情感"
            return llm_result
        
        # 加权平均
        intensity = (
            rule_result.intensity * self._rule_weight
            + llm_result.intensity * self._llm_weight
        )
        confidence = (
            rule_result.confidence * self._rule_weight
            + llm_result.confidence * self._llm_weight
        )
        
        # 选择置信度更高的主情感
        if llm_result.confidence > rule_result.confidence:
            primary = llm_result.primary
            secondary = llm_result.secondary
        else:
            primary = rule_result.primary
            secondary = rule_result.secondary
        
        return EmotionAnalysisResult(
            primary=primary,
            secondary=secondary,
            intensity=intensity,
            confidence=confidence,
            is_sarcastic=llm_result.is_sarcastic,
            is_complex=llm_result.is_complex,
            context_note=llm_result.context_note,
            source="hybrid",
            reason="混合分析",
        )
    
    # ===== 向后兼容方法 =====
    
    async def analyze_emotion(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """兼容原有接口"""
        result = await self.detect(text, context)
        return {
            "primary": result.primary,
            "secondary": result.secondary,
            "intensity": result.intensity,
            "confidence": result.confidence,
            "contextual_correction": result.is_sarcastic,
            "is_complex": result.is_complex,
            "context_note": result.context_note,
        }
