"""
LLM增强情感分析器
使用LLM进行深度情感分析，识别复杂情感表达
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_memory.core.types import EmotionType
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.processing.llm_enhanced_base import (
    DetectionMode,
    LLMEnhancedBase,
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


@dataclass
class EmotionAnalysisResult:
    """情感分析结果"""
    primary: EmotionType
    secondary: List[EmotionType]
    intensity: float
    confidence: float
    is_sarcastic: bool
    is_complex: bool
    context_note: str
    source: str  # "rule" | "llm" | "hybrid"


class LLMEmotionAnalyzer(LLMEnhancedBase):
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
        self._llm_weight = max(0.0, min(1.0, llm_weight))
        self._rule_weight = 1.0 - self._llm_weight
        self._enable_context_aware = enable_context_aware
        self._rule_analyzer = EmotionAnalyzer()
    
    async def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """分析文本情感
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            EmotionAnalysisResult: 分析结果
        """
        if not text:
            return EmotionAnalysisResult(
                primary=EmotionType.NEUTRAL,
                secondary=[],
                intensity=0.5,
                confidence=0.5,
                is_sarcastic=False,
                is_complex=False,
                context_note="空文本",
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
    ) -> EmotionAnalysisResult:
        """规则检测（同步版本）"""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            self._rule_analyzer.analyze_emotion(text, context)
        ) if not asyncio.iscoroutine_function(self._rule_analyzer.analyze_emotion) else None
        
        if result is None:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result = self._sync_rule_detect(text, context)
            else:
                result = loop.run_until_complete(
                    self._rule_analyzer.analyze_emotion(text, context)
                )
        
        if result is None:
            result = self._sync_rule_detect(text, context)
        
        return EmotionAnalysisResult(
            primary=result.get("primary", EmotionType.NEUTRAL),
            secondary=result.get("secondary", []),
            intensity=result.get("intensity", 0.5),
            confidence=result.get("confidence", 0.5),
            is_sarcastic=result.get("contextual_correction", False),
            is_complex=False,
            context_note="",
            source="rule"
        )
    
    def _sync_rule_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """同步规则检测"""
        return {
            "primary": EmotionType.NEUTRAL,
            "secondary": [],
            "intensity": 0.5,
            "confidence": 0.3,
            "contextual_correction": False
        }
    
    async def _rule_detect_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """规则检测（异步版本）"""
        result = await self._rule_analyzer.analyze_emotion(text, context)
        
        return EmotionAnalysisResult(
            primary=result.get("primary", EmotionType.NEUTRAL),
            secondary=result.get("secondary", []),
            intensity=result.get("intensity", 0.5),
            confidence=result.get("confidence", 0.5),
            is_sarcastic=result.get("contextual_correction", False),
            is_complex=False,
            context_note="",
            source="rule"
        )
    
    async def _llm_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """LLM检测"""
        prompt = EMOTION_ANALYSIS_PROMPT.format(text=text[:500])
        result = await self._call_llm(prompt)
        
        if not result.success or not result.parsed_json:
            logger.debug(f"LLM emotion analysis failed, falling back to rule")
            return await self._rule_detect_async(text, context)
        
        return self._parse_llm_result(result.parsed_json)
    
    async def _hybrid_detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionAnalysisResult:
        """混合检测：规则 + LLM加权"""
        rule_result = await self._rule_detect_async(text, context)
        
        if self._enable_context_aware and self._might_need_context_awareness(text):
            llm_result = await self._llm_detect(text, context)
            return self._merge_results(rule_result, llm_result)
        
        if rule_result.intensity > 0.7 and rule_result.confidence > 0.7:
            rule_result.source = "hybrid"
            return rule_result
        
        if rule_result.intensity < 0.3 and rule_result.confidence < 0.5:
            llm_result = await self._llm_detect(text, context)
            return self._merge_results(rule_result, llm_result)
        
        rule_result.source = "hybrid"
        return rule_result
    
    def _might_need_context_awareness(self, text: str) -> bool:
        """检查是否可能需要上下文感知"""
        sarcasm_indicators = [
            "真是", "太棒了", "呵呵", "哼", "切",
            "你说呢", "是吧", "对吧", "好呀"
        ]
        complex_indicators = [
            "又", "但", "不过", "虽然", "尽管",
            "矛盾", "纠结", "复杂"
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in sarcasm_indicators + complex_indicators)
    
    def _merge_results(
        self,
        rule_result: EmotionAnalysisResult,
        llm_result: EmotionAnalysisResult
    ) -> EmotionAnalysisResult:
        """合并规则和LLM结果"""
        if llm_result.is_sarcastic:
            return EmotionAnalysisResult(
                primary=llm_result.primary,
                secondary=llm_result.secondary,
                intensity=llm_result.intensity,
                confidence=llm_result.confidence,
                is_sarcastic=True,
                is_complex=llm_result.is_complex,
                context_note=llm_result.context_note or "检测到讽刺/反讽",
                source="hybrid"
            )
        
        if llm_result.is_complex:
            return EmotionAnalysisResult(
                primary=llm_result.primary,
                secondary=llm_result.secondary,
                intensity=llm_result.intensity,
                confidence=llm_result.confidence,
                is_sarcastic=llm_result.is_sarcastic,
                is_complex=True,
                context_note=llm_result.context_note or "复杂情感",
                source="hybrid"
            )
        
        rule_weight = self._rule_weight
        llm_weight = self._llm_weight
        
        intensity = rule_result.intensity * rule_weight + llm_result.intensity * llm_weight
        confidence = rule_result.confidence * rule_weight + llm_result.confidence * llm_weight
        
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
            source="hybrid"
        )
    
    def _parse_llm_result(self, data: Dict[str, Any]) -> EmotionAnalysisResult:
        """解析LLM结果"""
        primary_str = data.get("primary", "neutral").lower()
        emotion_map = {
            "joy": EmotionType.JOY,
            "sadness": EmotionType.SADNESS,
            "anger": EmotionType.ANGER,
            "fear": EmotionType.FEAR,
            "anxiety": EmotionType.ANXIETY,
            "excitement": EmotionType.EXCITEMENT,
            "calm": EmotionType.CALM,
            "neutral": EmotionType.NEUTRAL,
        }
        primary = emotion_map.get(primary_str, EmotionType.NEUTRAL)
        
        secondary_strs = data.get("secondary", [])
        secondary = []
        for s in secondary_strs:
            if isinstance(s, str):
                sec_emotion = emotion_map.get(s.lower())
                if sec_emotion and sec_emotion != primary:
                    secondary.append(sec_emotion)
        
        intensity = float(data.get("intensity", 0.5))
        intensity = max(0.0, min(1.0, intensity))
        
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        
        return EmotionAnalysisResult(
            primary=primary,
            secondary=secondary[:2],
            intensity=intensity,
            confidence=confidence,
            is_sarcastic=data.get("is_sarcastic", False),
            is_complex=data.get("is_complex", False),
            context_note=data.get("context_note", ""),
            source="llm"
        )
    
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
