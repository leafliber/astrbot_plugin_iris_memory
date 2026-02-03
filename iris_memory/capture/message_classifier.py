"""
消息分类器 - 支持本地规则和LLM分类
"""
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger
from iris_memory.capture.trigger_detector import TriggerDetector
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.processing.llm_processor import (
    LLMMessageProcessor, LLMClassificationResult
)

logger = get_logger("message_classifier")


class ProcessingLayer(Enum):
    """处理层级"""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    DISCARD = "discard"


@dataclass
class ClassificationResult:
    """分类结果"""
    layer: ProcessingLayer
    confidence: float
    reason: str
    metadata: Dict[str, Any]
    source: str  # "local" or "llm"


class MessageClassifier:
    """消息分类器 - 混合模式"""
    
    def __init__(
        self,
        trigger_detector: Optional[TriggerDetector] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        llm_processor: Optional[LLMMessageProcessor] = None,
        config: Optional[Dict] = None
    ):
        self.trigger_detector = trigger_detector or TriggerDetector()
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.llm_processor = llm_processor
        self.config = config or {}
        
        # 配置
        self.mode = self.config.get("llm_processing_mode", "hybrid")
        self.immediate_trigger_confidence = self.config.get(
            "immediate_trigger_confidence", 0.8
        )
        self.immediate_emotion_intensity = self.config.get(
            "immediate_emotion_intensity", 0.7
        )
        
        # 统计
        self.stats = {
            "local_classifications": 0,
            "llm_classifications": 0,
            "llm_fallbacks": 0
        }
    
    async def classify(
        self,
        message: str,
        context: Dict = None
    ) -> ClassificationResult:
        """分类消息 - 根据配置选择策略"""
        context = context or {}
        
        # 策略1: 仅本地规则
        if self.mode == "local":
            return await self._classify_local(message, context)
        
        # 策略2: 仅LLM
        if self.mode == "llm" and self.llm_processor and self.llm_processor.is_available():
            return await self._classify_llm(message, context)
        
        # 策略3: 混合模式（默认）
        return await self._classify_hybrid(message, context)
    
    async def _classify_local(
        self,
        message: str,
        context: Dict
    ) -> ClassificationResult:
        """本地规则分类"""
        self.stats["local_classifications"] += 1
        
        # 1. 负样本检查
        if self.trigger_detector._is_negative_sample(message):
            return ClassificationResult(
                layer=ProcessingLayer.DISCARD,
                confidence=1.0,
                reason="negative_sample",
                metadata={},
                source="local"
            )
        
        # 2. 检测触发器
        triggers = self.trigger_detector.detect_triggers(message)
        
        # 3. 分析情感
        emotion_result = await self.emotion_analyzer.analyze_emotion(message, context)
        
        # 4. 决策逻辑
        if triggers:
            max_confidence = max(t["confidence"] for t in triggers)
            if max_confidence >= self.immediate_trigger_confidence:
                return ClassificationResult(
                    layer=ProcessingLayer.IMMEDIATE,
                    confidence=max_confidence,
                    reason="high_confidence_trigger",
                    metadata={"triggers": triggers, "emotion": emotion_result},
                    source="local"
                )
        
        if emotion_result.get("intensity", 0) >= self.immediate_emotion_intensity:
            return ClassificationResult(
                layer=ProcessingLayer.IMMEDIATE,
                confidence=emotion_result["intensity"],
                reason="high_emotion_intensity",
                metadata={"triggers": triggers, "emotion": emotion_result},
                source="local"
            )
        
        return ClassificationResult(
            layer=ProcessingLayer.BATCH,
            confidence=0.5,
            reason="normal_message",
            metadata={"triggers": triggers, "emotion": emotion_result},
            source="local"
        )
    
    async def _classify_llm(
        self,
        message: str,
        context: Dict
    ) -> ClassificationResult:
        """LLM分类"""
        if not self.llm_processor or not self.llm_processor.is_available():
            self.stats["llm_fallbacks"] += 1
            return await self._classify_local(message, context)
        
        self.stats["llm_classifications"] += 1
        
        llm_result = await self.llm_processor.classify_message(message, context)
        
        if llm_result:
            return ClassificationResult(
                layer=ProcessingLayer(llm_result.layer),
                confidence=llm_result.confidence,
                reason=f"llm: {llm_result.reason}",
                metadata=llm_result.metadata,
                source="llm"
            )
        
        self.stats["llm_fallbacks"] += 1
        return await self._classify_local(message, context)
    
    async def _classify_hybrid(
        self,
        message: str,
        context: Dict
    ) -> ClassificationResult:
        """混合分类策略"""
        # 先本地分类
        local_result = await self._classify_local(message, context)
        
        # 如果本地置信度很高，直接使用
        if local_result.confidence >= 0.9 or local_result.confidence <= 0.3:
            return local_result
        
        # 边缘情况，使用LLM确认
        if self.llm_processor and self.llm_processor.is_available():
            try:
                llm_result = await self.llm_processor.classify_message(message, context)
                
                if llm_result:
                    self.stats["llm_classifications"] += 1
                    if llm_result.confidence >= 0.6:
                        return ClassificationResult(
                            layer=ProcessingLayer(llm_result.layer),
                            confidence=llm_result.confidence,
                            reason=f"llm_confirmed: {llm_result.reason}",
                            metadata={
                                **llm_result.metadata,
                                "local_result": {
                                    "layer": local_result.layer.value,
                                    "confidence": local_result.confidence
                                }
                            },
                            source="llm"
                        )
            except Exception as e:
                # LLM失败，回退到本地结果
                logger.warning(f"LLM classification failed, falling back to local: {e}")
                self.stats["llm_fallbacks"] += 1
        
        return local_result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
