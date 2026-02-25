"""
消息分类器 - 支持本地规则和LLM分类

优化措施：
1. 收紧混合模式阈值（0.6-0.8），减少LLM调用频率
2. 增加冷却时间机制（同一上下文60秒内只调用一次LLM）
3. 增加消息指纹缓存，避免重复处理相似消息
4. 增加采样率限制，批量处理时每批最多处理5条
"""
from enum import Enum
from typing import Dict, Any, Optional, Final
from dataclasses import dataclass, field
import time
import hashlib

from iris_memory.utils.logger import get_logger
from iris_memory.utils.rate_limiter import CooldownTracker
from iris_memory.utils.fingerprint import compute_message_fingerprint
from iris_memory.capture.detector.trigger_detector import TriggerDetector
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.processing.llm_processor import (
    LLMMessageProcessor, LLMClassificationResult
)
from iris_memory.core.constants import SourceType

logger = get_logger("message_classifier")


class ProcessingLayer(Enum):
    """处理层级"""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    DISCARD = "discard"


@dataclass(frozen=True)
class ClassificationResult:
    """分类结果"""
    layer: ProcessingLayer
    confidence: float
    reason: str
    metadata: Dict[str, Any]
    source: str  # "local" or "llm"


class MessageClassifier:
    """消息分类器 - 混合模式
    
    优化策略：
    - 阈值收紧：只有置信度在0.6-0.8之间的边缘情况才调用LLM
    - 冷却机制：同一用户/群聊60秒内最多调用一次LLM
    - 指纹去重：相似消息直接复用之前的结果
    - 采样限制：批量处理时每批最多处理5条消息
    """
    
    # 默认配置常量
    DEFAULT_MODE: Final[str] = "hybrid"
    DEFAULT_IMMEDIATE_TRIGGER_CONFIDENCE: Final[float] = 0.8
    DEFAULT_IMMEDIATE_EMOTION_INTENSITY: Final[float] = 0.7
    DEFAULT_HYBRID_UPPER_THRESHOLD: Final[float] = 0.8
    DEFAULT_HYBRID_LOWER_THRESHOLD: Final[float] = 0.6
    DEFAULT_LLM_COOLDOWN_SECONDS: Final[int] = 60
    DEFAULT_BATCH_SAMPLE_LIMIT: Final[int] = 5
    DEFAULT_FINGERPRINT_CACHE_SIZE: Final[int] = 100
    DEFAULT_FINGERPRINT_SIMILARITY_THRESHOLD: Final[float] = 0.85
    DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # 1小时
    
    def __init__(
        self,
        trigger_detector: Optional[TriggerDetector] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        llm_processor: Optional[LLMMessageProcessor] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.trigger_detector: TriggerDetector = trigger_detector or TriggerDetector()
        self.emotion_analyzer: Optional[EmotionAnalyzer] = emotion_analyzer
        self.llm_processor: Optional[LLMMessageProcessor] = llm_processor
        
        # 配置合并
        cfg: Dict[str, Any] = config or {}
        self.mode: str = cfg.get("llm_processing_mode", self.DEFAULT_MODE)
        self.immediate_trigger_confidence: float = cfg.get(
            "immediate_trigger_confidence", self.DEFAULT_IMMEDIATE_TRIGGER_CONFIDENCE
        )
        self.immediate_emotion_intensity: float = cfg.get(
            "immediate_emotion_intensity", self.DEFAULT_IMMEDIATE_EMOTION_INTENSITY
        )
        
        # 性能优化配置
        self.hybrid_upper_threshold: float = cfg.get(
            "hybrid_upper_threshold", self.DEFAULT_HYBRID_UPPER_THRESHOLD
        )
        self.hybrid_lower_threshold: float = cfg.get(
            "hybrid_lower_threshold", self.DEFAULT_HYBRID_LOWER_THRESHOLD
        )
        self.llm_cooldown_seconds: int = cfg.get(
            "llm_cooldown_seconds", self.DEFAULT_LLM_COOLDOWN_SECONDS
        )
        self.batch_sample_limit: int = cfg.get(
            "batch_sample_limit", self.DEFAULT_BATCH_SAMPLE_LIMIT
        )
        self.fingerprint_cache_size: int = cfg.get(
            "fingerprint_cache_size", self.DEFAULT_FINGERPRINT_CACHE_SIZE
        )
        self.fingerprint_similarity_threshold: float = cfg.get(
            "fingerprint_similarity_threshold", self.DEFAULT_FINGERPRINT_SIMILARITY_THRESHOLD
        )
        
        # 缓存机制
        self._llm_cooldown = CooldownTracker(self.llm_cooldown_seconds)
        self._fingerprint_cache: Dict[str, tuple[ClassificationResult, float]] = {}
        
        # 统计
        self.stats: Dict[str, int] = {
            "local_classifications": 0,
            "llm_classifications": 0,
            "llm_fallbacks": 0,
            "llm_skipped_cooldown": 0,
            "llm_skipped_cached": 0,
            "llm_skipped_throttle": 0,
        }
    
    def _get_context_key(self, context: Dict) -> str:
        """获取上下文标识键（用于冷却时间判断）
        
        优先使用 user_id，其次是 group_id
        """
        user_id = context.get("user_id", "")
        group_id = context.get("group_id", "")
        if user_id:
            return f"user:{user_id}"
        if group_id:
            return f"group:{group_id}"
        return "global"
    
    def _get_message_fingerprint(self, message: str) -> str:
        """生成消息指纹（用于去重）
        
        使用简化后的消息内容计算哈希，忽略标点、空格、大小写
        """
        return compute_message_fingerprint(message, max_length=100, hash_length=16)
    
    def _check_fingerprint_cache(self, message: str) -> Optional[ClassificationResult]:
        """
        检查指纹缓存
        
        如果找到相似的近期消息，直接返回缓存结果
        
        Args:
            message: 消息内容
            
        Returns:
            Optional[ClassificationResult]: 缓存的分类结果
        """
        fingerprint = self._get_message_fingerprint(message)
        
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self._fingerprint_cache.items()
            if current_time - ts > self.DEFAULT_CACHE_TTL_SECONDS
        ]
        for k in expired_keys:
            del self._fingerprint_cache[k]
        
        # 检查完全匹配
        if fingerprint in self._fingerprint_cache:
            cached_result, _ = self._fingerprint_cache[fingerprint]
            self.stats["llm_skipped_cached"] += 1
            logger.debug(f"Fingerprint cache hit: {fingerprint[:8]}")
            return cached_result
        
        return None
    
    def _cache_result(self, message: str, result: ClassificationResult):
        """缓存分类结果"""
        fingerprint = self._get_message_fingerprint(message)
        self._fingerprint_cache[fingerprint] = (result, time.time())
        
        # 限制缓存大小
        if len(self._fingerprint_cache) > self.fingerprint_cache_size:
            # 删除最旧的条目
            oldest_key = min(
                self._fingerprint_cache.keys(),
                key=lambda k: self._fingerprint_cache[k][1]
            )
            del self._fingerprint_cache[oldest_key]
    
    def _check_cooldown(self, context_key: str) -> bool:
        """检查LLM冷却时间
        
        Returns:
            bool: True if can call LLM, False if in cooldown
        """
        if not self._llm_cooldown.is_ready(context_key):
            self.stats["llm_skipped_cooldown"] += 1
            logger.debug(f"LLM cooldown active for {context_key}, skipping")
            return False
        return True
    
    def _record_llm_call(self, context_key: str):
        """记录LLM调用时间"""
        self._llm_cooldown.record(context_key)
    
    async def classify(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        分类消息 - 根据配置选择策略
        
        Args:
            message: 消息内容
            context: 上下文信息
            
        Returns:
            ClassificationResult: 分类结果
        """
        ctx: Dict[str, Any] = context or {}
        
        try:
            # 策略1: 仅本地规则（最省token）
            if self.mode == "local":
                return await self._classify_local(message, ctx)
            
            # 策略2: 仅LLM（最准确但最费token）
            if self.mode == "llm":
                return await self._classify_llm(message, ctx)
            
            # 策略3: 混合模式（默认，优化版）
            return await self._classify_hybrid(message, ctx)
            
        except Exception as e:
            logger.debug(f"Classification failed, falling back to local: {e}")
            return await self._classify_local(message, ctx)
    
    async def _classify_local(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> ClassificationResult:
        """
        本地规则分类
        
        Args:
            message: 消息内容
            context: 上下文信息
            
        Returns:
            ClassificationResult: 分类结果
        """
        self.stats["local_classifications"] += 1
        
        try:
            # 1. 负样本检查
            if hasattr(self.trigger_detector, '_is_negative_sample'):
                if self.trigger_detector._is_negative_sample(message):
                    return ClassificationResult(
                        layer=ProcessingLayer.DISCARD,
                        confidence=1.0,
                        reason="negative_sample",
                        metadata={},
                        source=SourceType.LOCAL
                    )
            
            # 2. 检测触发器
            triggers: List[Dict[str, Any]] = []
            if hasattr(self.trigger_detector, 'detect_triggers'):
                triggers = self.trigger_detector.detect_triggers(message) or []
            
            # 3. 分析情感
            emotion_result: Dict[str, Any] = {}
            if self.emotion_analyzer and hasattr(self.emotion_analyzer, 'analyze_emotion'):
                try:
                    emotion_result = await self.emotion_analyzer.analyze_emotion(message, context)
                except Exception as e:
                    logger.debug(f"Emotion analysis failed: {e}")
            
            # 4. 决策逻辑
            if triggers:
                max_confidence = max(t.confidence for t in triggers)
                if max_confidence >= self.immediate_trigger_confidence:
                    return ClassificationResult(
                        layer=ProcessingLayer.IMMEDIATE,
                        confidence=max_confidence,
                        reason="high_confidence_trigger",
                        metadata={"triggers": triggers, "emotion": emotion_result},
                        source=SourceType.LOCAL
                    )
            
            if emotion_result.get("intensity", 0) >= self.immediate_emotion_intensity:
                return ClassificationResult(
                    layer=ProcessingLayer.IMMEDIATE,
                    confidence=emotion_result["intensity"],
                    reason="high_emotion_intensity",
                    metadata={"triggers": triggers, "emotion": emotion_result},
                    source=SourceType.LOCAL
                )
            
            return ClassificationResult(
                layer=ProcessingLayer.BATCH,
                confidence=0.5,
                reason="normal_message",
                metadata={"triggers": triggers, "emotion": emotion_result},
                source=SourceType.LOCAL
            )
            
        except Exception as e:
            logger.warning(f"Local classification failed: {e}")
            return ClassificationResult(
                layer=ProcessingLayer.BATCH,
                confidence=0.5,
                reason="classification_error",
                metadata={"error": str(e)},
                source=SourceType.LOCAL
            )
    
    async def _classify_llm(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> ClassificationResult:
        """
        LLM分类（带缓存和冷却）
        
        Args:
            message: 消息内容
            context: 上下文信息
            
        Returns:
            ClassificationResult: 分类结果
        """
        # 检查LLM处理器可用性
        if not self.llm_processor or not hasattr(self.llm_processor, 'is_available'):
            self.stats["llm_fallbacks"] += 1
            return await self._classify_local(message, context)
        
        try:
            if not self.llm_processor.is_available():
                self.stats["llm_fallbacks"] += 1
                return await self._classify_local(message, context)
        except Exception as e:
            logger.debug(f"LLM availability check failed: {e}")
            self.stats["llm_fallbacks"] += 1
            return await self._classify_local(message, context)
        
        context_key = self._get_context_key(context)
        
        # 检查指纹缓存
        cached = self._check_fingerprint_cache(message)
        if cached:
            return cached
        
        # 检查冷却时间
        if not self._check_cooldown(context_key):
            local_result = await self._classify_local(message, context)
            return ClassificationResult(
                layer=local_result.layer,
                confidence=local_result.confidence * 0.9,
                reason=f"{local_result.reason} (llm_cooldown)",
                metadata={**local_result.metadata, "llm_skipped": "cooldown"},
                source=SourceType.LOCAL
            )
        
        # 调用LLM
        try:
            self.stats["llm_classifications"] += 1
            self._record_llm_call(context_key)
            
            llm_result = await self.llm_processor.classify_message(message, context)
            
            if llm_result:
                result = ClassificationResult(
                    layer=ProcessingLayer(llm_result.layer),
                    confidence=llm_result.confidence,
                    reason=f"llm: {llm_result.reason}",
                    metadata=llm_result.metadata,
                    source=SourceType.LLM
                )
                self._cache_result(message, result)
                return result
            
        except Exception as e:
            logger.warning(f"LLM classification call failed: {e}")
        
        self.stats["llm_fallbacks"] += 1
        return await self._classify_local(message, context)
    
    async def _classify_hybrid(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> ClassificationResult:
        """
        混合分类策略
        
        优化点：
        1. 收紧阈值范围（默认0.6-0.8），减少边缘情况
        2. 增加冷却时间检查，避免频繁调用
        3. 增加指纹缓存，复用相似消息结果
        
        Args:
            message: 消息内容
            context: 上下文信息
            
        Returns:
            ClassificationResult: 分类结果
        """
        context_key = self._get_context_key(context)
        
        # 先检查指纹缓存
        cached = self._check_fingerprint_cache(message)
        if cached:
            return cached
        
        # 本地分类
        local_result = await self._classify_local(message, context)
        
        # 收紧阈值范围：只有真正不确定的消息才会走LLM
        if local_result.confidence >= self.hybrid_upper_threshold:
            return local_result
        
        if local_result.confidence <= self.hybrid_lower_threshold:
            return local_result
        
        # 边缘情况：检查冷却时间
        if not self._check_cooldown(context_key):
            self.stats["llm_skipped_cooldown"] += 1
            logger.debug(f"LLM in cooldown for {context_key}, using local result")
            return ClassificationResult(
                layer=local_result.layer,
                confidence=local_result.confidence,
                reason=f"{local_result.reason} (llm_cooldown)",
                metadata={**local_result.metadata, "llm_skipped": "cooldown"},
                source=SourceType.LOCAL
            )
        
        # 边缘情况且不在冷却中，尝试使用LLM确认
        if self.llm_processor and hasattr(self.llm_processor, 'is_available'):
            try:
                if not self.llm_processor.is_available():
                    return local_result
                
                self._record_llm_call(context_key)
                llm_result = await self.llm_processor.classify_message(message, context)
                
                if llm_result and llm_result.confidence >= 0.6:
                    self.stats["llm_classifications"] += 1
                    result = ClassificationResult(
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
                        source=SourceType.LLM
                    )
                    self._cache_result(message, result)
                    return result
                    
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to local: {e}")
                self.stats["llm_fallbacks"] += 1
        
        return local_result
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息
        
        Returns:
            Dict[str, int]: 统计信息字典
        """
        return self.stats.copy()
    
    def clear_cache(self) -> None:
        """
        清除缓存（用于测试或重置）
        """
        self._fingerprint_cache.clear()
        self._last_llm_call.clear()
        logger.debug("Message classifier cache cleared")
