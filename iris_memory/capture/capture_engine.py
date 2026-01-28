"""
记忆捕获引擎
根据companion-memory框架实现智能记忆捕获
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.utils.logger import logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import (
    MemoryType, ModalityType, QualityLevel, SensitivityLevel,
    StorageLayer, VerificationMethod, TriggerType
)
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.capture.trigger_detector import TriggerDetector
from iris_memory.capture.sensitivity_detector import SensitivityDetector
from iris_memory.analysis.rif_scorer import RIFScorer


class MemoryCaptureEngine:
    """记忆捕获引擎
    
    实现智能记忆捕获：
    - 触发器检测
    - 情感分析
    - 敏感度检测
    - 去重检查
    - 冲突检测
    - 质量评估
    - RIF评分计算
    """
    
    def __init__(
        self,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        rif_scorer: Optional[RIFScorer] = None
    ):
        """初始化记忆捕获引擎
        
        Args:
            emotion_analyzer: 情感分析器（可选）
            rif_scorer: RIF评分器（可选）
        """
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.rif_scorer = rif_scorer or RIFScorer()
        self.trigger_detector = TriggerDetector()
        self.sensitivity_detector = SensitivityDetector()
        
        # 配置
        self.auto_capture = True
        self.min_confidence = 0.3
        self.rif_threshold = 0.4
    
    async def capture_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        is_user_requested: bool = False
    ) -> Optional[Memory]:
        """捕获记忆
        
        Args:
            message: 用户消息
            user_id: 用户ID
            group_id: 群组ID（可选）
            context: 上下文信息（可选）
            is_user_requested: 是否用户显式请求
            
        Returns:
            Optional[Memory]: 捕获的记忆对象，如果不应该捕获则返回None
        """
        try:
            # 1. 检查负样本
            if self.trigger_detector._is_negative_sample(message):
                logger.debug("Message identified as negative sample, skipping capture")
                return None
            
            # 2. 检测触发器
            triggers = self.trigger_detector.detect_triggers(message)
            if not triggers and not is_user_requested and not self.auto_capture:
                logger.debug("No trigger detected and auto_capture is False, skipping")
                return None
            
            # 3. 敏感度检测
            sensitivity_level, detected_entities = \
                self.sensitivity_detector.detect_sensitivity(message, context)
            
            # 4. 判断是否应该过滤
            if self.sensitivity_detector.should_filter(sensitivity_level):
                logger.warning(f"Memory filtered due to high sensitivity: {sensitivity_level}")
                return None
            
            # 5. 情感分析
            emotion_result = await self.emotion_analyzer.analyze_emotion(
                message, context
            )
            
            # 6. 确定记忆类型
            memory_type = self._determine_memory_type(triggers, emotion_result)
            
            # 7. 创建记忆对象
            memory = Memory(
                user_id=user_id,
                group_id=group_id,
                type=memory_type,
                modality=ModalityType.TEXT,
                content=message,
                sensitivity_level=sensitivity_level,
                detected_entities=detected_entities,
                is_user_requested=is_user_requested
            )
            
            # 8. 设置情感信息
            if memory_type == MemoryType.EMOTION:
                memory.subtype = emotion_result["primary"].value
                memory.emotional_weight = emotion_result["intensity"]
            
            # 9. 设置摘要
            memory.summary = self._generate_summary(message, triggers)
            
            # 10. 质量评估
            self._assess_quality(memory, triggers, emotion_result)
            
            # 11. 计算RIF评分
            self.rif_scorer.calculate_rif(memory)
            
            # 12. 确定初始存储层
            self._determine_storage_layer(memory, is_user_requested)
            
            logger.info(f"Memory captured: {memory.id}, type: {memory_type}, confidence: {memory.confidence}")
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to capture memory: {e}")
            return None
    
    def _determine_memory_type(
        self,
        triggers: List[Dict[str, Any]],
        emotion_result: Dict[str, Any]
    ) -> MemoryType:
        """确定记忆类型
        
        Args:
            triggers: 触发器列表
            emotion_result: 情感分析结果
            
        Returns:
            MemoryType: 记忆类型
        """
        trigger_types = [t["type"] for t in triggers]
        
        # 根据触发器类型确定记忆类型
        if TriggerType.EMOTION in trigger_types:
            return MemoryType.EMOTION
        elif TriggerType.PREFERENCE in trigger_types:
            return MemoryType.FACT
        elif TriggerType.RELATIONSHIP in trigger_types:
            return MemoryType.RELATIONSHIP
        elif TriggerType.BOUNDARY in trigger_types:
            return MemoryType.FACT
        elif TriggerType.FACT in trigger_types:
            return MemoryType.FACT
        elif TriggerType.EXPLICIT in trigger_types:
            # 显式触发器根据内容判断
            if emotion_result["intensity"] > 0.6:
                return MemoryType.EMOTION
            else:
                return MemoryType.FACT
        else:
            # 没有明确触发器，根据情感强度判断
            if emotion_result["intensity"] > 0.7:
                return MemoryType.EMOTION
            else:
                return MemoryType.INTERACTION
    
    def _generate_summary(
        self,
        message: str,
        triggers: List[Dict[str, Any]]
    ) -> Optional[str]:
        """生成记忆摘要
        
        Args:
            message: 原始消息
            triggers: 触发器列表
            
        Returns:
            Optional[str]: 摘要文本
        """
        # 简单实现：取消息的前100个字符
        if len(message) <= 100:
            return None  # 不需要摘要
        return message[:97] + "..."
    
    def _assess_quality(
        self,
        memory: Memory,
        triggers: List[Dict[str, Any]],
        emotion_result: Dict[str, Any]
    ):
        """评估记忆质量
        
        Args:
            memory: 记忆对象
            triggers: 触发器列表
            emotion_result: 情感分析结果
        """
        # 计算置信度
        confidence_factors = []
        
        # 1. 触发器置信度
        if triggers:
            max_trigger_confidence = max(t["confidence"] for t in triggers)
            confidence_factors.append(max_trigger_confidence)
        else:
            confidence_factors.append(0.3)  # 无触发器的默认置信度
        
        # 2. 情感分析置信度
        confidence_factors.append(emotion_result["confidence"])
        
        # 3. 上下文一致性（简化：假设0.5）
        confidence_factors.append(0.5)
        
        # 综合置信度
        memory.confidence = sum(confidence_factors) / len(confidence_factors)
        
        # 4. 确定质量等级
        if memory.confidence >= 0.9:
            memory.quality_level = QualityLevel.CONFIRMED
        elif memory.confidence >= 0.75:
            memory.quality_level = QualityLevel.HIGH_CONFIDENCE
        elif memory.confidence >= 0.5:
            memory.quality_level = QualityLevel.MODERATE
        elif memory.confidence >= 0.3:
            memory.quality_level = QualityLevel.LOW_CONFIDENCE
        else:
            memory.quality_level = QualityLevel.UNCERTAIN
        
        # 5. 设置验证方法
        if memory.is_user_requested:
            memory.verification_method = VerificationMethod.USER_EXPLICIT
        elif any(t["type"] == TriggerType.EXPLICIT for t in triggers):
            memory.verification_method = VerificationMethod.USER_EXPLICIT
        elif len(triggers) > 1:
            memory.verification_method = VerificationMethod.MULTIPLE_MENTIONS
        else:
            memory.verification_method = VerificationMethod.SYSTEM_INFERRED
    
    def _determine_storage_layer(self, memory: Memory, is_user_requested: bool):
        """确定初始存储层
        
        Args:
            memory: 记忆对象
            is_user_requested: 是否用户显式请求
        """
        if is_user_requested and memory.importance_score > 0.7:
            # 用户请求且重要性高，直接存到情景记忆
            memory.storage_layer = StorageLayer.EPISODIC
        elif memory.quality_level == QualityLevel.CONFIRMED:
            # 已确认的记忆，存到语义记忆
            memory.storage_layer = StorageLayer.SEMANTIC
        elif memory.confidence >= self.min_confidence:
            # 满足最小置信度，存到工作记忆
            memory.storage_layer = StorageLayer.WORKING
        else:
            # 不满足条件，不存储
            memory.storage_layer = StorageLayer.WORKING
    
    async def check_duplicate(
        self,
        memory: Memory,
        existing_memories: List[Memory],
        similarity_threshold: float = 0.9
    ) -> Optional[Memory]:
        """检查重复记忆
        
        Args:
            memory: 新记忆
            existing_memories: 已有记忆列表
            similarity_threshold: 相似度阈值
            
        Returns:
            Optional[Memory]: 如果找到重复记忆则返回，否则返回None
        """
        for existing in existing_memories:
            # 简单的内容相似度检查
            if self._calculate_similarity(memory.content, existing.content) > similarity_threshold:
                logger.info(f"Duplicate memory detected: {memory.id} similar to {existing.id}")
                return existing
        return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度（0-1）
        """
        # 使用简单的Jaccard相似度
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def check_conflicts(
        self,
        memory: Memory,
        existing_memories: List[Memory]
    ) -> List[Memory]:
        """检查记忆冲突
        
        Args:
            memory: 新记忆
            existing_memories: 已有记忆列表
            
        Returns:
            List[Memory]: 冲突的记忆列表
        """
        conflicts = []
        
        for existing in existing_memories:
            # 检查是否为相同类型但内容相反
            if memory.type == existing.type and self._is_opposite(memory.content, existing.content):
                conflicts.append(existing)
                memory.add_conflict(existing.id)
        
        return conflicts
    
    def _is_opposite(self, text1: str, text2: str) -> bool:
        """判断两个文本是否相反
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            bool: 是否相反
        """
        # 简化实现：检查否定词
        negation_words = ["不", "没", "不是", "don't", "not", "no"]
        
        for neg in negation_words:
            if neg in text1 and neg not in text2:
                # 移除否定词后比较
                text1_clean = text1.replace(neg, "")
                if text1_clean in text2 or text2 in text1_clean:
                    return True
        
        return False
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置
        
        Args:
            config: 配置字典
        """
        self.auto_capture = config.get("auto_capture", True)
        self.min_confidence = config.get("min_confidence", 0.3)
        self.rif_threshold = config.get("rif_threshold", 0.4)
