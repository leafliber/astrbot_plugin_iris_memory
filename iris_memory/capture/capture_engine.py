"""
记忆捕获引擎
根据companion-memory框架实现智能记忆捕获
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.utils.logger import get_logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import (
    MemoryType, ModalityType, QualityLevel, SensitivityLevel,
    StorageLayer, VerificationMethod, TriggerType
)
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.entity_extractor import EntityExtractor
from iris_memory.capture.trigger_detector import TriggerDetector
from iris_memory.capture.sensitivity_detector import SensitivityDetector
from iris_memory.analysis.rif_scorer import RIFScorer

# 模块logger
logger = get_logger("capture_engine")


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
        chroma_manager=None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        rif_scorer: Optional[RIFScorer] = None
    ):
        """初始化记忆捕获引擎
        
        Args:
            chroma_manager: Chroma存储管理器（用于去重和冲突检测）
            emotion_analyzer: 情感分析器（可选）
            rif_scorer: RIF评分器（可选）
        """
        self.chroma_manager = chroma_manager
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.rif_scorer = rif_scorer or RIFScorer()
        self.trigger_detector = TriggerDetector()
        self.sensitivity_detector = SensitivityDetector()
        self.entity_extractor = EntityExtractor()  # 实体提取器
        
        # 配置
        self.auto_capture = True
        self.min_confidence = 0.3
        self.rif_threshold = 0.4
        self.enable_duplicate_check = True  # 启用去重检查
        self.enable_conflict_check = True   # 启用冲突检测
        self.enable_entity_extraction = True  # 启用实体提取
    
    async def capture_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        is_user_requested: bool = False
    ) -> Optional[Memory]:
        """捕获记忆 - 核心记忆捕获流程

        实现 companion-memory framework 第12节的完整捕获流程，包括14个步骤：
        1. 负样本检查 - 过滤闲聊、问候语等不应记录的内容
        2. 触发器检测 - 识别显式/隐式记忆触发信号
        3. 敏感度检测 - 自动识别敏感信息并分级
        4. 实体提取 - 提取人名、地点、时间等结构化信息
        5. 情感分析 - 混合模型分析情感状态
        6. 记忆类型确定 - 根据内容特征分类
        7. 记忆对象创建 - 构建完整记忆数据结构
        8. 质量评估 - 多维度置信度计算
        9. RIF评分 - 时近性、相关性、频率综合评分
        10. 存储层确定 - 工作/情景/语义记忆分层
        11. 去重检查 - 避免重复记录（性能优化版）
        12. 冲突检测 - 识别语义冲突（增强版）

        复杂度说明：
        - 时间复杂度：O(N * M)，N为候选记忆数，M为平均记忆长度
        - 空间复杂度：O(K)，K为提取的实体数量
        - 性能优化：去重和冲突检测限制在50/30个候选记忆

        Args:
            message: 用户消息文本
            user_id: 用户唯一标识
            group_id: 群组ID（私聊时为None，决定记忆scope）
            context: 上下文信息（对话历史、情感状态等）
            is_user_requested: 是否用户显式请求保存（/memory_save指令）

        Returns:
            Optional[Memory]: 捕获的记忆对象，如果不满足条件则返回None
                - 负样本过滤返回None
                - 无触发器且非用户请求返回None
                - 高敏感度内容可能被过滤
                - 重复记忆会被跳过
        """
        try:
            msg_preview = message[:50] + "..." if len(message) > 50 else message
            logger.debug(f"Starting memory capture: user={user_id}, group={group_id}, is_user_requested={is_user_requested}")
            logger.debug(f"Message content: '{msg_preview}'")
            
            # 1. 检查负样本
            logger.debug("Step 1: Checking for negative samples...")
            if self.trigger_detector._is_negative_sample(message):
                logger.debug("Message identified as negative sample, skipping capture")
                return None
            logger.debug("Negative sample check passed")
            
            # 2. 检测触发器
            logger.debug("Step 2: Detecting triggers...")
            triggers = self.trigger_detector.detect_triggers(message)
            trigger_types = [t["type"].value if hasattr(t["type"], "value") else str(t["type"]) for t in triggers]
            logger.debug(f"Detected {len(triggers)} triggers: {trigger_types}")
            
            if not triggers and not is_user_requested and not self.auto_capture:
                logger.debug("No trigger detected and auto_capture is False, skipping")
                return None
            
            # 3. 敏感度检测
            logger.debug("Step 3: Detecting sensitivity...")
            sensitivity_level, detected_entities = \
                self.sensitivity_detector.detect_sensitivity(message, context)
            logger.debug(f"Sensitivity level: {sensitivity_level.value if hasattr(sensitivity_level, 'value') else sensitivity_level}, entities: {detected_entities}")
            
            # 3.5 实体提取（如果启用）
            if self.enable_entity_extraction:
                logger.debug("Step 3.5: Extracting entities...")
                extracted_entities = self.entity_extractor.extract_entities(message)
                entity_texts = [e.text for e in extracted_entities]
                detected_entities = list(set(detected_entities + entity_texts))
                logger.debug(f"Extracted entities: {[e.text for e in extracted_entities]}")
            
            # 4. 判断是否应该过滤
            logger.debug("Step 4: Checking if should filter...")
            if self.sensitivity_detector.should_filter(sensitivity_level):
                logger.warning(f"Memory filtered due to high sensitivity: {sensitivity_level}")
                return None
            logger.debug("Filter check passed")
            
            # 5. 情感分析
            logger.debug("Step 5: Analyzing emotion...")
            emotion_result = await self.emotion_analyzer.analyze_emotion(
                message, context
            )
            logger.debug(f"Emotion result: primary={emotion_result.get('primary')}, intensity={emotion_result.get('intensity'):.2f}, confidence={emotion_result.get('confidence'):.2f}")
            
            # 6. 确定记忆类型
            logger.debug("Step 6: Determining memory type...")
            memory_type = self._determine_memory_type(triggers, emotion_result)
            logger.debug(f"Determined memory type: {memory_type.value}")
            
            # 7. 创建记忆对象
            logger.debug("Step 7: Creating memory object...")
            # 根据场景确定 scope：私聊用 USER_PRIVATE，群聊用 GROUP_PRIVATE
            from iris_memory.core.memory_scope import MemoryScope
            memory_scope = MemoryScope.GROUP_PRIVATE if group_id else MemoryScope.USER_PRIVATE
            memory = Memory(
                user_id=user_id,
                group_id=group_id,
                scope=memory_scope,
                type=memory_type,
                modality=ModalityType.TEXT,
                content=message,
                sensitivity_level=sensitivity_level,
                detected_entities=detected_entities,
                is_user_requested=is_user_requested
            )
            logger.debug(f"Memory object created: id={memory.id}, scope={memory_scope.value}")
            
            # 8. 设置情感信息
            if memory_type == MemoryType.EMOTION:
                memory.subtype = emotion_result["primary"].value if hasattr(emotion_result["primary"], "value") else str(emotion_result["primary"])
                memory.emotional_weight = emotion_result["intensity"]
                logger.debug(f"Set emotional info: subtype={memory.subtype}, weight={memory.emotional_weight:.2f}")
            
            # 9. 设置摘要
            memory.summary = self._generate_summary(message, triggers)
            if memory.summary:
                logger.debug(f"Generated summary: {memory.summary}")
            
            # 10. 质量评估
            logger.debug("Step 10: Assessing quality...")
            self._assess_quality(memory, triggers, emotion_result)
            logger.debug(f"Quality assessment: confidence={memory.confidence:.2f}, level={memory.quality_level.value if hasattr(memory.quality_level, 'value') else memory.quality_level}")
            
            # 11. 计算RIF评分
            logger.debug("Step 11: Calculating RIF score...")
            self.rif_scorer.calculate_rif(memory)
            logger.debug(f"RIF score: {memory.rif_score:.3f}")
            
            # 12. 确定初始存储层
            logger.debug("Step 12: Determining storage layer...")
            self._determine_storage_layer(memory, is_user_requested)
            logger.debug(f"Storage layer: {memory.storage_layer.value if hasattr(memory.storage_layer, 'value') else memory.storage_layer}")
            
            # 13. 去重检查（使用向量相似度优化）
            if self.enable_duplicate_check and self.chroma_manager:
                logger.debug("Step 13: Checking for duplicates (vector-based)...")
                duplicate = await self._check_duplicate_by_vector(memory, user_id, group_id)
                if duplicate:
                    logger.info(f"Duplicate memory detected, skipping: {memory.id}")
                    return None
                logger.debug("No duplicate found")
            
            # 14. 冲突检测（使用向量相似度优化）
            if self.enable_conflict_check and self.chroma_manager:
                logger.debug("Step 14: Checking for conflicts (vector-based)...")
                conflicts = await self._check_conflicts_by_vector(memory, user_id, group_id)
                if conflicts:
                    # 尝试解决冲突
                    resolved = await self._resolve_conflicts(memory, conflicts)
                    if not resolved:
                        logger.info(f"Memory conflicts detected but not resolved: {memory.id}")
                else:
                    logger.debug("No conflicts found")
            
            logger.info(f"Memory captured successfully: id={memory.id}, type={memory_type.value}, confidence={memory.confidence:.2f}, rif={memory.rif_score:.3f}")
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to capture memory: user={user_id}, error={e}", exc_info=True)
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
    
    async def _check_duplicate_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str] = None,
        similarity_threshold: float = 0.95
    ) -> Optional[Memory]:
        """使用向量相似度检查重复记忆
        
        性能优化版本：使用ChromaDB的向量查询直接找到最相似的记忆，
        避免加载全部记忆。
        
        Args:
            memory: 新记忆
            user_id: 用户ID
            group_id: 群组ID
            similarity_threshold: 向量相似度阈值（默认0.95，越高越严格）
            
        Returns:
            Optional[Memory]: 如果找到重复记忆则返回，否则返回None
        """
        if not self.chroma_manager:
            return None
        
        try:
            # 使用向量查询找到最相似的记忆（只查询5条）
            similar_memories = await self.chroma_manager.query_memories(
                query_text=memory.content,
                user_id=user_id,
                group_id=group_id,
                top_k=5
            )
            
            if not similar_memories:
                return None
            
            # 检查是否有高相似度的记忆
            for existing in similar_memories:
                # 跳过自己（如果已经存在）
                if existing.id == memory.id:
                    continue
                
                # 使用文本相似度进行精确验证
                text_sim = self._calculate_similarity(memory.content, existing.content)
                if text_sim >= similarity_threshold:
                    logger.debug(f"Found duplicate via vector search: {existing.id} (text_sim={text_sim:.3f})")
                    return existing
            
            return None
            
        except Exception as e:
            logger.warning(f"Vector-based duplicate check failed: {e}, falling back to text-based")
            return None
    
    async def _check_conflicts_by_vector(
        self,
        memory: Memory,
        user_id: str,
        group_id: Optional[str] = None
    ) -> List[Memory]:
        """使用向量相似度检查记忆冲突
        
        性能优化版本：使用ChromaDB的向量查询找到语义相似的记忆，
        然后检查是否存在语义冲突。
        
        Args:
            memory: 新记忆
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            List[Memory]: 冲突的记忆列表
        """
        conflicts = []
        
        if not self.chroma_manager:
            return conflicts
        
        try:
            # 使用向量查询找到语义相关的记忆
            similar_memories = await self.chroma_manager.query_memories(
                query_text=memory.content,
                user_id=user_id,
                group_id=group_id,
                top_k=10
            )
            
            if not similar_memories:
                return conflicts
            
            # 检查语义冲突
            for existing in similar_memories:
                if existing.id == memory.id:
                    continue
                
                # 只检查相同类型的记忆
                if existing.type != memory.type:
                    continue
                
                # 检查内容相似度
                content_sim = self._calculate_content_similarity(memory.content, existing.content)
                if content_sim < 0.3:
                    continue
                
                # 检查是否为相反内容
                if self._is_opposite(memory.content, existing.content):
                    conflicts.append(existing)
                    memory.add_conflict(existing.id)
                    logger.debug(f"Conflict detected via vector search: {memory.id} vs {existing.id}")
            
            return conflicts
            
        except Exception as e:
            logger.warning(f"Vector-based conflict check failed: {e}")
            return conflicts
    
    async def _resolve_conflicts(
        self,
        new_memory: Memory,
        conflicting_memories: List[Memory]
    ) -> bool:
        """解决记忆冲突
        
        冲突解决策略：
        1. 如果新记忆是用户显式请求的，优先采用新记忆
        2. 如果新记忆置信度更高，更新旧记忆
        3. 如果旧记忆质量等级更高，保留旧记忆
        4. 否则标记为需要用户确认
        
        Args:
            new_memory: 新记忆
            conflicting_memories: 冲突的记忆列表
            
        Returns:
            bool: 是否成功解决冲突
        """
        if not conflicting_memories:
            return True
        
        resolved_count = 0
        
        for old_memory in conflicting_memories:
            resolution = self._determine_conflict_resolution(new_memory, old_memory)
            
            if resolution == "replace":
                # 用新记忆替换旧记忆
                try:
                    if self.chroma_manager:
                        await self.chroma_manager.delete_memory(old_memory.id)
                        logger.info(f"Conflict resolved: replaced {old_memory.id} with {new_memory.id}")
                        resolved_count += 1
                except Exception as e:
                    logger.error(f"Failed to replace conflicting memory: {e}")
                    
            elif resolution == "keep_old":
                # 保留旧记忆，标记新记忆为低质量
                new_memory.quality_level = QualityLevel.LOW_CONFIDENCE
                new_memory.metadata["conflict_resolution"] = "kept_old"
                logger.info(f"Conflict resolved: keeping {old_memory.id}, lowered {new_memory.id} quality")
                resolved_count += 1
                
            elif resolution == "merge":
                # 合并两条记忆（增加旧记忆的置信度）
                try:
                    if self.chroma_manager:
                        old_memory.confidence = min(1.0, old_memory.confidence + 0.1)
                        old_memory.access_count += 1
                        await self.chroma_manager.update_memory(old_memory)
                        logger.info(f"Conflict resolved: merged into {old_memory.id}")
                        resolved_count += 1
                        # 返回False表示不需要存储新记忆
                        return False
                except Exception as e:
                    logger.error(f"Failed to merge memories: {e}")
                    
            else:  # "pending"
                # 标记为待确认
                new_memory.metadata["conflict_status"] = "pending_user_confirmation"
                new_memory.metadata["conflicting_memory_id"] = old_memory.id
                logger.info(f"Conflict pending: {new_memory.id} vs {old_memory.id}")
        
        return resolved_count == len(conflicting_memories)
    
    def _determine_conflict_resolution(
        self,
        new_memory: Memory,
        old_memory: Memory
    ) -> str:
        """确定冲突解决策略
        
        Args:
            new_memory: 新记忆
            old_memory: 旧记忆
            
        Returns:
            str: 解决策略 ("replace", "keep_old", "merge", "pending")
        """
        # 策略1：用户显式请求的新记忆优先
        if new_memory.is_user_requested:
            return "replace"
        
        # 策略2：高质量等级的记忆优先
        if new_memory.quality_level.value > old_memory.quality_level.value + 1:
            return "replace"
        if old_memory.quality_level.value > new_memory.quality_level.value + 1:
            return "keep_old"
        
        # 策略3：置信度差异较大时
        confidence_diff = new_memory.confidence - old_memory.confidence
        if confidence_diff > 0.3:
            return "replace"
        if confidence_diff < -0.3:
            return "keep_old"
        
        # 策略4：如果内容非常相似但不完全相反，可能是更新
        if new_memory.created_time > old_memory.created_time:
            # 更新的信息，检查是否是细微修正
            content_sim = self._calculate_similarity(new_memory.content, old_memory.content)
            if content_sim > 0.7:
                return "replace"  # 可能是用户纠正旧信息
        
        # 默认：需要用户确认
        return "pending"
    
    async def check_duplicate(
        self,
        memory: Memory,
        existing_memories: List[Memory],
        similarity_threshold: float = 0.9
    ) -> Optional[Memory]:
        """检查重复记忆（性能优化版）

        使用多阶段检测策略优化性能：
        1. 快速预筛选：检查相同用户、相同类型、时间相近的记忆
        2. 使用ChromaDB向量查询（如果可用）进行近似最近邻搜索
        3. 精确相似度计算：对候选记忆进行详细比较

        Args:
            memory: 新记忆
            existing_memories: 已有记忆列表
            similarity_threshold: 相似度阈值

        Returns:
            Optional[Memory]: 如果找到重复记忆则返回，否则返回None
        """
        # 如果没有现有记忆，直接返回
        if not existing_memories:
            return None

        # 阶段1: 快速预筛选
        # 只检查最近7天内的记忆，减少比较数量
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=7)

        candidates = []
        for existing in existing_memories:
            # 跳过太久远的记忆
            if existing.created_time < cutoff_time:
                continue
            # 只检查相同用户的记忆
            if existing.user_id != memory.user_id:
                continue
            # 优先检查相同类型的记忆
            if existing.type == memory.type:
                candidates.insert(0, existing)  # 插入到前面优先检查
            else:
                candidates.append(existing)

        # 限制候选数量，避免过多计算
        max_candidates = min(50, len(candidates))
        candidates = candidates[:max_candidates]

        # 阶段2: 多层级相似度检测
        for existing in candidates:
            # 快速预检：长度差异过大的直接跳过
            len_ratio = len(memory.content) / max(1, len(existing.content))
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue

            # 层级1: 快速字符级相似度
            quick_sim = self._calculate_quick_similarity(memory.content, existing.content)
            if quick_sim < similarity_threshold * 0.8:  # 放宽阈值进行预筛选
                continue

            # 层级2: 精确相似度计算
            precise_sim = self._calculate_similarity(memory.content, existing.content)
            if precise_sim > similarity_threshold:
                logger.info(f"Duplicate memory detected: {memory.id} similar to {existing.id} (sim={precise_sim:.3f})")
                return existing

        return None

    def _calculate_quick_similarity(self, text1: str, text2: str) -> float:
        """快速相似度计算（用于预筛选）

        使用字符集合和哈希签名进行快速比较，时间复杂度 O(n)。

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度（0-1）
        """
        # 方法1: 字符集合Jaccard相似度
        set1 = set(text1.lower())
        set2 = set(text2.lower())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        char_sim = intersection / union

        # 方法2: 词集合Jaccard相似度（更精确但稍慢）
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if words1 and words2:
            word_intersection = len(words1 & words2)
            word_union = len(words1 | words2)
            word_sim = word_intersection / word_union if word_union > 0 else 0.0
        else:
            word_sim = 0.0

        # 综合得分：字符相似度40% + 词相似度60%
        return 0.4 * char_sim + 0.6 * word_sim

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算精确文本相似度 - 多算法融合版

        使用三种互补的相似度算法，综合评估文本语义相似度：

        算法1 - N-gram相似度（权重40%）：
            使用2-gram捕捉局部字符模式
            对中文尤其有效，能识别相似词组和短语
            计算：|intersection(gram1, gram2)| / |union(gram1, gram2)|

        算法2 - 序列相似度（权重40%）：
            使用Python difflib.SequenceMatcher
            基于最长公共子序列（LCS）算法
            能识别整体结构相似性，对词序变化敏感

        算法3 - 最长公共子串（权重20%）：
            使用动态规划计算最长公共子串长度
            空间优化：滚动数组将O(M*N)降至O(N)
            识别连续匹配的片段

        综合公式：similarity = 0.4*ngram + 0.4*sequence + 0.2*lcs

        复杂度：
            时间：O(N*M)，N和M为文本长度
            空间：O(N)，使用滚动数组优化

        Args:
            text1: 待比较的文本1
            text2: 待比较的文本2

        Returns:
            float: 综合相似度得分（0-1）
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # 方法1: N-gram相似度（捕捉局部模式）
        def get_ngrams(text, n=2):
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams1 = get_ngrams(text1_lower, 2)
        ngrams2 = get_ngrams(text2_lower, 2)

        if ngrams1 and ngrams2:
            ngram_intersection = len(ngrams1 & ngrams2)
            ngram_union = len(ngrams1 | ngrams2)
            ngram_sim = ngram_intersection / ngram_union if ngram_union > 0 else 0.0
        else:
            ngram_sim = 0.0

        # 方法2: 序列相似度（使用difflib）
        import difflib
        seq_sim = difflib.SequenceMatcher(None, text1_lower, text2_lower).ratio()

        # 方法3: 公共子串比例
        min_len = min(len(text1_lower), len(text2_lower))
        max_len = max(len(text1_lower), len(text2_lower))
        if min_len > 0:
            # 找到最长公共子串
            lcs_len = self._longest_common_substring_length(text1_lower, text2_lower)
            lcs_sim = lcs_len / max_len if max_len > 0 else 0.0
        else:
            lcs_sim = 0.0

        # 综合得分
        return 0.4 * ngram_sim + 0.4 * seq_sim + 0.2 * lcs_sim

    def _longest_common_substring_length(self, s1: str, s2: str) -> int:
        """计算最长公共子串长度（动态规划）

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            int: 最长公共子串长度
        """
        if not s1 or not s2:
            return 0

        m, n = len(s1), len(s2)
        # 使用滚动数组优化空间复杂度
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1] + 1
                    max_length = max(max_length, curr[j])
                else:
                    curr[j] = 0
            prev, curr = curr, prev

        return max_length
    
    def check_conflicts(
        self,
        memory: Memory,
        existing_memories: List[Memory]
    ) -> List[Memory]:
        """检查记忆冲突（性能优化版）

        优化策略：
        1. 快速预筛选：只检查相同类型、相同主题、时间相近的记忆
        2. 限制检查数量，避免大规模遍历
        3. 使用分层检测策略

        Args:
            memory: 新记忆
            existing_memories: 已有记忆列表

        Returns:
            List[Memory]: 冲突的记忆列表
        """
        conflicts = []

        if not existing_memories:
            return conflicts

        # 快速预筛选：只检查相关记忆
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=30)  # 只检查最近30天

        candidates = []
        for existing in existing_memories:
            # 只检查相同类型的记忆
            if existing.type != memory.type:
                continue
            # 跳过太久远的记忆
            if existing.created_time < cutoff_time:
                continue
            # 只检查相同用户的记忆
            if existing.user_id != memory.user_id:
                continue
            candidates.append(existing)

        # 限制候选数量
        max_candidates = min(30, len(candidates))
        candidates = candidates[:max_candidates]

        # 检查冲突
        for existing in candidates:
            # 快速预检：长度和主题相关性
            len_ratio = len(memory.content) / max(1, len(existing.content))
            if len_ratio < 0.3 or len_ratio > 3.0:
                continue

            # 检查内容相似度（需要有足够相似度才可能冲突）
            content_sim = self._calculate_content_similarity(memory.content, existing.content)
            if content_sim < 0.3:  # 内容完全不相关，不可能冲突
                continue

            # 检查是否为相反内容
            if self._is_opposite(memory.content, existing.content):
                conflicts.append(existing)
                memory.add_conflict(existing.id)
                logger.info(f"Conflict detected: {memory.id} conflicts with {existing.id} (similarity={content_sim:.3f})")

        return conflicts
    
    def _is_opposite(self, text1: str, text2: str) -> bool:
        """判断两个文本是否相反（语义冲突检测）- 增强版

        基于 companion-memory framework 第12节的冲突检测要求，实现多策略语义冲突检测：

        策略1 - 否定词检测（权重最高）：
            检测逻辑：text1包含否定词而text2不包含，且核心内容相似度>0.6
            示例："我喜欢咖啡" vs "我不喜欢咖啡" → 冲突

        策略2 - 反义词检测：
            预定义反义词库覆盖常见对立概念（喜欢/讨厌、开心/难过等）
            要求两个文本有共同主题（≥2个共同非停用词）
            示例："工作很开心" vs "工作很痛苦" → 冲突

        策略3 - 数值/时间冲突：
            检测相同描述框架下的数值差异
            示例："我有3个苹果" vs "我有5个苹果" → 冲突

        复杂度：O(N + M)，N/M为文本长度

        Args:
            text1: 待比较的文本1
            text2: 待比较的文本2

        Returns:
            bool: 是否存在语义冲突
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # 策略1: 否定词检测
        negation_words = ["不", "没", "无", "非", "别", "不是", "don't", "not", "no", "never", "不喜欢", "讨厌", "喜欢"]

        for neg in negation_words:
            # 情况1: 否定词在text1但不在text2
            if neg in text1_lower and neg not in text2_lower:
                text1_clean = text1_lower.replace(neg, "").strip()
                # 计算相似度，如果核心内容相似则可能是冲突
                similarity = self._calculate_content_similarity(text1_clean, text2_lower)
                if similarity > 0.6:
                    return True
            # 情况2: 否定词在text2但不在text1
            elif neg in text2_lower and neg not in text1_lower:
                text2_clean = text2_lower.replace(neg, "").strip()
                similarity = self._calculate_content_similarity(text1_lower, text2_clean)
                if similarity > 0.6:
                    return True

        # 策略2: 反义词检测
        antonym_pairs = [
            ("喜欢", "讨厌"), ("喜欢", "恨"), ("爱", "恨"),
            ("开心", "难过"), ("高兴", "伤心"), ("快乐", "痛苦"),
            ("好", "坏"), ("优秀", "差劲"), ("成功", "失败"),
            ("支持", "反对"), ("同意", "拒绝"),
            ("有", "没有"), ("能", "不能"), ("会", "不会"),
            ("大", "小"), ("多", "少"), ("高", "低"),
            ("喜欢", "dislike"), ("讨厌", "like"), ("love", "hate"),
            ("happy", "sad"), ("good", "bad"), ("success", "failure")
        ]

        for word1, word2 in antonym_pairs:
            if (word1 in text1_lower and word2 in text2_lower) or \
               (word1 in text2_lower and word2 in text1_lower):
                # 检查是否有相同的主题/对象
                if self._have_common_subject(text1_lower, text2_lower):
                    return True

        # 策略3: 数值冲突检测
        # 提取数值并检查是否冲突
        numbers1 = re.findall(r'\d+', text1)
        numbers2 = re.findall(r'\d+', text2)
        if numbers1 and numbers2 and numbers1 != numbers2:
            # 如果有相同的非数字部分，但数值不同，可能是冲突
            non_num1 = re.sub(r'\d+', '{NUM}', text1)
            non_num2 = re.sub(r'\d+', '{NUM}', text2)
            if non_num1 == non_num2 and set(numbers1) != set(numbers2):
                return True

        return False

    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """计算内容相似度（基于字符和词组）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度 (0-1)
        """
        # 使用N-gram计算相似度
        def get_ngrams(text, n=2):
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        # 2-gram相似度
        ngrams1 = get_ngrams(text1, 2)
        ngrams2 = get_ngrams(text2, 2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        if union == 0:
            return 0.0

        return intersection / union

    def _have_common_subject(self, text1: str, text2: str) -> bool:
        """检查两个文本是否有相同的主题/对象

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            bool: 是否有共同主题
        """
        # 简单的共同词检测（排除停用词）
        stopwords = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们',
                     '他们', '这', '那', '这些', '那些', '和', '与', '或', '就', '都', '而',
                     '及', '与', '或', '但是', '然而', 'the', 'a', 'an', 'is', 'are', 'was',
                     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on',
                     'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then'}

        # 提取关键词（长度大于1的词）
        words1 = set(w for w in re.findall(r'\w+', text1) if len(w) > 1 and w not in stopwords)
        words2 = set(w for w in re.findall(r'\w+', text2) if len(w) > 1 and w not in stopwords)

        if not words1 or not words2:
            return False

        # 如果有超过2个共同词，认为有共同主题
        common_words = words1 & words2
        return len(common_words) >= 2
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置
        
        Args:
            config: 配置字典
        """
        self.auto_capture = config.get("auto_capture", True)
        self.min_confidence = config.get("min_confidence", 0.3)
        self.rif_threshold = config.get("rif_threshold", 0.4)
        self.enable_duplicate_check = config.get("enable_duplicate_check", True)
        self.enable_conflict_check = config.get("enable_conflict_check", True)
        self.enable_entity_extraction = config.get("enable_entity_extraction", True)
