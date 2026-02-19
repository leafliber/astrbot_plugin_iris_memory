"""
记忆捕获引擎
根据companion-memory框架实现智能记忆捕获
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.models.memory import Memory
from iris_memory.core.types import (
    MemoryType, ModalityType, QualityLevel, SensitivityLevel,
    StorageLayer, VerificationMethod, TriggerType, TriggerMatch
)
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.entity.entity_extractor import EntityExtractor
from iris_memory.capture.detector.trigger_detector import TriggerDetector
from iris_memory.capture.detector.sensitivity_detector import SensitivityDetector
from iris_memory.capture.conflict.similarity_calculator import sanitize_for_log
from iris_memory.capture.conflict.conflict_resolver import ConflictResolver
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.core.defaults import DEFAULTS
from iris_memory.capture.capture_logger import capture_log
from iris_memory.utils.logger import get_logger

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
    
    支持LLM增强：
    - LLM敏感度检测（语义层面）
    - LLM触发器检测（意图理解）
    - LLM冲突解决（语义判断）
    """
    
    def __init__(
        self,
        chroma_manager=None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        rif_scorer: Optional[RIFScorer] = None,
        llm_sensitivity_detector=None,
        llm_trigger_detector=None,
        llm_conflict_resolver=None,
    ):
        """初始化记忆捕获引擎
        
        Args:
            chroma_manager: Chroma存储管理器（用于去重和冲突检测）
            emotion_analyzer: 情感分析器（可选）
            rif_scorer: RIF评分器（可选）
            llm_sensitivity_detector: LLM敏感度检测器（可选）
            llm_trigger_detector: LLM触发器检测器（可选）
            llm_conflict_resolver: LLM冲突解决器（可选）
        """
        self.chroma_manager = chroma_manager
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.rif_scorer = rif_scorer or RIFScorer()
        self.trigger_detector = TriggerDetector()
        self.sensitivity_detector = SensitivityDetector()
        self.entity_extractor = EntityExtractor()
        self.conflict_resolver = ConflictResolver()
        
        # LLM增强组件
        self._llm_sensitivity_detector = llm_sensitivity_detector
        self._llm_trigger_detector = llm_trigger_detector
        self._llm_conflict_resolver = llm_conflict_resolver
        
        # 配置
        self.auto_capture = True
        self.min_confidence = DEFAULTS.memory.min_confidence
        self.rif_threshold = DEFAULTS.memory.rif_threshold
        self.enable_duplicate_check = DEFAULTS.memory.enable_duplicate_check
        self.enable_conflict_check = DEFAULTS.memory.enable_conflict_check
        self.enable_entity_extraction = DEFAULTS.memory.enable_entity_extraction
    
    async def capture_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        is_user_requested: bool = False,
        sender_name: Optional[str] = None
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
        11. 去重检查 - 避免重复记录
        12. 冲突检测 - 识别语义冲突

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
            capture_log.capture_start(user_id, message, is_user_requested)
            
            if self.trigger_detector._is_negative_sample(message):
                capture_log.negative_sample(user_id)
                return None
            
            triggers = await self._detect_triggers(message)
            capture_log.trigger_detected(user_id, triggers)
            if not triggers and not is_user_requested and not self.auto_capture:
                capture_log.capture_skip(user_id, "no_trigger_auto_disabled")
                return None
            
            sensitivity_level, detected_entities = await self._detect_sensitivity(message, context)
            capture_log.sensitivity_detected(
                user_id, 
                sensitivity_level.name if hasattr(sensitivity_level, 'name') else str(sensitivity_level),
                detected_entities
            )
            
            if self.enable_entity_extraction:
                extracted_entities = self.entity_extractor.extract_entities(message)
                entity_texts = [e.text for e in extracted_entities]
                detected_entities = list(set(detected_entities + entity_texts))
            
            if self.sensitivity_detector.should_filter(sensitivity_level):
                capture_log.sensitivity_filtered(
                    user_id, 
                    sensitivity_level.name if hasattr(sensitivity_level, 'name') else str(sensitivity_level)
                )
                return None
            
            emotion_result = await self.emotion_analyzer.analyze_emotion(message, context)
            capture_log.emotion_analyzed(
                user_id,
                emotion_result["primary"].value if hasattr(emotion_result["primary"], 'value') else str(emotion_result["primary"]),
                emotion_result["intensity"],
                emotion_result["confidence"]
            )
            
            memory_type = self._determine_memory_type(triggers, emotion_result)
            
            from iris_memory.core.memory_scope import MemoryScope
            memory_scope = self._determine_memory_scope(message, group_id, triggers)
            requires_encryption = self.sensitivity_detector.get_encryption_required(sensitivity_level)
            
            memory = Memory(
                user_id=user_id,
                sender_name=sender_name,
                group_id=group_id,
                scope=memory_scope,
                type=memory_type,
                modality=ModalityType.TEXT,
                content=message,
                sensitivity_level=sensitivity_level,
                detected_entities=detected_entities,
                is_user_requested=is_user_requested
            )
            
            if requires_encryption:
                memory.metadata["requires_encryption"] = True
            
            capture_log.memory_created(
                user_id,
                memory_type.value if hasattr(memory_type, 'value') else str(memory_type),
                memory_scope.value if hasattr(memory_scope, 'value') else str(memory_scope)
            )
            
            memory.emotional_weight = emotion_result["intensity"]
            if memory_type == MemoryType.EMOTION:
                memory.subtype = emotion_result["primary"].value if hasattr(emotion_result["primary"], "value") else str(emotion_result["primary"])
            
            memory.summary = self._generate_summary(message, triggers)
            
            self._assess_quality(memory, triggers, emotion_result)
            capture_log.quality_assessed(
                user_id,
                memory.confidence,
                memory.quality_level.value if hasattr(memory.quality_level, 'value') else str(memory.quality_level)
            )
            
            self.rif_scorer.calculate_rif(memory)
            capture_log.rif_calculated(user_id, memory.rif_score)
            
            self._determine_storage_layer(memory, is_user_requested)
            capture_log.storage_determined(
                user_id,
                memory.storage_layer.value if hasattr(memory.storage_layer, 'value') else str(memory.storage_layer),
                "user_requested" if is_user_requested else "auto"
            )
            
            if self.chroma_manager and (self.enable_duplicate_check or self.enable_conflict_check):
                similar_memories = await self.chroma_manager.query_memories(
                    query_text=memory.content,
                    user_id=user_id,
                    group_id=group_id,
                    top_k=10
                )
                
                if self.enable_duplicate_check and similar_memories:
                    duplicate = self.conflict_resolver.find_duplicate_from_results(memory, similar_memories)
                    if duplicate:
                        capture_log.duplicate_found(user_id, memory.id)
                        return None
                
                if self.enable_conflict_check and similar_memories:
                    conflicts = self.conflict_resolver.find_conflicts_from_results(memory, similar_memories)
                    if conflicts:
                        resolved = await self.conflict_resolver.resolve_conflicts(
                            memory, conflicts, self.chroma_manager
                        )
                        capture_log.conflict_detected(user_id, len(conflicts), resolved)
            
            capture_log.capture_ok(
                user_id,
                memory.id,
                memory_type.value if hasattr(memory_type, 'value') else str(memory_type),
                memory.confidence,
                memory.rif_score,
                memory.storage_layer.value if hasattr(memory.storage_layer, 'value') else str(memory.storage_layer)
            )
            
            return memory
            
        except (ValueError, TypeError) as e:
            # 数据格式/类型错误：可恢复，仅记录
            capture_log.capture_error(user_id, e)
            return None
        except Exception as e:
            # 未预期异常：记录完整堆栈以便排查
            logger.error(f"Unexpected capture error for user={user_id}", exc_info=True)
            capture_log.capture_error(user_id, e)
            return None
    
    def _determine_memory_scope(
        self,
        message: str,
        group_id: Optional[str],
        triggers: List[TriggerMatch]
    ) -> 'MemoryScope':
        """确定记忆的可见性范围
        
        在群聊场景中区分群组共享知识和个人知识：
        - 群组共享：涉及群活动、群规则、群通知、大家共同参与的话题
        - 个人知识：涉及个人偏好、个人经历、自我描述
        
        Args:
            message: 消息内容
            group_id: 群组ID（私聊为None）
            triggers: 触发器列表
            
        Returns:
            MemoryScope: 记忆可见性范围
        """
        from iris_memory.core.memory_scope import MemoryScope
        
        if not group_id:
            return MemoryScope.USER_PRIVATE
        
        # 群组共享知识的关键词模式
        group_shared_patterns = [
            # 群活动和通知
            r'大家|各位|群里|群内|群友|所有人|通知|公告',
            r'我们(一起|都|聊|约|组织|参加|去)',
            r'群规|群主|管理员',
            # 共同话题 / 事件
            r'(咱们?|我们)(群|这边|这里)',
            r'群名|群头像|群文件',
            # 集体活动
            r'(一起|约|集合|组队)(玩|吃|去|做|打|聊)',
        ]
        
        # 个人知识的关键词模式
        personal_patterns = [
            r'^我(是|在|有|喜欢|讨厌|觉得|认为|想|要|不)',
            r'我(自己|个人|一个人)',
            r'我的(名字|工作|家|手机|电脑|生日|爱好|习惯)',
        ]
        
        msg_lower = message.lower()
        
        # 检查是否匹配群共享模式
        is_group_shared = any(
            re.search(pattern, msg_lower) for pattern in group_shared_patterns
        )
        
        # 检查是否匹配个人模式
        is_personal = any(
            re.search(pattern, msg_lower) for pattern in personal_patterns
        )
        
        # 如果同时匹配，个人模式优先（更保守的隐私策略）
        if is_personal:
            return MemoryScope.GROUP_PRIVATE
        
        if is_group_shared:
            return MemoryScope.GROUP_SHARED
        
        # 默认：群内个人知识（保护隐私）
        return MemoryScope.GROUP_PRIVATE

    def _determine_memory_type(
        self,
        triggers: List[TriggerMatch],
        emotion_result: Dict[str, Any]
    ) -> MemoryType:
        """确定记忆类型
        
        Args:
            triggers: 触发器列表
            emotion_result: 情感分析结果
            
        Returns:
            MemoryType: 记忆类型
        """
        trigger_types = [t.type for t in triggers]
        
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
        triggers: List[TriggerMatch]
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
        triggers: List[TriggerMatch],
        emotion_result: Dict[str, Any]
    ):
        """评估记忆质量
        
        Args:
            memory: 记忆对象
            triggers: 触发器列表
            emotion_result: 情感分析结果
        """
        # 计算置信度（加权平均）
        # 权重分配：触发器 40%，情感分析 30%，上下文一致性 30%
        
        # 1. 触发器置信度（权重 40%）
        if triggers:
            max_trigger_confidence = max(t.confidence for t in triggers)
            # 多触发器加成：多信号一致表示更高置信度
            trigger_bonus = min(0.15, len(triggers) * 0.05)
            trigger_conf = min(1.0, max_trigger_confidence + trigger_bonus)
        else:
            trigger_conf = 0.3  # 无触发器的默认置信度
        
        # 2. 情感分析置信度（权重 30%）
        emotion_conf = emotion_result["confidence"]
        
        # 3. 上下文一致性（权重 30%）
        # 基于多个因素动态计算，而非硬编码
        consistency_score = self._calculate_consistency_score(memory, triggers, emotion_result)
        
        # 综合置信度（加权平均）
        memory.confidence = trigger_conf * 0.4 + emotion_conf * 0.3 + consistency_score * 0.3
        
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
        elif any(t.type == TriggerType.EXPLICIT for t in triggers):
            memory.verification_method = VerificationMethod.USER_EXPLICIT
        elif len(triggers) > 1:
            memory.verification_method = VerificationMethod.MULTIPLE_MENTIONS
        else:
            memory.verification_method = VerificationMethod.SYSTEM_INFERRED
        
        # 6. 计算重要性评分（基于触发器质量和情感强度）
        # 用于 RIF 评分的 Frequency 维度差异化
        if triggers:
            # 触发器置信度作为基础重要性
            max_trigger_conf = max(t.confidence for t in triggers)
            trigger_importance = max_trigger_conf
        else:
            trigger_importance = 0.3  # 无触发器的默认重要性
        
        # 情感强度加成：高情感内容更重要
        emotion_importance = emotion_result["intensity"] * 0.3
        
        # 综合重要性评分（用户请求的记忆已在 _determine_storage_layer 中设置为 0.8+）
        if not memory.is_user_requested:
            memory.importance_score = min(1.0, trigger_importance * 0.7 + emotion_importance)
        
        # 7. 计算一致性评分（基于触发器数量和类型）
        # 用于 RIF 评分的 Relevance 维度差异化
        if triggers:
            # 多触发器表示信息一致性高（多种信号指向同一内容）
            trigger_consistency = min(1.0, len(triggers) * 0.2 + 0.4)
            
            # 显式触发器（如"记住"、"重要"）表示高一致性
            if any(t.type == TriggerType.EXPLICIT for t in triggers):
                trigger_consistency = min(1.0, trigger_consistency + 0.2)
            
            # 偏好/事实触发器表示信息明确
            if any(t.type in [TriggerType.PREFERENCE, TriggerType.FACT] for t in triggers):
                trigger_consistency = min(1.0, trigger_consistency + 0.1)
        else:
            trigger_consistency = 0.3  # 无触发器的默认一致性
        
        memory.consistency_score = trigger_consistency
    
    def _calculate_consistency_score(
        self,
        memory: Memory,
        triggers: List[TriggerMatch],
        emotion_result: Dict[str, Any]
    ) -> float:
        """计算上下文一致性评分
        
        基于多个信号的一致性来评估置信度，而非固定值：
        - 触发器类型多样性
        - 情感强度与记忆类型匹配度
        - 内容信息密度（实体数量）
        
        Args:
            memory: 记忆对象
            triggers: 触发器列表
            emotion_result: 情感分析结果
            
        Returns:
            float: 一致性评分 (0-1)
        """
        score = 0.5  # 基础分
        
        # 1. 触发器类型多样性加成
        if triggers:
            trigger_types = set(t.type for t in triggers)
            # 多种类型触发器表示信息更可靠
            diversity_bonus = min(0.2, len(trigger_types) * 0.08)
            score += diversity_bonus
            
            # 显式触发器（如"记住"、"重要"）加成
            if any(t.type == TriggerType.EXPLICIT for t in triggers):
                score += 0.15
        
        # 2. 情感强度与记忆类型匹配
        # 高情感强度 + EMOTION 类型 = 一致性高
        # 低情感强度 + FACT 类型 = 一致性高
        emotion_intensity = emotion_result.get("intensity", 0.5)
        if memory.type == MemoryType.EMOTION and emotion_intensity > 0.6:
            score += 0.1
        elif memory.type == MemoryType.FACT and emotion_intensity < 0.4:
            score += 0.08
        
        # 3. 内容信息密度（检测到的实体数量）
        entity_count = len(memory.detected_entities) if memory.detected_entities else 0
        if entity_count > 0:
            # 有实体信息表示内容更明确
            entity_bonus = min(0.12, entity_count * 0.04)
            score += entity_bonus
        
        # 4. 敏感度与重要性匹配
        # 高敏感度信息通常更重要
        if memory.sensitivity_level.value >= 2:  # PRIVATE 或更高
            score += 0.05
        
        return min(1.0, score)
    
    def _determine_storage_layer(self, memory: Memory, is_user_requested: bool):
        """确定初始存储层
        
        优化后的存储层判断逻辑，降低 EPISODIC 存储门槛：
        - 用户请求 -> 直接 EPISODIC
        - 高质量 (HIGH_CONFIDENCE/CONFIRMED) -> EPISODIC 或 SEMANTIC
        - 中高置信度 (>=0.5) -> EPISODIC
        - 普通置信度 -> WORKING
        
        Args:
            memory: 记忆对象
            is_user_requested: 是否用户显式请求
        """
        if is_user_requested:
            memory.storage_layer = StorageLayer.EPISODIC
            memory.importance_score = max(memory.importance_score, 0.8)
            return
        
        if memory.quality_level == QualityLevel.CONFIRMED:
            memory.storage_layer = StorageLayer.SEMANTIC
            return
        
        if memory.quality_level == QualityLevel.HIGH_CONFIDENCE:
            memory.storage_layer = StorageLayer.EPISODIC
            return
        
        if memory.emotional_weight > 0.6:
            memory.storage_layer = StorageLayer.EPISODIC
            return
        
        if memory.confidence >= 0.5:
            memory.storage_layer = StorageLayer.EPISODIC
            return
        
        if memory.confidence >= self.min_confidence:
            memory.storage_layer = StorageLayer.WORKING
            return
        
        memory.storage_layer = StorageLayer.WORKING
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置
        
        Args:
            config: 配置字典
        """
        self.auto_capture = config.get("auto_capture", True)
        self.min_confidence = config.get("min_confidence", DEFAULTS.memory.min_confidence)
        self.rif_threshold = config.get("rif_threshold", DEFAULTS.memory.rif_threshold)
        self.enable_duplicate_check = config.get("enable_duplicate_check", DEFAULTS.memory.enable_duplicate_check)
        self.enable_conflict_check = config.get("enable_conflict_check", DEFAULTS.memory.enable_conflict_check)
        self.enable_entity_extraction = config.get("enable_entity_extraction", DEFAULTS.memory.enable_entity_extraction)
    
    async def _detect_triggers(self, message: str) -> List[TriggerMatch]:
        """检测触发器（支持LLM增强）
        
        Args:
            message: 消息文本
            
        Returns:
            触发器列表
        """
        if self._llm_trigger_detector:
            try:
                result = await self._llm_trigger_detector.detect(message)
                if result.should_remember and result.trigger_type:
                    return [TriggerMatch(
                        type=result.trigger_type,
                        pattern="llm_detected",
                        confidence=result.confidence,
                        position=0,
                    )]
                elif not result.should_remember and result.confidence >= 0.7:
                    return []
            except Exception as e:
                capture_log.llm_trigger_detection_failed("", str(e))
        
        return self.trigger_detector.detect_triggers(message)
    
    async def _detect_sensitivity(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """检测敏感度（支持LLM增强）
        
        Args:
            message: 消息文本
            context: 上下文
            
        Returns:
            (敏感度等级, 检测到的实体列表)
        """
        if self._llm_sensitivity_detector:
            try:
                result = await self._llm_sensitivity_detector.detect(message, context)
                return (result.level, result.entities)
            except Exception as e:
                capture_log.llm_sensitivity_detection_failed("", str(e))
        
        return self.sensitivity_detector.detect_sensitivity(message, context)
