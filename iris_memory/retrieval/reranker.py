"""
结果重排序器
对检索结果进行重排序
"""

from typing import List, Optional
from datetime import datetime, timedelta

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, QualityLevel, EmotionType, RerankContext
from iris_memory.core.constants import NEGATIVE_EMOTIONS_CORE


class Reranker:
    """结果重排序器

    对检索结果进行重排序：
    - 质量等级优先：CONFIRMED > HIGH_CONFIDENCE > MODERATE
    - RIF评分：高RIF得分优先（内含时近性40%+相关性30%+频率30%）
    - 时间衰减：新记忆优先（仅作为RIF之外的微调补充）
    - 情感一致性：与当前情感一致的记忆优先
    - 访问频率：高频访问的记忆优先
    - 向量相似度：Chroma向量检索的补充权重
    - 发送者匹配：与当前对话者相关的记忆加权
    - 活跃度权重：活跃成员的记忆优先

    合并后的权重分配（注意：RIF已包含时近性，time_score仅做微调）：
    - 质量等级：0.25
    - RIF评分：0.25
    - 时间衰减：0.05（微调，避免与RIF内时近性双重加权）
    - 向量相似度：0.15
    - 发送者匹配：0.10
    - 活跃度权重：0.05
    - 访问频率：0.10
    - 情感一致性：0.05
    """

    def __init__(self, enable_vector_score: bool = True):
        """初始化重排序器

        Args:
            enable_vector_score: 是否启用向量相似度权重（默认True）
                如果记忆对象中没有vector_similarity字段，则自动忽略
        """
        # 质量等级权重
        self.quality_weights = {
            QualityLevel.CONFIRMED: 1.5,
            QualityLevel.HIGH_CONFIDENCE: 1.3,
            QualityLevel.MODERATE: 1.0,
            QualityLevel.LOW_CONFIDENCE: 0.7,
            QualityLevel.UNCERTAIN: 0.4
        }

        # 配置
        self.enable_vector_score = enable_vector_score
    
    def rerank(
        self,
        memories: List[Memory],
        query: Optional[str] = None,
        context: Optional[RerankContext] = None
    ) -> List[Memory]:
        """重排序记忆列表
        
        Args:
            memories: 记忆列表
            query: 查询文本（可选）
            context: 上下文信息（可选）
            
        Returns:
            List[Memory]: 重排序后的记忆列表
        """
        if not memories:
            return []
        
        # 为每个记忆计算综合得分
        scored_memories = []
        for memory in memories:
            score = self._calculate_rerank_score(memory, query, context)
            scored_memories.append((memory, score))
        
        # 按得分降序排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的记忆列表
        return [memory for memory, score in scored_memories]
    
    def _calculate_rerank_score(
        self,
        memory: Memory,
        query: Optional[str],
        context: Optional[RerankContext]
    ) -> float:
        """计算重排序得分

        合并后的权重分配（RIF已包含时近性，time_score仅做微调）：
        - 质量等级：0.25
        - RIF评分：0.25
        - 时间衰减：0.05（微调补充）
        - 向量相似度：0.15
        - 发送者匹配：0.10
        - 活跃度权重：0.05
        - 访问频率：0.10
        - 情感一致性：0.05

        Args:
            memory: 记忆对象
            query: 查询文本（可选，暂未使用）
            context: 上下文信息（可选，包含emotional_state, current_user_id）

        Returns:
            float: 综合得分
        """
        # 1. 质量等级得分
        quality_score = self.quality_weights.get(memory.quality_level, 1.0)

        # 2. RIF评分
        rif_score = memory.rif_score

        # 3. 时间衰减得分
        time_score = self._calculate_time_score(memory)

        # 4. 访问频率得分
        access_score = min(1.0, memory.access_count / 10.0)

        # 5. 情感一致性得分
        emotion_score = self._calculate_emotion_score(memory, context)

        # 6. 向量相似度得分（如果有）
        vector_score = 0.0
        if self.enable_vector_score and hasattr(memory, 'vector_similarity'):
            vector_score = memory.vector_similarity
        elif self.enable_vector_score and hasattr(memory, 'similarity'):
            vector_score = memory.similarity

        # 7. 发送者匹配得分
        sender_score = self._calculate_sender_score(memory, context)

        # 8. 活跃度得分
        activity_score = self._calculate_activity_score(memory, context)

        # 综合得分（注意：RIF已包含时近性40%，time_score仅作微调补充）
        comprehensive_score = (
            0.25 * quality_score +
            0.25 * rif_score +
            0.05 * time_score +
            0.10 * sender_score +
            0.05 * activity_score +
            0.10 * access_score +
            0.05 * emotion_score
        )

        # 如果有向量相似度，加入计算
        if vector_score > 0:
            comprehensive_score += 0.15 * vector_score

        return comprehensive_score

    def _calculate_sender_score(
        self,
        memory: Memory,
        context: Optional[RerankContext]
    ) -> float:
        """计算发送者匹配得分

        当记忆的 user_id 与当前对话者的 user_id 匹配时给予高分，
        使AI更容易引用与当前对话者相关的记忆。

        Args:
            memory: 记忆对象
            context: 上下文信息（包含 current_user_id）

        Returns:
            float: 发送者匹配得分（0-1）
        """
        if not context:
            return 0.5

        current_user_id = context.get('current_user_id')
        if not current_user_id:
            return 0.5

        if memory.user_id == current_user_id:
            return 1.0

        return 0.3

    def _calculate_activity_score(
        self,
        memory: Memory,
        context: Optional[dict]
    ) -> float:
        """计算记忆来源成员的活跃度得分

        活跃成员的记忆更可能被正确引用，不活跃成员的
        记忆可能已过时。

        Args:
            memory: 记忆对象
            context: 上下文信息（包含 member_identity_service）

        Returns:
            float: 活跃度得分（0-1）
        """
        if not context:
            return 0.5

        identity_service = context.get('member_identity_service')
        if not identity_service:
            return 0.5

        return identity_service.get_activity_score(memory.user_id)
    
    def _calculate_time_score(self, memory: Memory) -> float:
        """计算时间得分

        委托给 Memory.calculate_time_score()，基于最后访问时间排序。

        Args:
            memory: 记忆对象

        Returns:
            float: 时间得分（0-1）
        """
        return memory.calculate_time_score(use_created_time=False)
    
    def _calculate_emotion_score(
        self,
        memory: Memory,
        context: Optional[RerankContext]
    ) -> float:
        """计算情感一致性得分

        合并了两种实现的逻辑：
        1. 负面情感时避免高强度正面记忆
        2. 情感类型三级评分（完全一致/相似/不一致）

        Args:
            memory: 记忆对象
            context: 上下文信息（包含emotional_state）

        Returns:
            float: 情感一致性得分（0-1）
        """
        if not context or 'emotional_state' not in context:
            return 0.5  # 默认中等得分

        emotional_state = context['emotional_state']

        # 如果记忆不是情感类型，返回中等得分
        if memory.type != "emotion":
            return 0.5

        # 获取当前情感和记忆情感
        if not hasattr(emotional_state, 'current'):
            return 0.5

        current_emotion = emotional_state.current.primary
        memory_emotion = memory.subtype

        # 特殊规则：如果当前情感是负面，避免高强度正面记忆
        if current_emotion in NEGATIVE_EMOTIONS_CORE:
            if memory.emotional_weight > 0.8:
                if memory_emotion in ["joy", "excitement", "calm", "contentment", "amusement"]:
                    return 0.0  # 负面情感时，高强度正面记忆相关性为0

        # 三级评分系统
        current_emotion_str = current_emotion.value if hasattr(current_emotion, 'value') else str(current_emotion)

        if current_emotion_str == memory_emotion:
            return 1.0  # 完全一致
        elif self._is_similar_emotion(current_emotion_str, memory_emotion):
            return 0.7  # 相似情感
        else:
            return 0.3  # 不一致的情感
    
    def _is_similar_emotion(self, emotion1: str, emotion2: str) -> bool:
        """判断两个情感是否相似
        
        Args:
            emotion1: 情感1
            emotion2: 情感2
            
        Returns:
            bool: 是否相似
        """
        # 定义相似情感组
        similar_groups = [
            {"joy", "excitement", "calm"},  # 正面情感
            {"sadness", "fear", "anxiety"},  # 负面低能量情感
            {"anger", "disgust"},  # 负面高能量情感
        ]
        
        for group in similar_groups:
            if emotion1 in group and emotion2 in group:
                return True
        
        return False
    
    def filter_by_quality(
        self,
        memories: List[Memory],
        min_quality: QualityLevel = QualityLevel.MODERATE
    ) -> List[Memory]:
        """按质量等级过滤
        
        Args:
            memories: 记忆列表
            min_quality: 最小质量等级
            
        Returns:
            List[Memory]: 过滤后的记忆列表
        """
        return [
            m for m in memories
            if m.quality_level.value >= min_quality.value
        ]
    
    def filter_by_storage_layer(
        self,
        memories: List[Memory],
        storage_layer: StorageLayer
    ) -> List[Memory]:
        """按存储层过滤
        
        Args:
            memories: 记忆列表
            storage_layer: 存储层
            
        Returns:
            List[Memory]: 过滤后的记忆列表
        """
        return [m for m in memories if m.storage_layer == storage_layer]
    
    def group_by_type(self, memories: List[Memory]) -> dict:
        """按类型分组
        
        Args:
            memories: 记忆列表
            
        Returns:
            dict: 分组后的记忆字典
            {
                "fact": List[Memory],
                "emotion": List[Memory],
                ...
            }
        """
        grouped = {}
        for memory in memories:
            memory_type = memory.type.value
            if memory_type not in grouped:
                grouped[memory_type] = []
            grouped[memory_type].append(memory)
        
        return grouped
    
    def deduplicate(self, memories: List[Memory], similarity_threshold: float = 0.9) -> List[Memory]:
        """去重
        
        Args:
            memories: 记忆列表
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Memory]: 去重后的记忆列表
        """
        seen = []
        unique = []
        
        for memory in memories:
            is_duplicate = False
            for seen_memory in seen:
                if self._calculate_similarity(memory.content, seen_memory.content) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.append(memory)
                unique.append(memory)
        
        return unique
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版Jaccard）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度（0-1）
        """
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
