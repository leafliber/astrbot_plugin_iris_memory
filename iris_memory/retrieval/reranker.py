"""
结果重排序器
对检索结果进行重排序
"""

from typing import List, Optional
from datetime import datetime, timedelta

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, QualityLevel


class Reranker:
    """结果重排序器
    
    对检索结果进行重排序：
    - 质量等级优先：CONFIRMED > HIGH_CONFIDENCE > MODERATE
    - RIF评分：高RIF得分优先
    - 时间衰减：新记忆优先
    - 情感一致性：与当前情感一致的记忆优先
    - 访问频率：高频访问的记忆优先
    """
    
    def __init__(self):
        """初始化重排序器"""
        # 质量等级权重
        self.quality_weights = {
            QualityLevel.CONFIRMED: 1.5,
            QualityLevel.HIGH_CONFIDENCE: 1.3,
            QualityLevel.MODERATE: 1.0,
            QualityLevel.LOW_CONFIDENCE: 0.7,
            QualityLevel.UNCERTAIN: 0.4
        }
    
    def rerank(
        self,
        memories: List[Memory],
        query: Optional[str] = None,
        context: Optional[dict] = None
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
        context: Optional[dict]
    ) -> float:
        """计算重排序得分
        
        综合因素：
        - 质量等级：0.35权重
        - RIF评分：0.25权重
        - 时间得分：0.2权重
        - 访问频率：0.15权重
        - 情感一致性：0.05权重
        
        Args:
            memory: 记忆对象
            query: 查询文本（可选）
            context: 上下文信息（可选）
            
        Returns:
            float: 综合得分
        """
        # 1. 质量等级得分
        quality_score = self.quality_weights.get(memory.quality_level, 1.0)
        
        # 2. RIF评分
        rif_score = memory.rif_score
        
        # 3. 时间得分
        time_score = self._calculate_time_score(memory)
        
        # 4. 访问频率得分
        access_score = min(1.0, memory.access_count / 10.0)
        
        # 5. 情感一致性得分
        emotion_score = self._calculate_emotion_score(memory, context)
        
        # 综合得分
        comprehensive_score = (
            0.35 * quality_score +
            0.25 * rif_score +
            0.20 * time_score +
            0.15 * access_score +
            0.05 * emotion_score
        )
        
        return comprehensive_score
    
    def _calculate_time_score(self, memory: Memory) -> float:
        """计算时间得分
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 时间得分（0-1）
        """
        days = (datetime.now() - memory.last_access_time).days
        
        # 时间衰减函数
        if days < 1:
            return 1.0
        elif days < 7:
            return 0.9
        elif days < 30:
            return 0.7
        elif days < 90:
            return 0.5
        else:
            return 0.3
    
    def _calculate_emotion_score(
        self,
        memory: Memory,
        context: Optional[dict]
    ) -> float:
        """计算情感一致性得分
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 情感一致性得分（0-1）
        """
        if not context or 'emotional_state' not in context:
            return 0.5  # 默认中等得分
        
        emotional_state = context['emotional_state']
        
        # 如果记忆不是情感类型，返回中等得分
        if memory.type != "emotion":
            return 0.5
        
        # 检查情感是否一致
        if hasattr(emotional_state, 'current'):
            current_emotion = emotional_state.current.primary.value
            memory_emotion = memory.subtype
            
            if current_emotion == memory_emotion:
                return 1.0  # 完全一致
            elif self._is_similar_emotion(current_emotion, memory_emotion):
                return 0.7  # 相似情感
            else:
                return 0.3  # 不一致的情感
        
        return 0.5
    
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
