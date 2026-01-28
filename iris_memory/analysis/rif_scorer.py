"""
RIF评分器
根据companion-memory框架文档实现RIF（Recency, Relevance, Frequency）评分系统
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional

from iris_memory.core.types import MemoryType, DecayRate
from iris_memory.models.memory import Memory


class RIFScorer:
    """RIF评分器
    
    基于科学遗忘曲线，使用Recency（时近性）、Relevance（相关性）、Frequency（频率）
    三个维度评估记忆价值，实现选择性遗忘机制。
    """
    
    def __init__(self):
        """初始化RIF评分器"""
        # 权重配置（根据框架文档）
        self.recency_weight = 0.4  # 时近性权重
        self.relevance_weight = 0.3  # 相关性权重
        self.frequency_weight = 0.3  # 频率权重
        
        # 时近性计算配置
        self.time_weights = {
            'new': 1.2,      # 7天内
            'medium': 1.0,    # 7-30天
            'old': 0.8,      # 30-90天
            'very_old': 0.6  # >90天
        }
    
    def calculate_rif(self, memory: Memory) -> float:
        """计算记忆的RIF评分
        
        RIF = 0.4×时近性 + 0.3×相关性 + 0.3×频率
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: RIF评分（0-1）
        """
        # 计算各个维度
        recency = self._calculate_recency(memory)
        relevance = self._calculate_relevance(memory)
        frequency = self._calculate_frequency(memory)
        
        # 加权综合评分
        rif_score = (
            self.recency_weight * recency +
            self.relevance_weight * relevance +
            self.frequency_weight * frequency
        )
        
        # 更新记忆的RIF评分
        memory.rif_score = rif_score
        
        return rif_score
    
    def _calculate_recency(self, memory: Memory) -> float:
        """计算时近性得分（40%权重）
        
        使用指数衰减模型：
        - 根据记忆类型获取衰减率
        - 计算距离上次访问的时间
        - 应用指数衰减公式：exp(-λ * t)
        - 叠加时间权重
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 时近性得分（0-1）
        """
        # 获取基础衰减率
        decay_rate = self._get_decay_rate(memory)
        
        # 计算时间差（天数）
        time_delta = (datetime.now() - memory.last_access_time).days
        
        # 指数衰减模型
        exponential_decay = math.exp(-decay_rate * time_delta)
        
        # 应用时间权重
        time_weight = self._get_time_weight(memory.last_access_time)
        
        # 综合时近性得分
        recency_score = exponential_decay * time_weight
        
        # 归一化到0-1
        return min(1.0, max(0.0, recency_score))
    
    def _calculate_relevance(self, memory: Memory) -> float:
        """计算相关性得分（30%权重）
        
        基于访问频率和一致性分数：
        - 多次访问的记忆相关性得分提升
        - 一致性分数高的记忆相关性更高
        - 用户请求的记性相关性最高
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 相关性得分（0-1）
        """
        # 基础相关性：基于一致性分数
        base_relevance = memory.consistency_score
        
        # 访问频率加成（最多提升20%）
        access_bonus = min(0.2, memory.access_count * 0.02)
        
        # 用户请求加成（+30%）
        user_request_bonus = 0.3 if memory.is_user_requested else 0.0
        
        # 综合相关性得分
        relevance_score = base_relevance + access_bonus + user_request_bonus
        
        # 归一化到0-1
        return min(1.0, max(0.0, relevance_score))
    
    def _calculate_frequency(self, memory: Memory) -> float:
        """计算频率得分（30%权重）
        
        基于访问频率和重要性：
        - 高频访问降低衰减
        - 重要性高的记性频率得分更高
        - 访问加成：每次访问降低衰减常数（最多50%）
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 频率得分（0-1）
        """
        # 基础频率得分：基于访问频率
        base_frequency = min(1.0, memory.access_frequency * 10)
        
        # 重要性加成
        importance_bonus = memory.importance_score * 0.3
        
        # 访问加成：降低衰减常数（已在personalized_decay中体现）
        decay_reduction = min(0.5, memory.access_count * 0.05)
        adjusted_decay = max(0.0, memory.base_decay * (1.0 - decay_reduction))
        
        # 计算衰减修正因子
        decay_factor = math.exp(-adjusted_decay * (datetime.now() - memory.created_time).days)
        
        # 综合频率得分
        frequency_score = (base_frequency + importance_bonus) * decay_factor
        
        # 归一化到0-1
        return min(1.0, max(0.0, frequency_score))
    
    def _get_decay_rate(self, memory: Memory) -> float:
        """获取记忆类型的衰减率
        
        根据companion-memory框架：
        - 兴趣：30天半衰期
        - 习惯：90天半衰期
        - 人格：365天半衰期
        - 价值观：730天半衰期
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 衰减率（λ值）
        """
        return DecayRate.get_decay_rate(memory.type)
    
    def _get_time_weight(self, access_time: datetime) -> float:
        """根据时间差获取时间权重
        
        根据framework文档：
        - 新记忆（7天内）：时间权重×1.2
        - 中期记忆（7-30天）：时间权重×1.0
        - 旧记忆（30-90天）：时间权重×0.8
        - 远期记忆（>90天）：时间权重×0.6
        
        Args:
            access_time: 上次访问时间
            
        Returns:
            float: 时间权重
        """
        days = (datetime.now() - access_time).days
        
        if days < 7:
            return self.time_weights['new']
        elif days < 30:
            return self.time_weights['medium']
        elif days < 90:
            return self.time_weights['old']
        else:
            return self.time_weights['very_old']
    
    def calculate_batch_rif(self, memories: list[Memory]) -> Dict[str, float]:
        """批量计算RIF评分
        
        Args:
            memories: 记忆对象列表
            
        Returns:
            Dict[str, float]: 记忆ID到RIF评分的映射
        """
        rif_scores = {}
        for memory in memories:
            rif_scores[memory.id] = self.calculate_rif(memory)
        
        return rif_scores
    
    def should_delete(self, memory: Memory, threshold: float = 0.4) -> bool:
        """判断记忆是否应该删除
        
        根据framework文档的删除门槛：
        - CRITICAL敏感度：RIF < 0.2
        - 普通记忆：RIF < 0.4
        
        Args:
            memory: 记忆对象
            threshold: 删除阈值（默认0.4）
            
        Returns:
            bool: 是否应该删除
        """
        # CRITICAL敏感度的记忆使用更严格的阈值
        if memory.sensitivity_level.value >= 4:
            return memory.rif_score < 0.2
        else:
            return memory.rif_score < threshold
    
    def get_low_value_memories(
        self,
        memories: list[Memory],
        threshold: float = 0.4
    ) -> list[Memory]:
        """获取低价值记忆列表（用于归档或删除）
        
        Args:
            memories: 记忆对象列表
            threshold: RIF评分阈值
            
        Returns:
            list[Memory]: RIF评分低于阈值的记忆列表
        """
        return [m for m in memories if self.should_delete(m, threshold)]
    
    def update_personalized_decay(self, memory: Memory):
        """更新个性化衰减常数
        
        根据访问次数降低衰减常数（最多降低50%）
        每次访问使记忆更稳固，衰减更慢
        
        Args:
            memory: 记忆对象
        """
        # 计算访问加成：最多降低50%
        decay_reduction = min(0.5, memory.access_count * 0.05)
        
        # 更新个性化衰减常数
        memory.personalized_decay = memory.base_decay * (1.0 - decay_reduction)
    
    def get_recency_ranking(self, memories: list[Memory]) -> list[tuple[str, float]]:
        """获取记忆的时近性排名
        
        Args:
            memories: 记忆对象列表
            
        Returns:
            list[tuple[str, float]]: (记忆ID, 时近性得分)的排序列表
        """
        recency_scores = [(m.id, self._calculate_recency(m)) for m in memories]
        return sorted(recency_scores, key=lambda x: x[1], reverse=True)
    
    def get_relevance_ranking(self, memories: list[Memory]) -> list[tuple[str, float]]:
        """获取记忆的相关性排名
        
        Args:
            memories: 记忆对象列表
            
        Returns:
            list[tuple[str, float]]: (记忆ID, 相关性得分)的排序列表
        """
        relevance_scores = [(m.id, self._calculate_relevance(m)) for m in memories]
        return sorted(relevance_scores, key=lambda x: x[1], reverse=True)
    
    def get_frequency_ranking(self, memories: list[Memory]) -> list[tuple[str, float]]:
        """获取记忆的频率排名
        
        Args:
            memories: 记忆对象列表
            
        Returns:
            list[tuple[str, float]]: (记忆ID, 频率得分)的排序列表
        """
        frequency_scores = [(m.id, self._calculate_frequency(m)) for m in memories]
        return sorted(frequency_scores, key=lambda x: x[1], reverse=True)
