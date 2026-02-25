"""
RIF评分器
根据companion-memory框架文档实现RIF（Recency, Relevance, Frequency）评分系统
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from iris_memory.core.types import MemoryType, DecayRate
from iris_memory.models.memory import Memory
from iris_memory.utils.logger import get_logger

# 延迟导入，避免循环依赖
logger = get_logger("rif_scorer")


class RIFScorer:
    """RIF评分器
    
    基于科学遗忘曲线，使用Recency（时近性）、Relevance（相关性）、Frequency（频率）
    三个维度评估记忆价值，实现选择性遗忘机制。
    """
    
    def __init__(self):
        """初始化RIF评分器"""
        # RIF权重配置
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
        
        # 统计信息
        self.stats = {
            'calculations': 0
        }
        
        logger.debug("RIF Scorer initialized")
    
    def calculate_rif(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算记忆的RIF评分
        
        RIF = 0.4×时近性 + 0.3×相关性 + 0.3×频率
        
        Args:
            memory: 记忆对象
            context: 上下文信息（保留用于接口一致性）
            
        Returns:
            float: RIF评分（0-1）
        """
        try:
            self.stats['calculations'] += 1
            
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
            
            logger.debug(f"Traditional RIF calculated for memory {memory.id[:8]}: "
                        f"score={rif_score:.3f} (R:{recency:.2f}, R:{relevance:.2f}, F:{frequency:.2f})")
            
            return rif_score
            
        except Exception as e:
            logger.error(f"RIF calculation failed for memory {memory.id}: {e}")
            # 返回默认中等评分
            memory.rif_score = 0.5
            return 0.5
    
    def _calculate_recency(self, memory: Memory) -> float:
        """计算时近性得分（40%权重）
        
        改进的时近性计算：
        - 新创建的记忆获得较高的初始时近性
        - 根据记忆类型调整衰减速度
        - 考虑记忆的情感权重（高情感记忆衰减更慢）
        - 用户主动请求的记忆衰减更慢
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 时近性得分（0-1）
        """
        # 获取基础衰减率
        decay_rate = self._get_decay_rate(memory)
        
        # 根据记忆特征调整衰减率
        # 高情感权重记忆衰减更慢
        if memory.emotional_weight > 0.7:
            decay_rate *= 0.7  # 减慢 30%
        elif memory.emotional_weight > 0.4:
            decay_rate *= 0.85  # 减慢 15%
        
        # 用户主动请求的记忆衰减更慢
        if memory.is_user_requested:
            decay_rate *= 0.6  # 减慢 40%
        
        # 高重要性记忆衰减更慢
        if memory.importance_score > 0.8:
            decay_rate *= 0.75
        elif memory.importance_score > 0.5:
            decay_rate *= 0.9
        
        # 计算时间差（小时，更精确）
        time_delta_hours = (datetime.now() - memory.last_access_time).total_seconds() / 3600
        time_delta_days = time_delta_hours / 24
        
        # 指数衰减模型（基于小时更平滑）
        exponential_decay = math.exp(-decay_rate * time_delta_days)
        
        # 应用时间权重（更细粒度）
        time_weight = self._get_detailed_time_weight(memory.last_access_time)
        
        # 新创建记忆保护期（24小时内获得额外加成）
        creation_hours = (datetime.now() - memory.created_time).total_seconds() / 3600
        new_memory_bonus = 1.0
        if creation_hours < 24:
            new_memory_bonus = 1.0 + (24 - creation_hours) / 48  # 最大 +0.5
        
        # 综合时近性得分
        recency_score = exponential_decay * time_weight * new_memory_bonus
        
        # 归一化到0-1
        return min(1.0, max(0.0, recency_score))
    
    def _calculate_relevance(self, memory: Memory) -> float:
        """计算相关性得分（30%权重）
        
        改进的相关性计算：
        - 考虑记忆类型的重要性
        - 多次访问的记忆相关性得分提升
        - 一致性分数高的记忆相关性更高
        - 用户请求的记性相关性最高
        - 情感类记忆有额外加成
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 相关性得分（0-1）
        """
        # 基础相关性：基于一致性分数
        base_relevance = memory.consistency_score
        
        # 记忆类型权重
        type_weights = {
            'fact': 0.9,        # 事实类很重要
            'emotion': 0.95,    # 情感类非常重要
            'relationship': 0.95, # 关系类非常重要
            'interaction': 0.7,  # 互动类一般
            'inferred': 0.6     # 推断类较低
        }
        type_weight = type_weights.get(memory.type.value, 0.7)
        
        # 访问频率加成（非线性增长，最多提升25%）
        access_bonus = min(0.25, math.log1p(memory.access_count) * 0.1)
        
        # 用户请求加成（+35%）
        user_request_bonus = 0.35 if memory.is_user_requested else 0.0
        
        # 高置信度加成
        confidence_bonus = 0.0
        if memory.confidence >= 0.9:
            confidence_bonus = 0.15
        elif memory.confidence >= 0.75:
            confidence_bonus = 0.08
        elif memory.confidence >= 0.5:
            confidence_bonus = 0.03
        
        # 高情感强度加成
        emotion_bonus = 0.0
        if memory.emotional_weight > 0.8:
            emotion_bonus = 0.12
        elif memory.emotional_weight > 0.5:
            emotion_bonus = 0.06
        
        # 综合相关性得分
        relevance_score = (
            base_relevance * 0.3 +           # 一致性占 30%
            type_weight * 0.25 +             # 类型权重占 25%
            access_bonus +                   # 访问频率加成
            user_request_bonus +             # 用户请求加成
            confidence_bonus +               # 置信度加成
            emotion_bonus                    # 情感加成
        )
        
        # 归一化到0-1
        return min(1.0, max(0.0, relevance_score))
    
    def _calculate_frequency(self, memory: Memory) -> float:
        """计算频率得分（30%权重）
        
        改进的频率计算：
        - 基于访问频率和访问模式
        - 近期频繁访问比历史访问更有价值
        - 重要性高的记忆频率得分更高
        - 考虑记忆的新鲜度
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 频率得分（0-1）
        """
        # 基础频率得分：基于访问频率（使用 log1p 平滑）
        base_frequency = min(1.0, math.log1p(memory.access_count) / math.log1p(50))
        
        # 近期访问权重（最近7天的访问更有价值）
        days_since_last_access = (datetime.now() - memory.last_access_time).days
        recency_multiplier = 1.0
        if days_since_last_access <= 1:
            recency_multiplier = 1.3  # 24小时内访问过
        elif days_since_last_access <= 7:
            recency_multiplier = 1.15  # 一周内访问过
        elif days_since_last_access <= 30:
            recency_multiplier = 1.0
        else:
            recency_multiplier = 0.8  # 超过一个月
        
        # 重要性加成
        importance_bonus = memory.importance_score * 0.2
        
        # 用户请求加成
        user_request_bonus = 0.15 if memory.is_user_requested else 0.0
        
        # 高质量记忆加成
        quality_bonus = 0.0
        if memory.quality_level.value >= 4:  # HIGH_CONFIDENCE 或 CONFIRMED
            quality_bonus = 0.1
        elif memory.quality_level.value >= 3:  # MODERATE
            quality_bonus = 0.05
        
        # 情感权重加成
        emotion_bonus = min(0.1, memory.emotional_weight * 0.1)
        
        # 时间衰减因子（基于创建时间）
        age_days = (datetime.now() - memory.created_time).days
        if age_days <= 7:
            age_factor = 1.0
        elif age_days <= 30:
            age_factor = 0.95
        elif age_days <= 90:
            age_factor = 0.9
        else:
            age_factor = max(0.7, 1.0 - (age_days - 90) / 1000)
        
        # 综合频率得分
        frequency_score = (
            base_frequency * 0.5 +           # 基础频率占 50%
            importance_bonus +
            user_request_bonus +
            quality_bonus +
            emotion_bonus
        ) * recency_multiplier * age_factor
        
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
    
    def _get_detailed_time_weight(self, access_time: datetime) -> float:
        """获取详细的时间权重
        
        提供更细粒度的时间权重：
        - 1小时内：1.5（非常新鲜）
        - 24小时内：1.3（新鲜）
        - 7天内：1.2（新记忆）
        - 7-30天：1.0（中期）
        - 30-90天：0.8（旧记忆）
        - 90-365天：0.6（远期）
        - >365天：0.4（陈旧）
        
        Args:
            access_time: 上次访问时间
            
        Returns:
            float: 时间权重
        """
        hours = (datetime.now() - access_time).total_seconds() / 3600
        days = hours / 24
        
        if hours < 1:
            return 1.5  # 1小时内
        elif hours < 24:
            return 1.3  # 24小时内
        elif days < 7:
            return 1.2  # 7天内
        elif days < 30:
            return 1.0  # 7-30天
        elif days < 90:
            return 0.8  # 30-90天
        elif days < 365:
            return 0.6  # 90-365天
        else:
            return 0.4  # 超过一年
