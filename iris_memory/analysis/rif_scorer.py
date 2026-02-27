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
        
        统一使用 Memory.calculate_time_score() 作为基础时间得分，
        再根据记忆特征（情感权重、用户请求、重要性）进行调整，
        确保与 Reranker 等其他模块的时间算法一致。
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 时近性得分（0-1）
        """
        # 统一的基础时间得分（基于 last_access_time）
        base_time_score = memory.calculate_time_score(use_created_time=False)
        
        # 根据记忆特征调整（乘性修正）
        modifier = 1.0
        
        # 高情感权重记忆衰减更慢
        if memory.emotional_weight > 0.7:
            modifier *= 1.15  # 提升 15%
        elif memory.emotional_weight > 0.4:
            modifier *= 1.08  # 提升 8%
        
        # 用户主动请求的记忆衰减更慢
        if memory.is_user_requested:
            modifier *= 1.2  # 提升 20%
        
        # 高重要性记忆衰减更慢
        if memory.importance_score > 0.8:
            modifier *= 1.12
        elif memory.importance_score > 0.5:
            modifier *= 1.05
        
        # 新创建记忆保护期（24小时内获得额外加成）
        creation_hours = (datetime.now() - memory.created_time).total_seconds() / 3600
        if creation_hours < 24:
            new_memory_bonus = 1.0 + (24 - creation_hours) / 120  # 最大 +0.2
            modifier *= new_memory_bonus
        
        # 限制修正因子上限，防止累积过高（理论最大约 1.85 → 限制到 1.5）
        modifier = min(modifier, 1.5)
        
        recency_score = base_time_score * modifier
        
        # 归一化到0-1
        return min(1.0, max(0.0, recency_score))
    
    def _calculate_relevance(self, memory: Memory) -> float:
        """计算相关性得分（30%权重）
        
        改进的相关性计算（归一化权重体系）：
        - 基础相关性（一致性 + 类型）占 55%
        - 访问频率加成占 15%
        - 用户请求加成占 15%
        - 置信度加成占 10%
        - 情感加成占 5%
        
        所有分量在加权前已归一化到0-1，保证最终结果∈[0,1]。
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 相关性得分（0-1）
        """
        # 基础相关性：基于一致性分数 (0-1)
        base_relevance = memory.consistency_score
        
        # 记忆类型权重 (0-1)
        type_weights = {
            'fact': 0.9,
            'emotion': 0.95,
            'relationship': 0.95,
            'interaction': 0.7,
            'inferred': 0.6
        }
        type_weight = type_weights.get(memory.type.value, 0.7)
        
        # 访问频率得分 (0-1, 使用 log 平滑)
        access_score = min(1.0, math.log1p(memory.access_count) / math.log1p(20))
        
        # 用户请求得分 (0 或 1)
        user_request_score = 1.0 if memory.is_user_requested else 0.0
        
        # 置信度得分 (0-1, 直接使用)
        confidence_score = memory.confidence
        
        # 情感强度得分 (0-1)
        emotion_score = min(1.0, memory.emotional_weight)
        
        # 归一化加权求和（各权重之和 = 1.0）
        relevance_score = (
            base_relevance * 0.25 +      # 一致性占 25%
            type_weight * 0.30 +          # 类型权重占 30%
            access_score * 0.15 +         # 访问频率占 15%
            user_request_score * 0.15 +   # 用户请求占 15%
            confidence_score * 0.10 +     # 置信度占 10%
            emotion_score * 0.05          # 情感占 5%
        )
        
        # 保险归一化（理论上不应超过1.0）
        return min(1.0, max(0.0, relevance_score))
    
    def _calculate_frequency(self, memory: Memory) -> float:
        """计算频率得分（30%权重）
        
        归一化权重体系：
        - 基础频率（访问次数）占 40%
        - 重要性占 20%
        - 用户请求占 15%
        - 质量等级占 10%
        - 情感权重占 5%
        - 近期访问作为乘性修正
        - 时间衰减作为乘性修正
        
        所有分量在加权前已归一化到0-1，保证最终结果∈[0,1]。
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: 频率得分（0-1）
        """
        # 基础频率得分 (0-1)
        base_frequency = min(1.0, math.log1p(memory.access_count) / math.log1p(50))
        
        # 重要性得分 (0-1)
        importance_score = memory.importance_score
        
        # 用户请求得分 (0 或 1)
        user_request_score = 1.0 if memory.is_user_requested else 0.0
        
        # 质量得分 (0-1, 归一化枚举值)
        quality_score = min(1.0, memory.quality_level.value / 5.0)
        
        # 情感权重得分 (0-1)
        emotion_score = min(1.0, memory.emotional_weight)
        
        # 归一化加权求和（权重之和 = 0.90，留0.10给乘性修正因子的影响空间）
        weighted_sum = (
            base_frequency * 0.40 +
            importance_score * 0.20 +
            user_request_score * 0.15 +
            quality_score * 0.10 +
            emotion_score * 0.05
        )
        
        # 近期访问乘性修正 (0.8-1.15)
        days_since_last_access = (datetime.now() - memory.last_access_time).days
        if days_since_last_access <= 1:
            recency_multiplier = 1.15
        elif days_since_last_access <= 7:
            recency_multiplier = 1.08
        elif days_since_last_access <= 30:
            recency_multiplier = 1.0
        else:
            recency_multiplier = 0.85
        
        # 时间衰减乘性修正 (0.7-1.0)
        age_days = (datetime.now() - memory.created_time).days
        if age_days <= 7:
            age_factor = 1.0
        elif age_days <= 30:
            age_factor = 0.95
        elif age_days <= 90:
            age_factor = 0.9
        else:
            age_factor = max(0.7, 1.0 - (age_days - 90) / 1000)
        
        frequency_score = weighted_sum * recency_multiplier * age_factor
        
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
