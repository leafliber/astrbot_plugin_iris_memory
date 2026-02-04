"""
RIF评分器
根据companion-memory框架文档实现RIF（Recency, Relevance, Frequency）评分系统
现已扩展支持多维度评分系统
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
    
    现已支持多维度评分系统，可通过配置切换到更精细的五维度评分。
    """
    
    def __init__(self, use_multidimensional: bool = False, **multidimensional_kwargs):
        """初始化RIF评分器
        
        Args:
            use_multidimensional: 是否使用多维度评分系统（默认False，保持向后兼容）
            **multidimensional_kwargs: 传递给MultidimensionalScorer的参数
        """
        self.use_multidimensional = use_multidimensional
        self._multidimensional_scorer = None
        self._multidimensional_kwargs = multidimensional_kwargs
        
        # 传统RIF权重配置
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
            'traditional_calculations': 0,
            'multidimensional_calculations': 0,
            'fallbacks': 0
        }
        
        if use_multidimensional:
            logger.info("RIF Scorer initialized with multidimensional scoring enabled")
        else:
            logger.info("RIF Scorer initialized with traditional RIF scoring")
    
    def calculate_rif(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算记忆的RIF评分
        
        根据配置选择传统RIF评分或多维度评分
        
        Args:
            memory: 记忆对象
            context: 上下文信息（用于多维度评分）
            
        Returns:
            float: RIF评分（0-1）
        """
        if self.use_multidimensional:
            return self._calculate_multidimensional_rif(memory, context)
        else:
            return self._calculate_traditional_rif(memory)
    
    def _calculate_multidimensional_rif(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """使用多维度评分系统计算RIF分数
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 多维度评分结果（0-1）
        """
        try:
            # 延迟导入和初始化MultidimensionalScorer
            if self._multidimensional_scorer is None:
                from iris_memory.analysis.multidimensional_scorer import MultidimensionalScorer
                self._multidimensional_scorer = MultidimensionalScorer(
                    fallback_to_rif=False,  # 避免循环回退
                    **self._multidimensional_kwargs
                )
                logger.debug("MultidimensionalScorer initialized")
            
            # 计算多维度得分
            result = self._multidimensional_scorer.calculate_score(memory, context)
            self.stats['multidimensional_calculations'] += 1
            
            # 更新记忆对象的RIF评分
            memory.rif_score = result.final_score
            
            logger.debug(f"Multidimensional RIF calculated for memory {memory.id[:8]}: "
                        f"score={result.final_score:.3f}, scenario={result.scenario_type.value}")
            
            return result.final_score
            
        except Exception as e:
            logger.warning(f"Multidimensional scoring failed for memory {memory.id}, "
                          f"falling back to traditional RIF: {e}")
            self.stats['fallbacks'] += 1
            return self._calculate_traditional_rif(memory)
    
    def _calculate_traditional_rif(self, memory: Memory) -> float:
        """计算传统的RIF评分
        
        RIF = 0.4×时近性 + 0.3×相关性 + 0.3×频率
        
        Args:
            memory: 记忆对象
            
        Returns:
            float: RIF评分（0-1）
        """
        try:
            self.stats['traditional_calculations'] += 1
            
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
            logger.error(f"Traditional RIF calculation failed for memory {memory.id}: {e}")
            # 返回默认中等评分
            memory.rif_score = 0.5
            return 0.5
    
    def enable_multidimensional_mode(self, **kwargs):
        """启用多维度评分模式
        
        Args:
            **kwargs: 传递给MultidimensionalScorer的参数
        """
        self.use_multidimensional = True
        self._multidimensional_kwargs.update(kwargs)
        self._multidimensional_scorer = None  # 重置，下次使用时重新初始化
        logger.info("Multidimensional scoring mode enabled")
    
    def disable_multidimensional_mode(self):
        """禁用多维度评分模式，回退到传统RIF"""
        self.use_multidimensional = False
        self._multidimensional_scorer = None
        logger.info("Multidimensional scoring mode disabled, using traditional RIF")
    
    def get_scoring_mode(self) -> str:
        """获取当前评分模式"""
        return "multidimensional" if self.use_multidimensional else "traditional"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取评分统计信息"""
        base_stats = {
            'scoring_mode': self.get_scoring_mode(),
            'traditional_calculations': self.stats['traditional_calculations'],
            'multidimensional_calculations': self.stats['multidimensional_calculations'],
            'fallbacks': self.stats['fallbacks'],
            'total_calculations': (
                self.stats['traditional_calculations'] + 
                self.stats['multidimensional_calculations']
            )
        }
        
        # 如果有多维度评分器，添加其统计信息
        if self._multidimensional_scorer:
            base_stats['multidimensional_stats'] = self._multidimensional_scorer.stats
        
        return base_stats
    
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
    
    def _get_time_weight(self, access_time: datetime) -> float:
        """根据时间差获取时间权重（基础版本）
        
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
    
    def _get_detailed_time_weight(self, access_time: datetime) -> float:
        """获取更详细的时间权重（改进版本）
        
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
