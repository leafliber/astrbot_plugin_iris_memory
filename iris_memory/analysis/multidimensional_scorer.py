"""
多维度记忆评分系统
替代传统RIF评分，提供更精细的记忆价值评估

基于认知科学和记忆研究，将记忆价值分解为五个独立维度：
- Temporal: 时间维度 - 考虑时近性、持久性、时间模式
- Semantic: 语义维度 - 考虑内容质量、一致性、关联性
- Social: 社交维度 - 考虑人际关系、社交重要性、影响力
- Emotional: 情感维度 - 考虑情感强度、情感类型、情感持续性
- Quality: 质量维度 - 考虑可信度、验证状态、准确性

每个维度独立计算，然后根据上下文场景动态聚合。
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from iris_memory.models.memory import Memory
from iris_memory.core.types import MemoryType, QualityLevel, SensitivityLevel, DecayRate
from iris_memory.utils.logger import get_logger

logger = get_logger("multidimensional_scorer")


class ScenarioType(str, Enum):
    """场景类型 - 用于动态调整权重"""
    EMOTIONAL_DIALOGUE = "emotional_dialogue"      # 情感对话
    FACTUAL_QUERY = "factual_query"                # 事实查询
    SOCIAL_INTERACTION = "social_interaction"      # 社交互动
    PERSONAL_REFLECTION = "personal_reflection"    # 个人反思
    CREATIVE_CONTEXT = "creative_context"          # 创意场景
    ROUTINE_CHAT = "routine_chat"                  # 日常闲聊
    EMERGENCY = "emergency"                        # 紧急情况
    DEFAULT = "default"                            # 默认平衡


@dataclass
class DimensionWeights:
    """维度权重配置"""
    temporal: float = 0.20     # 时间维度权重
    semantic: float = 0.20     # 语义维度权重
    social: float = 0.20       # 社交维度权重
    emotional: float = 0.20    # 情感维度权重
    quality: float = 0.20      # 质量维度权重
    
    def normalize(self) -> 'DimensionWeights':
        """归一化权重，确保总和为1.0"""
        total = self.temporal + self.semantic + self.social + self.emotional + self.quality
        if total == 0:
            return DimensionWeights()
        
        return DimensionWeights(
            temporal=self.temporal / total,
            semantic=self.semantic / total,
            social=self.social / total,
            emotional=self.emotional / total,
            quality=self.quality / total
        )


@dataclass
class MultidimensionalScore:
    """多维度评分结果"""
    temporal_score: float = 0.0      # 时间维度得分 [0,1]
    semantic_score: float = 0.0      # 语义维度得分 [0,1]
    social_score: float = 0.0        # 社交维度得分 [0,1]
    emotional_score: float = 0.0     # 情感维度得分 [0,1]
    quality_score: float = 0.0       # 质量维度得分 [0,1]
    
    # 聚合得分
    weighted_score: float = 0.0      # 加权聚合得分 [0,1]
    final_score: float = 0.0         # 最终评分（可包含调整因子）[0,1]
    
    # 元数据
    scenario_type: ScenarioType = ScenarioType.DEFAULT
    weights_used: DimensionWeights = field(default_factory=DimensionWeights)
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'temporal_score': self.temporal_score,
            'semantic_score': self.semantic_score,
            'social_score': self.social_score,
            'emotional_score': self.emotional_score,
            'quality_score': self.quality_score,
            'weighted_score': self.weighted_score,
            'final_score': self.final_score,
            'scenario_type': self.scenario_type.value,
            'weights_used': {
                'temporal': self.weights_used.temporal,
                'semantic': self.weights_used.semantic,
                'social': self.weights_used.social,
                'emotional': self.weights_used.emotional,
                'quality': self.weights_used.quality
            },
            'metadata': self.calculation_metadata
        }


class MultidimensionalScorer:
    """多维度记忆评分系统
    
    提供更精细和科学的记忆价值评估，替代传统的RIF三维评分。
    每个维度独立计算，支持根据不同场景动态调整权重。
    """
    
    def __init__(
        self,
        enable_advanced_features: bool = True,
        enable_context_adaptation: bool = True,
        fallback_to_rif: bool = False
    ):
        """初始化多维度评分器
        
        Args:
            enable_advanced_features: 是否启用高级特性（如非线性聚合）
            enable_context_adaptation: 是否启用上下文自适应权重调整
            fallback_to_rif: 在计算失败时是否回退到RIF评分
        """
        self.enable_advanced_features = enable_advanced_features
        self.enable_context_adaptation = enable_context_adaptation
        self.fallback_to_rif = fallback_to_rif
        
        # 场景权重预设
        self.scenario_weights = self._initialize_scenario_weights()
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'by_scenario': {scenario.value: 0 for scenario in ScenarioType},
            'dimension_distributions': {
                'temporal': [],
                'semantic': [],
                'social': [],
                'emotional': [],
                'quality': []
            }
        }
        
        logger.info(f"Multidimensional scorer initialized (advanced={enable_advanced_features}, "
                   f"context_adaptive={enable_context_adaptation})")
    
    def calculate_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> MultidimensionalScore:
        """计算记忆的多维度评分
        
        Args:
            memory: 记忆对象
            context: 上下文信息，包含场景类型、当前状态等
            
        Returns:
            MultidimensionalScore: 多维度评分结果
        """
        self.stats['total_calculations'] += 1
        
        try:
            # 1. 确定场景类型
            scenario_type = self._determine_scenario(context)
            self.stats['by_scenario'][scenario_type.value] += 1
            
            # 2. 计算各维度得分
            temporal_score = self._calculate_temporal_score(memory, context)
            semantic_score = self._calculate_semantic_score(memory, context)
            social_score = self._calculate_social_score(memory, context)
            emotional_score = self._calculate_emotional_score(memory, context)
            quality_score = self._calculate_quality_score(memory, context)
            
            # 3. 获取场景权重
            weights = self._get_scenario_weights(scenario_type, context)
            
            # 4. 计算加权得分
            weighted_score = self._calculate_weighted_score({
                'temporal': temporal_score,
                'semantic': semantic_score,
                'social': social_score,
                'emotional': emotional_score,
                'quality': quality_score
            }, weights)
            
            # 5. 应用高级调整（如果启用）
            final_score = self._apply_advanced_adjustments(
                weighted_score, memory, context, scenario_type
            ) if self.enable_advanced_features else weighted_score
            
            # 6. 收集统计信息
            self._update_statistics(temporal_score, semantic_score, social_score,
                                  emotional_score, quality_score)
            
            # 7. 构建结果
            result = MultidimensionalScore(
                temporal_score=temporal_score,
                semantic_score=semantic_score,
                social_score=social_score,
                emotional_score=emotional_score,
                quality_score=quality_score,
                weighted_score=weighted_score,
                final_score=final_score,
                scenario_type=scenario_type,
                weights_used=weights,
                calculation_metadata={
                    'calculation_time': datetime.now().isoformat(),
                    'memory_age_hours': (datetime.now() - memory.created_time).total_seconds() / 3600,
                    'access_pattern': 'frequent' if memory.access_count > 10 else 'normal',
                    'context_signals': self._extract_context_signals(context)
                }
            )
            
            # 8. 更新记忆对象的评分（保持RIF兼容性）
            memory.rif_score = final_score
            
            logger.debug(f"Calculated multidimensional score for memory {memory.id[:8]}: "
                        f"final={final_score:.3f} (T:{temporal_score:.2f}, S:{semantic_score:.2f}, "
                        f"So:{social_score:.2f}, E:{emotional_score:.2f}, Q:{quality_score:.2f}) "
                        f"scenario={scenario_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating multidimensional score for memory {memory.id}: {e}")
            
            # 回退策略
            if self.fallback_to_rif:
                from iris_memory.analysis.rif_scorer import RIFScorer
                rif_scorer = RIFScorer()
                fallback_score = rif_scorer.calculate_rif(memory)
                logger.warning(f"Using RIF fallback score: {fallback_score:.3f}")
                return self._create_fallback_result(fallback_score)
            else:
                # 返回默认中等评分
                return self._create_fallback_result(0.5)
    
    def _determine_scenario(self, context: Optional[Dict[str, Any]]) -> ScenarioType:
        """确定当前场景类型
        
        分析上下文信息，智能判断当前处于什么类型的交互场景，
        以便选择合适的维度权重配置。
        """
        if not context:
            return ScenarioType.DEFAULT
        
        # 检查明确的场景指示
        if 'scenario_type' in context:
            try:
                return ScenarioType(context['scenario_type'])
            except ValueError:
                pass
        
        # 基于情感状态判断
        emotional_state = context.get('emotional_state', {})
        if emotional_state:
            intensity = emotional_state.get('intensity', 0.0)
            emotion_type = emotional_state.get('type', 'neutral')
            
            if intensity > 0.7:
                return ScenarioType.EMOTIONAL_DIALOGUE
            elif emotion_type in ['anger', 'fear', 'sadness'] and intensity > 0.5:
                return ScenarioType.EMERGENCY
        
        # 基于查询类型判断
        query_type = context.get('query_type', '')
        if query_type in ['fact_lookup', 'information_retrieval', 'verification']:
            return ScenarioType.FACTUAL_QUERY
        elif query_type in ['social_advice', 'relationship_help']:
            return ScenarioType.SOCIAL_INTERACTION
        
        # 基于消息特征判断
        message_text = context.get('current_message', '').lower()
        
        # 情感对话关键词
        emotional_keywords = ['感觉', '情绪', '心情', '难过', '开心', '愤怒', '焦虑', '压力']
        if any(keyword in message_text for keyword in emotional_keywords):
            return ScenarioType.EMOTIONAL_DIALOGUE
        
        # 社交关系关键词
        social_keywords = ['朋友', '家人', '同事', '关系', '社交', '相处']
        if any(keyword in message_text for keyword in social_keywords):
            return ScenarioType.SOCIAL_INTERACTION
        
        # 事实查询关键词
        fact_keywords = ['什么是', '如何', '为什么', '哪里', '何时', '谁']
        if any(keyword in message_text for keyword in fact_keywords):
            return ScenarioType.FACTUAL_QUERY
        
        # 个人反思关键词
        reflection_keywords = ['我觉得', '我认为', '我的想法', '反思', '思考']
        if any(keyword in message_text for keyword in reflection_keywords):
            return ScenarioType.PERSONAL_REFLECTION
        
        # 创意场景关键词
        creative_keywords = ['创作', '想象', '设计', '创意', '灵感']
        if any(keyword in message_text for keyword in creative_keywords):
            return ScenarioType.CREATIVE_CONTEXT
        
        return ScenarioType.DEFAULT
    
    def _calculate_temporal_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算时间维度得分
        
        时间维度考虑：
        1. 时近性 (Recency) - 最近访问时间的重要性
        2. 持久性 (Durability) - 记忆保持价值的时间跨度
        3. 时间模式 (Temporal Pattern) - 访问的时间规律
        4. 衰减抵抗 (Decay Resistance) - 对时间衰减的抵抗能力
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 时间维度得分 [0,1]
        """
        try:
            # 1. 基础时近性计算
            hours_since_access = (datetime.now() - memory.last_access_time).total_seconds() / 3600
            days_since_access = hours_since_access / 24
            
            # 使用改进的双指数衰减模型
            # 短期衰减（0-24小时）：快速下降
            # 长期衰减（>24小时）：缓慢下降
            if hours_since_access < 24:
                short_term_factor = math.exp(-hours_since_access / 12)  # 12小时半衰期
                recency_score = 0.7 * short_term_factor + 0.3  # 基础分0.3
            else:
                long_term_factor = math.exp(-days_since_access / 30)  # 30天半衰期
                recency_score = 0.7 * long_term_factor + 0.1  # 基础分0.1
            
            # 2. 持久性评估
            age_days = (datetime.now() - memory.created_time).days
            
            # 年龄-价值曲线：新记忆和老记忆都有价值，中期记忆价值较低
            if age_days < 7:
                age_value = 1.0  # 新记忆高价值
            elif age_days < 30:
                age_value = 0.6 + 0.4 * math.exp(-(age_days - 7) / 10)  # 逐渐衰减
            elif age_days < 180:
                age_value = 0.4  # 中期记忆稳定低价值
            else:
                # 长期记忆价值回升（历史价值）
                age_value = 0.4 + 0.4 * min(1.0, (age_days - 180) / 365)
            
            # 3. 时间模式分析
            pattern_score = self._analyze_temporal_pattern(memory)
            
            # 4. 衰减抵抗能力
            decay_resistance = self._calculate_decay_resistance(memory)
            
            # 综合时间得分
            temporal_score = (
                0.4 * recency_score +      # 时近性权重40%
                0.25 * age_value +         # 持久性权重25%
                0.2 * pattern_score +      # 时间模式权重20%
                0.15 * decay_resistance    # 衰减抵抗权重15%
            )
            
            return max(0.0, min(1.0, temporal_score))
            
        except Exception as e:
            logger.error(f"Error calculating temporal score: {e}")
            return 0.5
    
    def _calculate_semantic_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算语义维度得分
        
        语义维度考虑：
        1. 内容质量 (Content Quality) - 信息的丰富度和准确性
        2. 一致性 (Consistency) - 与已有记忆的一致性
        3. 关联性 (Relatedness) - 与其他记忆的关联强度
        4. 语义完整性 (Semantic Completeness) - 信息的完整度
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 语义维度得分 [0,1]
        """
        try:
            # 1. 内容质量评估
            content_quality = self._assess_content_quality(memory)
            
            # 2. 一致性得分（直接使用memory的consistency_score）
            consistency_score = memory.consistency_score
            
            # 3. 关联性计算
            relatedness_score = self._calculate_semantic_relatedness(memory)
            
            # 4. 语义完整性
            completeness_score = self._assess_semantic_completeness(memory)
            
            # 综合语义得分
            semantic_score = (
                0.3 * content_quality +     # 内容质量权重30%
                0.25 * consistency_score +  # 一致性权重25%
                0.25 * relatedness_score +  # 关联性权重25%
                0.2 * completeness_score    # 完整性权重20%
            )
            
            return max(0.0, min(1.0, semantic_score))
            
        except Exception as e:
            logger.error(f"Error calculating semantic score: {e}")
            return 0.5
    
    def _calculate_social_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算社交维度得分
        
        社交维度考虑：
        1. 人际重要性 (Interpersonal Importance) - 涉及的人际关系重要程度
        2. 社交影响力 (Social Influence) - 对社交互动的影响
        3. 关系深度 (Relationship Depth) - 所涉及关系的深度
        4. 群体相关性 (Group Relevance) - 与群体/社区的相关性
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 社交维度得分 [0,1]
        """
        try:
            # 1. 记忆类型的社交权重
            social_type_weights = {
                MemoryType.RELATIONSHIP: 1.0,    # 关系类记忆最高
                MemoryType.EMOTION: 0.7,         # 情感类中等（可能涉及人际）
                MemoryType.INTERACTION: 0.8,     # 互动类较高
                MemoryType.FACT: 0.3,            # 事实类较低（除非涉及人）
                MemoryType.INFERRED: 0.4         # 推断类较低
            }
            type_weight = social_type_weights.get(memory.type, 0.5)
            
            # 2. 人际重要性评估
            interpersonal_importance = self._assess_interpersonal_importance(memory)
            
            # 3. 社交影响力
            social_influence = self._calculate_social_influence(memory, context)
            
            # 4. 关系深度
            relationship_depth = self._assess_relationship_depth(memory)
            
            # 5. 群体相关性
            group_relevance = self._assess_group_relevance(memory, context)
            
            # 综合社交得分
            social_score = (
                0.2 * type_weight +               # 类型权重20%
                0.3 * interpersonal_importance +  # 人际重要性30%
                0.2 * social_influence +          # 社交影响力20%
                0.15 * relationship_depth +       # 关系深度15%
                0.15 * group_relevance            # 群体相关性15%
            )
            
            return max(0.0, min(1.0, social_score))
            
        except Exception as e:
            logger.error(f"Error calculating social score: {e}")
            return 0.5
    
    def _calculate_emotional_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算情感维度得分
        
        情感维度考虑：
        1. 情感强度 (Emotional Intensity) - 情感的强烈程度
        2. 情感类型 (Emotional Type) - 情感类型的重要性
        3. 情感持续性 (Emotional Durability) - 情感影响的持续时间
        4. 情感共鸣 (Emotional Resonance) - 与当前情感状态的共鸣
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 情感维度得分 [0,1]
        """
        try:
            # 1. 基础情感强度（直接使用memory的emotional_weight）
            emotional_intensity = memory.emotional_weight
            
            # 2. 情感类型重要性
            emotion_type_importance = self._assess_emotion_type_importance(memory)
            
            # 3. 情感持续性评估
            emotional_durability = self._calculate_emotional_durability(memory)
            
            # 4. 与当前情感状态的共鸣
            emotional_resonance = self._calculate_emotional_resonance(memory, context)
            
            # 5. 记忆类型的情感加成
            type_emotional_bonus = 1.0 if memory.type == MemoryType.EMOTION else 0.5
            
            # 综合情感得分
            emotional_score = (
                0.35 * emotional_intensity +      # 情感强度权重35%
                0.2 * emotion_type_importance +   # 情感类型权重20%
                0.2 * emotional_durability +      # 情感持续性权重20%
                0.15 * emotional_resonance +      # 情感共鸣权重15%
                0.1 * type_emotional_bonus        # 类型加成权重10%
            )
            
            return max(0.0, min(1.0, emotional_score))
            
        except Exception as e:
            logger.error(f"Error calculating emotional score: {e}")
            return 0.5
    
    def _calculate_quality_score(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """计算质量维度得分
        
        质量维度考虑：
        1. 可信度 (Credibility) - 信息的可信程度
        2. 验证状态 (Verification Status) - 验证程度
        3. 准确性 (Accuracy) - 信息的准确性
        4. 来源质量 (Source Quality) - 信息来源的质量
        
        Args:
            memory: 记忆对象
            context: 上下文信息
            
        Returns:
            float: 质量维度得分 [0,1]
        """
        try:
            # 1. 置信度评分（直接使用memory的confidence）
            credibility_score = memory.confidence
            
            # 2. 质量等级评分
            quality_level_scores = {
                QualityLevel.CONFIRMED: 1.0,
                QualityLevel.HIGH_CONFIDENCE: 0.8,
                QualityLevel.MODERATE: 0.6,
                QualityLevel.LOW_CONFIDENCE: 0.4,
                QualityLevel.UNCERTAIN: 0.2
            }
            quality_score = quality_level_scores.get(memory.quality_level, 0.5)
            
            # 3. 验证状态评分
            verification_scores = {
                'user_explicit': 1.0,
                'multiple_mentions': 0.8,
                'cross_validation': 0.7,
                'system_inferred': 0.4,
                'unverified': 0.2
            }
            verification_score = verification_scores.get(
                memory.verification_method.value, 0.3
            )
            
            # 4. 来源质量评分
            source_quality = self._assess_source_quality(memory)
            
            # 5. 一致性影响（与已有记忆的一致性）
            consistency_bonus = memory.consistency_score * 0.2
            
            # 综合质量得分
            final_quality_score = (
                0.3 * credibility_score +    # 可信度权重30%
                0.25 * quality_score +       # 质量等级权重25%
                0.2 * verification_score +   # 验证状态权重20%
                0.15 * source_quality +      # 来源质量权重15%
                0.1 * consistency_bonus      # 一致性加成权重10%
            )
            
            return max(0.0, min(1.0, final_quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _get_scenario_weights(
        self,
        scenario_type: ScenarioType,
        context: Optional[Dict[str, Any]] = None
    ) -> DimensionWeights:
        """获取场景对应的维度权重配置
        
        Args:
            scenario_type: 场景类型
            context: 上下文信息
            
        Returns:
            DimensionWeights: 归一化的维度权重
        """
        return self.scenario_weights.get(scenario_type, DimensionWeights()).normalize()
    
    def _calculate_weighted_score(
        self,
        dimension_scores: Dict[str, float],
        weights: DimensionWeights
    ) -> float:
        """计算加权综合得分
        
        Args:
            dimension_scores: 各维度得分字典
            weights: 维度权重
            
        Returns:
            float: 加权综合得分 [0,1]
        """
        weighted_score = (
            weights.temporal * dimension_scores['temporal'] +
            weights.semantic * dimension_scores['semantic'] +
            weights.social * dimension_scores['social'] +
            weights.emotional * dimension_scores['emotional'] +
            weights.quality * dimension_scores['quality']
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def _apply_advanced_adjustments(
        self,
        base_score: float,
        memory: Memory,
        context: Optional[Dict[str, Any]],
        scenario_type: ScenarioType
    ) -> float:
        """应用高级调整因子
        
        Args:
            base_score: 基础加权得分
            memory: 记忆对象
            context: 上下文信息
            scenario_type: 场景类型
            
        Returns:
            float: 调整后的最终得分 [0,1]
        """
        adjusted_score = base_score
        
        try:
            # 1. 用户明确请求加成
            if memory.is_user_requested:
                adjusted_score *= 1.15  # 15%加成
            
            # 2. 高敏感度记忆保护
            if memory.sensitivity_level.value >= 3:  # SENSITIVE或CRITICAL
                adjusted_score *= 1.1  # 10%加成
            
            # 3. 非线性聚合调整（避免中等分扎堆）
            if 0.4 <= adjusted_score <= 0.6:
                # 对中等分进行非线性压缩，增加区分度
                center = 0.5
                distance = abs(adjusted_score - center)
                adjusted_score = center + distance * 0.8  # 压缩20%
            
            # 4. 场景特定调整
            scenario_multipliers = {
                ScenarioType.EMERGENCY: 1.2,         # 紧急情况优先
                ScenarioType.EMOTIONAL_DIALOGUE: 1.1, # 情感对话重要
                ScenarioType.ROUTINE_CHAT: 0.9        # 日常闲聊降权
            }
            
            multiplier = scenario_multipliers.get(scenario_type, 1.0)
            adjusted_score *= multiplier
            
            # 5. 访问模式调整
            if memory.access_count > 10:  # 高频访问
                adjusted_score *= 1.05
            elif memory.access_count == 0:  # 从未访问
                adjusted_score *= 0.95
            
            return max(0.0, min(1.0, adjusted_score))
            
        except Exception as e:
            logger.error(f"Error applying advanced adjustments: {e}")
            return base_score
    
    def _initialize_scenario_weights(self) -> Dict[ScenarioType, DimensionWeights]:
        """初始化各场景的权重配置"""
        return {
            ScenarioType.EMOTIONAL_DIALOGUE: DimensionWeights(
                temporal=0.15, semantic=0.20, social=0.15, emotional=0.35, quality=0.15
            ),
            ScenarioType.FACTUAL_QUERY: DimensionWeights(
                temporal=0.15, semantic=0.30, social=0.10, emotional=0.10, quality=0.35
            ),
            ScenarioType.SOCIAL_INTERACTION: DimensionWeights(
                temporal=0.10, semantic=0.20, social=0.35, emotional=0.20, quality=0.15
            ),
            ScenarioType.PERSONAL_REFLECTION: DimensionWeights(
                temporal=0.20, semantic=0.25, social=0.10, emotional=0.30, quality=0.15
            ),
            ScenarioType.CREATIVE_CONTEXT: DimensionWeights(
                temporal=0.15, semantic=0.25, social=0.15, emotional=0.25, quality=0.20
            ),
            ScenarioType.ROUTINE_CHAT: DimensionWeights(
                temporal=0.25, semantic=0.20, social=0.20, emotional=0.15, quality=0.20
            ),
            ScenarioType.EMERGENCY: DimensionWeights(
                temporal=0.30, semantic=0.15, social=0.20, emotional=0.20, quality=0.15
            ),
            ScenarioType.DEFAULT: DimensionWeights()  # 均衡权重
        }
    
    # ===== 辅助计算方法 =====
    
    def _analyze_temporal_pattern(self, memory: Memory) -> float:
        """分析时间访问模式
        
        返回时间模式得分，考虑访问的规律性和预测性
        """
        try:
            # 简化实现：基于访问频率和时间分布
            if memory.access_count < 2:
                return 0.5  # 无足够数据
            
            # 访问间隔一致性（假设实现）
            # 这里简化为基于访问次数的启发式
            if memory.access_count >= 5:
                return 0.8  # 高频访问表示规律性
            elif memory.access_count >= 3:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _calculate_decay_resistance(self, memory: Memory) -> float:
        """计算衰减抵抗能力"""
        try:
            resistance = 0.5  # 基础抵抗力
            
            # 用户请求的记忆抵抗力更强
            if memory.is_user_requested:
                resistance += 0.3
            
            # 高重要性记忆抵抗力更强
            resistance += memory.importance_score * 0.2
            
            # 高质量记忆抵抗力更强
            if memory.quality_level.value >= 4:
                resistance += 0.2
            
            # 关键记忆类型抵抗力更强
            if memory.type in [MemoryType.RELATIONSHIP, MemoryType.EMOTION]:
                resistance += 0.1
            
            return max(0.0, min(1.0, resistance))
            
        except Exception:
            return 0.5
    
    def _assess_content_quality(self, memory: Memory) -> float:
        """评估内容质量"""
        try:
            quality = 0.5  # 基础质量
            
            # 内容长度影响（适中最好）
            content_length = len(memory.content) if memory.content else 0
            if 50 <= content_length <= 500:
                quality += 0.2
            elif content_length < 10:
                quality -= 0.2
            
            # 关键词丰富度
            if memory.keywords and len(memory.keywords) > 2:
                quality += 0.1
            
            # 是否有摘要
            if memory.summary:
                quality += 0.1
            
            # 多模态内容
            if memory.has_image or memory.audio:
                quality += 0.1
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
    
    def _calculate_semantic_relatedness(self, memory: Memory) -> float:
        """计算语义关联性"""
        try:
            # 基于关联记忆数量
            related_count = len(memory.related_memories) if memory.related_memories else 0
            
            if related_count == 0:
                return 0.3  # 孤立记忆
            elif related_count <= 3:
                return 0.6  # 少量关联
            elif related_count <= 10:
                return 0.8  # 丰富关联
            else:
                return 1.0  # 核心记忆
                
        except Exception:
            return 0.5
    
    def _assess_semantic_completeness(self, memory: Memory) -> float:
        """评估语义完整性"""
        try:
            completeness = 0.3  # 基础完整性
            
            # 检查必要字段
            if memory.content and len(memory.content.strip()) > 5:
                completeness += 0.3
            
            if memory.type != MemoryType.INFERRED:
                completeness += 0.2
            
            if memory.keywords:
                completeness += 0.1
            
            if memory.summary:
                completeness += 0.1
            
            return max(0.0, min(1.0, completeness))
            
        except Exception:
            return 0.5
    
    def _assess_interpersonal_importance(self, memory: Memory) -> float:
        """评估人际重要性"""
        try:
            importance = 0.3  # 基础重要性
            
            # 关系类记忆优先
            if memory.type == MemoryType.RELATIONSHIP:
                importance += 0.4
            
            # 基于敏感度（人际信息通常敏感）
            if memory.sensitivity_level.value >= 2:  # PERSONAL及以上
                importance += 0.2
            
            # 检查内容中的人际关键词
            content = memory.content.lower() if memory.content else ""
            interpersonal_keywords = ['朋友', '家人', '同事', '伴侣', '父母', '孩子', '兄弟', '姐妹']
            
            if any(keyword in content for keyword in interpersonal_keywords):
                importance += 0.3
            
            return max(0.0, min(1.0, importance))
            
        except Exception:
            return 0.5
    
    def _calculate_social_influence(self, memory: Memory, context: Optional[Dict[str, Any]]) -> float:
        """计算社交影响力"""
        try:
            influence = 0.4  # 基础影响力
            
            # 群聊记忆影响力更大
            if memory.group_id:
                influence += 0.3
            
            # 高访问频率表示社交价值
            if memory.access_count > 5:
                influence += 0.2
            
            # 用户明确请求的记忆社交影响力更大
            if memory.is_user_requested:
                influence += 0.1
            
            return max(0.0, min(1.0, influence))
            
        except Exception:
            return 0.5
    
    def _assess_relationship_depth(self, memory: Memory) -> float:
        """评估关系深度"""
        try:
            depth = 0.3  # 基础深度
            
            # 关系类记忆本身深度较高
            if memory.type == MemoryType.RELATIONSHIP:
                depth += 0.4
            
            # 情感类记忆可能涉及深层关系
            if memory.type == MemoryType.EMOTION and memory.emotional_weight > 0.6:
                depth += 0.2
            
            # 高敏感度暗示深度关系
            if memory.sensitivity_level.value >= 3:
                depth += 0.1
            
            return max(0.0, min(1.0, depth))
            
        except Exception:
            return 0.5
    
    def _assess_group_relevance(self, memory: Memory, context: Optional[Dict[str, Any]]) -> float:
        """评估群体相关性"""
        try:
            relevance = 0.5  # 基础相关性
            
            # 群聊记忆群体相关性高
            if memory.group_id:
                relevance += 0.3
            
            # 当前是否在群聊上下文中
            if context and context.get('group_id') == memory.group_id:
                relevance += 0.2
            
            return max(0.0, min(1.0, relevance))
            
        except Exception:
            return 0.5
    
    def _assess_emotion_type_importance(self, memory: Memory) -> float:
        """评估情感类型重要性"""
        try:
            # 基于情感强度和类型
            if memory.type != MemoryType.EMOTION:
                return memory.emotional_weight * 0.5  # 非情感记忆降权
            
            # 情感记忆的类型重要性
            importance = 0.7  # 基础重要性
            
            # 强情感更重要
            if memory.emotional_weight > 0.8:
                importance += 0.2
            elif memory.emotional_weight < 0.3:
                importance -= 0.2
            
            return max(0.0, min(1.0, importance))
            
        except Exception:
            return 0.5
    
    def _calculate_emotional_durability(self, memory: Memory) -> float:
        """计算情感持续性"""
        try:
            durability = 0.5  # 基础持续性
            
            # 基于记忆年龄和访问模式
            age_days = (datetime.now() - memory.created_time).days
            
            # 如果是老记忆但仍有情感权重，说明持续性强
            if age_days > 30 and memory.emotional_weight > 0.5:
                durability += 0.3
            
            # 多次访问的情感记忆持续性强
            if memory.access_count > 3 and memory.type == MemoryType.EMOTION:
                durability += 0.2
            
            return max(0.0, min(1.0, durability))
            
        except Exception:
            return 0.5
    
    def _calculate_emotional_resonance(
        self,
        memory: Memory,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """计算情感共鸣"""
        try:
            if not context or 'emotional_state' not in context:
                return 0.5  # 无上下文情感信息
            
            current_emotion = context['emotional_state']
            current_intensity = current_emotion.get('intensity', 0.0)
            current_type = current_emotion.get('type', 'neutral')
            
            # 情感强度相似度
            intensity_similarity = 1.0 - abs(current_intensity - memory.emotional_weight)
            
            # 情感类型匹配（简化实现）
            type_match = 0.7 if memory.type == MemoryType.EMOTION else 0.5
            
            resonance = 0.6 * intensity_similarity + 0.4 * type_match
            
            return max(0.0, min(1.0, resonance))
            
        except Exception:
            return 0.5
    
    def _assess_source_quality(self, memory: Memory) -> float:
        """评估信息来源质量"""
        try:
            quality = 0.5  # 基础来源质量
            
            # 用户明确提供的信息质量高
            if memory.is_user_requested:
                quality += 0.3
            
            # 验证状态影响来源质量
            if memory.verification_method.value == 'user_explicit':
                quality += 0.2
            elif memory.verification_method.value == 'multiple_mentions':
                quality += 0.1
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
    
    def _extract_context_signals(self, context: Optional[Dict[str, Any]]) -> List[str]:
        """提取上下文信号用于调试"""
        signals = []
        
        if not context:
            return ['no_context']
        
        if 'emotional_state' in context:
            signals.append('emotional_state')
        
        if 'query_type' in context:
            signals.append(f"query_type_{context['query_type']}")
        
        if 'group_id' in context:
            signals.append('group_context')
        
        if 'current_message' in context:
            signals.append('message_context')
        
        return signals if signals else ['basic_context']
    
    def _update_statistics(self, *scores):
        """更新统计信息"""
        try:
            score_names = ['temporal', 'semantic', 'social', 'emotional', 'quality']
            for name, score in zip(score_names, scores):
                self.stats['dimension_distributions'][name].append(score)
                
                # 保持统计数据大小
                if len(self.stats['dimension_distributions'][name]) > 1000:
                    self.stats['dimension_distributions'][name] = \
                        self.stats['dimension_distributions'][name][-500:]
                        
        except Exception as e:
            logger.debug(f"Error updating statistics: {e}")
    
    def _create_fallback_result(self, fallback_score: float) -> MultidimensionalScore:
        """创建回退结果"""
        return MultidimensionalScore(
            temporal_score=fallback_score,
            semantic_score=fallback_score,
            social_score=fallback_score,
            emotional_score=fallback_score,
            quality_score=fallback_score,
            weighted_score=fallback_score,
            final_score=fallback_score,
            scenario_type=ScenarioType.DEFAULT,
            weights_used=DimensionWeights(),
            calculation_metadata={
                'calculation_time': datetime.now().isoformat(),
                'fallback_used': True,
                'error_recovery': True
            }
        )