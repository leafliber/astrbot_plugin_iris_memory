"""
检索路由器
根据查询复杂度自动选择最优检索策略
"""

import re
from typing import Optional

from iris_memory.core.types import RetrievalStrategy


class RetrievalRouter:
    """检索路由器
    
    根据查询复杂度自动选择最优检索策略：
    - 简单查询（单关键词、短文本）：纯向量检索
    - 多跳推理查询（涉及实体关系）：图遍历检索（暂未实现）
    - 时间感知查询（包含时间线索）：时间向量编码检索
    - 情感感知查询（当前情感相关）：情感过滤检索
    - 复杂查询（多维度约束）：混合检索
    """
    
    def __init__(self):
        """初始化检索路由器"""
        # 时间相关关键词
        self.time_keywords = [
            r'昨天|今天|明天|上周|下周|上个月|下个月',
            r'最近|之前|以前|那时|那时候',
            r'去年|今年|明年',
            r'几天前|几周前|几个月前',
            r'yesterday|today|tomorrow|last week|next week',
            r'recently|before|ago|last year|this year'
        ]
        
        # 关系相关关键词
        self.relation_keywords = [
            r'谁是|谁是.*的上司|.*的上司是谁|.*的同事|.*的朋友',
            r'who is|boss of|colleague|friend of',
            r'关系|认识|了解'
        ]
    
    def route(self, query: str, context: Optional[dict] = None) -> RetrievalStrategy:
        """路由查询到最优策略
        
        Args:
            query: 查询文本
            context: 上下文信息（可选）
            
        Returns:
            RetrievalStrategy: 推荐的检索策略
        """
        query = query.strip()
        
        # 1. 检查是否为复杂查询
        if self._is_complex_query(query, context):
            return RetrievalStrategy.HYBRID
        
        # 2. 检查是否为时间感知查询
        if self._is_time_aware_query(query):
            return RetrievalStrategy.TIME_AWARE
        
        # 3. 检查是否为情感感知查询
        if self._is_emotion_aware_query(context):
            return RetrievalStrategy.EMOTION_AWARE
        
        # 4. 检查是否为多跳推理查询
        if self._is_multi_hop_query(query):
            return RetrievalStrategy.GRAPH_ONLY
        
        # 5. 默认：简单查询使用纯向量检索
        return RetrievalStrategy.VECTOR_ONLY
    
    def _is_complex_query(self, query: str, context: Optional[dict]) -> bool:
        """判断是否为复杂查询
        
        复杂查询特征：
        - 包含多个约束条件
        - 包含时间线索
        - 包含关系线索
        
        Args:
            query: 查询文本
            context: 上下文信息
            
        Returns:
            bool: 是否为复杂查询
        """
        # 检查多个关键词
        keywords = re.findall(r'\b\w{2,}\b', query)
        if len(keywords) >= 5:  # 5个以上关键词
            return True
        
        # 检查包含时间和关系
        has_time = self._is_time_aware_query(query)
        has_relation = self._is_multi_hop_query(query)
        
        if has_time and has_relation:
            return True
        
        return False
    
    def _is_time_aware_query(self, query: str) -> bool:
        """判断是否为时间感知查询
        
        Args:
            query: 查询文本
            
        Returns:
            bool: 是否为时间感知查询
        """
        for pattern in self.time_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_emotion_aware_query(self, context: Optional[dict]) -> bool:
        """判断是否为情感感知查询
        
        Args:
            context: 上下文信息
            
        Returns:
            bool: 是否为情感感知查询
        """
        if not context:
            return False
        
        # 检查当前情感状态
        if 'emotional_state' in context:
            emotional_state = context['emotional_state']
            
            # 如果用户心情不好，需要情感感知
            if hasattr(emotional_state, 'current'):
                primary_emotion = emotional_state.current.primary.value
                negative_emotions = ['sadness', 'anger', 'anxiety', 'fear']
                
                if primary_emotion in negative_emotions:
                    return True
                
                # 如果情感强度很高
                if emotional_state.current.intensity > 0.7:
                    return True
        
        return False
    
    def _is_multi_hop_query(self, query: str) -> bool:
        """判断是否为多跳推理查询
        
        Args:
            query: 查询文本
            
        Returns:
            bool: 是否为多跳推理查询
        """
        for pattern in self.relation_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def analyze_query_complexity(self, query: str) -> dict:
        """分析查询复杂度
        
        Args:
            query: 查询文本
            
        Returns:
            dict: 复杂度分析结果
            {
                "complexity": "simple|medium|complex",
                "features": {
                    "time_aware": bool,
                    "emotion_aware": bool,
                    "multi_hop": bool,
                    "keyword_count": int
                },
                "recommended_strategy": RetrievalStrategy
            }
        """
        features = {
            "time_aware": self._is_time_aware_query(query),
            "emotion_aware": False,  # 需要上下文
            "multi_hop": self._is_multi_hop_query(query),
            "keyword_count": len(re.findall(r'\b\w{2,}\b', query))
        }
        
        # 判断复杂度
        feature_count = sum([
            features["time_aware"],
            features["multi_hop"],
            features["keyword_count"] >= 5
        ])
        
        if feature_count >= 2 or features["keyword_count"] >= 7:
            complexity = "complex"
        elif feature_count == 1:
            complexity = "medium"
        else:
            complexity = "simple"
        
        return {
            "complexity": complexity,
            "features": features,
            "recommended_strategy": self.route(query)
        }
