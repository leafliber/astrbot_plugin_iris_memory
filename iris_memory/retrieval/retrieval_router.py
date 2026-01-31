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
            r'who is|boss of|my boss|colleague|friend of',
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

        # 先检查各个特征
        is_multi_hop = self._is_multi_hop_query(query)
        is_emotion_aware = self._is_emotion_aware_query(context)
        is_time_aware = self._is_time_aware_query(query)

        # 计算关键词数量（根据语言使用不同的计算方法）
        # 判断是否主要为英文
        is_english = len(re.findall(r'[a-zA-Z]', query)) > len(query) * 0.5

        if is_english:
            # 英文：计算单词数
            keyword_count = len(query.split())
        else:
            # 中文和其他：简化为长度/2
            keyword_count = len(query) // 2

        # 时间 + 关系 = 复杂查询（优先级最高）
        if is_time_aware and is_multi_hop:
            return RetrievalStrategy.HYBRID

        # 情感感知查询（优先级第二）
        if is_emotion_aware:
            return RetrievalStrategy.EMOTION_AWARE

        # 多跳推理查询（优先级第三）
        if is_multi_hop:
            return RetrievalStrategy.GRAPH_ONLY

        # 时间感知查询（优先级第四）- 纯时间查询优先
        if is_time_aware:
            # 检查是否有其他语义特征（多个实体或复杂主题）
            semantic_keywords = [
                r'公司.*同事|同事.*公司|项目.*讨论|讨论.*项目',
                r'company.*colleague|colleague.*company|project.*discussion'
            ]
            has_semantic_features = any(
                re.search(pattern, query, re.IGNORECASE)
                for pattern in semantic_keywords
            )

            # 纯时间查询（无复杂语义特征）使用时间感知检索
            if not has_semantic_features and not is_multi_hop:
                return RetrievalStrategy.TIME_AWARE
            
            # 有时间特征且复杂：使用混合检索
            return RetrievalStrategy.HYBRID

        # 复杂查询判断（优先级第五）
        # 检查是否包含大量特殊字符（不是真正的复杂查询）
        special_char_ratio = len(re.sub(r'[\w\u4e00-\u9fff]', '', query)) / len(query) if len(query) > 0 else 0

        # 需要同时满足长度和关键词条件，或者非常长
        if special_char_ratio < 0.3:
            if (len(query) > 15 and keyword_count >= 5) or len(query) > 20:
                return RetrievalStrategy.HYBRID

        # 默认：简单查询使用纯向量检索
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
        # 检查包含时间和关系
        has_time = self._is_time_aware_query(query)
        has_relation = self._is_multi_hop_query(query)

        if has_time and has_relation:
            return True

        # 检查查询长度（长查询通常是复杂的）
        if len(query) > 15:
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
            # 使用长度作为关键字计数的替代
            "keyword_count": len(query) // 2  # 简化：每2个字符算一个词
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
