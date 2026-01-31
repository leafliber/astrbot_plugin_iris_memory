"""
记忆检索引擎
根据companion-memory框架实现混合检索
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.utils.logger import logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, RetrievalStrategy, EmotionType
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.token_manager import TokenBudget, MemoryCompressor, DynamicMemorySelector
from iris_memory.utils.persona_coordinator import PersonaCoordinator, CoordinationStrategy
from iris_memory.retrieval.reranker import Reranker


class MemoryRetrievalEngine:
    """记忆检索引擎

    实现混合检索：
    - 简单查询：纯向量检索
    - 多跳推理：图遍历检索（暂未实现）
    - 时间感知：时间向量编码检索
    - 情感感知：情感过滤检索
    - 复杂查询：混合检索

    结果重排序（使用统一的Reranker模块）：
    - 质量等级：0.25 (CONFIRMED > HIGH_CONFIDENCE > MODERATE)
    - RIF评分：0.25 (基于时近性、相关性、频率的科学评分)
    - 时间衰减：0.20 (新记忆优先)
    - 向量相似度：0.15 (Chroma检索结果的补充)
    - 访问频率：0.10 (高频访问的记忆优先)
    - 情感一致性：0.05 (与当前情感一致的记忆优先)
    """

    def __init__(
        self,
        chroma_manager,
        rif_scorer: Optional[RIFScorer] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        reranker: Optional[Reranker] = None
    ):
        """初始化记忆检索引擎

        Args:
            chroma_manager: Chroma存储管理器
            rif_scorer: RIF评分器（可选）
            emotion_analyzer: 情感分析器（可选）
            reranker: 重排序器（可选，默认创建新实例）
        """
        self.chroma_manager = chroma_manager
        self.rif_scorer = rif_scorer or RIFScorer()
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()

        # 配置
        self.max_context_memories = 3
        self.enable_time_aware = True
        self.enable_emotion_aware = True
        self.enable_token_budget = True  # 启用token预算管理

        # Token管理器
        self.token_budget = TokenBudget(total_budget=512)
        self.memory_compressor = MemoryCompressor(max_summary_length=100)
        self.memory_selector = DynamicMemorySelector(
            token_budget=self.token_budget,
            compressor=self.memory_compressor
        )

        # 人格协调器
        self.persona_coordinator = PersonaCoordinator(
            strategy=CoordinationStrategy.HYBRID
        )

        # 重排序器
        self.reranker = reranker or Reranker(enable_vector_score=True)
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        storage_layer: Optional[StorageLayer] = None,
        emotional_state: Optional[EmotionalState] = None
    ) -> List[Memory]:
        """检索相关记忆
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID（可选）
            top_k: 返回的最大数量
            storage_layer: 存储层过滤（可选）
            emotional_state: 情感状态（用于情感感知检索）
            
        Returns:
            List[Memory]: 相关记忆列表（已排序）
        """
        try:
            # 1. 从Chroma检索候选记忆
            candidate_memories = await self.chroma_manager.query_memories(
                query_text=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k * 2,  # 获取更多候选
                storage_layer=storage_layer
            )
            
            if not candidate_memories:
                logger.debug(f"No memories found for query: {query[:50]}...")
                return []
            
            # 2. 应用情感过滤
            if self.enable_emotion_aware and emotional_state:
                candidate_memories = self._apply_emotion_filter(
                    candidate_memories,
                    emotional_state
                )
            
            # 3. 更新访问统计
            for memory in candidate_memories:
                memory.update_access()
            
            # 4. 结果重排序
            ranked_memories = self._rerank_memories(
                candidate_memories,
                query,
                emotional_state
            )
            
            # 5. 返回Top-N结果
            result = ranked_memories[:top_k]
            
            logger.info(f"Retrieved {len(result)} memories for query: {query[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def _apply_emotion_filter(
        self,
        memories: List[Memory],
        emotional_state: EmotionalState
    ) -> List[Memory]:
        """应用情感过滤
        
        Args:
            memories: 记忆列表
            emotional_state: 情感状态
            
        Returns:
            List[Memory]: 过滤后的记忆列表
        """
        # 检查是否应该过滤高强度正面记忆
        if not self.emotion_analyzer.should_filter_positive_memories(emotional_state):
            return memories
        
        # 过滤掉高强度正面记忆
        filtered = []
        for memory in memories:
            if memory.type == "emotion" and memory.subtype in ["joy", "excitement"]:
                if memory.emotional_weight > 0.8:
                    logger.debug(f"Filtered high-intensity positive memory: {memory.id}")
                    continue
            filtered.append(memory)
        
        return filtered
    
    def _rerank_memories(
        self,
        memories: List[Memory],
        query: str,
        emotional_state: Optional[EmotionalState] = None
    ) -> List[Memory]:
        """重排序记忆（使用统一的Reranker）

        Args:
            memories: 记忆列表
            query: 查询文本
            emotional_state: 情感状态（可选）

        Returns:
            List[Memory]: 排序后的记忆列表
        """
        # 构建上下文
        context = {}
        if emotional_state:
            context['emotional_state'] = emotional_state

        # 使用Reranker重排序
        return self.reranker.rerank(memories, query, context)

    async def retrieve_with_strategy(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = 10,
        emotional_state: Optional[EmotionalState] = None
    ) -> List[Memory]:
        """使用指定策略检索记忆
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID（可选）
            strategy: 检索策略
            top_k: 返回的最大数量
            emotional_state: 情感状态（可选）
            
        Returns:
            List[Memory]: 相关记忆列表
        """
        # 根据策略选择检索方法
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return await self._vector_only_retrieval(
                query, user_id, group_id, top_k
            )
        elif strategy == RetrievalStrategy.TIME_AWARE:
            return await self._time_aware_retrieval(
                query, user_id, group_id, top_k
            )
        elif strategy == RetrievalStrategy.EMOTION_AWARE:
            return await self._emotion_aware_retrieval(
                query, user_id, group_id, top_k, emotional_state
            )
        else:  # HYBRID
            return await self.retrieve(
                query, user_id, group_id, top_k, None, emotional_state
            )
    
    async def _vector_only_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int
    ) -> List[Memory]:
        """纯向量检索"""
        return await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k
        )
    
    async def _time_aware_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int
    ) -> List[Memory]:
        """时间感知检索"""
        memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2
        )
        
        # 按时间得分重新排序
        scored = [(m, self._calculate_time_score(m)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, s in scored[:top_k]]

    def _calculate_time_score(self, memory: Memory) -> float:
        """计算时间得分

        基于记忆的新旧程度计算得分，越新的记忆得分越高

        Args:
            memory: 记忆对象

        Returns:
            时间得分 (0-1)
        """
        from datetime import datetime, timedelta

        now = datetime.now()
        days_ago = (now - memory.created_time).total_seconds() / 86400  # 转换为天数

        # 使用指数衰减：越新的记忆得分越高
        # 30天内：得分0.9-1.0
        # 30-90天：得分0.7-0.9
        # 90-365天：得分0.4-0.7
        # 365天以上：得分0-0.4
        if days_ago < 30:
            return 1.0 - (days_ago / 30) * 0.1
        elif days_ago < 90:
            return 0.9 - ((days_ago - 30) / 60) * 0.2
        elif days_ago < 365:
            return 0.7 - ((days_ago - 90) / 275) * 0.3
        else:
            return max(0, 0.4 - ((days_ago - 365) / 365) * 0.4)
    
    async def _emotion_aware_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int,
        emotional_state: Optional[EmotionalState]
    ) -> List[Memory]:
        """情感感知检索"""
        memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2
        )
        
        # 应用情感过滤
        if emotional_state:
            memories = self._apply_emotion_filter(memories, emotional_state)
        
        return memories[:top_k]
    
    def format_memories_for_llm(
        self,
        memories: List[Memory],
        use_token_budget: bool = True,
        user_persona: Optional[Dict[str, Any]] = None
    ) -> str:
        """格式化记忆用于注入到LLM上下文
        
        Args:
            memories: 记忆列表
            use_token_budget: 是否使用token预算管理（默认True）
            user_persona: 用户画像（可选，用于人格协调）
            
        Returns:
            str: 格式化的记忆文本
        """
        if not memories:
            return ""
        
        # 如果启用token预算，使用动态选择器
        if self.enable_token_budget and use_token_budget:
            return self.memory_selector.get_memory_context(
                memories,
                target_count=self.max_context_memories
            )
        
        # 否则使用传统的格式化方法
        formatted = "【相关记忆】\n"
        for i, memory in enumerate(memories, 1):
            time_str = memory.created_time.strftime("%Y-%m-%d %H:%M")
            # 处理type可能是枚举或字符串的情况
            if hasattr(memory.type, 'value'):
                type_label = memory.type.value.upper()
            else:
                type_label = str(memory.type).upper()

            formatted += f"{i}. [{type_label}] {time_str}\n"
            formatted += f"   内容: {memory.content}\n"

            if memory.summary:
                formatted += f"   摘要: {memory.summary}\n"

            if memory.emotional_weight > 0.5:
                formatted += f"   情感强度: {memory.emotional_weight:.2f}\n"

            formatted += "\n"
        
        # 如果有用户画像，应用人格协调
        if user_persona:
            formatted = self.persona_coordinator.format_context_with_persona(
                formatted,
                user_persona,
                bot_persona="friendly"  # 默认Bot人格
            )
        
        return formatted
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置
        
        Args:
            config: 配置字典
        """
        self.max_context_memories = config.get("max_context_memories", 3)
        self.enable_time_aware = config.get("enable_time_aware", True)
        self.enable_emotion_aware = config.get("enable_emotion_aware", True)
        self.enable_token_budget = config.get("enable_token_budget", True)
        
        # 更新token预算
        token_budget = config.get("token_budget", 512)
        if token_budget != self.token_budget.total_budget:
            self.token_budget = TokenBudget(total_budget=token_budget)
            self.memory_selector.token_budget = self.token_budget
        
        # 更新人格协调策略
        coordination_strategy = config.get("coordination_strategy", "hybrid")
        self.persona_coordinator.set_strategy(
            CoordinationStrategy(coordination_strategy)
        )
        self.enable_token_budget = config.get("enable_token_budget", True)
        
        # 更新token预算
        token_budget = config.get("token_budget", 512)
        if token_budget != self.token_budget.total_budget:
            self.token_budget = TokenBudget(total_budget=token_budget)
            self.memory_selector.token_budget = self.token_budget
