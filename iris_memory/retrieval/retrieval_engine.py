"""
记忆检索引擎
根据companion-memory框架实现混合检索
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.utils.logger import get_logger

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, RetrievalStrategy, EmotionType, MemoryType
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.token_manager import TokenBudget, MemoryCompressor, DynamicMemorySelector
from iris_memory.utils.persona_coordinator import PersonaCoordinator, CoordinationStrategy
from iris_memory.retrieval.reranker import Reranker
from iris_memory.retrieval.retrieval_router import RetrievalRouter

# 模块logger
logger = get_logger("retrieval_engine")


class MemoryRetrievalEngine:
    """记忆检索引擎 - 实现 companion-memory framework 第13节

    提供多种检索策略，根据查询复杂度自动选择最优方案：

    检索策略体系：
    ─────────────────────────────────────────
    1. VECTOR_ONLY (纯向量检索)
       适用：简单关键词查询、短文本查询
       优势：速度快，适合直接语义匹配
       示例："我喜欢的颜色"、"工作地点"

    2. TIME_AWARE (时间感知检索)
       适用：包含时间线索的查询
       算法：时间衰减函数加权 + 向量相似度
       公式：score = 0.7*semantic + 0.3*time_decay
       示例："上周说的"、"最近有什么安排"

    3. EMOTION_AWARE (情感感知检索)
       适用：用户情感状态需要考虑的查询
       机制：负面情感时过滤高强度正面记忆
       示例：用户难过时避免检索"最快乐的时刻"

    4. GRAPH_ONLY (图遍历检索) [规划中]
       适用：多跳关系推理查询
       示例："小王的上司是谁"
       状态：暂未完整实现，fallback到HYBRID

    5. HYBRID (混合检索) [默认]
       适用：复杂多维度查询
       流程：向量检索 → 情感过滤 → Reranker重排序
       综合权重见下

    结果重排序权重（Reranker）：
    ─────────────────────────────────────────
    - 质量等级：    0.25  (CONFIRMED > HIGH_CONFIDENCE > ...)
    - RIF评分：     0.25  (时近性40% + 相关性30% + 频率30%)
    - 时间衰减：    0.20  (新记忆优先)
    - 向量相似度：  0.15  (语义匹配度)
    - 访问频率：    0.10  (热度加权)
    - 情感一致性：  0.05  (情绪匹配)

    Token预算管理：
    ─────────────────────────────────────────
    - 总预算可配置（默认512 tokens）
    - 动态记忆选择器优化上下文使用
    - 记忆压缩减少token消耗
    """

    def __init__(
        self,
        chroma_manager,
        rif_scorer: Optional[RIFScorer] = None,
        emotion_analyzer: Optional[EmotionAnalyzer] = None,
        reranker: Optional[Reranker] = None,
        session_manager=None
    ):
        """初始化记忆检索引擎

        Args:
            chroma_manager: Chroma存储管理器
            rif_scorer: RIF评分器（可选）
            emotion_analyzer: 情感分析器（可选）
            reranker: 重排序器（可选，默认创建新实例）
            session_manager: 会话管理器（可选，用于合并工作记忆）
        """
        self.chroma_manager = chroma_manager
        self.rif_scorer = rif_scorer or RIFScorer()
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.session_manager = session_manager

        # 配置
        self.max_context_memories = 3
        self.enable_time_aware = True
        self.enable_emotion_aware = True
        self.enable_token_budget = True  # 启用token预算管理
        self.enable_routing = True  # 启用检索路由
        self.enable_working_memory_merge = True  # 启用工作记忆合并

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
        
        # 检索路由器
        self.router = RetrievalRouter()
    
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
            query_preview = query[:50] + "..." if len(query) > 50 else query
            logger.debug(f"Starting memory retrieval: user={user_id}, group={group_id}, top_k={top_k}")
            logger.debug(f"Query: '{query_preview}'")
            
            if emotional_state:
                logger.debug(f"Emotional state: primary={emotional_state.current.primary.value}, intensity={emotional_state.current.intensity:.2f}")
            
            # 使用检索路由器选择策略
            strategy = RetrievalStrategy.HYBRID
            if self.enable_routing:
                context = {'emotional_state': emotional_state}
                strategy = self.router.route(query, context)
                logger.debug(f"Retrieval router selected strategy: {strategy}")
            else:
                logger.debug("Routing disabled, using HYBRID strategy")
            
            # 使用选定策略检索
            result = await self.retrieve_with_strategy(
                query, user_id, group_id, strategy, top_k, emotional_state, storage_layer
            )
            
            logger.info(f"Retrieved {len(result)} memories for user={user_id}, strategy={strategy}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: user={user_id}, error={e}", exc_info=True)
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
            if memory.type == MemoryType.EMOTION and memory.subtype in ["joy", "excitement"]:
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
        emotional_state: Optional[EmotionalState] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """使用指定策略检索记忆

        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID（可选）
            strategy: 检索策略
            top_k: 返回的最大数量
            emotional_state: 情感状态（可选）
            storage_layer: 存储层过滤（可选）

        Returns:
            List[Memory]: 相关记忆列表
        """
        # 根据策略选择检索方法
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return await self._vector_only_retrieval(
                query, user_id, group_id, top_k, storage_layer
            )
        elif strategy == RetrievalStrategy.TIME_AWARE:
            return await self._time_aware_retrieval(
                query, user_id, group_id, top_k, storage_layer
            )
        elif strategy == RetrievalStrategy.EMOTION_AWARE:
            return await self._emotion_aware_retrieval(
                query, user_id, group_id, top_k, emotional_state, storage_layer
            )
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            # 图检索暂未完整实现，降级到混合检索并记录警告
            logger.warning(
                f"GRAPH_ONLY strategy requested but not fully implemented. "
                f"Falling back to HYBRID retrieval. Query: '{query[:50]}...'"
            )
            return await self._hybrid_retrieval(
                query, user_id, group_id, top_k, emotional_state, storage_layer
            )
        else:  # HYBRID
            return await self._hybrid_retrieval(
                query, user_id, group_id, top_k, emotional_state, storage_layer
            )
    
    async def _hybrid_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        emotional_state: Optional[EmotionalState] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """混合检索（原来的核心逻辑）
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID（可选）
            top_k: 返回的最大数量
            emotional_state: 情感状态（用于情感感知检索）
            storage_layer: 存储层过滤（可选）
            
        Returns:
            List[Memory]: 相关记忆列表（已排序）
        """
        # 1. 从Chroma检索候选记忆（EPISODIC/SEMANTIC）
        candidate_memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2,  # 获取更多候选
            storage_layer=storage_layer
        )
        
        # 2. 合并工作记忆（如果启用且有session_manager）
        if self.enable_working_memory_merge and self.session_manager:
            working_memories = await self._get_relevant_working_memories(
                query, user_id, group_id, storage_layer
            )
            if working_memories:
                logger.debug(f"Merging {len(working_memories)} working memories")
                candidate_memories = self._merge_memories(
                    candidate_memories, working_memories
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
    
    async def _vector_only_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """纯向量检索"""
        memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            storage_layer=storage_layer
        )
        # 更新访问统计
        for memory in memories:
            memory.update_access()
        return memories
    
    async def _time_aware_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """时间感知检索"""
        memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2,
            storage_layer=storage_layer
        )
        
        # 更新访问统计
        for memory in memories:
            memory.update_access()
        
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
        emotional_state: Optional[EmotionalState],
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """情感感知检索"""
        memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2,
            storage_layer=storage_layer
        )
        
        # 更新访问统计
        for memory in memories:
            memory.update_access()
        
        # 应用情感过滤
        if emotional_state:
            memories = self._apply_emotion_filter(memories, emotional_state)
        
        return memories[:top_k]
    
    async def _get_relevant_working_memories(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """获取相关的工作记忆
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID（可选）
            storage_layer: 存储层过滤（可选）
            
        Returns:
            List[Memory]: 工作记忆列表
        """
        if not self.session_manager:
            return []
        
        # 如果指定了非WORKING的存储层，不返回工作记忆
        if storage_layer and storage_layer != StorageLayer.WORKING:
            return []
        
        try:
            working_memories = await self.session_manager.get_working_memory(user_id, group_id)
            
            if not working_memories:
                return []
            
            # 简单的关键词匹配过滤（工作记忆通常数量较少，不需要向量检索）
            query_lower = query.lower()
            query_keywords = set(query_lower.split())
            
            relevant = []
            for memory in working_memories:
                content_lower = memory.content.lower()
                # 检查是否有关键词匹配
                content_words = set(content_lower.split())
                overlap = query_keywords & content_words
                
                # 如果有关键词重叠或者查询在内容中，认为相关
                if overlap or query_lower in content_lower or content_lower in query_lower:
                    relevant.append(memory)
                # 对于短查询，包含所有工作记忆以保持上下文
                elif len(query) < 20:
                    relevant.append(memory)
            
            logger.debug(f"Found {len(relevant)} relevant working memories out of {len(working_memories)}")
            return relevant
            
        except Exception as e:
            logger.warning(f"Failed to get working memories: {e}")
            return []
    
    def _merge_memories(
        self,
        persistent_memories: List[Memory],
        working_memories: List[Memory]
    ) -> List[Memory]:
        """合并持久记忆和工作记忆
        
        Args:
            persistent_memories: 来自Chroma的持久记忆
            working_memories: 来自SessionManager的工作记忆
            
        Returns:
            List[Memory]: 合并后的记忆列表（去重）
        """
        # 使用ID去重
        seen_ids = set()
        merged = []
        
        # 工作记忆优先（更新鲜的上下文）
        for memory in working_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                merged.append(memory)
        
        # 添加持久记忆
        for memory in persistent_memories:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                merged.append(memory)
        
        return merged
    
    def set_session_manager(self, session_manager):
        """设置会话管理器（用于延迟注入）
        
        Args:
            session_manager: 会话管理器实例
        """
        self.session_manager = session_manager
        logger.debug("SessionManager injected into MemoryRetrievalEngine")
    
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
        self.enable_routing = config.get("enable_routing", True)
        self.enable_working_memory_merge = config.get("enable_working_memory_merge", True)
        
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
