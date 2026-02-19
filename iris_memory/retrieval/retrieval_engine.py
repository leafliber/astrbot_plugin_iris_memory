"""
记忆检索引擎
根据companion-memory框架实现混合检索
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from iris_memory.models.memory import Memory
from iris_memory.core.types import StorageLayer, RetrievalStrategy, EmotionType, MemoryType, RerankContext
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.token_manager import TokenBudget, MemoryCompressor, DynamicMemorySelector
from iris_memory.analysis.persona.persona_coordinator import PersonaCoordinator, CoordinationStrategy
from iris_memory.utils.member_utils import format_member_tag
from iris_memory.retrieval.reranker import Reranker
from iris_memory.retrieval.retrieval_router import RetrievalRouter
from iris_memory.retrieval.retrieval_logger import retrieval_log


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
        session_manager=None,
        llm_retrieval_router=None,
    ):
        """初始化记忆检索引擎

        Args:
            chroma_manager: Chroma存储管理器
            rif_scorer: RIF评分器（可选）
            emotion_analyzer: 情感分析器（可选）
            reranker: 重排序器（可选，默认创建新实例）
            session_manager: 会话管理器（可选，用于合并工作记忆）
            llm_retrieval_router: LLM增强检索路由器（可选）
        """
        self.chroma_manager = chroma_manager
        self.rif_scorer = rif_scorer or RIFScorer()
        self.emotion_analyzer = emotion_analyzer or EmotionAnalyzer()
        self.session_manager = session_manager

        # 配置
        self.max_context_memories = 3
        self.enable_time_aware = True
        self.enable_emotion_aware = True
        self.enable_token_budget = True
        self.enable_routing = True
        self.enable_working_memory_merge = True

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
        
        # 检索路由器（支持LLM增强）
        self._llm_router = llm_retrieval_router
        self.router = RetrievalRouter()

        # 知识图谱模块（可选，由外部注入）
        self._kg_module = None
    
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
            retrieval_log.retrieve_start(user_id, query, group_id, top_k)
            
            if emotional_state:
                retrieval_log.emotional_state(
                    user_id,
                    emotional_state.current.primary.value,
                    emotional_state.current.intensity
                )
            
            strategy = RetrievalStrategy.HYBRID
            if self.enable_routing:
                context = {'emotional_state': emotional_state}
                strategy = await self._route_query(query, context)
                retrieval_log.strategy_selected(user_id, strategy.value if hasattr(strategy, 'value') else str(strategy))
            else:
                retrieval_log.strategy_selected(user_id, "HYBRID", "default")
            
            result = await self.retrieve_with_strategy(
                query, user_id, group_id, strategy, top_k, emotional_state, storage_layer
            )
            
            retrieval_log.retrieve_ok(user_id, len(result), strategy.value if hasattr(strategy, 'value') else str(strategy))
            return result
            
        except Exception as e:
            retrieval_log.retrieve_error(user_id, e)
            return []
    
    def _apply_emotion_filter(
        self,
        memories: List[Memory],
        emotional_state: EmotionalState,
        user_id: str = ""
    ) -> List[Memory]:
        if not emotional_state:
            return memories
        
        filtered = []
        for memory in memories:
            if emotional_state.current.primary in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]:
                if memory.emotional_weight > 0.7 and memory.type == MemoryType.EMOTION:
                    if hasattr(memory, 'subtype') and memory.subtype:
                        if memory.subtype in ['joy', 'excitement']:
                            retrieval_log.memory_filtered(user_id, memory.id, "emotion_mismatch")
                            continue
            
            filtered.append(memory)
        
        return filtered
    
    async def _route_query(self, query: str, context: Optional[Dict] = None) -> RetrievalStrategy:
        """路由查询（支持LLM增强）
        
        Args:
            query: 查询文本
            context: 上下文
            
        Returns:
            RetrievalStrategy: 检索策略
        """
        if self._llm_router:
            try:
                result = await self._llm_router.detect(query, context)
                if result.confidence >= 0.6:
                    return result.strategy
            except Exception as e:
                retrieval_log.routing_failed("", str(e))
        
        return self.router.route(query, context)
    
    def _rerank_memories(
        self,
        memories: List[Memory],
        query: str,
        emotional_state: Optional[EmotionalState] = None,
        user_id: Optional[str] = None
    ) -> List[Memory]:
        """重排序记忆（使用统一的Reranker）

        Args:
            memories: 记忆列表
            query: 查询文本
            emotional_state: 情感状态（可选）
            user_id: 当前对话者ID（可选，用于sender匹配权重）

        Returns:
            List[Memory]: 排序后的记忆列表
        """
        # 构建上下文
        context: RerankContext = {}
        if emotional_state:
            context['emotional_state'] = emotional_state
        if user_id:
            context['current_user_id'] = user_id

        # 注入 MemberIdentityService（如果可用）
        from iris_memory.utils.member_utils import get_identity_service
        identity_service = get_identity_service()
        if identity_service:
            context['member_identity_service'] = identity_service

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
            if self._kg_module and self._kg_module.enabled:
                return await self._graph_retrieval(
                    query, user_id, group_id, top_k, emotional_state, storage_layer
                )
            else:
                retrieval_log.graph_fallback(user_id, query)
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
        candidate_memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2,
            storage_layer=storage_layer
        )
        retrieval_log.vector_query(user_id, len(candidate_memories), storage_layer.value if storage_layer and hasattr(storage_layer, 'value') else None)
        
        if self.enable_working_memory_merge and self.session_manager:
            working_memories = await self._get_relevant_working_memories(
                query, user_id, group_id, storage_layer
            )
            if working_memories:
                retrieval_log.working_memory_merged(user_id, len(working_memories), len(candidate_memories) + len(working_memories))
                candidate_memories = self._merge_memories(
                    candidate_memories, working_memories
                )
        
        if not candidate_memories:
            retrieval_log.no_memories_found(user_id, query)
            return []
        
        if self.enable_emotion_aware and emotional_state:
            before_count = len(candidate_memories)
            candidate_memories = self._apply_emotion_filter(
                candidate_memories,
                emotional_state,
                user_id
            )
            retrieval_log.emotion_filter_applied(
                user_id, before_count, len(candidate_memories), "negative_state"
            )
        
        ranked_memories = self._rerank_memories(
            candidate_memories,
            query,
            emotional_state,
            user_id
        )
        
        result = ranked_memories[:top_k]
        
        # 仅对最终返回的记忆更新访问统计
        for memory in result:
            memory.update_access()
        
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
        
        # 按时间得分重新排序
        scored = [(m, self._calculate_time_score(m)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [m for m, s in scored[:top_k]]
        # 仅对最终返回的记忆更新访问统计
        for memory in result:
            memory.update_access()
        return result

    def _calculate_time_score(self, memory: Memory) -> float:
        """计算时间得分

        委托给 Memory.calculate_time_score()，基于创建时间排序。

        Args:
            memory: 记忆对象

        Returns:
            时间得分 (0-1)
        """
        return memory.calculate_time_score(use_created_time=True)
    
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
        
        if emotional_state:
            memories = self._apply_emotion_filter(memories, emotional_state, user_id)
        
        result = memories[:top_k]
        # 仅对最终返回的记忆更新访问统计
        for memory in result:
            memory.update_access()
        return result

    def set_kg_module(self, kg_module) -> None:
        """注入知识图谱模块

        Args:
            kg_module: KnowledgeGraphModule 实例
        """
        self._kg_module = kg_module

    async def _graph_retrieval(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        top_k: int = 10,
        emotional_state: Optional[EmotionalState] = None,
        storage_layer: Optional[StorageLayer] = None
    ) -> List[Memory]:
        """知识图谱检索 + 向量检索混合

        流程：
        1. 图推理获取相关 memory_id
        2. 向量检索获取候选记忆
        3. 图结果提升相关记忆的排序权重
        4. 合并 + 重排序

        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID
            top_k: 最大返回数
            emotional_state: 情感状态
            storage_layer: 存储层过滤

        Returns:
            List[Memory]: 记忆列表
        """
        # 向量检索部分
        vector_memories = await self.chroma_manager.query_memories(
            query_text=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k * 2,
            storage_layer=storage_layer
        )

        # 图推理部分
        kg_memory_ids: set = set()
        if self._kg_module and self._kg_module.reasoning:
            try:
                reasoning_result = await self._kg_module.graph_retrieve(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                )
                # 收集推理路径涉及的 memory_id
                for edge in reasoning_result.get_all_edges():
                    if edge.memory_id:
                        kg_memory_ids.add(edge.memory_id)
            except Exception as e:
                retrieval_log.graph_fallback(user_id, f"KG reason error: {e}")

        # 工作记忆合并
        if self.enable_working_memory_merge and self.session_manager:
            working_memories = await self._get_relevant_working_memories(
                query, user_id, group_id, storage_layer
            )
            if working_memories:
                vector_memories = self._merge_memories(vector_memories, working_memories)

        if not vector_memories:
            retrieval_log.no_memories_found(user_id, query)
            return []

        # KG boost: 如果记忆的 ID 出现在图推理路径中，提升其 importance_score
        if kg_memory_ids:
            for mem in vector_memories:
                if mem.id in kg_memory_ids:
                    mem.importance_score = min(1.0, mem.importance_score + 0.3)
                    mem.rif_score = min(1.0, mem.rif_score + 0.2)

        # 情感过滤
        if self.enable_emotion_aware and emotional_state:
            vector_memories = self._apply_emotion_filter(
                vector_memories, emotional_state, user_id
            )

        # 重排序
        ranked = self._rerank_memories(
            vector_memories, query, emotional_state, user_id
        )

        result = ranked[:top_k]
        for memory in result:
            memory.update_access()

        return result
    
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
            
            return relevant
            
        except Exception as e:
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
        self.session_manager = session_manager
    
    def format_memories_for_llm(
        self,
        memories: List[Memory],
        use_token_budget: bool = True,
        user_persona: Optional[Dict[str, Any]] = None,
        persona_style: str = "default",
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> str:
        """格式化记忆用于注入到LLM上下文
        
        Args:
            memories: 记忆列表
            use_token_budget: 是否使用token预算管理（默认True）
            user_persona: 用户画像（可选，用于人格协调）
            persona_style: 人格风格 (default/natural/roleplay)
            group_id: 群组ID（用于区分群聊/个人知识）
            current_sender_name: 当前发言者名称（用于群成员识别）
            
        Returns:
            str: 格式化的记忆文本
        """
        if not memories:
            return ""
        
        # 如果启用token预算，使用动态选择器
        if self.enable_token_budget and use_token_budget:
            return self.memory_selector.get_memory_context(
                memories,
                target_count=self.max_context_memories,
                persona_style=persona_style,
                group_id=group_id,
                current_sender_name=current_sender_name
            )
        
        # 根据人格风格选择格式化方式
        if persona_style == "natural":
            formatted = self._format_natural_style(memories, group_id, current_sender_name)
        elif persona_style == "roleplay":
            formatted = self._format_roleplay_style(memories, group_id, current_sender_name)
        else:
            formatted = self._format_default_style(memories, group_id, current_sender_name)
        
        # 如果有用户画像，应用人格协调
        if user_persona:
            formatted = self.persona_coordinator.format_context_with_persona(
                formatted,
                user_persona,
                bot_persona="friendly"  # 默认Bot人格
            )
        
        return formatted
    
    def _format_memory_label(self, memory: Memory, group_id: Optional[str] = None) -> str:
        """为记忆生成来源标签，明确区分群聊知识和个人知识
        
        Args:
            memory: 记忆对象
            group_id: 当前群组ID
            
        Returns:
            str: 来源标签字符串
        """
        parts = []
        
        # 来源标注
        from iris_memory.core.memory_scope import MemoryScope
        if memory.scope == MemoryScope.GROUP_SHARED:
            parts.append("群聊共识")
        elif memory.scope == MemoryScope.GROUP_PRIVATE:
            sender_tag = self._format_sender_tag(memory, group_id)
            if not sender_tag:
                # 强制使用 user_id 生成标识，确保记忆来源明确
                sender_tag = format_member_tag(None, memory.user_id, group_id)
            if sender_tag:
                parts.append(f"{sender_tag}的个人信息")
            else:
                parts.append("个人信息")
        elif memory.scope == MemoryScope.USER_PRIVATE:
            parts.append("私聊记忆")
        
        return "｜".join(parts) if parts else ""
    
    def _format_natural_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> str:
        """自然群友风格格式化"""
        formatted = "【你记得的事情】\n"
        formatted += "以下是你和群友之间的往事，请用自己的话自然提及，不要暴露'记录'、'数据'等概念：\n"
        
        if group_id:
            formatted += "（注意区分群共识和个人信息，不要把A的事情说成B的）\n"
        
        for memory in memories:
            label = self._format_memory_label(memory, group_id)
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f"（{sender_tag}说的）" if sender_tag else ""
            prefix = f"[{label}]" if label else ""
            formatted += f"- {prefix}{sender}{memory.content}\n"
        
        return formatted
    
    def _format_roleplay_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> str:
        """角色扮演风格格式化"""
        formatted = "【你的记忆】\n"
        formatted += "这些都是你亲身经历的事情，回复时可以自然地说'我记得...'、'你之前说过...'：\n"
        for memory in memories:
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f"（{sender_tag}）" if sender_tag else ""
            formatted += f"· {sender}{memory.content}\n"
        return formatted
    
    def _format_default_style(
        self,
        memories: List[Memory],
        group_id: Optional[str] = None,
        current_sender_name: Optional[str] = None
    ) -> str:
        """默认格式化"""
        formatted = "【相关记忆】\n"
        for i, memory in enumerate(memories, 1):
            time_str = memory.created_time.strftime("%Y-%m-%d %H:%M")
            if hasattr(memory.type, 'value'):
                type_label = memory.type.value.upper()
            else:
                type_label = str(memory.type).upper()
            
            label = self._format_memory_label(memory, group_id)
            sender_tag = self._format_sender_tag(memory, group_id)
            sender = f" @{sender_tag}" if sender_tag else ""
            
            formatted += f"{i}. [{type_label}]{sender} {time_str}"
            if label:
                formatted += f" ({label})"
            formatted += f"\n   内容: {memory.content}\n"
            
            if memory.summary:
                formatted += f"   摘要: {memory.summary}\n"
            
            if memory.emotional_weight > 0.5:
                formatted += f"   情感强度: {memory.emotional_weight:.2f}\n"
            
            formatted += "\n"
        
        return formatted

    def _format_sender_tag(self, memory: Memory, group_id: Optional[str]) -> str:
        """Format a stable sender tag for group disambiguation."""
        if group_id:
            return format_member_tag(memory.sender_name, memory.user_id, group_id)
        return (memory.sender_name or "").strip()
    
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
