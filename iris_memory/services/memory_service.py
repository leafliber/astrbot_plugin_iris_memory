"""
记忆业务服务层 - 封装核心业务逻辑

职责：
1. 记忆捕获、存储、检索的业务逻辑
2. 组件协调与生命周期管理
3. 异常处理和错误回显
"""
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import asyncio

from astrbot.api.star import Context
from astrbot.api import AstrBotConfig, logger

from iris_memory.models.memory import Memory
from iris_memory.models.user_persona import UserPersona
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.storage.session_manager import SessionManager
from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
from iris_memory.storage.cache import CacheManager
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.utils.hook_manager import MemoryInjector, InjectionMode, HookPriority
from iris_memory.utils.logger import get_logger
from iris_memory.core.config_manager import ConfigManager, init_config_manager
from iris_memory.core.defaults import DEFAULTS
from iris_memory.core.constants import (
    StorageLayer, SessionScope, PersonaStyle, SourceType,
    LogTemplates, KVStoreKeys, NumericDefaults, Separators
)
from iris_memory.utils.command_utils import SessionKeyBuilder

# 可选组件（可能未启用）
from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
from iris_memory.capture.batch_processor import MessageBatchProcessor
from iris_memory.processing.llm_processor import LLMMessageProcessor
from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
from iris_memory.proactive.reply_generator import ProactiveReplyGenerator
from iris_memory.proactive.proactive_manager import ProactiveReplyManager
from iris_memory.multimodal.image_analyzer import ImageAnalyzer


class MemoryService:
    """
    记忆业务服务层
    
    封装所有与记忆相关的业务逻辑，供Handler层调用
    """
    
    def __init__(self, context: Context, config: AstrBotConfig, plugin_data_path: Path):
        """
        初始化记忆服务
        
        Args:
            context: AstrBot上下文
            config: 插件配置
            plugin_data_path: 插件数据目录
        """
        self.context = context
        self.config = config
        self.plugin_data_path = plugin_data_path
        self.cfg = init_config_manager(config)
        self.logger = get_logger("memory_service")
        
        # 核心组件
        self._chroma_manager: Optional[ChromaManager] = None
        self._capture_engine: Optional[MemoryCaptureEngine] = None
        self._retrieval_engine: Optional[MemoryRetrievalEngine] = None
        self._session_manager: Optional[SessionManager] = None
        self._lifecycle_manager: Optional[SessionLifecycleManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._emotion_analyzer: Optional[EmotionAnalyzer] = None
        self._rif_scorer: Optional[RIFScorer] = None
        self._memory_injector: Optional[MemoryInjector] = None
        
        # 可选组件
        self._message_classifier: Optional[MessageClassifier] = None
        self._batch_processor: Optional[MessageBatchProcessor] = None
        self._llm_processor: Optional[LLMMessageProcessor] = None
        self._reply_detector: Optional[ProactiveReplyDetector] = None
        self._reply_generator: Optional[ProactiveReplyGenerator] = None
        self._proactive_manager: Optional[ProactiveReplyManager] = None
        self._image_analyzer: Optional[ImageAnalyzer] = None
        
        # 用户状态缓存
        self._user_emotional_states: Dict[str, EmotionalState] = {}
        self._user_personas: Dict[str, UserPersona] = {}
    
    # ========== 属性访问器 ==========
    
    @property
    def chroma_manager(self) -> Optional[ChromaManager]:
        return self._chroma_manager
    
    @property
    def capture_engine(self) -> Optional[MemoryCaptureEngine]:
        return self._capture_engine
    
    @property
    def retrieval_engine(self) -> Optional[MemoryRetrievalEngine]:
        return self._retrieval_engine
    
    @property
    def session_manager(self) -> Optional[SessionManager]:
        return self._session_manager
    
    @property
    def lifecycle_manager(self) -> Optional[SessionLifecycleManager]:
        return self._lifecycle_manager
    
    @property
    def batch_processor(self) -> Optional[MessageBatchProcessor]:
        return self._batch_processor
    
    @property
    def message_classifier(self) -> Optional[MessageClassifier]:
        return self._message_classifier
    
    @property
    def image_analyzer(self) -> Optional[ImageAnalyzer]:
        return self._image_analyzer
    
    @property
    def proactive_manager(self) -> Optional[ProactiveReplyManager]:
        return self._proactive_manager
    
    @property
    def emotion_analyzer(self) -> Optional[EmotionAnalyzer]:
        return self._emotion_analyzer
    
    # ========== 初始化方法 ==========
    
    async def initialize(self) -> None:
        """异步初始化所有组件"""
        try:
            self.plugin_data_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(LogTemplates.PLUGIN_INIT_START)
            
            await self._init_core_components()
            await self._init_message_processing()
            await self._init_proactive_reply()
            await self._init_image_analyzer()
            await self._apply_config()
            await self._init_batch_processor()
            
            self.logger.info(LogTemplates.PLUGIN_INIT_SUCCESS)
            
        except Exception as e:
            self.logger.error(LogTemplates.PLUGIN_INIT_FAILED.format(error=e), exc_info=True)
            raise
    
    async def _init_core_components(self) -> None:
        """初始化核心组件"""
        # 情感分析器
        self._emotion_analyzer = EmotionAnalyzer(self.config)
        
        # RIF评分器
        self._rif_scorer = RIFScorer()
        
        # Chroma管理器
        self._chroma_manager = ChromaManager(
            self.config, self.plugin_data_path, self.context
        )
        await self._chroma_manager.initialize()
        
        # 会话管理器
        self._session_manager = SessionManager(
            max_working_memory=self.cfg.max_working_memory,
            max_sessions=DEFAULTS.session.max_sessions,
            ttl=self.cfg.session_timeout
        )
        
        # 缓存管理器
        self._cache_manager = CacheManager({})
        
        # 生命周期管理器
        self._lifecycle_manager = SessionLifecycleManager(
            session_manager=self._session_manager,
            chroma_manager=self._chroma_manager,
            upgrade_mode=self.cfg.upgrade_mode,
            llm_upgrade_batch_size=DEFAULTS.memory.llm_upgrade_batch_size,
            llm_upgrade_threshold=DEFAULTS.memory.llm_upgrade_threshold
        )
        await self._lifecycle_manager.start()
        
        # 捕获引擎
        self._capture_engine = MemoryCaptureEngine(
            chroma_manager=self._chroma_manager,
            emotion_analyzer=self._emotion_analyzer,
            rif_scorer=self._rif_scorer
        )
        
        # 检索引擎
        self._retrieval_engine = MemoryRetrievalEngine(
            chroma_manager=self._chroma_manager,
            rif_scorer=self._rif_scorer,
            emotion_analyzer=self._emotion_analyzer,
            session_manager=self._session_manager
        )
        
        # 记忆注入器
        self._memory_injector = MemoryInjector(
            injection_mode=InjectionMode.SUFFIX,
            priority=HookPriority.NORMAL,
            namespace="iris_memory"
        )
    
    async def _init_message_processing(self) -> None:
        """初始化分层消息处理组件"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="message processing"))
        
        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self.cfg.use_llm
        
        if not enable_batch:
            self.logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Batch processing"))
            return
        
        # 初始化LLM处理器
        if use_llm:
            self._llm_processor = LLMMessageProcessor(
                astrbot_context=self.context,
                max_tokens=DEFAULTS.message_processing.llm_max_tokens_for_summary
            )
            llm_ready = await self._llm_processor.initialize()
            if llm_ready and self._lifecycle_manager:
                self._lifecycle_manager.set_llm_provider(self._llm_processor)
                self.logger.info("LLM processor ready")
            else:
                self.logger.warning("LLM context not available, LLM features disabled")
                self._llm_processor = None
        
        # 初始化消息分类器
        self._message_classifier = MessageClassifier(
            trigger_detector=self._capture_engine.trigger_detector if self._capture_engine else None,
            emotion_analyzer=self._emotion_analyzer,
            llm_processor=self._llm_processor,
            config={
                "llm_processing_mode": DEFAULTS.message_processing.llm_processing_mode,
                "immediate_trigger_confidence": DEFAULTS.message_processing.immediate_trigger_confidence,
                "immediate_emotion_intensity": DEFAULTS.message_processing.immediate_emotion_intensity
            }
        )
        
        self.logger.info("Message classifier initialized")
    
    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))
        
        if not self.cfg.proactive_reply_enabled:
            self.logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return
        
        self._reply_detector = ProactiveReplyDetector(
            emotion_analyzer=self._emotion_analyzer,
            config={
                "high_emotion_threshold": DEFAULTS.proactive_reply.high_emotion_threshold,
                "question_threshold": DEFAULTS.proactive_reply.question_threshold
            }
        )
        
        self._reply_generator = ProactiveReplyGenerator(
            astrbot_context=self.context,
            retrieval_engine=self._retrieval_engine,
            config={
                "max_reply_tokens": DEFAULTS.proactive_reply.max_reply_tokens,
                "reply_temperature": DEFAULTS.proactive_reply.reply_temperature
            }
        )
        await self._reply_generator.initialize()
        
        self._proactive_manager = ProactiveReplyManager(
            astrbot_context=self.context,
            reply_detector=self._reply_detector,
            reply_generator=self._reply_generator,
            config={
                "enable_proactive_reply": self.cfg.proactive_reply_enabled,
                "reply_cooldown": DEFAULTS.proactive_reply.cooldown_seconds,
                "max_daily_replies": self.cfg.proactive_reply_max_daily
            }
        )
        await self._proactive_manager.initialize()
        
        self.logger.info("Proactive reply components initialized")
    
    async def _init_image_analyzer(self) -> None:
        """初始化图片分析器"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="image analyzer"))
        
        if not self.cfg.image_analysis_enabled:
            self.logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Image analysis"))
            return
        
        daily_budget = self.cfg.image_analysis_daily_budget
        session_budget = self.cfg.image_analysis_session_budget
        
        self._image_analyzer = ImageAnalyzer(
            astrbot_context=self.context,
            config={
                "enable_image_analysis": self.cfg.image_analysis_enabled,
                "default_level": self.cfg.image_analysis_mode,
                "max_images_per_message": self.cfg.image_analysis_max_images,
                "skip_sticker": DEFAULTS.image_analysis.skip_sticker,
                "analysis_cooldown": DEFAULTS.image_analysis.analysis_cooldown,
                "cache_ttl": DEFAULTS.image_analysis.cache_ttl,
                "max_cache_size": DEFAULTS.image_analysis.max_cache_size,
                "daily_analysis_budget": daily_budget if daily_budget > 0 else 999999,
                "session_analysis_budget": session_budget if session_budget > 0 else 999999,
                "similar_image_window": DEFAULTS.image_analysis.similar_image_window,
                "recent_image_limit": DEFAULTS.image_analysis.recent_image_limit,
                "require_context_relevance": self.cfg.image_analysis_require_context
            }
        )
        
        self.logger.info(f"Image analyzer initialized: mode={self.cfg.image_analysis_mode}")
    
    async def _init_batch_processor(self) -> None:
        """初始化批量处理器"""
        use_llm = self.cfg.use_llm
        
        # 使用 ConfigManager 的便捷属性获取配置
        batch_config = {
            "short_message_threshold": self.cfg.short_message_threshold,
            "merge_time_window": self.cfg.merge_time_window,
            "max_merge_count": self.cfg.max_merge_count,
            "llm_cooldown_seconds": 60,  # 使用默认值
            "summary_interval_seconds": 300,  # 使用默认值
        }
        
        threshold_count = self.cfg.batch_threshold_count
        
        self._batch_processor = MessageBatchProcessor(
            capture_engine=self._capture_engine,
            llm_processor=self._llm_processor,
            proactive_manager=self._proactive_manager,
            threshold_count=threshold_count,
            threshold_interval=DEFAULTS.message_processing.batch_threshold_interval,
            processing_mode=DEFAULTS.message_processing.batch_processing_mode,
            use_llm_summary=use_llm and self._llm_processor is not None,
            on_save_callback=self._save_batch_queues,
            config=batch_config
        )
        await self._batch_processor.start()
        
        self.logger.info(f"Batch processor initialized (threshold={threshold_count})")
    
    async def _apply_config(self) -> None:
        """应用配置到各组件"""
        # 更新缓存管理器
        if self._cache_manager:
            self._cache_manager = CacheManager({
                'embedding_cache': {
                    'max_size': DEFAULTS.cache.embedding_cache_size,
                    'strategy': DEFAULTS.cache.embedding_cache_strategy
                },
                'working_cache': {
                    'max_sessions': DEFAULTS.session.max_sessions,
                    'max_memories_per_session': self.cfg.max_working_memory,
                    'ttl': DEFAULTS.cache.working_cache_ttl
                },
                'compression': {
                    'max_length': DEFAULTS.cache.compression_max_length
                }
            })
        
        # 应用捕获引擎配置
        if self._capture_engine:
            self._capture_engine.set_config({
                "auto_capture": self.cfg.enable_memory,
                "min_confidence": DEFAULTS.memory.min_confidence,
                "rif_threshold": self.cfg.rif_threshold
            })
        
        # 更新会话管理器
        if self._session_manager:
            self._session_manager.max_working_memory = self.cfg.max_working_memory
            self._session_manager.max_sessions = DEFAULTS.session.max_sessions
            self._session_manager.ttl = self.cfg.session_timeout
        
        # 应用生命周期管理器配置
        if self._lifecycle_manager:
            self._lifecycle_manager.cleanup_interval = DEFAULTS.session.session_cleanup_interval
            self._lifecycle_manager.session_timeout = self.cfg.session_timeout
            self._lifecycle_manager.inactive_timeout = DEFAULTS.session.session_inactive_timeout
        
        # 应用检索引擎配置
        if self._retrieval_engine:
            self._retrieval_engine.set_config({
                "max_context_memories": self.cfg.max_context_memories,
                "enable_time_aware": DEFAULTS.llm_integration.enable_time_aware,
                "enable_emotion_aware": DEFAULTS.llm_integration.enable_emotion_aware,
                "enable_token_budget": self.cfg.enable_inject,
                "token_budget": self.cfg.token_budget,
                "coordination_strategy": DEFAULTS.llm_integration.coordination_strategy
            })
        
        # 配置记忆注入器
        if self._memory_injector:
            self._memory_injector.injection_mode = InjectionMode(DEFAULTS.llm_integration.injection_mode)
    
    # ========== 业务方法 ==========
    
    async def capture_and_store_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        is_user_requested: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Memory]:
        """
        捕获并存储记忆
        
        Args:
            message: 消息内容
            user_id: 用户ID
            group_id: 群聊ID
            is_user_requested: 是否用户主动请求
            context: 额外上下文
            
        Returns:
            Optional[Memory]: 捕获的记忆对象
        """
        if not self._capture_engine:
            return None
        
        try:
            memory = await self._capture_engine.capture_memory(
                message=message,
                user_id=user_id,
                group_id=group_id,
                is_user_requested=is_user_requested,
                context=context
            )
            
            if not memory:
                return None
            
            # 分层存储
            await self._store_memory_by_layer(memory)
            
            return memory
            
        except Exception as e:
            self.logger.warning(f"Failed to capture memory: {e}")
            return None
    
    async def _store_memory_by_layer(self, memory: Memory) -> None:
        """根据层级存储记忆"""
        if memory.storage_layer.value == StorageLayer.WORKING:
            if self._session_manager:
                await self._session_manager.add_working_memory(memory)
        else:
            if self._chroma_manager:
                await self._chroma_manager.add_memory(memory)
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int = NumericDefaults.TOP_K_SEARCH
    ) -> List[Memory]:
        """
        搜索记忆
        
        Args:
            query: 查询内容
            user_id: 用户ID
            group_id: 群聊ID
            top_k: 返回数量
            
        Returns:
            List[Memory]: 记忆列表
        """
        if not self._retrieval_engine:
            return []
        
        try:
            emotional_state = self._get_or_create_emotional_state(user_id)
            
            memories = await self._retrieval_engine.retrieve(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                emotional_state=emotional_state
            )
            
            return memories
            
        except Exception as e:
            self.logger.warning(f"Failed to search memories: {e}")
            return []
    
    async def clear_memories(self, user_id: str, group_id: Optional[str]) -> bool:
        """
        清除用户记忆
        
        Args:
            user_id: 用户ID
            group_id: 群聊ID
            
        Returns:
            bool: 是否成功
        """
        try:
            if self._chroma_manager:
                await self._chroma_manager.delete_session(user_id, group_id)
            
            if self._session_manager:
                await self._session_manager.clear_working_memory(user_id, group_id)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to clear memories: {e}")
            return False
    
    async def delete_private_memories(self, user_id: str) -> Tuple[bool, int]:
        """
        删除用户私聊记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            if not self._chroma_manager:
                return False, 0
            
            success, count = await self._chroma_manager.delete_user_memories(
                user_id, in_private_only=True
            )
            
            if self._session_manager:
                await self._session_manager.clear_working_memory(user_id, None)
            
            return success, count
            
        except Exception as e:
            self.logger.warning(f"Failed to delete private memories: {e}")
            return False, 0
    
    async def delete_group_memories(
        self,
        group_id: str,
        scope_filter: Optional[str],
        user_id: Optional[str] = None
    ) -> Tuple[bool, int]:
        """
        删除群聊记忆
        
        Args:
            group_id: 群聊ID
            scope_filter: 范围过滤
            user_id: 用户ID（用于清除缓存）
            
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            if not self._chroma_manager:
                return False, 0
            
            success, count = await self._chroma_manager.delete_group_memories(
                group_id, scope_filter
            )
            
            # 清除工作记忆缓存
            if user_id and scope_filter != SessionScope.GROUP_SHARED:
                if self._session_manager:
                    await self._session_manager.clear_working_memory(user_id, group_id)
            
            return success, count
            
        except Exception as e:
            self.logger.warning(f"Failed to delete group memories: {e}")
            return False, 0
    
    async def delete_all_memories(self) -> Tuple[bool, int]:
        """
        删除所有记忆
        
        Returns:
            Tuple[bool, int]: (是否成功, 删除数量)
        """
        try:
            if not self._chroma_manager:
                return False, 0
            
            success, count = await self._chroma_manager.delete_all_memories()
            
            # 重置会话管理器
            if self._session_manager:
                self._session_manager = SessionManager(
                    max_working_memory=self.cfg.max_working_memory,
                    max_sessions=DEFAULTS.session.max_sessions,
                    ttl=self.cfg.session_timeout
                )
            
            return success, count
            
        except Exception as e:
            self.logger.warning(f"Failed to delete all memories: {e}")
            return False, 0
    
    async def get_memory_stats(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        获取记忆统计
        
        Args:
            user_id: 用户ID
            group_id: 群聊ID
            
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        stats = {
            "working_count": 0,
            "episodic_count": 0,
            "image_analyzed": 0,
            "cache_hits": 0
        }
        
        try:
            if self._session_manager:
                working_memories = await self._session_manager.get_working_memory(user_id, group_id)
                stats["working_count"] = len(working_memories)
            
            if self._chroma_manager:
                stats["episodic_count"] = await self._chroma_manager.count_memories(
                    user_id=user_id,
                    group_id=group_id
                )
            
            if self._image_analyzer:
                image_stats = self._image_analyzer.get_statistics()
                stats["image_analyzed"] = image_stats.get('total_analyzed', 0)
                stats["cache_hits"] = image_stats.get('cache_hits', 0)
                
        except Exception as e:
            self.logger.warning(f"Failed to get memory stats: {e}")
        
        return stats
    
    async def prepare_llm_context(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        image_context: str = ""
    ) -> str:
        """
        准备LLM上下文
        
        Args:
            query: 查询内容
            user_id: 用户ID
            group_id: 群聊ID
            image_context: 图片上下文
            
        Returns:
            str: 格式化的上下文
        """
        if not self._retrieval_engine:
            return ""
        
        try:
            # 更新情感状态
            emotional_state = self._get_or_create_emotional_state(user_id)
            if self._emotion_analyzer:
                emotion_result = await self._emotion_analyzer.analyze_emotion(query)
                self._emotion_analyzer.update_emotional_state(
                    emotional_state,
                    emotion_result["primary"],
                    emotion_result["intensity"],
                    emotion_result["confidence"],
                    emotion_result["secondary"]
                )
            
            # 检索记忆
            memories = await self._retrieval_engine.retrieve(
                query=query,
                user_id=user_id,
                group_id=group_id,
                top_k=self.cfg.max_context_memories,
                emotional_state=emotional_state
            )
            
            context_parts = []
            
            # 添加记忆上下文
            if memories:
                memory_context = self._retrieval_engine.format_memories_for_llm(
                    memories,
                    persona_style=PersonaStyle.NATURAL
                )
                context_parts.append(memory_context)
                self.logger.debug(LogTemplates.MEMORY_INJECTED.format(count=len(memories)))
            
            # 添加图片上下文
            if image_context:
                context_parts.append(image_context)
                self.logger.debug("Injected image context into LLM prompt")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare LLM context: {e}")
            return ""
    
    async def analyze_images(
        self,
        message_chain: List[Any],
        user_id: str,
        context_text: str,
        umo: str,
        session_id: str
    ) -> Tuple[str, str]:
        """
        分析图片
        
        Args:
            message_chain: 消息链
            user_id: 用户ID
            context_text: 上下文文本
            umo: 统一消息来源
            session_id: 会话ID
            
        Returns:
            Tuple[str, str]: (LLM上下文格式, 记忆格式)
        """
        if not self._image_analyzer:
            return "", ""
        
        try:
            image_results = await self._image_analyzer.analyze_message_images(
                message_chain=message_chain,
                user_id=user_id,
                context_text=context_text,
                umo=umo,
                session_id=session_id
            )
            
            if not image_results:
                return "", ""
            
            llm_context = self._image_analyzer.format_for_llm_context(image_results)
            memory_format = self._image_analyzer.format_for_memory(image_results)
            
            return llm_context, memory_format
            
        except Exception as e:
            self.logger.warning(f"Image analysis failed: {e}")
            return "", ""
    
    async def process_message_batch(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        context: Dict[str, Any],
        umo: str,
        image_description: str = ""
    ) -> None:
        """
        处理消息批次
        
        Args:
            message: 消息内容
            user_id: 用户ID
            group_id: 群聊ID
            context: 上下文
            umo: 统一消息来源
            image_description: 图片描述
        """
        if not self._batch_processor or not self._message_classifier:
            return
        
        try:
            # 合并图片信息
            full_message = message
            if image_description:
                full_message = f"{message} {image_description}".strip()
                context["has_image"] = True
                context["image_description"] = image_description
            
            # 分类消息
            classification = await self._message_classifier.classify(full_message, context)
            
            self.logger.debug(
                f"Message classified: {classification.layer.value} "
                f"(confidence: {classification.confidence:.2f}, source: {classification.source})"
            )
            
            # 根据层级处理
            if classification.layer == ProcessingLayer.DISCARD:
                return
            
            if classification.layer == ProcessingLayer.IMMEDIATE:
                await self._handle_immediate_memory(
                    full_message, user_id, group_id, classification
                )
            else:
                # BATCH 层
                await self._batch_processor.add_message(
                    content=full_message,
                    user_id=user_id,
                    group_id=group_id,
                    context=context,
                    umo=umo
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to process message batch: {e}")
    
    async def _handle_immediate_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        classification: Any
    ) -> None:
        """处理立即层级的记忆"""
        memory = await self.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context={
                "classification": classification.metadata,
                "source": classification.source
            }
        )
        
        if memory:
            self.logger.debug(LogTemplates.IMMEDIATE_MEMORY_CAPTURED.format(memory_id=memory.id))
    
    # ========== 会话管理 ==========
    
    def _get_or_create_emotional_state(self, user_id: str) -> EmotionalState:
        """获取或创建用户情感状态"""
        if user_id not in self._user_emotional_states:
            self._user_emotional_states[user_id] = EmotionalState()
        return self._user_emotional_states[user_id]
    
    def get_or_create_user_persona(self, user_id: str) -> UserPersona:
        """获取或创建用户画像"""
        if user_id not in self._user_personas:
            self._user_personas[user_id] = UserPersona(user_id=user_id)
        return self._user_personas[user_id]
    
    def update_session_activity(self, user_id: str, group_id: Optional[str]) -> None:
        """更新会话活动"""
        if self._session_manager:
            self._session_manager.update_session_activity(user_id, group_id)
    
    async def activate_session(self, user_id: str, group_id: Optional[str]) -> None:
        """激活会话"""
        if self._lifecycle_manager:
            await self._lifecycle_manager.activate_session(user_id, group_id)
    
    # ========== 持久化 ==========
    
    async def load_from_kv(self, get_kv_data) -> None:
        """从KV存储加载数据"""
        try:
            # 加载会话数据
            if self._session_manager:
                sessions_data = await get_kv_data(KVStoreKeys.SESSIONS, {})
                if sessions_data:
                    await self._session_manager.deserialize_from_kv_storage(sessions_data)
                    self.logger.info(LogTemplates.SESSION_LOADED.format(
                        count=self._session_manager.get_session_count()
                    ))
            
            # 加载生命周期状态
            if self._lifecycle_manager:
                lifecycle_state = await get_kv_data(KVStoreKeys.LIFECYCLE_STATE, {})
                if lifecycle_state:
                    await self._lifecycle_manager.deserialize_state(lifecycle_state)
                    self.logger.info("Loaded lifecycle state")
            
            # 加载批量处理器队列
            if self._batch_processor:
                batch_queues = await get_kv_data(KVStoreKeys.BATCH_QUEUES, {})
                if batch_queues:
                    await self._batch_processor.deserialize_queues(batch_queues)
                    self.logger.info("Loaded batch processor queues")
            
        except Exception as e:
            self.logger.warning(f"Failed to load from KV: {e}")
    
    async def save_to_kv(self, put_kv_data) -> None:
        """保存到KV存储"""
        try:
            # 保存会话数据
            if self._session_manager:
                sessions_data = await self._session_manager.serialize_for_kv_storage()
                await put_kv_data(KVStoreKeys.SESSIONS, sessions_data)
                self.logger.info(LogTemplates.SESSION_SAVED.format(
                    count=self._session_manager.get_session_count()
                ))
            
            # 保存批量处理器队列
            if self._batch_processor:
                batch_queues = await self._batch_processor.serialize_queues()
                await put_kv_data(KVStoreKeys.BATCH_QUEUES, batch_queues)
                self.logger.info("Saved batch processor queues")
                
        except Exception as e:
            self.logger.warning(f"Failed to save to KV: {e}")
    
    async def _save_batch_queues(self) -> None:
        """保存批量队列（供回调使用）"""
        # 实际保存由 Star 类的 _save_sessions 处理
        pass
    
    # ========== 销毁方法 ==========
    
    async def terminate(self) -> None:
        """销毁服务"""
        try:
            if self._batch_processor:
                await self._batch_processor.stop()
            
            if self._proactive_manager:
                await self._proactive_manager.stop()
            
            if self._lifecycle_manager:
                await self._lifecycle_manager.stop()
            
            if self._chroma_manager:
                await self._chroma_manager.close()
            
            self._log_final_stats()
            
            self.logger.info(LogTemplates.PLUGIN_TERMINATED)
            
        except Exception as e:
            self.logger.error(LogTemplates.PLUGIN_TERMINATE_ERROR.format(error=e), exc_info=True)
    
    def _log_final_stats(self) -> None:
        """输出最终统计"""
        self.logger.info(LogTemplates.FINAL_STATS_HEADER)
        
        components = [
            ("Message Classifier", self._message_classifier),
            ("Batch Processor", self._batch_processor),
            ("LLM Processor", self._llm_processor),
            ("Proactive Manager", self._proactive_manager),
            ("Image Analyzer", self._image_analyzer),
        ]
        
        for name, component in components:
            if component and hasattr(component, 'get_stats'):
                try:
                    stats = component.get_stats()
                    self.logger.info(f"{name}: {stats}")
                except Exception:
                    pass
