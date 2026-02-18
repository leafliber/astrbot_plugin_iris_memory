"""
MemoryService 初始化模块

将初始化逻辑从 MemoryService 中拆分出来，提高代码可维护性。
"""

from typing import TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.core.defaults import DEFAULTS
from iris_memory.core.constants import LogTemplates

if TYPE_CHECKING:
    from astrbot.api.star import Context
    from astrbot.api import AstrBotConfig
    from pathlib import Path

logger = get_logger("memory_service.init")


class ServiceInitializer:
    """MemoryService 初始化器 Mixin
    
    职责：
    1. 核心组件初始化
    2. 可选组件初始化
    3. LLM增强组件初始化
    4. 配置应用
    """

    async def _init_core_components(self) -> None:
        """初始化核心组件"""
        self._emotion_analyzer = self._create_emotion_analyzer()
        self._rif_scorer = self._create_rif_scorer()
        self._chroma_manager = await self._create_chroma_manager()
        self._session_manager = self._create_session_manager()
        self._cache_manager = self._create_cache_manager()
        self._lifecycle_manager = await self._create_lifecycle_manager()
        self._capture_engine = self._create_capture_engine()
        self._retrieval_engine = self._create_retrieval_engine()
        self._chat_history_buffer = self._create_chat_history_buffer()
        self._member_identity = self._create_member_identity()
        
        logger.info("MemberIdentityService initialized")

    def _create_emotion_analyzer(self):
        from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
        return EmotionAnalyzer(self.config)

    def _create_rif_scorer(self):
        from iris_memory.analysis.rif_scorer import RIFScorer
        return RIFScorer()

    async def _create_chroma_manager(self):
        from iris_memory.storage.chroma_manager import ChromaManager
        manager = ChromaManager(
            self.config, self.plugin_data_path, self.context
        )
        await manager.initialize()
        return manager

    def _create_session_manager(self):
        from iris_memory.storage.session_manager import SessionManager
        return SessionManager(
            max_working_memory=self.cfg.max_working_memory,
            max_sessions=DEFAULTS.session.max_sessions,
            ttl=self.cfg.session_timeout
        )

    def _create_cache_manager(self):
        from iris_memory.storage.cache import CacheManager
        return CacheManager({})

    async def _create_lifecycle_manager(self):
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
        manager = SessionLifecycleManager(
            session_manager=self._session_manager,
            chroma_manager=self._chroma_manager,
            upgrade_mode=self.cfg.upgrade_mode,
            llm_upgrade_batch_size=DEFAULTS.memory.llm_upgrade_batch_size,
            llm_upgrade_threshold=DEFAULTS.memory.llm_upgrade_threshold
        )
        await manager.start()
        return manager

    def _create_capture_engine(self):
        from iris_memory.capture.engine import MemoryCaptureEngine
        return MemoryCaptureEngine(
            chroma_manager=self._chroma_manager,
            emotion_analyzer=self._emotion_analyzer,
            rif_scorer=self._rif_scorer,
            llm_sensitivity_detector=self._llm_sensitivity_detector,
            llm_trigger_detector=self._llm_trigger_detector,
            llm_conflict_resolver=self._llm_conflict_resolver,
        )

    def _create_retrieval_engine(self):
        from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
        return MemoryRetrievalEngine(
            chroma_manager=self._chroma_manager,
            rif_scorer=self._rif_scorer,
            emotion_analyzer=self._emotion_analyzer,
            session_manager=self._session_manager,
            llm_retrieval_router=self._llm_retrieval_router,
        )

    def _create_chat_history_buffer(self):
        from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer
        return ChatHistoryBuffer(
            max_messages=self.cfg.chat_context_count
        )

    def _create_member_identity(self):
        from iris_memory.utils.member_utils import set_identity_service
        from iris_memory.utils.member_identity_service import MemberIdentityService
        service = MemberIdentityService()
        set_identity_service(service)
        return service

    async def _init_activity_adaptive(self) -> None:
        """初始化场景自适应组件（活跃度追踪器 + 配置提供者）"""
        logger.info(LogTemplates.COMPONENT_INIT.format(component="activity adaptive"))
        
        from iris_memory.core.activity_config import GroupActivityTracker, ActivityAwareConfigProvider
        
        self._activity_tracker = GroupActivityTracker()
        
        if self._session_manager:
            self._session_manager._activity_tracker = self._activity_tracker
        
        enabled = self.cfg.enable_activity_adaptive
        self._activity_provider = self.cfg.init_activity_provider(
            tracker=self._activity_tracker,
            enabled=enabled,
        )
        
        status = "enabled" if enabled else "disabled"
        logger.info(f"Activity adaptive system {status}")

    async def _init_message_processing(self) -> None:
        """初始化分层消息处理组件"""
        logger.info(LogTemplates.COMPONENT_INIT.format(component="message processing"))
        
        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self.cfg.use_llm
        
        if not enable_batch:
            logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Batch processing"))
            return
        
        if use_llm:
            await self._init_llm_processor()
        
        self._init_message_classifier()
        
        logger.info("Message classifier initialized")

    async def _init_llm_processor(self):
        from iris_memory.processing.llm_processor import LLMMessageProcessor
        
        self._llm_processor = LLMMessageProcessor(
            astrbot_context=self.context,
            max_tokens=DEFAULTS.message_processing.llm_max_tokens_for_summary,
            provider_id=self.cfg.llm_provider_id
        )
        llm_ready = await self._llm_processor.initialize()
        if llm_ready and self._lifecycle_manager:
            self._lifecycle_manager.set_llm_provider(self._llm_processor)
            logger.info("LLM processor ready")
        else:
            logger.warning("LLM context not available, LLM features disabled")
            self._llm_processor = None

    def _init_message_classifier(self):
        from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
        
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

    async def _init_persona_extractor(self) -> None:
        """初始化画像提取器"""
        from pathlib import Path
        from iris_memory.analysis.persona.extractor import PersonaExtractor
        from iris_memory.analysis.persona.keyword_maps import KeywordMaps
        
        mode = self.cfg.persona_extraction_mode
        logger.info(f"Initializing persona extractor (mode={mode})")

        keyword_yaml = self.plugin_data_path.parent / "data" / "keyword_maps.yaml"
        if not keyword_yaml.exists():
            keyword_yaml = Path(__file__).resolve().parent.parent.parent / "data" / "keyword_maps.yaml"
        kw_maps = KeywordMaps(yaml_path=keyword_yaml if keyword_yaml.exists() else None)

        self._persona_extractor = PersonaExtractor(
            extraction_mode=mode,
            keyword_maps=kw_maps,
            astrbot_context=self.context if mode in ("llm", "hybrid") else None,
            llm_provider_id=self.cfg.persona_llm_provider,
            llm_max_tokens=self.cfg.persona_llm_max_tokens,
            llm_daily_limit=self.cfg.persona_llm_daily_limit,
            enable_interest=self.cfg.persona_enable_interest,
            enable_style=self.cfg.persona_enable_style,
            enable_preference=self.cfg.persona_enable_preference,
            fallback_to_rule=self.cfg.persona_fallback_to_rule,
        )
        logger.info(f"Persona extractor ready (mode={mode})")

    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        logger.info(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))
        
        if not self.cfg.proactive_reply_enabled:
            logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return
        
        reply_detector = self._get_reply_detector()
        event_queue = getattr(self.context, '_event_queue', None)
        
        if event_queue is None:
            logger.warning(
                "Cannot access context._event_queue, "
                "proactive reply event dispatch may not work"
            )
        
        await self._create_proactive_manager(reply_detector, event_queue)
        
        logger.info("Proactive reply components initialized")

    def _get_reply_detector(self):
        if self._llm_proactive_reply_detector:
            logger.info("Using LLM-enhanced proactive reply detector")
            return self._llm_proactive_reply_detector
        
        from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
        self._reply_detector = ProactiveReplyDetector(
            emotion_analyzer=self._emotion_analyzer,
            config={
                "high_emotion_threshold": DEFAULTS.proactive_reply.high_emotion_threshold,
                "question_threshold": DEFAULTS.proactive_reply.question_threshold
            }
        )
        return self._reply_detector

    async def _create_proactive_manager(self, reply_detector, event_queue):
        from iris_memory.proactive.proactive_manager import ProactiveReplyManager
        
        self._proactive_manager = ProactiveReplyManager(
            astrbot_context=self.context,
            reply_detector=reply_detector,
            event_queue=event_queue,
            config={
                "enable_proactive_reply": self.cfg.proactive_reply_enabled,
                "reply_cooldown": DEFAULTS.proactive_reply.cooldown_seconds,
                "max_daily_replies": self.cfg.proactive_reply_max_daily,
                "group_whitelist_mode": self.cfg.proactive_reply_group_whitelist_mode
            },
            config_manager=self.cfg
        )
        await self._proactive_manager.initialize()

    async def _init_image_analyzer(self) -> None:
        """初始化图片分析器"""
        logger.info(LogTemplates.COMPONENT_INIT.format(component="image analyzer"))
        
        if not self.cfg.image_analysis_enabled:
            logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Image analysis"))
            return
        
        from iris_memory.multimodal.image_analyzer import ImageAnalyzer
        
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
            },
            provider_id=self.cfg.image_analysis_provider_id
        )
        
        logger.info(f"Image analyzer initialized: mode={self.cfg.image_analysis_mode}")

    async def _init_llm_enhanced(self) -> None:
        """初始化LLM智能增强组件"""
        logger.info(LogTemplates.COMPONENT_INIT.format(component="LLM enhanced"))
        
        if not self.cfg.llm_enhanced_enabled:
            logger.info("LLM enhanced: all modules using rule mode")
            return
        
        from iris_memory.core.detection.llm_enhanced_base import DetectionMode
        from iris_memory.capture.detector.llm_sensitivity_detector import LLMSensitivityDetector
        from iris_memory.capture.detector.llm_trigger_detector import LLMTriggerDetector
        from iris_memory.analysis.emotion.llm_emotion_analyzer import LLMEmotionAnalyzer
        from iris_memory.proactive.llm_proactive_reply_detector import LLMProactiveReplyDetector
        from iris_memory.capture.conflict.llm_conflict_resolver import LLMConflictResolver
        from iris_memory.retrieval.llm_retrieval_router import LLMRetrievalRouter
        
        provider_id = self.cfg.llm_enhanced_provider_id
        modes = []
        
        if self.cfg.sensitivity_mode != "rule":
            self._llm_sensitivity_detector = LLMSensitivityDetector(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.sensitivity_mode),
            )
            modes.append(f"sensitivity={self.cfg.sensitivity_mode}")
        
        if self.cfg.trigger_mode != "rule":
            self._llm_trigger_detector = LLMTriggerDetector(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.trigger_mode),
            )
            modes.append(f"trigger={self.cfg.trigger_mode}")
        
        if self.cfg.emotion_mode != "rule":
            self._llm_emotion_analyzer = LLMEmotionAnalyzer(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.emotion_mode),
            )
            modes.append(f"emotion={self.cfg.emotion_mode}")
        
        if self.cfg.proactive_mode != "rule":
            self._llm_proactive_reply_detector = LLMProactiveReplyDetector(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.proactive_mode),
            )
            modes.append(f"proactive={self.cfg.proactive_mode}")
        
        if self.cfg.conflict_mode != "rule":
            self._llm_conflict_resolver = LLMConflictResolver(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.conflict_mode),
            )
            modes.append(f"conflict={self.cfg.conflict_mode}")
        
        if self.cfg.retrieval_mode != "rule":
            self._llm_retrieval_router = LLMRetrievalRouter(
                astrbot_context=self.context,
                provider_id=provider_id,
                mode=DetectionMode(self.cfg.retrieval_mode),
            )
            modes.append(f"retrieval={self.cfg.retrieval_mode}")
        
        if modes:
            logger.info(f"LLM enhanced enabled: {', '.join(modes)}")
        else:
            logger.info("LLM enhanced: all modules using rule mode")

    async def _init_batch_processor(self) -> None:
        """初始化批量处理器"""
        from iris_memory.capture.batch_processor import MessageBatchProcessor
        
        use_llm = self.cfg.use_llm
        
        batch_config = {
            "short_message_threshold": self.cfg.short_message_threshold,
            "merge_time_window": self.cfg.merge_time_window,
            "max_merge_count": self.cfg.max_merge_count,
            "llm_cooldown_seconds": 60,
            "summary_interval_seconds": 300,
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
            config=batch_config,
            config_manager=self.cfg
        )
        await self._batch_processor.start()
        
        logger.info(f"Batch processor initialized (threshold={threshold_count})")

    async def _apply_config(self) -> None:
        """应用配置到各组件"""
        self._apply_cache_config()
        self._apply_capture_engine_config()
        self._apply_session_manager_config()
        self._apply_lifecycle_manager_config()
        self._apply_retrieval_engine_config()
        self._apply_chat_history_buffer_config()

    def _apply_cache_config(self):
        from iris_memory.storage.cache import CacheManager
        
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

    def _apply_capture_engine_config(self):
        if self._capture_engine:
            self._capture_engine.set_config({
                "auto_capture": self.cfg.enable_memory,
                "min_confidence": DEFAULTS.memory.min_confidence,
                "rif_threshold": self.cfg.rif_threshold
            })

    def _apply_session_manager_config(self):
        if self._session_manager:
            self._session_manager.max_working_memory = self.cfg.max_working_memory
            self._session_manager.max_sessions = DEFAULTS.session.max_sessions
            self._session_manager.ttl = self.cfg.session_timeout

    def _apply_lifecycle_manager_config(self):
        if self._lifecycle_manager:
            self._lifecycle_manager.cleanup_interval = DEFAULTS.session.session_cleanup_interval
            self._lifecycle_manager.session_timeout = self.cfg.session_timeout
            self._lifecycle_manager.inactive_timeout = DEFAULTS.session.session_inactive_timeout

    def _apply_retrieval_engine_config(self):
        if self._retrieval_engine:
            self._retrieval_engine.set_config({
                "max_context_memories": self.cfg.max_context_memories,
                "enable_time_aware": DEFAULTS.llm_integration.enable_time_aware,
                "enable_emotion_aware": DEFAULTS.llm_integration.enable_emotion_aware,
                "enable_token_budget": self.cfg.enable_inject,
                "token_budget": self.cfg.token_budget,
                "coordination_strategy": DEFAULTS.llm_integration.coordination_strategy
            })

    def _apply_chat_history_buffer_config(self):
        if self._chat_history_buffer:
            self._chat_history_buffer.set_max_messages(self.cfg.chat_context_count)
