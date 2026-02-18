"""
记忆业务服务层 - 封装核心业务逻辑

职责：
1. 记忆捕获、存储、检索的业务逻辑
2. 组件协调与生命周期管理
3. 异常处理和错误回显
"""
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import asyncio

from astrbot.api.star import Context
from astrbot.api import AstrBotConfig, logger

from iris_memory.models.memory import Memory
from iris_memory.models.user_persona import UserPersona
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.capture.engine import MemoryCaptureEngine
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.storage.session_manager import SessionManager
from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
from iris_memory.storage.cache import CacheManager
from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer, ChatMessage
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
# MemoryInjector/InjectionMode/HookPriority 保留在 hook_manager 模块中
# 但本服务不再实例化 MemoryInjector，注入通过 req.system_prompt += 完成
from iris_memory.utils.logger import get_logger
from iris_memory.core.config_manager import ConfigManager, init_config_manager
from iris_memory.core.defaults import DEFAULTS
from iris_memory.core.constants import (
    SessionScope, PersonaStyle, SourceType,
    LogTemplates, KVStoreKeys, NumericDefaults, Separators
)
from iris_memory.core.types import StorageLayer
from iris_memory.utils.command_utils import SessionKeyBuilder
from iris_memory.utils.member_utils import format_member_tag, set_identity_service
from iris_memory.utils.member_identity_service import MemberIdentityService
from iris_memory.analysis.persona.logger import persona_log
from iris_memory.core.activity_config import GroupActivityTracker, ActivityAwareConfigProvider

# 可选组件（可能未启用）
from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
from iris_memory.capture.batch_processor import MessageBatchProcessor
from iris_memory.processing.llm_processor import LLMMessageProcessor
from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
from iris_memory.proactive.proactive_manager import ProactiveReplyManager
from iris_memory.multimodal.image_analyzer import ImageAnalyzer
from iris_memory.analysis.persona.extractor import PersonaExtractor
from iris_memory.analysis.persona.keyword_maps import KeywordMaps

# LLM增强组件
from iris_memory.core.detection.llm_enhanced_base import DetectionMode
from iris_memory.capture.detector.llm_sensitivity_detector import LLMSensitivityDetector
from iris_memory.capture.detector.llm_trigger_detector import LLMTriggerDetector
from iris_memory.analysis.emotion.llm_emotion_analyzer import LLMEmotionAnalyzer
from iris_memory.proactive.llm_proactive_reply_detector import LLMProactiveReplyDetector
from iris_memory.capture.conflict.llm_conflict_resolver import LLMConflictResolver
from iris_memory.retrieval.llm_retrieval_router import LLMRetrievalRouter


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
        
        # 初始化状态跟踪（支持热更新场景）
        self._is_initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()
        
        # 核心组件
        self._chroma_manager: Optional[ChromaManager] = None
        self._capture_engine: Optional[MemoryCaptureEngine] = None
        self._retrieval_engine: Optional[MemoryRetrievalEngine] = None
        self._session_manager: Optional[SessionManager] = None
        self._lifecycle_manager: Optional[SessionLifecycleManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._emotion_analyzer: Optional[EmotionAnalyzer] = None
        self._rif_scorer: Optional[RIFScorer] = None
        self._chat_history_buffer: Optional[ChatHistoryBuffer] = None
        
        # 可选组件
        self._message_classifier: Optional[MessageClassifier] = None
        self._batch_processor: Optional[MessageBatchProcessor] = None
        self._llm_processor: Optional[LLMMessageProcessor] = None
        self._reply_detector: Optional[ProactiveReplyDetector] = None
        self._proactive_manager: Optional[ProactiveReplyManager] = None
        self._image_analyzer: Optional[ImageAnalyzer] = None
        self._member_identity: Optional[MemberIdentityService] = None
        self._persona_extractor: Optional[PersonaExtractor] = None
        self._activity_tracker: Optional[GroupActivityTracker] = None
        self._activity_provider: Optional[ActivityAwareConfigProvider] = None
        
        # LLM增强组件
        self._llm_sensitivity_detector: Optional[LLMSensitivityDetector] = None
        self._llm_trigger_detector: Optional[LLMTriggerDetector] = None
        self._llm_emotion_analyzer: Optional[LLMEmotionAnalyzer] = None
        self._llm_proactive_reply_detector: Optional[LLMProactiveReplyDetector] = None
        self._llm_conflict_resolver: Optional[LLMConflictResolver] = None
        self._llm_retrieval_router: Optional[LLMRetrievalRouter] = None
        
        # 用户状态缓存
        self._user_emotional_states: Dict[str, EmotionalState] = {}
        self._user_personas: Dict[str, UserPersona] = {}
        
        # 最近注入记忆跟踪（用于避免重复提及）
        # session_key -> [memory_id, ...]
        self._recently_injected: Dict[str, List[str]] = {}
        self._max_recent_track: int = 20  # 每个会话跟踪最近20条注入记忆
    
    # ========== 属性访问器 ==========
    
    @property
    def is_initialized(self) -> bool:
        """检查服务是否已完成初始化（热更新场景）"""
        return self._is_initialized
    
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
    def chat_history_buffer(self) -> Optional[ChatHistoryBuffer]:
        return self._chat_history_buffer
    
    @property
    def proactive_manager(self) -> Optional[ProactiveReplyManager]:
        return self._proactive_manager
    
    @property
    def emotion_analyzer(self) -> Optional[EmotionAnalyzer]:
        return self._emotion_analyzer
    
    @property
    def member_identity(self) -> Optional[MemberIdentityService]:
        return self._member_identity
    
    @property
    def activity_tracker(self) -> Optional[GroupActivityTracker]:
        return self._activity_tracker
    
    @property
    def activity_provider(self) -> Optional[ActivityAwareConfigProvider]:
        return self._activity_provider
    
    @property
    def llm_sensitivity_detector(self) -> Optional[LLMSensitivityDetector]:
        return self._llm_sensitivity_detector
    
    @property
    def llm_trigger_detector(self) -> Optional[LLMTriggerDetector]:
        return self._llm_trigger_detector
    
    @property
    def llm_emotion_analyzer(self) -> Optional[LLMEmotionAnalyzer]:
        return self._llm_emotion_analyzer
    
    @property
    def llm_proactive_reply_detector(self) -> Optional[LLMProactiveReplyDetector]:
        return self._llm_proactive_reply_detector
    
    @property
    def llm_conflict_resolver(self) -> Optional[LLMConflictResolver]:
        return self._llm_conflict_resolver
    
    @property
    def llm_retrieval_router(self) -> Optional[LLMRetrievalRouter]:
        return self._llm_retrieval_router
    
    def is_embedding_ready(self) -> bool:
        """检查 embedding 系统是否就绪
        
        Returns:
            bool: embedding 提供者是否已加载完成，可正常使用
        """
        if not self._chroma_manager:
            return False
        return self._chroma_manager.embedding_manager.is_ready
    
    # ========== 初始化方法 ==========
    
    async def initialize(self) -> None:
        """异步初始化所有组件
        
        使用 asyncio.Lock 防止并发初始化（热更新场景下可能发生）。
        初始化完成后设置 _is_initialized = True。
        """
        async with self._init_lock:
            try:
                self._is_initialized = False
                self.plugin_data_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(LogTemplates.PLUGIN_INIT_START)
                
                await self._init_llm_enhanced()
                await self._init_core_components()
                await self._init_activity_adaptive()
                await self._init_message_processing()
                await self._init_persona_extractor()
                await self._init_proactive_reply()
                await self._init_image_analyzer()
                await self._apply_config()
                await self._init_batch_processor()
                
                self._is_initialized = True
                self.logger.info(LogTemplates.PLUGIN_INIT_SUCCESS)
                
            except Exception as e:
                self._is_initialized = False
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
        
        # 会话管理器（activity_tracker 延迟设置，在 _init_activity_adaptive 中完成）
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
            rif_scorer=self._rif_scorer,
            llm_sensitivity_detector=self._llm_sensitivity_detector,
            llm_trigger_detector=self._llm_trigger_detector,
            llm_conflict_resolver=self._llm_conflict_resolver,
        )
        
        # 检索引擎
        self._retrieval_engine = MemoryRetrievalEngine(
            chroma_manager=self._chroma_manager,
            rif_scorer=self._rif_scorer,
            emotion_analyzer=self._emotion_analyzer,
            session_manager=self._session_manager,
            llm_retrieval_router=self._llm_retrieval_router,
        )
        
        # 聊天记录缓冲区
        self._chat_history_buffer = ChatHistoryBuffer(
            max_messages=self.cfg.chat_context_count
        )
        
        # 成员身份服务
        self._member_identity = MemberIdentityService()
        set_identity_service(self._member_identity)
        self.logger.info("MemberIdentityService initialized")
    
    async def _init_activity_adaptive(self) -> None:
        """初始化场景自适应组件（活跃度追踪器 + 配置提供者）"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="activity adaptive"))
        
        # 创建活跃度追踪器
        self._activity_tracker = GroupActivityTracker()
        
        # 将追踪器注入 SessionManager
        if self._session_manager:
            self._session_manager._activity_tracker = self._activity_tracker
        
        # 通过 ConfigManager 初始化配置提供者
        enabled = self.cfg.enable_activity_adaptive
        self._activity_provider = self.cfg.init_activity_provider(
            tracker=self._activity_tracker,
            enabled=enabled,
        )
        
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"Activity adaptive system {status}")
    
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
                max_tokens=DEFAULTS.message_processing.llm_max_tokens_for_summary,
                provider_id=self.cfg.llm_provider_id
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
    
    async def _init_persona_extractor(self) -> None:
        """初始化画像提取器"""
        mode = self.cfg.persona_extraction_mode
        self.logger.info(f"Initializing persona extractor (mode={mode})")

        # 加载外置关键词配置
        keyword_yaml = self.plugin_data_path.parent / "data" / "keyword_maps.yaml"
        if not keyword_yaml.exists():
            # 尝试插件根目录
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
        self.logger.info(f"Persona extractor ready (mode={mode})")

    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="proactive reply"))
        
        if not self.cfg.proactive_reply_enabled:
            self.logger.info(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Proactive reply"))
            return
        
        reply_detector = None
        if self._llm_proactive_reply_detector:
            reply_detector = self._llm_proactive_reply_detector
            self.logger.info("Using LLM-enhanced proactive reply detector")
        else:
            self._reply_detector = ProactiveReplyDetector(
                emotion_analyzer=self._emotion_analyzer,
                config={
                    "high_emotion_threshold": DEFAULTS.proactive_reply.high_emotion_threshold,
                    "question_threshold": DEFAULTS.proactive_reply.question_threshold
                }
            )
            reply_detector = self._reply_detector
        
        event_queue = getattr(self.context, '_event_queue', None)
        if event_queue is None:
            self.logger.warning(
                "Cannot access context._event_queue, "
                "proactive reply event dispatch may not work"
            )
        
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
            },
            provider_id=self.cfg.image_analysis_provider_id
        )
        
        self.logger.info(f"Image analyzer initialized: mode={self.cfg.image_analysis_mode}")
    
    async def _init_llm_enhanced(self) -> None:
        """初始化LLM智能增强组件"""
        self.logger.info(LogTemplates.COMPONENT_INIT.format(component="LLM enhanced"))
        
        if not self.cfg.llm_enhanced_enabled:
            self.logger.info("LLM enhanced: all modules using rule mode")
            return
        
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
            self.logger.info(f"LLM enhanced enabled: {', '.join(modes)}")
        else:
            self.logger.info("LLM enhanced: all modules using rule mode")
    
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
            config=batch_config,
            config_manager=self.cfg
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
        
        # 配置聊天记录缓冲区
        if self._chat_history_buffer:
            self._chat_history_buffer.set_max_messages(self.cfg.chat_context_count)
    
    # ========== 业务方法 ==========
    
    async def capture_and_store_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        is_user_requested: bool = False,
        context: Optional[Dict[str, Any]] = None,
        sender_name: Optional[str] = None
    ) -> Optional[Memory]:
        """
        捕获并存储记忆
        
        Args:
            message: 消息内容
            user_id: 用户ID
            group_id: 群聊ID
            is_user_requested: 是否用户主动请求
            context: 额外上下文
            sender_name: 发送者显示名称
            
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
                context=context,
                sender_name=sender_name
            )
            
            if not memory:
                return None
            
            # 分层存储
            await self._store_memory_by_layer(memory)
            
            # 画像闭环：从新捕获的记忆更新用户画像
            await self._update_persona_from_memory(memory, user_id)
            
            return memory
            
        except Exception as e:
            self.logger.warning(f"Failed to capture memory: {e}")
            return None
    
    async def _store_memory_by_layer(self, memory: Memory) -> None:
        """根据层级存储记忆"""
        if memory.storage_layer == StorageLayer.WORKING:
            if self._session_manager:
                await self._session_manager.add_working_memory(memory)
        else:
            if self._chroma_manager:
                await self._chroma_manager.add_memory(memory)

    async def _update_persona_from_memory(self, memory: Memory, user_id: str) -> None:
        """画像闭环：从记忆更新用户画像并记录 DEBUG 日志
        
        在每次成功捕获记忆后调用，使画像始终与最新记忆保持同步。
        支持三种模式：rule / llm / hybrid，由 persona.extraction_mode 配置控制。
        """
        try:
            persona = self.get_or_create_user_persona(user_id)
            mem_id = getattr(memory, "id", None)
            persona_log.update_start(user_id, mem_id)

            changes = []
            content = getattr(memory, "content", "") or ""
            summary = getattr(memory, "summary", None)
            mem_type_raw = getattr(memory, "type", None)
            mem_type = mem_type_raw.value if hasattr(mem_type_raw, "value") else str(mem_type_raw)
            confidence = getattr(memory, "confidence", 0.5)

            # 情感维度始终使用内置规则（与 LLM 提取无关）
            if mem_type in ("emotion",):
                changes.extend(persona._update_emotional(memory, mem_id, confidence))

            # 事实 / 关系 / 交互维度 → 使用 PersonaExtractor
            if self._persona_extractor and self.cfg.persona_extraction_mode != "rule":
                # 使用提取器（llm / hybrid）
                result = await self._persona_extractor.extract(
                    content=content,
                    summary=summary,
                )
                if result.confidence > 0 or result.interests:
                    ext_changes = persona.apply_extraction_result(
                        result,
                        source_memory_id=mem_id,
                        memory_type=mem_type,
                        base_confidence=confidence,
                    )
                    changes.extend(ext_changes)
                else:
                    # 提取器无结果时走传统规则（纯兜底）
                    changes.extend(persona.update_from_memory(memory))
            else:
                # rule 模式 → 原始行为
                changes.extend(persona.update_from_memory(memory))

            # 更新活跃时段
            created = getattr(memory, "created_time", None)
            if created and isinstance(created, datetime):
                persona.hourly_distribution[created.hour] += 1.0

            if changes:
                persona_log.update_applied(
                    user_id,
                    [c.to_dict() for c in changes]
                )
                self.logger.debug(
                    f"Persona updated for user={user_id}: "
                    f"{len(changes)} change(s) from memory={mem_id}"
                )
            else:
                persona_log.update_skipped(user_id, "no_applicable_changes")

        except Exception as e:
            persona_log.update_error(user_id, e)
            self.logger.warning(f"Failed to update persona from memory: {e}")
    
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
        image_context: str = "",
        sender_name: Optional[str] = None
    ) -> str:
        """
        准备LLM上下文（包含聊天记录+记忆+图片）
        
        Args:
            query: 查询内容
            user_id: 用户ID
            group_id: 群聊ID
            image_context: 图片上下文
            sender_name: 当前发言者名称
            
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
            
            # 过滤最近已注入过的记忆（避免重复提及）
            session_key = SessionKeyBuilder.build(user_id, group_id)
            if memories:
                memories = self._filter_recently_injected(memories, session_key)
            
            context_parts = []
            
            # 1. 注入近期聊天记录（最高优先级 —— 让AI了解当前话题）
            chat_context = await self._build_chat_history_context(
                user_id, group_id
            )
            if chat_context:
                context_parts.append(chat_context)
            
            # 2. 添加记忆上下文（带scope和sender标注）
            if memories:
                # 获取用户画像注入视图
                persona = self.get_or_create_user_persona(user_id)
                persona_view = persona.to_injection_view()
                persona_log.inject_view(user_id, persona_view)

                memory_context = self._retrieval_engine.format_memories_for_llm(
                    memories,
                    persona_style=PersonaStyle.NATURAL,
                    user_persona=persona_view,
                    group_id=group_id,
                    current_sender_name=sender_name
                )
                context_parts.append(memory_context)
                self.logger.debug(LogTemplates.MEMORY_INJECTED.format(count=len(memories)))
                
                # 记录本次注入的记忆ID
                self._track_injected_memories(
                    session_key,
                    [m.id for m in memories]
                )

            # 添加群成员识别提示
            member_context = self._build_member_identity_context(
                memories,
                group_id,
                user_id,
                sender_name
            )
            if member_context:
                context_parts.append(member_context)
            
            # 添加图片上下文
            if image_context:
                context_parts.append(image_context)
                self.logger.debug("Injected image context into LLM prompt")
            
            # 添加行为指导（防止重复提及和过度反问）
            behavior_directives = self._build_behavior_directives(
                group_id,
                sender_name
            )
            if behavior_directives:
                context_parts.append(behavior_directives)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare LLM context: {e}")
            return ""
    
    def _filter_recently_injected(
        self,
        memories: List[Memory],
        session_key: str
    ) -> List[Memory]:
        """过滤最近已注入过的记忆，避免重复提及同一件事
        
        保留策略：
        - 如果过滤后没有剩余记忆，则保留原始列表（避免完全无上下文）
        - 最近3次注入中出现过的记忆会被降权/过滤
        
        Args:
            memories: 候选记忆列表
            session_key: 会话键
            
        Returns:
            List[Memory]: 过滤后的记忆列表
        """
        recent_ids = set(self._recently_injected.get(session_key, []))
        if not recent_ids:
            return memories
        
        filtered = [m for m in memories if m.id not in recent_ids]
        
        # 如果过滤后没有记忆了，返回原始列表（但最多返回一半，减少重复）
        if not filtered:
            return memories[:max(1, len(memories) // 2)]
        
        return filtered
    
    def _track_injected_memories(self, session_key: str, memory_ids: List[str]) -> None:
        """记录本次注入的记忆ID
        
        Args:
            session_key: 会话键
            memory_ids: 注入的记忆ID列表
        """
        if session_key not in self._recently_injected:
            self._recently_injected[session_key] = []
        
        self._recently_injected[session_key].extend(memory_ids)
        
        # 保留最近N条
        if len(self._recently_injected[session_key]) > self._max_recent_track:
            self._recently_injected[session_key] = \
                self._recently_injected[session_key][-self._max_recent_track:]
    
    def _build_behavior_directives(
        self,
        group_id: Optional[str],
        sender_name: Optional[str] = None
    ) -> str:
        """构建行为指导，与人格Prompt协同工作
        
        解决以下问题：
        1. 群聊知识/个人知识区分不清
        2. 重复提及同一件事
        3. 反问过于频繁且重复
        4. 与人格协调
        
        群成员识别细节由 _build_member_identity_context 提供。
        
        Args:
            group_id: 群组ID
            sender_name: 当前发言者名称（保留用于未来扩展）
            
        Returns:
            str: 行为指导文本
        """
        directives = []
        
        directives.append("【记忆使用规则】")
        
        # 1. 防止重复提及
        directives.append("◆ 禁止重复：不要反复提起同一件事或记忆。如果你刚才已经提到过某个话题，就自然地聊别的，不要翻来覆去说同一件事。")
        
        # 2. 防止过度反问
        directives.append("◆ 减少反问：不要频繁反问对方，尤其不要重复问同一个问题。用陈述、共鸣、接话的方式回应，像真人朋友那样自然接话。如果想了解更多，偶尔问一下就够了。")
        
        # 3. 回复风格
        directives.append("◆ 简短自然：回复尽量简短，像群里随手接话，一行结束。不要写长篇大论，不要列清单式回答日常闲聊。")
        
        if group_id:
            # 4. 群聊知识区分
            directives.append("◆ 知识区分：记忆中标注了「群聊共识」和「个人信息」。群聊共识是大家都知道的事，个人信息是某个人的私事。引用个人信息时要确认是当前对话者的，不要张冠李戴。")
            
            # 成员识别细节由 _build_member_identity_context 提供
        else:
            # 私聊场景
            directives.append("◆ 这是私聊对话，记忆都是你和对方之间的。")
        
        return "\n".join(directives)

    def _build_member_identity_context(
        self,
        memories: List[Memory],
        group_id: Optional[str],
        user_id: str,
        sender_name: Optional[str]
    ) -> str:
        """Build a compact member identity hint for group chats.

        增强功能：
        - 使用 MemberIdentityService 获取稳定标签
        - 注入群成员列表（最近活跃的前10位）
        - 标注名称变更提示
        """
        if not group_id:
            return ""

        current_tag = format_member_tag(sender_name, user_id, group_id)
        other_tags = []
        seen = set()

        for memory in memories:
            tag = format_member_tag(memory.sender_name, memory.user_id, group_id)
            if not tag:
                continue
            if tag == current_tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            other_tags.append(tag)

        lines = [
            "【群成员识别】",
            f"当前对话者: {current_tag}。回复时针对这个人，不要混淆成其他群友。",
        ]

        if other_tags:
            lines.append("记忆中涉及成员: " + ", ".join(other_tags[:5]))

        # 注入群成员列表（通过 MemberIdentityService）
        if self._member_identity:
            all_members = self._member_identity.get_group_members(group_id)
            # 过滤掉当前对话者和已列出的成员
            extra_members = [
                m for m in all_members
                if m != current_tag and m not in seen
            ]
            if extra_members:
                lines.append(
                    "群内其他已知成员: " + ", ".join(extra_members[:10])
                )

            # 名称变更提示
            history = self._member_identity.get_name_history(user_id)
            if history:
                last_change = history[-1]
                lines.append(
                    f"注意: 当前对话者曾用名 \"{last_change['old_name']}\"，"
                    f"现在叫 \"{last_change['new_name']}\"。"
                )

        lines.append(
            "同名以#后ID区分。不要把A说的话当成B说的，"
            "引用其他人的记忆时要明确说明。"
        )

        return "\n".join(lines)
    
    async def analyze_images(
        self,
        message_chain: List[Any],
        user_id: str,
        group_id: Optional[str],
        context_text: str,
        umo: str,
        session_id: str
    ) -> Tuple[str, str]:
        """
        分析图片
        
        Args:
            message_chain: 消息链
            user_id: 用户ID
            group_id: 群聊ID（用于动态预算）
            context_text: 上下文文本
            umo: 统一消息来源
            session_id: 会话ID
            
        Returns:
            Tuple[str, str]: (LLM上下文格式, 记忆格式)
        """
        if not self._image_analyzer:
            return "", ""
        
        try:
            daily_budget = self.cfg.get_daily_analysis_budget(group_id)
            effective_daily_budget = daily_budget if daily_budget > 0 else 999999

            image_results = await self._image_analyzer.analyze_message_images(
                message_chain=message_chain,
                user_id=user_id,
                context_text=context_text,
                umo=umo,
                session_id=session_id,
                daily_analysis_budget=effective_daily_budget,
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
                sender_name = context.get("sender_name")
                await self._handle_immediate_memory(
                    full_message, user_id, group_id, classification, sender_name
                )
            else:
                # BATCH 层
                sender_name = context.get("sender_name")
                await self._batch_processor.add_message(
                    content=full_message,
                    user_id=user_id,
                    sender_name=sender_name,
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
        classification: Any,
        sender_name: Optional[str] = None
    ) -> None:
        """处理立即层级的记忆"""
        memory = await self.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context={
                "classification": classification.metadata,
                "source": classification.source
            },
            sender_name=sender_name
        )
        
        if memory:
            self.logger.debug(LogTemplates.IMMEDIATE_MEMORY_CAPTURED.format(memory_id=memory.id))
    
    # ========== 聊天记录缓冲 ==========
    
    async def record_chat_message(
        self,
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        group_id: Optional[str] = None,
        is_bot: bool = False,
        session_user_id: Optional[str] = None
    ) -> None:
        """记录一条聊天消息到缓冲区
        
        在 on_all_messages 和 on_llm_response 中调用，
        确保Bot参与和未参与的消息都被记录。
        
        Args:
            sender_id: 发送者ID
            sender_name: 发送者昵称
            content: 消息内容
            group_id: 群组ID
            is_bot: 是否为Bot的消息
            session_user_id: 用于定位缓冲区的用户ID（Bot私聊回复时需要）
        """
        if self._chat_history_buffer:
            await self._chat_history_buffer.add_message(
                sender_id=sender_id,
                sender_name=sender_name,
                content=content,
                group_id=group_id,
                is_bot=is_bot,
                session_user_id=session_user_id
            )
    
    async def _build_chat_history_context(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> str:
        """构建聊天记录上下文
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            str: 格式化的聊天记录，为空则返回空字符串
        """
        if not self._chat_history_buffer:
            return ""

        chat_context_count = self.cfg.get_chat_context_count(group_id)
        if chat_context_count <= 0:
            return ""

        if self._chat_history_buffer.max_messages < chat_context_count:
            self._chat_history_buffer.set_max_messages(chat_context_count)
        
        messages = await self._chat_history_buffer.get_recent_messages(
            user_id=user_id,
            group_id=group_id,
            limit=chat_context_count
        )
        
        if not messages:
            return ""
        
        context = self._chat_history_buffer.format_for_llm(
            messages,
            group_id=group_id
        )
        
        if context:
            self.logger.debug(
                f"Injected {len(messages)} chat messages into context "
                f"(group={group_id is not None})"
            )
        
        return context
    
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
            
            # 加载聊天记录缓冲区
            if self._chat_history_buffer:
                chat_history = await get_kv_data(KVStoreKeys.CHAT_HISTORY, {})
                if chat_history:
                    await self._chat_history_buffer.deserialize(chat_history)
                    self.logger.info("Loaded chat history buffer")
            
            # 加载主动回复白名单
            if self._proactive_manager:
                whitelist_data = await get_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, [])
                if whitelist_data:
                    self._proactive_manager.deserialize_whitelist(whitelist_data)
                    self.logger.info("Loaded proactive reply whitelist")
            
            # 加载成员身份数据
            if self._member_identity:
                identity_data = await get_kv_data(KVStoreKeys.MEMBER_IDENTITY, {})
                if identity_data:
                    self._member_identity.deserialize(identity_data)
                    stats = self._member_identity.get_stats()
                    self.logger.info(
                        f"Loaded member identity data: "
                        f"{stats['total_profiles']} profiles, "
                        f"{stats['total_groups']} groups"
                    )
            
            # 加载群活跃度数据
            if self._activity_tracker:
                activity_data = await get_kv_data(KVStoreKeys.GROUP_ACTIVITY, {})
                if activity_data:
                    self._activity_tracker.deserialize(activity_data)
                    self.logger.info("Loaded group activity states")
            
            # 加载用户画像
            personas_data = await get_kv_data(KVStoreKeys.USER_PERSONAS, {})
            if personas_data:
                persona_log.restore_start(len(personas_data))
                for uid, pdata in personas_data.items():
                    try:
                        self._user_personas[uid] = UserPersona.from_dict(pdata)
                        persona_log.restore_ok(uid)
                    except Exception as e:
                        persona_log.restore_error(uid, e)
                self.logger.info(f"Loaded {len(self._user_personas)} user personas")
            
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
            
            # 保存聊天记录缓冲区
            if self._chat_history_buffer:
                chat_history = await self._chat_history_buffer.serialize()
                await put_kv_data(KVStoreKeys.CHAT_HISTORY, chat_history)
                self.logger.info("Saved chat history buffer")
            
            # 保存主动回复白名单
            if self._proactive_manager:
                whitelist_data = self._proactive_manager.serialize_whitelist()
                await put_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, whitelist_data)
                self.logger.info("Saved proactive reply whitelist")
            
            # 保存成员身份数据
            if self._member_identity:
                identity_data = self._member_identity.serialize()
                await put_kv_data(KVStoreKeys.MEMBER_IDENTITY, identity_data)
                self.logger.info("Saved member identity data")
            
            # 保存群活跃度数据
            if self._activity_tracker:
                activity_data = self._activity_tracker.serialize()
                await put_kv_data(KVStoreKeys.GROUP_ACTIVITY, activity_data)
                self.logger.info("Saved group activity states")
            
            # 保存用户画像
            if self._user_personas:
                personas_data = {}
                for uid, persona in self._user_personas.items():
                    try:
                        persona_log.persist_start(uid)
                        personas_data[uid] = persona.to_dict()
                        persona_log.persist_ok(uid, persona.update_count)
                    except Exception as e:
                        persona_log.persist_error(uid, e)
                await put_kv_data(KVStoreKeys.USER_PERSONAS, personas_data)
                self.logger.info(f"Saved {len(personas_data)} user personas")
                
        except Exception as e:
            self.logger.warning(f"Failed to save to KV: {e}")
    
    async def _save_batch_queues(self) -> None:
        """保存批量队列（供回调使用）"""
        # 实际保存由 Star 类的 _save_sessions 处理
        pass
    
    # ========== 销毁方法 ==========
    
    async def terminate(self) -> None:
        """销毁服务
        
        热更新友好：
        1. 立即标记为未初始化，阻止新操作进入
        2. 按依赖顺序停止后台任务（先停消费者，再停生产者）
        3. 等待所有任务完成
        4. 清理全局状态引用
        5. 关闭底层存储
        """
        self.logger.info("[Hot-Reload] Terminating memory service...")
        
        # 1. 立即标记为未初始化，阻止新请求
        self._is_initialized = False
        
        try:
            # 2. 停止后台任务（按依赖顺序：消费者 → 生产者）
            # 批量处理器（消费消息队列）
            if self._batch_processor:
                try:
                    await self._batch_processor.stop()
                    self.logger.debug("[Hot-Reload] Batch processor stopped")
                except Exception as e:
                    self.logger.warning(f"[Hot-Reload] Error stopping batch processor: {e}")
            
            # 主动回复管理器
            if self._proactive_manager:
                try:
                    await self._proactive_manager.stop()
                    self.logger.debug("[Hot-Reload] Proactive manager stopped")
                except Exception as e:
                    self.logger.warning(f"[Hot-Reload] Error stopping proactive manager: {e}")
            
            # 生命周期管理器（清理/升级定时任务）
            if self._lifecycle_manager:
                try:
                    await self._lifecycle_manager.stop()
                    self.logger.debug("[Hot-Reload] Lifecycle manager stopped")
                except Exception as e:
                    self.logger.warning(f"[Hot-Reload] Error stopping lifecycle manager: {e}")
            
            # 3. 清理全局状态引用
            set_identity_service(None)
            self.logger.debug("[Hot-Reload] Global identity service cleared")
            
            # 4. 关闭底层存储
            if self._chroma_manager:
                try:
                    await self._chroma_manager.close()
                    self.logger.debug("[Hot-Reload] Chroma manager closed")
                except Exception as e:
                    self.logger.warning(f"[Hot-Reload] Error closing Chroma manager: {e}")
            
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
