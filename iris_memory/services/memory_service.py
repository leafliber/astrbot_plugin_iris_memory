"""
记忆业务服务层 - 封装核心业务逻辑

职责：
1. 记忆捕获、存储、检索的业务逻辑
2. 组件协调与生命周期管理
3. 异常处理和错误回显

架构：
- 使用 Mixin 模式拆分功能模块
- initializers.py: 初始化逻辑
- business_operations.py: 业务操作
- persistence.py: 持久化逻辑
"""
from typing import Optional, List, Dict, Any
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
from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer, ChatMessage
from iris_memory.analysis.emotion.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
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
from iris_memory.analysis.persona.persona_logger import persona_log
from iris_memory.core.activity_config import GroupActivityTracker, ActivityAwareConfigProvider

from iris_memory.services.initializers import ServiceInitializer
from iris_memory.services.business_operations import BusinessOperations
from iris_memory.services.persistence import PersistenceOperations


class MemoryService(
    ServiceInitializer,
    BusinessOperations,
    PersistenceOperations
):
    """
    记忆业务服务层
    
    封装所有与记忆相关的业务逻辑，供Handler层调用
    
    使用 Mixin 模式组织代码：
    - ServiceInitializer: 初始化逻辑
    - BusinessOperations: 业务操作
    - PersistenceOperations: 持久化逻辑
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
        
        self._is_initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()
        
        self._init_core_attributes()
        self._init_component_attributes()
        self._init_optional_component_attributes()
        self._init_llm_enhanced_attributes()
        self._init_user_state_cache()

    def _init_core_attributes(self):
        """初始化核心属性"""
        self._user_emotional_states: Dict[str, EmotionalState] = {}
        self._user_personas: Dict[str, UserPersona] = {}
        self._recently_injected: Dict[str, List[str]] = {}
        self._max_recent_track: int = 20

    def _init_component_attributes(self):
        """初始化核心组件属性"""
        self._chroma_manager: Optional[ChromaManager] = None
        self._capture_engine: Optional[MemoryCaptureEngine] = None
        self._retrieval_engine: Optional[MemoryRetrievalEngine] = None
        self._session_manager: Optional[SessionManager] = None
        self._lifecycle_manager: Optional[SessionLifecycleManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._emotion_analyzer: Optional[EmotionAnalyzer] = None
        self._rif_scorer: Optional[RIFScorer] = None
        self._chat_history_buffer: Optional[ChatHistoryBuffer] = None

    def _init_optional_component_attributes(self):
        """初始化可选组件属性"""
        self._message_classifier = None
        self._batch_processor = None
        self._llm_processor = None
        self._reply_detector = None
        self._proactive_manager = None
        self._image_analyzer = None
        self._member_identity = None
        self._persona_extractor = None
        self._activity_tracker = None
        self._activity_provider = None

    def _init_llm_enhanced_attributes(self):
        """初始化LLM增强组件属性"""
        self._llm_sensitivity_detector = None
        self._llm_trigger_detector = None
        self._llm_emotion_analyzer = None
        self._llm_proactive_reply_detector = None
        self._llm_conflict_resolver = None
        self._llm_retrieval_router = None

    def _init_user_state_cache(self):
        """初始化用户状态缓存"""
        pass

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
    def batch_processor(self):
        return self._batch_processor
    
    @property
    def message_classifier(self):
        return self._message_classifier
    
    @property
    def image_analyzer(self):
        return self._image_analyzer
    
    @property
    def chat_history_buffer(self) -> Optional[ChatHistoryBuffer]:
        return self._chat_history_buffer
    
    @property
    def proactive_manager(self):
        return self._proactive_manager
    
    @property
    def emotion_analyzer(self) -> Optional[EmotionAnalyzer]:
        return self._emotion_analyzer
    
    @property
    def member_identity(self):
        return self._member_identity
    
    @property
    def activity_tracker(self):
        return self._activity_tracker
    
    @property
    def activity_provider(self):
        return self._activity_provider
    
    @property
    def llm_sensitivity_detector(self):
        return self._llm_sensitivity_detector
    
    @property
    def llm_trigger_detector(self):
        return self._llm_trigger_detector
    
    @property
    def llm_emotion_analyzer(self):
        return self._llm_emotion_analyzer
    
    @property
    def llm_proactive_reply_detector(self):
        return self._llm_proactive_reply_detector
    
    @property
    def llm_conflict_resolver(self):
        return self._llm_conflict_resolver
    
    @property
    def llm_retrieval_router(self):
        return self._llm_retrieval_router
    
    def is_embedding_ready(self) -> bool:
        """检查 embedding 系统是否就绪"""
        if not self._chroma_manager:
            return False
        return self._chroma_manager.embedding_manager.is_ready
    
    async def initialize(self) -> None:
        """异步初始化所有组件"""
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
