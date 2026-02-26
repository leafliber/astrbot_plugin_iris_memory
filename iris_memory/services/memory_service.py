"""
记忆业务服务层 - 封装核心业务逻辑

架构重构后，MemoryService 持有 7 个 Feature Module 而非 20+ 个原子组件：
- StorageModule:      存储基础设施
- AnalysisModule:     分析能力
- LLMEnhancedModule:  LLM 增强检测器 + LLM 处理器
- CaptureModule:      记忆捕获
- RetrievalModule:    记忆检索
- ProactiveModule:    主动回复
- KnowledgeGraphModule: 知识图谱（实体关系 + 多跳推理）

保留 Mixin 拆分：
- initializers.py:        初始化逻辑
- business_operations.py: 业务操作
- persistence.py:         持久化逻辑
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio

from astrbot.api.star import Context
from astrbot.api import AstrBotConfig

from iris_memory.models.user_persona import UserPersona
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.logger import get_logger
from iris_memory.utils.bounded_dict import BoundedDict
from iris_memory.core.config_manager import init_config_manager
from iris_memory.core.constants import LogTemplates
from iris_memory.utils.member_utils import set_identity_service
from iris_memory.utils.member_identity_service import MemberIdentityService

from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.proactive_module import ProactiveModule
from iris_memory.services.modules.kg_module import KnowledgeGraphModule

from iris_memory.services.initializers import ServiceInitializer
from iris_memory.services.business_operations import BusinessOperations
from iris_memory.services.persistence import PersistenceOperations


class MemoryService(
    ServiceInitializer,
    BusinessOperations,
    PersistenceOperations,
):
    """
    记忆业务服务层

    持有 7 个 Feature Module + 少量共享状态，
    通过 Mixin 拆分初始化 / 业务 / 持久化逻辑。
    """

    def __init__(self, context: Context, config: AstrBotConfig, plugin_data_path: Path):
        self.context = context
        self.config = config
        self.plugin_data_path = plugin_data_path
        self.cfg = init_config_manager(config)
        self.logger = get_logger("memory_service")

        self._is_initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._module_init_status: Dict[str, bool] = {}  # 各模块初始化状态

        # ── Feature Modules ──
        self.storage = StorageModule()
        self.analysis = AnalysisModule()
        self.llm_enhanced = LLMEnhancedModule()
        self.capture = CaptureModule()
        self.retrieval = RetrievalModule()
        self.proactive = ProactiveModule()
        self.kg = KnowledgeGraphModule()

        # ── 共享状态（有界 LRU，防止内存无限增长）──
        self._user_emotional_states: BoundedDict[str, EmotionalState] = BoundedDict(max_size=2000)
        self._user_personas: BoundedDict[str, UserPersona] = BoundedDict(max_size=2000)
        self._recently_injected: BoundedDict[str, List[str]] = BoundedDict(max_size=2000)
        self._max_recent_track: int = 20

        # ── 独立组件（不适合归入 Module 的辅助组件） ──
        self._member_identity: Optional[MemberIdentityService] = None
        self._image_analyzer: Optional[Any] = None
        self._activity_tracker: Optional[Any] = None
        self._activity_provider: Optional[Any] = None

    # ================================================================
    # 向后兼容属性
    # 这些属性代理到 Module，让 main.py / 旧代码无需修改即可访问
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def chroma_manager(self):
        return self.storage.chroma_manager

    @property
    def capture_engine(self):
        return self.capture.capture_engine

    @property
    def retrieval_engine(self):
        return self.retrieval.retrieval_engine

    @property
    def session_manager(self):
        return self.storage.session_manager

    @property
    def lifecycle_manager(self):
        return self.storage.lifecycle_manager

    @property
    def batch_processor(self):
        return self.capture.batch_processor

    @property
    def message_classifier(self):
        return self.capture.message_classifier

    @property
    def image_analyzer(self):
        return self._image_analyzer

    @property
    def chat_history_buffer(self):
        # 优先使用直接属性（测试兼容），否则代理到模块
        if '_chat_history_buffer' in self.__dict__:
            return self._chat_history_buffer
        return self.storage.chat_history_buffer

    @property
    def proactive_manager(self):
        return self.proactive.proactive_manager

    @property
    def emotion_analyzer(self):
        return self.analysis.emotion_analyzer

    @property
    def persona_extractor(self):
        return self.analysis.persona_extractor

    @property
    def _persona_batch_processor(self):
        return self.analysis.persona_batch_processor

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
        return self.llm_enhanced.sensitivity_detector

    @property
    def llm_trigger_detector(self):
        return self.llm_enhanced.trigger_detector

    @property
    def llm_emotion_analyzer(self):
        return self.llm_enhanced.emotion_analyzer

    @property
    def llm_proactive_reply_detector(self):
        return self.llm_enhanced.proactive_reply_detector

    @property
    def llm_conflict_resolver(self):
        return self.llm_enhanced.conflict_resolver

    @property
    def llm_retrieval_router(self):
        return self.llm_enhanced.retrieval_router

    def is_embedding_ready(self) -> bool:
        """检查 embedding 系统是否就绪"""
        return self.storage.is_embedding_ready()

    def health_check(self) -> Dict[str, Any]:
        """统一健康检查

        返回各模块和关键组件的运行状态，便于运维排查。

        Returns:
            Dict 包含:
              - status: "healthy" | "degraded" | "unhealthy"
              - initialized: bool
              - modules: Dict[str, bool]  各模块初始化状态
              - components: Dict[str, str]  关键组件可用性
        """
        components: Dict[str, str] = {}

        # 存储层
        components["chroma_manager"] = (
            "available" if self.chroma_manager else "unavailable"
        )
        components["session_manager"] = (
            "available" if self.session_manager else "unavailable"
        )
        components["embedding"] = (
            "ready" if self.is_embedding_ready() else "not_ready"
        )

        # 捕获 / 检索
        components["capture_engine"] = (
            "available" if self.capture_engine else "unavailable"
        )
        components["retrieval_engine"] = (
            "available" if self.retrieval_engine else "unavailable"
        )
        components["batch_processor"] = (
            "running" if self.batch_processor else "unavailable"
        )
        components["persona_batch_processor"] = (
            "running" if (self._persona_batch_processor
                         and self._persona_batch_processor.is_running)
            else "unavailable"
        )

        # 增强模块
        components["knowledge_graph"] = (
            "enabled" if (self.kg and self.kg.enabled) else "disabled"
        )
        components["proactive_manager"] = (
            "enabled" if self.proactive_manager else "disabled"
        )
        components["image_analyzer"] = (
            "enabled" if self.image_analyzer else "disabled"
        )

        # 综合状态判定
        failed_modules = [
            k for k, v in self._module_init_status.items() if not v
        ]
        if not self._is_initialized:
            status = "unhealthy"
        elif failed_modules:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "initialized": self._is_initialized,
            "modules": dict(self._module_init_status),
            "components": components,
            "failed_modules": failed_modules,
        }

    async def initialize(self) -> None:
        """异步初始化所有组件
        
        采用分阶段初始化策略：
        - 核心组件（storage/capture/retrieval）失败则整体失败
        - 增强组件（KG/proactive/image）失败则降级运行
        """
        async with self._init_lock:
            try:
                self._is_initialized = False
                self._module_init_status.clear()
                self.plugin_data_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(LogTemplates.PLUGIN_INIT_START)

                # ── 阶段1: 核心组件（失败则整体失败） ──
                await self._init_llm_enhanced()
                self._module_init_status["llm_enhanced"] = True

                await self._init_core_components()
                self._module_init_status["core"] = True

                # ── 阶段2: 增强组件（失败则降级运行） ──

                # 初始化记忆升级评估器
                try:
                    await self._init_upgrade_evaluator()
                    self._module_init_status["upgrade_evaluator"] = True
                except Exception as e:
                    self._module_init_status["upgrade_evaluator"] = False
                    self.logger.warning(
                        f"Upgrade evaluator initialization failed, using rule mode: {e}",
                        exc_info=True,
                    )

                for name, init_fn in [
                    ("knowledge_graph", self._init_knowledge_graph),
                    ("activity_adaptive", self._init_activity_adaptive),
                    ("message_processing", self._init_message_processing),
                    ("persona_extractor", self._init_persona_extractor),
                    ("proactive_reply", self._init_proactive_reply),
                    ("image_analyzer", self._init_image_analyzer),
                ]:
                    try:
                        await init_fn()
                        self._module_init_status[name] = True
                    except Exception as e:
                        self._module_init_status[name] = False
                        self.logger.warning(
                            f"Module '{name}' initialization failed, running in degraded mode: {e}",
                            exc_info=True,
                        )

                await self._apply_config()
                self._module_init_status["config"] = True

                # 批量处理器依赖其他组件，单独处理
                try:
                    await self._init_batch_processor()
                    self._module_init_status["batch_processor"] = True
                except Exception as e:
                    self._module_init_status["batch_processor"] = False
                    self.logger.warning(
                        f"Batch processor initialization failed: {e}",
                        exc_info=True,
                    )

                # 画像批量处理器依赖 persona_extractor
                try:
                    await self._init_persona_batch_processor()
                    self._module_init_status["persona_batch_processor"] = True
                except Exception as e:
                    self._module_init_status["persona_batch_processor"] = False
                    self.logger.warning(
                        f"Persona batch processor initialization failed: {e}",
                        exc_info=True,
                    )

                self._is_initialized = True

                failed = [k for k, v in self._module_init_status.items() if not v]
                if failed:
                    self.logger.warning(
                        f"{LogTemplates.PLUGIN_INIT_SUCCESS} "
                        f"(degraded mode, failed modules: {', '.join(failed)})"
                    )
                else:
                    self.logger.info(LogTemplates.PLUGIN_INIT_SUCCESS)

            except Exception as e:
                self._is_initialized = False
                self.logger.error(
                    LogTemplates.PLUGIN_INIT_FAILED.format(error=e),
                    exc_info=True,
                )
                raise
