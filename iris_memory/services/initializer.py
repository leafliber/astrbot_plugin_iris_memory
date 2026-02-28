"""
服务初始化编排器

从 MemoryService 抽取的初始化逻辑，负责：
- 编排所有组件的初始化顺序
- 管理初始化状态
- 处理降级逻辑

设计原则：
- 单一职责：仅负责初始化编排
- 依赖注入：通过构造函数接收所有依赖
- 可测试：独立于 MemoryService 可单独测试
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from astrbot.api import AstrBotConfig
from astrbot.api.star import Context

from iris_memory.core.config_manager import ConfigManager
from iris_memory.core.constants import LogTemplates, UNLIMITED_BUDGET
from iris_memory.core.defaults import DEFAULTS
from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.proactive_module import ProactiveModule
from iris_memory.services.modules.kg_module import KnowledgeGraphModule
from iris_memory.services.shared_state import SharedState
from iris_memory.utils.logger import get_logger
from iris_memory.utils.member_utils import set_identity_service
from iris_memory.utils.member_identity_service import MemberIdentityService


@dataclass
class InitializerDeps:
    """初始化器依赖项"""
    context: Context
    config: AstrBotConfig
    plugin_data_path: Path
    cfg: ConfigManager

    storage: StorageModule = field(default_factory=StorageModule)
    analysis: AnalysisModule = field(default_factory=AnalysisModule)
    llm_enhanced: LLMEnhancedModule = field(default_factory=LLMEnhancedModule)
    capture: CaptureModule = field(default_factory=CaptureModule)
    retrieval: RetrievalModule = field(default_factory=RetrievalModule)
    proactive: ProactiveModule = field(default_factory=ProactiveModule)
    kg: KnowledgeGraphModule = field(default_factory=KnowledgeGraphModule)

    shared_state: SharedState = field(default_factory=lambda: SharedState(max_size=2000, max_recent_track=20))


@dataclass
class InitializerResult:
    """初始化结果"""
    modules: InitializerDeps
    module_init_status: Dict[str, bool]
    member_identity: Optional[MemberIdentityService] = None
    image_analyzer: Optional[Any] = None
    activity_tracker: Optional[Any] = None
    activity_provider: Optional[Any] = None


class ServiceInitializer:
    """
    服务初始化编排器

    负责按正确顺序初始化所有组件，处理核心组件失败和增强组件降级。
    """

    def __init__(self, deps: InitializerDeps):
        self._deps = deps
        self._logger = get_logger("service_initializer")
        self._module_init_status: Dict[str, bool] = {}
        self._init_lock: asyncio.Lock = asyncio.Lock()

        self._member_identity: Optional[MemberIdentityService] = None
        self._image_analyzer: Optional[Any] = None
        self._activity_tracker: Optional[Any] = None
        self._activity_provider: Optional[Any] = None

    @property
    def module_init_status(self) -> Dict[str, bool]:
        return self._module_init_status

    async def initialize_all(self) -> InitializerResult:
        """
        执行完整初始化流程

        Returns:
            InitializerResult: 包含所有初始化后的模块和状态

        Raises:
            Exception: 核心组件初始化失败时抛出
        """
        async with self._init_lock:
            try:
                self._module_init_status.clear()
                self._deps.plugin_data_path.mkdir(parents=True, exist_ok=True)
                self._logger.info(LogTemplates.PLUGIN_INIT_START)

                await self._init_phase_core()
                await self._init_phase_enhanced()
                await self._init_phase_finalize()

                failed = [k for k, v in self._module_init_status.items() if not v]
                if failed:
                    self._logger.warning(
                        f"{LogTemplates.PLUGIN_INIT_SUCCESS} "
                        f"(degraded mode, failed modules: {', '.join(failed)})"
                    )
                else:
                    self._logger.info(LogTemplates.PLUGIN_INIT_SUCCESS)

                return InitializerResult(
                    modules=self._deps,
                    module_init_status=dict(self._module_init_status),
                    member_identity=self._member_identity,
                    image_analyzer=self._image_analyzer,
                    activity_tracker=self._activity_tracker,
                    activity_provider=self._activity_provider,
                )

            except Exception as e:
                self._logger.error(
                    LogTemplates.PLUGIN_INIT_FAILED.format(error=e),
                    exc_info=True,
                )
                raise

    async def _init_phase_core(self) -> None:
        """阶段1: 核心组件初始化（失败则整体失败）"""
        await self._init_llm_enhanced()
        self._module_init_status["llm_enhanced"] = True

        await self._init_core_components()
        self._module_init_status["core"] = True

    async def _init_phase_enhanced(self) -> None:
        """阶段2: 增强组件初始化（失败则降级运行）"""
        await self._init_upgrade_evaluator_safe()

        enhanced_modules = [
            ("knowledge_graph", self._init_knowledge_graph),
            ("activity_adaptive", self._init_activity_adaptive),
            ("message_processing", self._init_message_processing),
            ("persona_extractor", self._init_persona_extractor),
            ("proactive_reply", self._init_proactive_reply),
            ("image_analyzer", self._init_image_analyzer),
        ]

        for name, init_fn in enhanced_modules:
            await self._init_module_safe(name, init_fn)

    async def _init_phase_finalize(self) -> None:
        """阶段3: 收尾初始化"""
        await self._apply_config()
        self._module_init_status["config"] = True

        await self._init_module_safe("batch_processor", self._init_batch_processor)
        await self._init_module_safe("persona_batch_processor", self._init_persona_batch_processor)

    async def _init_module_safe(self, name: str, init_fn: Callable) -> None:
        """安全初始化模块，失败时记录并继续"""
        try:
            await init_fn()
            self._module_init_status[name] = True
        except Exception as e:
            self._module_init_status[name] = False
            self._logger.warning(
                f"Module '{name}' initialization failed, running in degraded mode: {e}",
                exc_info=True,
            )

    async def _init_llm_enhanced(self) -> None:
        """初始化 LLM 增强组件"""
        self._logger.debug(LogTemplates.COMPONENT_INIT.format(component="LLM enhanced"))
        await self._deps.llm_enhanced.initialize(self._deps.cfg, self._deps.context)

    async def _init_core_components(self) -> None:
        """初始化核心组件"""
        self._deps.analysis.initialize(self._deps.config)

        await self._deps.storage.initialize(
            config=self._deps.config,
            cfg=self._deps.cfg,
            plugin_data_path=self._deps.plugin_data_path,
            context=self._deps.context,
        )

        self._deps.capture.init_capture_engine(
            chroma_manager=self._deps.storage.chroma_manager,
            emotion_analyzer=self._deps.analysis.emotion_analyzer,
            rif_scorer=self._deps.analysis.rif_scorer,
            llm_sensitivity_detector=self._deps.llm_enhanced.sensitivity_detector,
            llm_trigger_detector=self._deps.llm_enhanced.trigger_detector,
            llm_conflict_resolver=self._deps.llm_enhanced.conflict_resolver,
        )

        self._deps.retrieval.initialize(
            chroma_manager=self._deps.storage.chroma_manager,
            rif_scorer=self._deps.analysis.rif_scorer,
            emotion_analyzer=self._deps.analysis.emotion_analyzer,
            session_manager=self._deps.storage.session_manager,
            llm_retrieval_router=self._deps.llm_enhanced.retrieval_router,
        )

        self._init_member_identity()
        self._logger.debug("Core components initialized")

    def _init_member_identity(self) -> None:
        """初始化成员身份服务"""
        self._member_identity = MemberIdentityService()
        set_identity_service(self._member_identity)
        self._logger.debug("MemberIdentityService initialized")

    async def _init_upgrade_evaluator_safe(self) -> None:
        """安全初始化记忆升级评估器"""
        try:
            await self._init_upgrade_evaluator()
            self._module_init_status["upgrade_evaluator"] = True
        except Exception as e:
            self._module_init_status["upgrade_evaluator"] = False
            self._logger.warning(
                f"Upgrade evaluator initialization failed, using rule mode: {e}",
                exc_info=True,
            )

    async def _init_upgrade_evaluator(self) -> None:
        """初始化记忆升级评估器的 LLM provider"""
        from iris_memory.core.upgrade_evaluator import UpgradeMode

        upgrade_mode = self._deps.cfg.upgrade_mode

        if upgrade_mode in ("llm", "hybrid"):
            self._logger.debug(f"Initializing upgrade evaluator LLM provider (mode={upgrade_mode})")

            if not self._deps.llm_enhanced.llm_processor:
                await self._deps.llm_enhanced.init_llm_processor(
                    context=self._deps.context,
                    cfg=self._deps.cfg,
                    lifecycle_manager=None,
                )

            if self._deps.llm_enhanced.llm_processor and self._deps.storage.lifecycle_manager:
                self._deps.storage.lifecycle_manager.set_llm_provider(
                    self._deps.llm_enhanced.llm_processor
                )
                self._logger.debug("Upgrade evaluator LLM provider set")
        else:
            self._logger.debug("Upgrade evaluator using rule mode (no LLM needed)")

    async def _init_knowledge_graph(self) -> None:
        """初始化知识图谱模块"""
        self._logger.debug(LogTemplates.COMPONENT_INIT.format(component="knowledge graph"))

        cfg = self._deps.cfg
        await self._deps.kg.initialize(
            plugin_data_path=self._deps.plugin_data_path,
            astrbot_context=self._deps.context,
            provider_id=cfg.knowledge_graph_provider_id,
            kg_mode=cfg.get("knowledge_graph.extraction_mode", "rule"),
            max_depth=cfg.get("knowledge_graph.max_depth", 3),
            max_nodes_per_hop=cfg.get("knowledge_graph.max_nodes_per_hop", 10),
            max_facts=cfg.get("knowledge_graph.max_facts", 8),
            enabled=cfg.get("knowledge_graph.enabled", True),
            auto_maintenance=cfg.get("knowledge_graph.auto_maintenance", True),
            maintenance_interval=cfg.get("knowledge_graph.maintenance_interval", 86400),
            auto_cleanup_orphans=cfg.get("knowledge_graph.auto_cleanup_orphans", True),
            auto_cleanup_low_confidence=cfg.get("knowledge_graph.auto_cleanup_low_confidence", True),
            low_confidence_threshold=cfg.get("knowledge_graph.low_confidence_threshold", 0.2),
            staleness_days=cfg.get("knowledge_graph.staleness_days", 30),
        )

        if self._deps.kg.enabled:
            self._deps.retrieval.set_kg_module(self._deps.kg)
            if self._deps.kg.storage:
                self._deps.capture.set_kg_storage(self._deps.kg.storage)

    async def _init_activity_adaptive(self) -> None:
        """初始化场景自适应组件"""
        self._logger.debug(LogTemplates.COMPONENT_INIT.format(component="activity adaptive"))

        from iris_memory.core.activity_config import GroupActivityTracker

        self._activity_tracker = GroupActivityTracker()

        if self._deps.storage.session_manager:
            self._deps.storage.session_manager._activity_tracker = self._activity_tracker

        enabled = self._deps.cfg.enable_activity_adaptive
        self._activity_provider = self._deps.cfg.init_activity_provider(
            tracker=self._activity_tracker,
            enabled=enabled,
        )

        status = "enabled" if enabled else "disabled"
        self._logger.debug(f"Activity adaptive system {status}")

    async def _init_message_processing(self) -> None:
        """初始化分层消息处理组件"""
        self._logger.debug(LogTemplates.COMPONENT_INIT.format(component="message processing"))

        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self._deps.cfg.use_llm

        if not enable_batch:
            self._logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Batch processing"))
            return

        if use_llm and not self._deps.llm_enhanced.llm_processor:
            await self._deps.llm_enhanced.init_llm_processor(
                context=self._deps.context,
                cfg=self._deps.cfg,
                lifecycle_manager=None,
            )

        self._deps.capture.init_message_classifier(
            emotion_analyzer=self._deps.analysis.emotion_analyzer,
            llm_processor=self._deps.llm_enhanced.llm_processor,
        )

        self._logger.debug("Message classifier initialized")

    async def _init_persona_extractor(self) -> None:
        """初始化画像提取器"""
        if not self._deps.cfg.get("persona.enabled", True):
            self._logger.info("Persona feature is disabled by config")
            return

        await self._deps.analysis.init_persona_extractor(
            cfg=self._deps.cfg,
            plugin_data_path=self._deps.plugin_data_path,
            context=self._deps.context,
        )

    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        await self._deps.proactive.initialize(
            cfg=self._deps.cfg,
            context=self._deps.context,
            emotion_analyzer=self._deps.analysis.emotion_analyzer,
            llm_proactive_reply_detector=self._deps.llm_enhanced.proactive_reply_detector,
        )

    async def _init_image_analyzer(self) -> None:
        """初始化图片分析器"""
        self._logger.debug(LogTemplates.COMPONENT_INIT.format(component="image analyzer"))

        if not self._deps.cfg.image_analysis_enabled:
            self._logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Image analysis"))
            return

        from iris_memory.multimodal.image_analyzer import ImageAnalyzer

        daily_budget = self._deps.cfg.image_analysis_daily_budget
        session_budget = self._deps.cfg.image_analysis_session_budget

        self._image_analyzer = ImageAnalyzer(
            astrbot_context=self._deps.context,
            config={
                "enable_image_analysis": self._deps.cfg.image_analysis_enabled,
                "default_level": self._deps.cfg.image_analysis_mode,
                "max_images_per_message": self._deps.cfg.image_analysis_max_images,
                "skip_sticker": DEFAULTS.image_analysis.skip_sticker,
                "analysis_cooldown": DEFAULTS.image_analysis.analysis_cooldown,
                "cache_ttl": DEFAULTS.image_analysis.cache_ttl,
                "max_cache_size": DEFAULTS.image_analysis.max_cache_size,
                "daily_analysis_budget": daily_budget if daily_budget > 0 else UNLIMITED_BUDGET,
                "session_analysis_budget": session_budget if session_budget > 0 else UNLIMITED_BUDGET,
                "similar_image_window": DEFAULTS.image_analysis.similar_image_window,
                "recent_image_limit": DEFAULTS.image_analysis.recent_image_limit,
                "require_context_relevance": self._deps.cfg.image_analysis_require_context,
            },
            provider_id=self._deps.cfg.image_analysis_provider_id,
        )

        self._logger.debug(f"Image analyzer initialized: mode={self._deps.cfg.image_analysis_mode}")

    async def _apply_config(self) -> None:
        """将配置应用到各模块"""
        self._deps.storage.apply_config(self._deps.cfg)
        self._deps.capture.apply_config(self._deps.cfg)
        self._deps.retrieval.apply_config(self._deps.cfg)

    async def _init_batch_processor(self) -> None:
        """初始化批量处理器"""
        await self._deps.capture.init_batch_processor(
            cfg=self._deps.cfg,
            llm_processor=self._deps.llm_enhanced.llm_processor,
            proactive_manager=self._deps.proactive.proactive_manager,
            on_save_callback=None,
        )

    async def _init_persona_batch_processor(self) -> None:
        """初始化画像批量处理器"""
        if not self._deps.cfg.get("persona.enabled", True):
            self._logger.debug("Persona batch processor skipped (disabled)")
            return

        await self._deps.analysis.init_persona_batch_processor(
            cfg=self._deps.cfg,
            apply_result_callback=None,
            working_memory_callback=None,
        )
