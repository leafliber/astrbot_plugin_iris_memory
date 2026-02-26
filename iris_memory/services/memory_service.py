"""
记忆业务服务层 — Facade（门面）模式

重构后的 MemoryService 是一个薄 Facade：
- 持有 7 个 Feature Module + SharedState
- 通过组合持有 BusinessService + PersistenceService（取代 Mixin 继承）
- 负责初始化编排（吸收原 ServiceInitializer 逻辑）
- 委托所有业务/持久化操作给对应 Service

架构变更（Mixin 继承 → 组合 + 依赖注入）：
Before:
  class MemoryService(ServiceInitializer, BusinessOperations, PersistenceOperations): ...
After:
  class MemoryService:
      self._business: BusinessService      # 显式组合
      self._persistence: PersistenceService  # 显式组合
      self._shared_state: SharedState        # 共享状态集中管理

优势：
- 消除 Mixin 间的隐式循环依赖
- 所有依赖通过构造函数显式注入，可独立测试
- SharedState 集中管理跨服务共享状态
- 回调不再跨 Mixin 边界引用方法
"""
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import asyncio

from astrbot.api.star import Context
from astrbot.api import AstrBotConfig

from iris_memory.utils.logger import get_logger
from iris_memory.utils.bounded_dict import BoundedDict
from iris_memory.core.config_manager import init_config_manager
from iris_memory.core.constants import LogTemplates, UNLIMITED_BUDGET
from iris_memory.core.defaults import DEFAULTS
from iris_memory.utils.member_utils import set_identity_service
from iris_memory.utils.member_identity_service import MemberIdentityService

from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.proactive_module import ProactiveModule
from iris_memory.services.modules.kg_module import KnowledgeGraphModule

from iris_memory.services.shared_state import SharedState
from iris_memory.services.business_service import BusinessService
from iris_memory.services.persistence_service import PersistenceService


class MemoryService:
    """
    记忆业务服务层 — Facade

    持有 7 个 Feature Module + SharedState，
    通过组合 BusinessService 和 PersistenceService 委托业务/持久化逻辑。
    初始化逻辑直接内联（无需 Mixin）。
    """

    def __init__(self, context: Context, config: AstrBotConfig, plugin_data_path: Path):
        self.context = context
        self.config = config
        self.plugin_data_path = plugin_data_path
        self.cfg = init_config_manager(config)
        self.logger = get_logger("memory_service")

        self._is_initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._module_init_status: Dict[str, bool] = {}

        # ── Feature Modules ──
        self.storage = StorageModule()
        self.analysis = AnalysisModule()
        self.llm_enhanced = LLMEnhancedModule()
        self.capture = CaptureModule()
        self.retrieval = RetrievalModule()
        self.proactive = ProactiveModule()
        self.kg = KnowledgeGraphModule()

        # ── 共享状态（集中管理，取代分散在 Mixin 中的属性） ──
        self._shared_state = SharedState(max_size=2000, max_recent_track=20)

        # ── 独立组件（不适合归入 Module 的辅助组件） ──
        self._member_identity: Optional[MemberIdentityService] = None
        self._image_analyzer: Optional[Any] = None
        self._activity_tracker: Optional[Any] = None
        self._activity_provider: Optional[Any] = None

        # ── 组合式服务（初始化后创建） ──
        self._business: Optional[BusinessService] = None
        self._persistence: Optional[PersistenceService] = None

    # ================================================================
    # 向后兼容属性
    # 这些属性代理到 Module / SharedState，让 main.py / 旧代码无需修改即可访问
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

    # ── SharedState 向后兼容属性（main.py 直接访问） ──

    @property
    def _user_emotional_states(self):
        return self._shared_state.user_emotional_states

    @property
    def _user_personas(self):
        return self._shared_state.user_personas

    @property
    def _recently_injected(self):
        return self._shared_state.recently_injected

    @property
    def _max_recent_track(self) -> int:
        return self._shared_state.max_recent_track

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

    # ================================================================
    # 初始化（整合原 ServiceInitializer 逻辑）
    # ================================================================

    async def initialize(self) -> None:
        """异步初始化所有组件

        采用分阶段初始化策略：
        - 核心组件（storage/capture/retrieval）失败则整体失败
        - 增强组件（KG/proactive/image）失败则降级运行
        - 初始化完成后创建 BusinessService 和 PersistenceService
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

                # ── 阶段3: 创建组合式服务 ──
                self._create_services()

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

    def _create_services(self) -> None:
        """创建组合式服务（在所有模块初始化完成后调用）

        BusinessService 和 PersistenceService 通过构造函数显式接收依赖，
        取代原 Mixin 模式下的 self.xxx 隐式访问。
        """
        self._business = BusinessService(
            shared_state=self._shared_state,
            cfg=self.cfg,
            storage=self.storage,
            analysis=self.analysis,
            llm_enhanced=self.llm_enhanced,
            capture=self.capture,
            retrieval=self.retrieval,
            kg=self.kg,
            image_analyzer=self._image_analyzer,
            member_identity=self._member_identity,
            activity_tracker=self._activity_tracker,
        )

        self._persistence = PersistenceService(
            shared_state=self._shared_state,
            cfg=self.cfg,
            storage=self.storage,
            analysis=self.analysis,
            capture=self.capture,
            retrieval=self.retrieval,
            proactive=self.proactive,
            llm_enhanced=self.llm_enhanced,
            kg=self.kg,
            member_identity=self._member_identity,
            image_analyzer=self._image_analyzer,
            activity_tracker=self._activity_tracker,
            activity_provider=self._activity_provider,
        )

        # 重新绑定批量处理器的保存回调，确保回调指向有效的 PersistenceService
        if self.capture.batch_processor:
            self.capture.batch_processor.on_save_callback = self._deferred_save_batch_queues

    # ── 初始化子方法（整合自原 ServiceInitializer） ──

    async def _init_core_components(self) -> None:
        """初始化核心组件"""
        # 分析模块（同步，无外部依赖）
        self.analysis.initialize(self.config)

        # 存储模块
        await self.storage.initialize(
            config=self.config,
            cfg=self.cfg,
            plugin_data_path=self.plugin_data_path,
            context=self.context,
        )

        # 捕获引擎（依赖 storage + analysis + llm_enhanced）
        self.capture.init_capture_engine(
            chroma_manager=self.storage.chroma_manager,
            emotion_analyzer=self.analysis.emotion_analyzer,
            rif_scorer=self.analysis.rif_scorer,
            llm_sensitivity_detector=self.llm_enhanced.sensitivity_detector,
            llm_trigger_detector=self.llm_enhanced.trigger_detector,
            llm_conflict_resolver=self.llm_enhanced.conflict_resolver,
        )

        # 检索引擎（依赖 storage + analysis + llm_enhanced）
        self.retrieval.initialize(
            chroma_manager=self.storage.chroma_manager,
            rif_scorer=self.analysis.rif_scorer,
            emotion_analyzer=self.analysis.emotion_analyzer,
            session_manager=self.storage.session_manager,
            llm_retrieval_router=self.llm_enhanced.retrieval_router,
        )

        # 成员身份服务
        self._init_member_identity()

        self.logger.debug("Core components initialized")

    def _init_member_identity(self) -> None:
        self._member_identity = MemberIdentityService()
        set_identity_service(self._member_identity)
        self.logger.debug("MemberIdentityService initialized")

    async def _init_knowledge_graph(self) -> None:
        """初始化知识图谱模块"""
        self.logger.debug(LogTemplates.COMPONENT_INIT.format(component="knowledge graph"))

        kg_enabled = self.cfg.get("knowledge_graph.enabled", True)
        kg_mode = self.cfg.get("knowledge_graph.extraction_mode", "rule")
        kg_max_depth = self.cfg.get("knowledge_graph.max_depth", 3)
        kg_max_nodes = self.cfg.get("knowledge_graph.max_nodes_per_hop", 10)
        kg_max_facts = self.cfg.get("knowledge_graph.max_facts", 8)

        await self.kg.initialize(
            plugin_data_path=self.plugin_data_path,
            astrbot_context=self.context,
            provider_id=self.cfg.knowledge_graph_provider_id,
            kg_mode=kg_mode,
            max_depth=kg_max_depth,
            max_nodes_per_hop=kg_max_nodes,
            max_facts=kg_max_facts,
            enabled=kg_enabled,
        )

        # 注入到检索引擎
        if self.kg.enabled:
            self.retrieval.set_kg_module(self.kg)

    async def _init_llm_enhanced(self) -> None:
        """初始化 LLM 增强组件"""
        self.logger.debug(LogTemplates.COMPONENT_INIT.format(component="LLM enhanced"))
        await self.llm_enhanced.initialize(self.cfg, self.context)

    async def _init_activity_adaptive(self) -> None:
        """初始化场景自适应组件"""
        self.logger.debug(LogTemplates.COMPONENT_INIT.format(component="activity adaptive"))

        from iris_memory.core.activity_config import GroupActivityTracker

        self._activity_tracker = GroupActivityTracker()

        if self.storage.session_manager:
            self.storage.session_manager._activity_tracker = self._activity_tracker

        enabled = self.cfg.enable_activity_adaptive
        self._activity_provider = self.cfg.init_activity_provider(
            tracker=self._activity_tracker,
            enabled=enabled,
        )

        status = "enabled" if enabled else "disabled"
        self.logger.debug(f"Activity adaptive system {status}")

    async def _init_message_processing(self) -> None:
        """初始化分层消息处理组件"""
        self.logger.debug(LogTemplates.COMPONENT_INIT.format(component="message processing"))

        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self.cfg.use_llm

        if not enable_batch:
            self.logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Batch processing"))
            return

        if use_llm and not self.llm_enhanced.llm_processor:
            await self.llm_enhanced.init_llm_processor(
                context=self.context,
                cfg=self.cfg,
                lifecycle_manager=None,
            )

        self.capture.init_message_classifier(
            emotion_analyzer=self.analysis.emotion_analyzer,
            llm_processor=self.llm_enhanced.llm_processor,
        )

        self.logger.debug("Message classifier initialized")

    async def _init_upgrade_evaluator(self) -> None:
        """初始化记忆升级评估器的 LLM provider"""
        from iris_memory.core.upgrade_evaluator import UpgradeMode

        upgrade_mode = self.cfg.upgrade_mode

        if upgrade_mode in ("llm", "hybrid"):
            self.logger.debug(f"Initializing upgrade evaluator LLM provider (mode={upgrade_mode})")

            if not self.llm_enhanced.llm_processor:
                await self.llm_enhanced.init_llm_processor(
                    context=self.context,
                    cfg=self.cfg,
                    lifecycle_manager=None,
                )

            if self.llm_enhanced.llm_processor and self.storage.lifecycle_manager:
                self.storage.lifecycle_manager.set_llm_provider(
                    self.llm_enhanced.llm_processor
                )
                self.logger.debug("Upgrade evaluator LLM provider set")
        else:
            self.logger.debug("Upgrade evaluator using rule mode (no LLM needed)")

    async def _init_persona_extractor(self) -> None:
        """初始化画像提取器"""
        if not self.cfg.get("persona.enabled", True):
            self.logger.info("Persona feature is disabled by config")
            return

        await self.analysis.init_persona_extractor(
            cfg=self.cfg,
            plugin_data_path=self.plugin_data_path,
            context=self.context,
        )

    async def _init_persona_batch_processor(self) -> None:
        """初始化画像批量处理器（依赖 persona_extractor）

        回调直接使用 SharedState，不再跨 Mixin 调用 BusinessOperations 方法。
        同时注入工作记忆查询回调，使后台处理能够获取会话上下文。
        """
        if not self.cfg.get("persona.enabled", True):
            self.logger.debug("Persona batch processor skipped (disabled)")
            return

        await self.analysis.init_persona_batch_processor(
            cfg=self.cfg,
            apply_result_callback=self._apply_batch_persona_result,
            working_memory_callback=self._get_working_memory_for_batch_processor,
        )

    async def _get_working_memory_for_batch_processor(
        self,
        user_id: str,
        group_id: Optional[str] = None,
    ) -> List[Any]:
        """为画像批量处理器查询工作记忆

        这是后台记忆管理查询工作记忆的接口，使批量画像提取
        能够获取会话的近期工作记忆作为上下文。

        Args:
            user_id: 用户 ID
            group_id: 群组 ID（私聊时为 None）

        Returns:
            List[Memory]: 工作记忆列表
        """
        try:
            if not self.storage.session_manager:
                return []
            return await self.storage.session_manager.get_working_memory(
                user_id=user_id,
                group_id=group_id,
                update_access=False,  # 后台查询不更新访问计数
            )
        except Exception as e:
            self.logger.debug(f"Failed to get working memory for batch processor: {e}")
            return []

    def _apply_batch_persona_result(
        self, user_id: str, session_key: str, result: Any, msg: Any
    ) -> None:
        """画像批量处理结果回调 — 将提取结果应用到用户画像

        此回调由 PersonaBatchProcessor 在完成批量提取后调用。
        直接使用 SharedState 访问用户画像，不再依赖 BusinessOperations Mixin。
        同时同步 sender_name 到 display_name。
        """
        from iris_memory.analysis.persona.persona_logger import persona_log

        try:
            persona = self._shared_state.get_or_create_user_persona(user_id)

            # 同步 sender_name 到 display_name（如果提供了且当前为空）
            sender_name = getattr(msg, "sender_name", None)
            if sender_name and not persona.display_name:
                persona.display_name = sender_name
                self.logger.debug(f"Batch update: display_name for user={user_id}: {sender_name}")

            changes = persona.apply_extraction_result(
                result,
                source_memory_id=msg.memory_id,
                memory_type=msg.memory_type,
                base_confidence=msg.confidence,
            )
            if not changes:
                changes = self._fallback_rule_update(persona, msg)

            if changes:
                persona_log.update_applied(
                    user_id,
                    [c.to_dict() for c in changes]
                )
                self.logger.debug(
                    f"Persona batch result applied for user={user_id}: "
                    f"{len(changes)} change(s)"
                )
        except Exception as e:
            persona_log.update_error(user_id, e)
            self.logger.warning(
                f"Failed to apply batch persona result for user={user_id}: {e}"
            )

    @staticmethod
    def _fallback_rule_update(persona: Any, msg: Any) -> list:
        """当批量提取结果为空时，尝试规则引擎更新"""
        from iris_memory.analysis.persona.persona_batch_processor import PersonaQueuedMessage

        class _MemoryLike:
            """轻量 memory 替身，供 update_from_memory 使用"""
            def __init__(self, m: PersonaQueuedMessage):
                self.id = m.memory_id
                self.content = m.content
                self.summary = m.summary
                self.type = m.memory_type
                self.confidence = m.confidence
                self.created_time = None

        return persona.update_from_memory(_MemoryLike(msg))

    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        await self.proactive.initialize(
            cfg=self.cfg,
            context=self.context,
            emotion_analyzer=self.analysis.emotion_analyzer,
            llm_proactive_reply_detector=self.llm_enhanced.proactive_reply_detector,
        )

    async def _init_image_analyzer(self) -> None:
        """初始化图片分析器"""
        self.logger.debug(LogTemplates.COMPONENT_INIT.format(component="image analyzer"))

        if not self.cfg.image_analysis_enabled:
            self.logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Image analysis"))
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
                "daily_analysis_budget": daily_budget if daily_budget > 0 else UNLIMITED_BUDGET,
                "session_analysis_budget": session_budget if session_budget > 0 else UNLIMITED_BUDGET,
                "similar_image_window": DEFAULTS.image_analysis.similar_image_window,
                "recent_image_limit": DEFAULTS.image_analysis.recent_image_limit,
                "require_context_relevance": self.cfg.image_analysis_require_context,
            },
            provider_id=self.cfg.image_analysis_provider_id,
        )

        self.logger.debug(f"Image analyzer initialized: mode={self.cfg.image_analysis_mode}")

    async def _apply_config(self) -> None:
        """将配置应用到各模块"""
        self.storage.apply_config(self.cfg)
        self.capture.apply_config(self.cfg)
        self.retrieval.apply_config(self.cfg)

    async def _init_batch_processor(self) -> None:
        """初始化批量处理器

        回调使用延迟绑定方式，因为 PersistenceService 在此时尚未创建。
        """
        await self.capture.init_batch_processor(
            cfg=self.cfg,
            llm_processor=self.llm_enhanced.llm_processor,
            proactive_manager=self.proactive.proactive_manager,
            on_save_callback=self._deferred_save_batch_queues,
        )

    async def _deferred_save_batch_queues(self, put_kv_data=None) -> None:
        """延迟绑定的批量队列保存回调

        批量处理器初始化在 PersistenceService 创建之前，
        因此使用此方法作为桥接，在运行时委托给 PersistenceService。
        """
        if self._persistence:
            await self._persistence._save_batch_queues(put_kv_data)

    # ================================================================
    # 业务操作委托（Facade 模式 → BusinessService）
    # ================================================================

    async def capture_and_store_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        is_user_requested: bool = False,
        context: Optional[Dict[str, Any]] = None,
        sender_name: Optional[str] = None
    ) -> Optional[Any]:
        """捕获并存储记忆"""
        if not self._is_initialized or not self._business:
            return None
        return await self._business.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=is_user_requested,
            context=context,
            sender_name=sender_name
        )

    async def search_memories(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int = 5
    ) -> List[Any]:
        """搜索记忆"""
        if not self._is_initialized or not self._business:
            return []
        return await self._business.search_memories(
            query=query, user_id=user_id, group_id=group_id, top_k=top_k
        )

    async def clear_memories(self, user_id: str, group_id: Optional[str]) -> bool:
        """清除用户记忆"""
        if not self._is_initialized or not self._business:
            return False
        return await self._business.clear_memories(user_id, group_id)

    async def delete_private_memories(self, user_id: str) -> Tuple[bool, int]:
        """删除用户私聊记忆"""
        if not self._is_initialized or not self._business:
            return False, 0
        return await self._business.delete_private_memories(user_id)

    async def delete_group_memories(
        self,
        group_id: str,
        scope_filter: Optional[str],
        user_id: Optional[str] = None
    ) -> Tuple[bool, int]:
        """删除群聊记忆"""
        if not self._is_initialized or not self._business:
            return False, 0
        return await self._business.delete_group_memories(
            group_id, scope_filter, user_id
        )

    async def delete_all_memories(self) -> Tuple[bool, int]:
        """删除所有记忆"""
        if not self._is_initialized or not self._business:
            return False, 0
        return await self._business.delete_all_memories()

    async def get_memory_stats(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> Dict[str, Any]:
        """获取记忆统计"""
        if not self._is_initialized or not self._business:
            return {}
        return await self._business.get_memory_stats(user_id, group_id)

    async def prepare_llm_context(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        image_context: str = "",
        sender_name: Optional[str] = None,
        reply_context: Optional[str] = None,
    ) -> str:
        """准备 LLM 上下文"""
        if not self._is_initialized or not self._business:
            return ""
        return await self._business.prepare_llm_context(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context,
            sender_name=sender_name,
            reply_context=reply_context,
        )

    async def analyze_images(
        self,
        message_chain: List[Any],
        user_id: str,
        group_id: Optional[str],
        context_text: str,
        umo: str,
        session_id: str
    ) -> Tuple[str, str]:
        """分析图片"""
        if not self._is_initialized or not self._business:
            return "", ""
        return await self._business.analyze_images(
            message_chain=message_chain,
            user_id=user_id,
            group_id=group_id,
            context_text=context_text,
            umo=umo,
            session_id=session_id,
        )

    async def process_message_batch(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        context: Dict[str, Any],
        umo: str,
        image_description: str = ""
    ) -> None:
        """处理消息批次"""
        if not self._is_initialized or not self._business:
            return
        await self._business.process_message_batch(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context=context,
            umo=umo,
            image_description=image_description,
        )

    async def record_chat_message(
        self,
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        group_id: Optional[str] = None,
        is_bot: bool = False,
        session_user_id: Optional[str] = None,
        reply_sender_name: Optional[str] = None,
        reply_sender_id: Optional[str] = None,
        reply_content: Optional[str] = None,
    ) -> None:
        """记录一条聊天消息到缓冲区"""
        if not self._is_initialized or not self._business:
            return
        await self._business.record_chat_message(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            group_id=group_id,
            is_bot=is_bot,
            session_user_id=session_user_id,
            reply_sender_name=reply_sender_name,
            reply_sender_id=reply_sender_id,
            reply_content=reply_content,
        )

    def update_session_activity(self, user_id: str, group_id: Optional[str]) -> None:
        """更新会话活动"""
        if not self._is_initialized or not self._business:
            return
        self._business.update_session_activity(user_id, group_id)

    async def activate_session(self, user_id: str, group_id: Optional[str]) -> None:
        """激活会话"""
        if not self._is_initialized or not self._business:
            return
        await self._business.activate_session(user_id, group_id)

    def get_or_create_user_persona(self, user_id: str):
        """获取或创建用户画像"""
        return self._shared_state.get_or_create_user_persona(user_id)

    def _get_or_create_emotional_state(self, user_id: str):
        """获取或创建用户情感状态"""
        return self._shared_state.get_or_create_emotional_state(user_id)

    # ================================================================
    # 持久化操作委托（Facade 模式 → PersistenceService）
    # ================================================================

    async def load_from_kv(self, get_kv_data) -> None:
        """从 KV 存储加载数据"""
        if not self._is_initialized or not self._persistence:
            return
        await self._persistence.load_from_kv(get_kv_data)

    async def save_to_kv(self, put_kv_data) -> None:
        """保存到 KV 存储"""
        if not self._is_initialized or not self._persistence:
            return
        await self._persistence.save_to_kv(put_kv_data)

    async def terminate(self) -> None:
        """销毁服务

        热更新友好：
        1. 立即标记为未初始化，阻止新操作进入
        2. 委托给 PersistenceService 执行停止/清理逻辑
        """
        self._is_initialized = False

        if self._persistence:
            await self._persistence.terminate()
