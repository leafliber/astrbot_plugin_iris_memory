"""
MemoryService 初始化模块

将初始化逻辑从 MemoryService 中拆分出来，委托给各 Feature Module。
"""

from typing import TYPE_CHECKING

from iris_memory.utils.logger import get_logger
from iris_memory.core.defaults import DEFAULTS
from iris_memory.core.constants import LogTemplates, UNLIMITED_BUDGET

if TYPE_CHECKING:
    from astrbot.api.star import Context
    from astrbot.api import AstrBotConfig
    from pathlib import Path

logger = get_logger("memory_service.init")


class ServiceInitializer:
    """MemoryService 初始化器 Mixin

    通过委托给各 Feature Module 完成初始化，
    本 Mixin 只负责编排顺序和跨模块粘合。
    """

    # ── 核心组件（存储 + 分析 + 捕获 + 检索） ──

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

        logger.debug("Core components initialized")

    def _init_member_identity(self) -> None:
        from iris_memory.utils.member_utils import set_identity_service
        from iris_memory.utils.member_identity_service import MemberIdentityService

        self._member_identity = MemberIdentityService()
        set_identity_service(self._member_identity)
        logger.debug("MemberIdentityService initialized")

    # ── 知识图谱 ──

    async def _init_knowledge_graph(self) -> None:
        """初始化知识图谱模块"""
        logger.debug(LogTemplates.COMPONENT_INIT.format(component="knowledge graph"))

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

    # ── LLM 增强 ──

    async def _init_llm_enhanced(self) -> None:
        """初始化 LLM 增强组件"""
        logger.debug(LogTemplates.COMPONENT_INIT.format(component="LLM enhanced"))
        await self.llm_enhanced.initialize(self.cfg, self.context)

    # ── 场景自适应 ──

    async def _init_activity_adaptive(self) -> None:
        """初始化场景自适应组件"""
        logger.debug(LogTemplates.COMPONENT_INIT.format(component="activity adaptive"))

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
        logger.debug(f"Activity adaptive system {status}")

    # ── 消息处理（LLM processor + classifier） ──

    async def _init_message_processing(self) -> None:
        """初始化分层消息处理组件"""
        logger.debug(LogTemplates.COMPONENT_INIT.format(component="message processing"))

        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self.cfg.use_llm

        if not enable_batch:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Batch processing"))
            return

        if use_llm:
            await self.llm_enhanced.init_llm_processor(
                context=self.context,
                cfg=self.cfg,
                lifecycle_manager=self.storage.lifecycle_manager,
            )

        self.capture.init_message_classifier(
            emotion_analyzer=self.analysis.emotion_analyzer,
            llm_processor=self.llm_enhanced.llm_processor,
        )

        logger.debug("Message classifier initialized")

    # ── 画像提取 ──

    async def _init_persona_extractor(self) -> None:
        """初始化画像提取器"""
        await self.analysis.init_persona_extractor(
            cfg=self.cfg,
            plugin_data_path=self.plugin_data_path,
            context=self.context,
        )

    # ── 画像批量处理器 ──

    async def _init_persona_batch_processor(self) -> None:
        """初始化画像批量处理器（依赖 persona_extractor）"""
        await self.analysis.init_persona_batch_processor(
            cfg=self.cfg,
            apply_result_callback=self._apply_batch_persona_result,
        )

    def _apply_batch_persona_result(
        self, user_id: str, session_key: str, result, msg
    ) -> None:
        """画像批量处理结果回调 — 将提取结果应用到用户画像

        此回调由 PersonaBatchProcessor 在完成批量提取后调用。
        复用现有 UserPersona.apply_extraction_result() 接口。
        """
        from iris_memory.analysis.persona.persona_logger import persona_log

        try:
            persona = self.get_or_create_user_persona(user_id)
            changes = persona.apply_extraction_result(
                result,
                source_memory_id=msg.memory_id,
                memory_type=msg.memory_type,
                base_confidence=msg.confidence,
            )
            if not changes:
                # 提取结果未匹配到可更新的字段，回退到规则更新
                # 构造一个轻量 memory-like 对象供 update_from_memory 使用
                changes = self._fallback_rule_update(persona, msg)

            if changes:
                persona_log.update_applied(
                    user_id,
                    [c.to_dict() for c in changes]
                )
                from iris_memory.utils.logger import get_logger
                _logger = get_logger("memory_service.business")
                _logger.debug(
                    f"Persona batch result applied for user={user_id}: "
                    f"{len(changes)} change(s)"
                )
        except Exception as e:
            persona_log.update_error(user_id, e)
            from iris_memory.utils.logger import get_logger
            _logger = get_logger("memory_service.business")
            _logger.warning(
                f"Failed to apply batch persona result for user={user_id}: {e}"
            )

    @staticmethod
    def _fallback_rule_update(persona, msg) -> list:
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

    # ── 主动回复 ──

    async def _init_proactive_reply(self) -> None:
        """初始化主动回复组件"""
        await self.proactive.initialize(
            cfg=self.cfg,
            context=self.context,
            emotion_analyzer=self.analysis.emotion_analyzer,
            llm_proactive_reply_detector=self.llm_enhanced.proactive_reply_detector,
        )

    # ── 图片分析 ──

    async def _init_image_analyzer(self) -> None:
        """初始化图片分析器"""
        logger.debug(LogTemplates.COMPONENT_INIT.format(component="image analyzer"))

        if not self.cfg.image_analysis_enabled:
            logger.debug(LogTemplates.COMPONENT_INIT_DISABLED.format(component="Image analysis"))
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

        logger.debug(f"Image analyzer initialized: mode={self.cfg.image_analysis_mode}")

    # ── 配置应用 ──

    async def _apply_config(self) -> None:
        """将配置应用到各模块"""
        self.storage.apply_config(self.cfg)
        self.capture.apply_config(self.cfg)
        self.retrieval.apply_config(self.cfg)

    # ── 批量处理器（最后初始化，依赖所有其他组件） ──

    async def _init_batch_processor(self) -> None:
        """初始化批量处理器"""
        await self.capture.init_batch_processor(
            cfg=self.cfg,
            llm_processor=self.llm_enhanced.llm_processor,
            proactive_manager=self.proactive.proactive_manager,
            on_save_callback=self._save_batch_queues,
        )
