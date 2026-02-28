"""
记忆业务服务层 — Facade（门面）模式

重构后的 MemoryService 是一个薄 Facade：
- 持有 7 个 Feature Module + SharedState
- 通过组合持有 BusinessService + PersistenceService（取代 Mixin 继承）
- 委托初始化逻辑给 ServiceInitializer
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

refs:
- AstrBot API: astrbot.api.star.Context
- AstrBot API: astrbot.api.AstrBotConfig
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import AstrBotConfig
from astrbot.api.star import Context

from iris_memory.core.config_manager import init_config_manager
from iris_memory.core.constants import LogTemplates
from iris_memory.services.initializer import InitializerDeps, ServiceInitializer
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.kg_module import KnowledgeGraphModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.proactive_module import ProactiveModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.shared_state import SharedState
from iris_memory.services.business_service import BusinessService, BusinessServiceDeps
from iris_memory.services.persistence_service import PersistenceService
from iris_memory.utils.logger import get_logger


class MemoryService:
    """
    记忆业务服务层 — Facade

    持有 7 个 Feature Module + SharedState，
    通过组合 BusinessService 和 PersistenceService 委托业务/持久化逻辑。
    初始化逻辑委托给 ServiceInitializer。
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

        self._deps = InitializerDeps(
            context=context,
            config=config,
            plugin_data_path=plugin_data_path,
            cfg=self.cfg,
        )

        self._initializer = ServiceInitializer(self._deps)

        self._member_identity: Optional[Any] = None
        self._image_analyzer: Optional[Any] = None
        self._activity_tracker: Optional[Any] = None
        self._activity_provider: Optional[Any] = None

        self._business: Optional[BusinessService] = None
        self._persistence: Optional[PersistenceService] = None

    @property
    def storage(self) -> StorageModule:
        return self._deps.storage

    @property
    def analysis(self) -> AnalysisModule:
        return self._deps.analysis

    @property
    def llm_enhanced(self) -> LLMEnhancedModule:
        return self._deps.llm_enhanced

    @property
    def capture(self) -> CaptureModule:
        return self._deps.capture

    @property
    def retrieval(self) -> RetrievalModule:
        return self._deps.retrieval

    @property
    def proactive(self) -> ProactiveModule:
        return self._deps.proactive

    @property
    def kg(self) -> KnowledgeGraphModule:
        return self._deps.kg

    @property
    def _shared_state(self) -> SharedState:
        return self._deps.shared_state

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

        components["chroma_manager"] = (
            "available" if self.chroma_manager else "unavailable"
        )
        components["session_manager"] = (
            "available" if self.session_manager else "unavailable"
        )
        components["embedding"] = (
            "ready" if self.is_embedding_ready() else "not_ready"
        )

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

        components["knowledge_graph"] = (
            "enabled" if (self.kg and self.kg.enabled) else "disabled"
        )
        components["proactive_manager"] = (
            "enabled" if self.proactive_manager else "disabled"
        )
        components["image_analyzer"] = (
            "enabled" if self.image_analyzer else "disabled"
        )

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
        - 初始化完成后创建 BusinessService 和 PersistenceService
        """
        async with self._init_lock:
            try:
                self._is_initialized = False
                self._module_init_status.clear()
                self.logger.info(LogTemplates.PLUGIN_INIT_START)

                result = await self._initializer.initialize_all()

                self._module_init_status = result.module_init_status
                self._member_identity = result.member_identity
                self._image_analyzer = result.image_analyzer
                self._activity_tracker = result.activity_tracker
                self._activity_provider = result.activity_provider

                self._create_services()

                self._is_initialized = True

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
        business_deps = BusinessServiceDeps(
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
        self._business = BusinessService(deps=business_deps)

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

        if self.capture.batch_processor:
            self.capture.batch_processor.on_save_callback = self._deferred_save_batch_queues

    async def _deferred_save_batch_queues(self, put_kv_data=None) -> None:
        """延迟绑定的批量队列保存回调"""
        if self._persistence:
            await self._persistence._save_batch_queues(put_kv_data)

    def _apply_batch_persona_result(
        self, user_id: str, session_key: str, result: Any, msg: Any
    ) -> None:
        """画像批量处理结果回调 — 将提取结果应用到用户画像"""
        from iris_memory.analysis.persona.persona_logger import persona_log

        try:
            persona = self._shared_state.get_or_create_user_persona(user_id)

            sender_name = getattr(msg, "sender_name", None)
            if sender_name and sender_name != persona.display_name:
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
            def __init__(self, m: PersonaQueuedMessage):
                self.id = m.memory_id
                self.content = m.content
                self.summary = m.summary
                self.type = m.memory_type
                self.confidence = m.confidence
                self.created_time = None

        return persona.update_from_memory(_MemoryLike(msg))

    async def capture_and_store_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        is_user_requested: bool = False,
        context: Optional[Dict[str, Any]] = None,
        sender_name: Optional[str] = None,
        persona_id: Optional[str] = None,
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
            sender_name=sender_name,
            persona_id=persona_id,
        )

    async def search_memories(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        top_k: int = 5,
        persona_id: Optional[str] = None,
    ) -> List[Any]:
        """搜索记忆"""
        if not self._is_initialized or not self._business:
            return []
        return await self._business.search_memories(
            query=query, user_id=user_id, group_id=group_id, top_k=top_k, persona_id=persona_id
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
        persona_id: Optional[str] = None,
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
            persona_id=persona_id,
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
        image_description: str = "",
        persona_id: Optional[str] = None,
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
            persona_id=persona_id,
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
