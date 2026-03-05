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

from iris_memory.config import ConfigManager
from iris_memory.core.constants import LogTemplates
from iris_memory.services.initializer import InitializerDeps, ServiceInitializer
from iris_memory.services.modules.analysis_module import AnalysisModule
from iris_memory.services.modules.capture_module import CaptureModule
from iris_memory.services.modules.kg_module import KnowledgeGraphModule
from iris_memory.services.modules.llm_enhanced_module import LLMEnhancedModule
from iris_memory.services.modules.proactive_module import ProactiveModule
from iris_memory.services.modules.retrieval_module import RetrievalModule
from iris_memory.services.modules.storage_module import StorageModule
from iris_memory.services.modules.cooldown_module import CooldownModule
from iris_memory.services.shared_state import SharedState
from iris_memory.services.business_service import BusinessService, BusinessServiceDeps
from iris_memory.services.persistence_service import PersistenceService
from iris_memory.utils.logger import get_logger

# 简单属性代理：attribute_name → (module_name, sub_attr)
# MemoryService.__getattr__ 使用此表把访问委托到 self._deps.<module>.<attr>
_PROXY_ATTRS: Dict[str, Tuple[str, str]] = {
    # storage module
    "chroma_manager": ("storage", "chroma_manager"),
    "session_manager": ("storage", "session_manager"),
    "lifecycle_manager": ("storage", "lifecycle_manager"),
    # capture module
    "capture_engine": ("capture", "capture_engine"),
    "batch_processor": ("capture", "batch_processor"),
    "message_classifier": ("capture", "message_classifier"),
    # retrieval module
    "retrieval_engine": ("retrieval", "retrieval_engine"),
    # proactive module
    "proactive_manager": ("proactive", "manager"),
    # analysis module
    "emotion_analyzer": ("analysis", "emotion_analyzer"),
    "persona_extractor": ("analysis", "persona_extractor"),
    "_persona_batch_processor": ("analysis", "persona_batch_processor"),
    # llm_enhanced module
    "llm_sensitivity_detector": ("llm_enhanced", "sensitivity_detector"),
    "llm_trigger_detector": ("llm_enhanced", "trigger_detector"),
    "llm_emotion_analyzer": ("llm_enhanced", "emotion_analyzer"),
    "llm_conflict_resolver": ("llm_enhanced", "conflict_resolver"),
    "llm_retrieval_router": ("llm_enhanced", "retrieval_router"),
}

# SharedState 属性代理
_STATE_ATTRS: Dict[str, str] = {
    "_user_emotional_states": "user_emotional_states",
    "_user_personas": "user_personas",
    "_recently_injected": "recently_injected",
    "_max_recent_track": "max_recent_track",
}


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
        self.cfg = ConfigManager(user_config=config, plugin_data_path=plugin_data_path)
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
    def cooldown(self) -> CooldownModule:
        return self._deps.cooldown

    @property
    def _shared_state(self) -> SharedState:
        return self._deps.shared_state

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def __getattr__(self, name: str) -> Any:
        """数据驱动的属性代理

        按 _PROXY_ATTRS 表把访问委托到 self._deps.<module>.<attr>，
        按 _STATE_ATTRS 表把访问委托到 self._deps.shared_state.<attr>。
        """
        if name in _PROXY_ATTRS:
            module_name, attr = _PROXY_ATTRS[name]
            module = getattr(self._deps, module_name)
            return getattr(module, attr)
        if name in _STATE_ATTRS:
            return getattr(self._deps.shared_state, _STATE_ATTRS[name])
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    @property
    def image_analyzer(self):
        return self._image_analyzer

    @property
    def chat_history_buffer(self):
        if '_chat_history_buffer' in self.__dict__:
            return self._chat_history_buffer
        return self.storage.chat_history_buffer

    @property
    def member_identity(self):
        return self._member_identity

    @property
    def activity_tracker(self):
        return self._activity_tracker

    @property
    def activity_provider(self):
        return self._activity_provider

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

                # 在所有组件和服务就绪后，注入主动回复发送器
                self._setup_proactive_reply_sender()

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

    def _setup_proactive_reply_sender(self) -> None:
        """创建并注入主动回复发送器

        在所有组件和服务完成初始化后调用。
        将 ProactiveReplySender 注入到 ProactiveManager，
        使其能够通过完整的 LLM 流程发送主动回复（记忆注入、画像注入等）。

        注意：不在此处解析 provider，因为 AstrBot 先加载插件后加载 provider，
        此时 provider 尚不可用。ProactiveReplySender 将在首次发送时懒加载解析。
        """
        manager = self.proactive_manager
        if not manager:
            return

        try:
            from iris_memory.proactive.reply_sender import ProactiveReplySender

            # 仅传递配置的 provider_id，不在初始化阶段解析 provider
            # ProactiveReplySender 将在实际发送时懒加载解析 provider
            configured_pid = self.cfg.llm_provider_id

            sender = ProactiveReplySender(
                astrbot_context=self.context,
                prepare_llm_context=self.prepare_llm_context,
                record_chat_message=self.record_chat_message,
                get_group_umo=manager.get_group_umo,
                configured_provider_id=configured_pid,
            )

            self.proactive.setup_reply_sender(sender)

            # 注入 CooldownModule 状态检查回调，确保主动回复遵守用户 /冷却 命令
            manager.set_cooldown_checker(self.cooldown.is_active)

            self.logger.info("Proactive reply sender wired successfully")
        except Exception as e:
            self.logger.warning(f"Failed to setup proactive reply sender: {e}")

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
