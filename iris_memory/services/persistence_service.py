"""
持久化操作服务

将持久化逻辑从 MemoryService Mixin 中分离为独立的组合式服务类，
通过构造函数显式接收所有依赖，消除隐式耦合。

设计动机：
- 原 PersistenceOperations 作为 Mixin 通过 self.xxx 隐式访问 MemoryService 属性
- 转为组合模式后，所有依赖通过构造函数显式注入
"""

from typing import Dict, Any, Optional

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import KVStoreKeys, LogTemplates
from iris_memory.utils.member_utils import set_identity_service
from iris_memory.services.shared_state import SharedState

logger = get_logger("memory_service.persistence")


class PersistenceService:
    """持久化操作服务

    通过构造函数注入所有依赖，取代原 Mixin 模式。

    职责：
    1. KV 存储加载
    2. KV 存储保存
    3. 服务销毁和状态清理

    Args:
        shared_state: 跨服务共享状态
        storage: 存储模块
        analysis: 分析模块
        capture: 捕获模块
        retrieval: 检索模块
        proactive: 主动回复模块
        llm_enhanced: LLM 增强模块
        kg: 知识图谱模块
        member_identity: 成员身份服务（可选）
        image_analyzer: 图片分析器（可选）
        activity_tracker: 群活跃度跟踪器（可选）
        activity_provider: 活跃度配置提供者（可选）
    """

    def __init__(
        self,
        *,
        shared_state: SharedState,
        cfg: Any,
        storage: Any,
        analysis: Any,
        capture: Any,
        retrieval: Any,
        proactive: Any,
        llm_enhanced: Any,
        kg: Any,
        member_identity: Any = None,
        image_analyzer: Any = None,
        activity_tracker: Any = None,
        activity_provider: Any = None,
    ) -> None:
        self._state = shared_state
        self._cfg = cfg
        self._storage = storage
        self._analysis = analysis
        self._capture = capture
        self._retrieval = retrieval
        self._proactive = proactive
        self._llm_enhanced = llm_enhanced
        self._kg = kg
        self._member_identity = member_identity
        self._image_analyzer = image_analyzer
        self._activity_tracker = activity_tracker
        self._activity_provider = activity_provider
        self._cached_put_kv_data: Optional[Any] = None

    # ── 可选组件更新 ──

    def set_member_identity(self, identity: Any) -> None:
        """更新成员身份服务引用"""
        self._member_identity = identity

    def set_image_analyzer(self, analyzer: Any) -> None:
        """更新图片分析器引用"""
        self._image_analyzer = analyzer

    def set_activity_tracker(self, tracker: Any) -> None:
        """更新群活跃度跟踪器引用"""
        self._activity_tracker = tracker

    def set_activity_provider(self, provider: Any) -> None:
        """更新活跃度配置提供者引用"""
        self._activity_provider = provider

    # ── KV 加载 ──

    async def load_from_kv(self, get_kv_data: Any) -> None:
        """从 KV 存储加载数据"""
        try:
            await self._load_session_data(get_kv_data)
            await self._load_lifecycle_state(get_kv_data)
            await self._load_batch_queues(get_kv_data)
            await self._load_chat_history(get_kv_data)
            await self._load_proactive_whitelist(get_kv_data)
            await self._load_member_identity(get_kv_data)
            await self._load_activity_data(get_kv_data)
            await self._load_user_personas(get_kv_data)
            await self._load_persona_batch_queues(get_kv_data)

        except Exception as e:
            logger.error(f"Failed to load from KV: {e}", exc_info=True)

    async def _load_session_data(self, get_kv_data: Any) -> None:
        """加载会话数据"""
        if not self._storage.session_manager:
            return

        sessions_data = await get_kv_data(KVStoreKeys.SESSIONS, {})
        if sessions_data:
            await self._storage.session_manager.deserialize_from_kv_storage(sessions_data)
            logger.debug(LogTemplates.SESSION_LOADED.format(
                count=self._storage.session_manager.get_session_count()
            ))

    async def _load_lifecycle_state(self, get_kv_data: Any) -> None:
        """加载生命周期状态"""
        if not self._storage.lifecycle_manager:
            return

        lifecycle_state = await get_kv_data(KVStoreKeys.LIFECYCLE_STATE, {})
        if lifecycle_state:
            await self._storage.lifecycle_manager.deserialize_state(lifecycle_state)
            logger.debug("Loaded lifecycle state")

    async def _load_batch_queues(self, get_kv_data: Any) -> None:
        """加载批量处理器队列"""
        if not self._capture.batch_processor:
            return

        batch_queues = await get_kv_data(KVStoreKeys.BATCH_QUEUES, {})
        if batch_queues:
            await self._capture.batch_processor.deserialize_queues(batch_queues)
            logger.debug("Loaded batch processor queues")

    async def _load_chat_history(self, get_kv_data: Any) -> None:
        """加载聊天记录缓冲区"""
        if not self._storage.chat_history_buffer:
            return

        chat_history = await get_kv_data(KVStoreKeys.CHAT_HISTORY, {})
        if chat_history:
            await self._storage.chat_history_buffer.deserialize(chat_history)
            logger.debug("Loaded chat history buffer")

    async def _load_proactive_whitelist(self, get_kv_data: Any) -> None:
        """加载主动回复白名单"""
        if not self._proactive.proactive_manager:
            return

        whitelist_data = await get_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, [])
        if whitelist_data:
            self._proactive.proactive_manager.deserialize_whitelist(whitelist_data)
            logger.debug("Loaded proactive reply whitelist")

    async def _load_member_identity(self, get_kv_data: Any) -> None:
        """加载成员身份数据"""
        if not self._member_identity:
            return

        identity_data = await get_kv_data(KVStoreKeys.MEMBER_IDENTITY, {})
        if identity_data:
            self._member_identity.deserialize(identity_data)
            stats = self._member_identity.get_stats()
            logger.debug(
                f"Loaded member identity data: "
                f"{stats['total_profiles']} profiles, "
                f"{stats['total_groups']} groups"
            )

    async def _load_activity_data(self, get_kv_data: Any) -> None:
        """加载群活跃度数据"""
        if not self._activity_tracker:
            return

        activity_data = await get_kv_data(KVStoreKeys.GROUP_ACTIVITY, {})
        if activity_data:
            self._activity_tracker.deserialize(activity_data)
            logger.debug("Loaded group activity states")

    async def _load_user_personas(self, get_kv_data: Any) -> None:
        """加载用户画像"""
        if not self._cfg.get("persona.enabled", True):
            return

        from iris_memory.models.user_persona import UserPersona
        from iris_memory.analysis.persona.persona_logger import persona_log

        personas_data = await get_kv_data(KVStoreKeys.USER_PERSONAS, {})
        if personas_data:
            persona_log.restore_start(len(personas_data))
            success_count = 0
            fail_count = 0
            for uid, pdata in personas_data.items():
                try:
                    self._state.user_personas[uid] = UserPersona.from_dict(pdata)
                    persona_log.restore_ok(uid)
                    success_count += 1
                except Exception as e:
                    persona_log.restore_error(uid, e)
                    fail_count += 1
            persona_log.restore_summary(len(personas_data), success_count, fail_count)
            logger.debug(f"Loaded {len(self._state.user_personas)} user personas")

    async def _load_persona_batch_queues(self, get_kv_data: Any) -> None:
        """加载画像批量处理器队列（重启时清空策略）"""
        if not self._cfg.get("persona.enabled", True):
            return

        persona_batch = self._analysis.persona_batch_processor
        if not persona_batch:
            return

        data = await get_kv_data(KVStoreKeys.PERSONA_BATCH_QUEUES, {})
        if data:
            await persona_batch.deserialize_and_clear(data)
            logger.debug("Loaded persona batch processor state (queues cleared)")

    # ── KV 保存 ──

    async def save_to_kv(self, put_kv_data: Any) -> None:
        """保存到 KV 存储"""
        try:
            # 缓存 put_kv_data 供批量处理器自动保存回调使用
            self._cached_put_kv_data = put_kv_data

            await self._save_session_data(put_kv_data)
            await self._save_batch_queues(put_kv_data)
            await self._save_chat_history(put_kv_data)
            await self._save_proactive_whitelist(put_kv_data)
            await self._save_member_identity(put_kv_data)
            await self._save_activity_data(put_kv_data)
            await self._save_user_personas(put_kv_data)
            await self._save_persona_batch_queues(put_kv_data)

        except Exception as e:
            logger.error(f"Failed to save to KV: {e}", exc_info=True)

    async def _save_session_data(self, put_kv_data: Any) -> None:
        """保存会话数据"""
        if not self._storage.session_manager:
            return

        sessions_data = await self._storage.session_manager.serialize_for_kv_storage()
        await put_kv_data(KVStoreKeys.SESSIONS, sessions_data)
        logger.debug(LogTemplates.SESSION_SAVED.format(
            count=self._storage.session_manager.get_session_count()
        ))

    async def _save_batch_queues(self, put_kv_data: Optional[Any] = None) -> None:
        """保存批量处理器队列

        Args:
            put_kv_data: KV 写入函数。为 None 时尝试使用上次缓存的函数，
                以支持批量处理器的无参回调。
        """
        if not self._capture.batch_processor:
            return

        put_func = put_kv_data or self._cached_put_kv_data
        if put_func is None:
            logger.debug("Skip saving batch queues: put_kv_data callback is unavailable")
            return

        batch_queues = await self._capture.batch_processor.serialize_queues()
        await put_func(KVStoreKeys.BATCH_QUEUES, batch_queues)
        logger.debug("Saved batch processor queues")

    async def _save_chat_history(self, put_kv_data: Any) -> None:
        """保存聊天记录缓冲区"""
        if not self._storage.chat_history_buffer:
            return

        chat_history = await self._storage.chat_history_buffer.serialize()
        await put_kv_data(KVStoreKeys.CHAT_HISTORY, chat_history)
        logger.debug("Saved chat history buffer")

    async def _save_proactive_whitelist(self, put_kv_data: Any) -> None:
        """保存主动回复白名单"""
        if not self._proactive.proactive_manager:
            return

        whitelist_data = self._proactive.proactive_manager.serialize_whitelist()
        await put_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, whitelist_data)
        logger.debug("Saved proactive reply whitelist")

    async def _save_member_identity(self, put_kv_data: Any) -> None:
        """保存成员身份数据"""
        if not self._member_identity:
            return

        identity_data = self._member_identity.serialize()
        await put_kv_data(KVStoreKeys.MEMBER_IDENTITY, identity_data)
        logger.debug("Saved member identity data")

    async def _save_activity_data(self, put_kv_data: Any) -> None:
        """保存群活跃度数据"""
        if not self._activity_tracker:
            return

        activity_data = self._activity_tracker.serialize()
        await put_kv_data(KVStoreKeys.GROUP_ACTIVITY, activity_data)
        logger.debug("Saved group activity states")

    async def _save_user_personas(self, put_kv_data: Any) -> None:
        """保存用户画像"""
        if not self._cfg.get("persona.enabled", True):
            return

        if not self._state.user_personas:
            return

        from iris_memory.analysis.persona.persona_logger import persona_log

        personas_data = {}
        for uid, persona in self._state.user_personas.items():
            try:
                persona_log.persist_start(uid)
                personas_data[uid] = persona.to_dict()
                persona_log.persist_ok(uid, persona.update_count)
            except Exception as e:
                persona_log.persist_error(uid, e)
        await put_kv_data(KVStoreKeys.USER_PERSONAS, personas_data)
        logger.debug(f"Saved {len(personas_data)} user personas")

    async def _save_persona_batch_queues(self, put_kv_data: Any) -> None:
        """保存画像批量处理器队列"""
        if not self._cfg.get("persona.enabled", True):
            return

        persona_batch = self._analysis.persona_batch_processor
        if not persona_batch:
            return

        data = await persona_batch.serialize_queues()
        await put_kv_data(KVStoreKeys.PERSONA_BATCH_QUEUES, data)
        logger.debug("Saved persona batch processor state")

    # ── 服务销毁 ──

    async def terminate(self) -> None:
        """销毁服务

        热更新友好：
        1. 按依赖顺序停止后台任务（先停消费者，再停生产者）
        2. 等待所有任务完成
        3. 清理全局状态引用
        4. 关闭底层存储
        """
        logger.debug("[Hot-Reload] Terminating memory service...")

        try:
            await self._capture.stop()
            await self._analysis.stop()
            await self._proactive.stop()
            await self._storage.stop()
            self._clear_global_state()

            self._log_final_stats()

            logger.debug(LogTemplates.PLUGIN_TERMINATED)

        except Exception as e:
            logger.error(LogTemplates.PLUGIN_TERMINATE_ERROR.format(error=e), exc_info=True)

    def _clear_global_state(self) -> None:
        """清理全局状态引用"""
        set_identity_service(None)

        from iris_memory.core.service_container import ServiceContainer
        ServiceContainer.instance().clear()
        logger.debug("[Hot-Reload] ServiceContainer and global state cleared")

    def _log_final_stats(self) -> None:
        """输出最终统计"""
        logger.debug(LogTemplates.FINAL_STATS_HEADER)

        components = [
            ("Message Classifier", self._capture.message_classifier),
            ("Batch Processor", self._capture.batch_processor),
            ("Persona Batch Processor", self._analysis.persona_batch_processor),
            ("LLM Processor", self._llm_enhanced.llm_processor if self._llm_enhanced else None),
            ("Proactive Manager", self._proactive.proactive_manager),
            ("Image Analyzer", self._image_analyzer),
        ]

        for name, component in components:
            if component and hasattr(component, 'get_stats'):
                try:
                    stats = component.get_stats()
                    logger.debug(f"{name}: {stats}")
                except Exception as e:
                    logger.debug(f"Failed to get stats from {name}: {e}")
