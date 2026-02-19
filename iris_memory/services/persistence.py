"""
MemoryService 持久化模块

将持久化逻辑从 MemoryService 中拆分出来，提高代码可维护性。
"""

from typing import Dict, Any

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import KVStoreKeys, LogTemplates
from iris_memory.utils.member_utils import set_identity_service

logger = get_logger("memory_service.persistence")


class PersistenceOperations:
    """MemoryService 持久化操作 Mixin
    
    职责：
    1. KV存储加载
    2. KV存储保存
    3. 服务销毁
    """

    async def load_from_kv(self, get_kv_data) -> None:
        """从KV存储加载数据"""
        try:
            await self._load_session_data(get_kv_data)
            await self._load_lifecycle_state(get_kv_data)
            await self._load_batch_queues(get_kv_data)
            await self._load_chat_history(get_kv_data)
            await self._load_proactive_whitelist(get_kv_data)
            await self._load_member_identity(get_kv_data)
            await self._load_activity_data(get_kv_data)
            await self._load_user_personas(get_kv_data)
            
        except Exception as e:
            logger.error(f"Failed to load from KV: {e}", exc_info=True)

    async def _load_session_data(self, get_kv_data) -> None:
        """加载会话数据"""
        if not self.session_manager:
            return
        
        from iris_memory.analysis.persona.persona_logger import persona_log
        
        sessions_data = await get_kv_data(KVStoreKeys.SESSIONS, {})
        if sessions_data:
            await self.session_manager.deserialize_from_kv_storage(sessions_data)
            logger.info(LogTemplates.SESSION_LOADED.format(
                count=self.session_manager.get_session_count()
            ))

    async def _load_lifecycle_state(self, get_kv_data) -> None:
        """加载生命周期状态"""
        if not self.lifecycle_manager:
            return
        
        lifecycle_state = await get_kv_data(KVStoreKeys.LIFECYCLE_STATE, {})
        if lifecycle_state:
            await self.lifecycle_manager.deserialize_state(lifecycle_state)
            logger.info("Loaded lifecycle state")

    async def _load_batch_queues(self, get_kv_data) -> None:
        """加载批量处理器队列"""
        if not self.batch_processor:
            return
        
        batch_queues = await get_kv_data(KVStoreKeys.BATCH_QUEUES, {})
        if batch_queues:
            await self.batch_processor.deserialize_queues(batch_queues)
            logger.info("Loaded batch processor queues")

    async def _load_chat_history(self, get_kv_data) -> None:
        """加载聊天记录缓冲区"""
        if not self.chat_history_buffer:
            return
        
        chat_history = await get_kv_data(KVStoreKeys.CHAT_HISTORY, {})
        if chat_history:
            await self.chat_history_buffer.deserialize(chat_history)
            logger.info("Loaded chat history buffer")

    async def _load_proactive_whitelist(self, get_kv_data) -> None:
        """加载主动回复白名单"""
        if not self.proactive_manager:
            return
        
        whitelist_data = await get_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, [])
        if whitelist_data:
            self.proactive_manager.deserialize_whitelist(whitelist_data)
            logger.info("Loaded proactive reply whitelist")

    async def _load_member_identity(self, get_kv_data) -> None:
        """加载成员身份数据"""
        if not self._member_identity:
            return
        
        identity_data = await get_kv_data(KVStoreKeys.MEMBER_IDENTITY, {})
        if identity_data:
            self._member_identity.deserialize(identity_data)
            stats = self._member_identity.get_stats()
            logger.info(
                f"Loaded member identity data: "
                f"{stats['total_profiles']} profiles, "
                f"{stats['total_groups']} groups"
            )

    async def _load_activity_data(self, get_kv_data) -> None:
        """加载群活跃度数据"""
        if not self._activity_tracker:
            return
        
        activity_data = await get_kv_data(KVStoreKeys.GROUP_ACTIVITY, {})
        if activity_data:
            self._activity_tracker.deserialize(activity_data)
            logger.info("Loaded group activity states")

    async def _load_user_personas(self, get_kv_data) -> None:
        """加载用户画像"""
        from iris_memory.models.user_persona import UserPersona
        from iris_memory.analysis.persona.persona_logger import persona_log
        
        personas_data = await get_kv_data(KVStoreKeys.USER_PERSONAS, {})
        if personas_data:
            persona_log.restore_start(len(personas_data))
            success_count = 0
            fail_count = 0
            for uid, pdata in personas_data.items():
                try:
                    self._user_personas[uid] = UserPersona.from_dict(pdata)
                    persona_log.restore_ok(uid)
                    success_count += 1
                except Exception as e:
                    persona_log.restore_error(uid, e)
                    fail_count += 1
            persona_log.restore_summary(len(personas_data), success_count, fail_count)
            logger.info(f"Loaded {len(self._user_personas)} user personas")

    async def save_to_kv(self, put_kv_data) -> None:
        """保存到KV存储"""
        try:
            await self._save_session_data(put_kv_data)
            await self._save_batch_queues(put_kv_data)
            await self._save_chat_history(put_kv_data)
            await self._save_proactive_whitelist(put_kv_data)
            await self._save_member_identity(put_kv_data)
            await self._save_activity_data(put_kv_data)
            await self._save_user_personas(put_kv_data)
            
        except Exception as e:
            logger.error(f"Failed to save to KV: {e}", exc_info=True)

    async def _save_session_data(self, put_kv_data) -> None:
        """保存会话数据"""
        if not self.session_manager:
            return
        
        sessions_data = await self.session_manager.serialize_for_kv_storage()
        await put_kv_data(KVStoreKeys.SESSIONS, sessions_data)
        logger.info(LogTemplates.SESSION_SAVED.format(
            count=self.session_manager.get_session_count()
        ))

    async def _save_batch_queues(self, put_kv_data) -> None:
        """保存批量处理器队列"""
        if not self.batch_processor:
            return
        
        batch_queues = await self.batch_processor.serialize_queues()
        await put_kv_data(KVStoreKeys.BATCH_QUEUES, batch_queues)
        logger.info("Saved batch processor queues")

    async def _save_chat_history(self, put_kv_data) -> None:
        """保存聊天记录缓冲区"""
        if not self.chat_history_buffer:
            return
        
        chat_history = await self.chat_history_buffer.serialize()
        await put_kv_data(KVStoreKeys.CHAT_HISTORY, chat_history)
        logger.info("Saved chat history buffer")

    async def _save_proactive_whitelist(self, put_kv_data) -> None:
        """保存主动回复白名单"""
        if not self.proactive_manager:
            return
        
        whitelist_data = self.proactive_manager.serialize_whitelist()
        await put_kv_data(KVStoreKeys.PROACTIVE_REPLY_WHITELIST, whitelist_data)
        logger.info("Saved proactive reply whitelist")

    async def _save_member_identity(self, put_kv_data) -> None:
        """保存成员身份数据"""
        if not self._member_identity:
            return
        
        identity_data = self._member_identity.serialize()
        await put_kv_data(KVStoreKeys.MEMBER_IDENTITY, identity_data)
        logger.info("Saved member identity data")

    async def _save_activity_data(self, put_kv_data) -> None:
        """保存群活跃度数据"""
        if not self._activity_tracker:
            return
        
        activity_data = self._activity_tracker.serialize()
        await put_kv_data(KVStoreKeys.GROUP_ACTIVITY, activity_data)
        logger.info("Saved group activity states")

    async def _save_user_personas(self, put_kv_data) -> None:
        """保存用户画像"""
        if not self._user_personas:
            return
        
        from iris_memory.analysis.persona.persona_logger import persona_log
        
        personas_data = {}
        for uid, persona in self._user_personas.items():
            try:
                persona_log.persist_start(uid)
                personas_data[uid] = persona.to_dict()
                persona_log.persist_ok(uid, persona.update_count)
            except Exception as e:
                persona_log.persist_error(uid, e)
        await put_kv_data(KVStoreKeys.USER_PERSONAS, personas_data)
        logger.info(f"Saved {len(personas_data)} user personas")

    async def _save_batch_queues(self) -> None:
        """保存批量队列（供回调使用）"""
        pass

    async def terminate(self) -> None:
        """销毁服务
        
        热更新友好：
        1. 立即标记为未初始化，阻止新操作进入
        2. 按依赖顺序停止后台任务（先停消费者，再停生产者）
        3. 等待所有任务完成
        4. 清理全局状态引用
        5. 关闭底层存储
        """
        logger.info("[Hot-Reload] Terminating memory service...")
        
        self._is_initialized = False
        
        try:
            await self.capture.stop()
            await self.proactive.stop()
            await self.storage.stop()
            self._clear_global_state()
            
            self._log_final_stats()
            
            logger.info(LogTemplates.PLUGIN_TERMINATED)
            
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
        logger.info(LogTemplates.FINAL_STATS_HEADER)
        
        components = [
            ("Message Classifier", self.message_classifier),
            ("Batch Processor", self.batch_processor),
            ("LLM Processor", self.llm_enhanced.llm_processor if hasattr(self, 'llm_enhanced') else None),
            ("Proactive Manager", self.proactive_manager),
            ("Image Analyzer", self._image_analyzer),
        ]
        
        for name, component in components:
            if component and hasattr(component, 'get_stats'):
                try:
                    stats = component.get_stats()
                    logger.info(f"{name}: {stats}")
                except Exception as e:
                    logger.debug(f"Failed to get stats from {name}: {e}")
