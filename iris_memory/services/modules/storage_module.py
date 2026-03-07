"""
存储模块 — 聚合所有存储基础设施组件

包含：ChromaManager, SessionManager, LifecycleManager, CacheManager, ChatHistoryBuffer
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.storage.chroma_manager import ChromaManager
    from iris_memory.storage.session_manager import SessionManager
    from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
    from iris_memory.storage.cache import CacheManager
    from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer

logger = get_logger("module.storage")


class StorageModule:
    """存储模块

    聚合底层存储组件，统一对外暴露存储能力。
    """

    def __init__(self) -> None:
        self._chroma_manager: Optional[ChromaManager] = None
        self._session_manager: Optional[SessionManager] = None
        self._lifecycle_manager: Optional[SessionLifecycleManager] = None
        self._cache_manager: Optional[CacheManager] = None
        self._chat_history_buffer: Optional[ChatHistoryBuffer] = None

    # ── 属性访问 ──

    @property
    def chroma_manager(self) -> Optional[ChromaManager]:
        return self._chroma_manager

    @property
    def session_manager(self) -> Optional[SessionManager]:
        return self._session_manager

    @property
    def lifecycle_manager(self) -> Optional[SessionLifecycleManager]:
        return self._lifecycle_manager

    @property
    def cache_manager(self) -> Optional[CacheManager]:
        return self._cache_manager

    @property
    def chat_history_buffer(self) -> Optional[ChatHistoryBuffer]:
        return self._chat_history_buffer

    # ── 初始化 ──

    async def initialize(
        self,
        config: Any,
        cfg: Any,
        plugin_data_path: Any,
        context: Any,
    ) -> None:
        """初始化所有存储组件"""
        from iris_memory.storage.chroma_manager import ChromaManager
        from iris_memory.storage.session_manager import SessionManager
        from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
        from iris_memory.storage.cache import CacheManager
        from iris_memory.storage.chat_history_buffer import ChatHistoryBuffer
        from iris_memory.config import get_store

        # ChromaManager
        self._chroma_manager = ChromaManager(config, plugin_data_path, context)
        await self._chroma_manager.initialize()

        # SessionManager
        self._session_manager = SessionManager(
            max_working_memory=cfg.get("memory.max_working_memory", 10),
            max_sessions=get_store().get("session.max_sessions"),
            ttl=cfg.get("session.session_timeout", 86400),
        )

        # CacheManager
        self._cache_manager = CacheManager({})

        # LifecycleManager
        self._lifecycle_manager = SessionLifecycleManager(
            session_manager=self._session_manager,
            chroma_manager=self._chroma_manager,
            upgrade_mode=cfg.get("memory.upgrade_mode", "rule"),
            llm_upgrade_batch_size=get_store().get("memory.llm_upgrade_batch_size"),
            llm_upgrade_threshold=get_store().get("memory.llm_upgrade_threshold"),
        )
        # 注入 AstrBot 上下文供语义提取 LLM 调用使用
        self._lifecycle_manager.set_astrbot_context(context, cfg.get("llm_providers.memory_provider_id", None))
        await self._lifecycle_manager.start()

        # ChatHistoryBuffer
        self._chat_history_buffer = ChatHistoryBuffer(
            max_messages=cfg.chat_context_count,
        )

        logger.debug("StorageModule initialized")

    def is_embedding_ready(self) -> bool:
        """检查 embedding 系统是否就绪"""
        if not self._chroma_manager:
            return False
        return self._chroma_manager.embedding_manager.is_ready

    # ── 配置应用 ──

    def apply_config(self, cfg: Any) -> None:
        """将用户配置应用到各存储组件"""
        from iris_memory.config import get_store
        from iris_memory.storage.cache import CacheManager

        if self._cache_manager:
            self._cache_manager = CacheManager({
                "embedding_cache": {
                    "max_size": get_store().get("cache.embedding_cache_size"),
                    "strategy": get_store().get("cache.embedding_cache_strategy"),
                },
                "working_cache": {
                    "max_sessions": get_store().get("session.max_sessions"),
                    "max_memories_per_session": cfg.get("memory.max_working_memory", 10),
                    "ttl": get_store().get("cache.working_cache_ttl"),
                },
                "compression": {
                    "max_length": get_store().get("cache.compression_max_length"),
                },
            })

        if self._session_manager:
            self._session_manager.max_working_memory = cfg.get("memory.max_working_memory", 10)
            self._session_manager.max_sessions = get_store().get("session.max_sessions")
            self._session_manager.ttl = cfg.get("session.session_timeout", 86400)

        if self._lifecycle_manager:
            self._lifecycle_manager.cleanup_interval = get_store().get("session.session_cleanup_interval")
            self._lifecycle_manager.session_timeout = cfg.get("session.session_timeout", 86400)
            self._lifecycle_manager.inactive_timeout = get_store().get("session.session_inactive_timeout")

        if self._chat_history_buffer:
            self._chat_history_buffer.set_max_messages(cfg.get("advanced.chat_context_count", 15))

    # ── 生命周期 ──

    async def stop(self) -> None:
        """停止所有后台任务并关闭存储"""
        if self._lifecycle_manager:
            try:
                await self._lifecycle_manager.stop()
                logger.debug("[Hot-Reload] Lifecycle manager stopped")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error stopping lifecycle manager: {e}")

        if self._chroma_manager:
            try:
                await self._chroma_manager.close()
                logger.debug("[Hot-Reload] Chroma manager closed")
            except Exception as e:
                logger.warning(f"[Hot-Reload] Error closing Chroma manager: {e}")
