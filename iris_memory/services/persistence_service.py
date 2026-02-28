"""
持久化操作服务

将持久化逻辑从 MemoryService Mixin 中分离为独立的组合式服务类，
通过构造函数显式接收所有依赖，消除隐式耦合。

设计动机：
- 原 PersistenceOperations 作为 Mixin 通过 self.xxx 隐式访问 MemoryService 属性
- 转为组合模式后，所有依赖通过构造函数显式注入

重构历史：
- 2026-03-01: 配置驱动化重构，将 18 个 _load_xxx/_save_xxx 方法简化为配置表驱动
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, Awaitable

from iris_memory.utils.logger import get_logger
from iris_memory.core.constants import KVStoreKeys, LogTemplates
from iris_memory.utils.member_utils import set_identity_service
from iris_memory.services.shared_state import SharedState

logger = get_logger("memory_service.persistence")


@dataclass
class KVLoaderConfig:
    """KV 加载配置"""
    key: str
    component_path: str
    deserialize_method: str
    default: Any = field(default_factory=dict)
    config_check: Optional[str] = None
    log_message: Optional[str] = None
    is_async: bool = True
    special_handler: Optional[str] = None


@dataclass
class KVSaveConfig:
    """KV 保存配置"""
    key: str
    component_path: str
    serialize_method: str
    config_check: Optional[str] = None
    log_message: Optional[str] = None
    is_async: bool = True
    special_handler: Optional[str] = None


_KV_LOADERS: list[KVLoaderConfig] = [
    KVLoaderConfig(
        key=KVStoreKeys.SESSIONS,
        component_path="_storage.session_manager",
        deserialize_method="deserialize_from_kv_storage",
        log_message="Loaded {count} sessions",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.LIFECYCLE_STATE,
        component_path="_storage.lifecycle_manager",
        deserialize_method="deserialize_state",
        log_message="Loaded lifecycle state",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.BATCH_QUEUES,
        component_path="_capture.batch_processor",
        deserialize_method="deserialize_queues",
        log_message="Loaded batch processor queues",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.CHAT_HISTORY,
        component_path="_storage.chat_history_buffer",
        deserialize_method="deserialize",
        log_message="Loaded chat history buffer",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
        component_path="_proactive.proactive_manager",
        deserialize_method="deserialize_whitelist",
        default=[],
        log_message="Loaded proactive reply whitelist",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.MEMBER_IDENTITY,
        component_path="_member_identity",
        deserialize_method="deserialize",
        log_message="Loaded member identity data",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.GROUP_ACTIVITY,
        component_path="_activity_tracker",
        deserialize_method="deserialize",
        log_message="Loaded group activity states",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.USER_PERSONAS,
        component_path="",
        deserialize_method="",
        config_check="persona.enabled",
        special_handler="_load_user_personas",
    ),
    KVLoaderConfig(
        key=KVStoreKeys.PERSONA_BATCH_QUEUES,
        component_path="",
        deserialize_method="",
        config_check="persona.enabled",
        special_handler="_load_persona_batch_queues",
    ),
]

_KV_SAVERS: list[KVSaveConfig] = [
    KVSaveConfig(
        key=KVStoreKeys.SESSIONS,
        component_path="_storage.session_manager",
        serialize_method="serialize_for_kv_storage",
        log_message="Saved {count} sessions",
    ),
    KVSaveConfig(
        key=KVStoreKeys.BATCH_QUEUES,
        component_path="_capture.batch_processor",
        serialize_method="serialize_queues",
        log_message="Saved batch processor queues",
    ),
    KVSaveConfig(
        key=KVStoreKeys.CHAT_HISTORY,
        component_path="_storage.chat_history_buffer",
        serialize_method="serialize",
        log_message="Saved chat history buffer",
    ),
    KVSaveConfig(
        key=KVStoreKeys.PROACTIVE_REPLY_WHITELIST,
        component_path="_proactive.proactive_manager",
        serialize_method="serialize_whitelist",
        log_message="Saved proactive reply whitelist",
    ),
    KVSaveConfig(
        key=KVStoreKeys.MEMBER_IDENTITY,
        component_path="_member_identity",
        serialize_method="serialize",
        log_message="Saved member identity data",
    ),
    KVSaveConfig(
        key=KVStoreKeys.GROUP_ACTIVITY,
        component_path="_activity_tracker",
        serialize_method="serialize",
        log_message="Saved group activity states",
    ),
    KVSaveConfig(
        key=KVStoreKeys.USER_PERSONAS,
        component_path="",
        serialize_method="",
        config_check="persona.enabled",
        special_handler="_save_user_personas",
    ),
    KVSaveConfig(
        key=KVStoreKeys.PERSONA_BATCH_QUEUES,
        component_path="",
        serialize_method="",
        config_check="persona.enabled",
        special_handler="_save_persona_batch_queues",
    ),
]


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
        """从 KV 存储加载数据（配置驱动）"""
        try:
            for config in _KV_LOADERS:
                await self._execute_loader(config, get_kv_data)
        except Exception as e:
            logger.error(f"Failed to load from KV: {e}", exc_info=True)

    async def _execute_loader(self, config: KVLoaderConfig, get_kv_data: Any) -> None:
        """执行单个加载配置"""
        if config.config_check and not self._cfg.get(config.config_check, True):
            return

        if config.special_handler:
            await getattr(self, config.special_handler)(get_kv_data)
            return

        component = self._resolve_component(config.component_path)
        if not component:
            return

        data = await get_kv_data(config.key, config.default)
        if not data:
            return

        method = getattr(component, config.deserialize_method)
        if config.is_async:
            await method(data)
        else:
            method(data)

        if config.log_message:
            self._log_loader_result(config, component)

    def _resolve_component(self, path: str) -> Any:
        """解析组件路径"""
        if not path:
            return None
        obj = self
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def _log_loader_result(self, config: KVLoaderConfig, component: Any) -> None:
        """记录加载结果日志"""
        if "{count}" in config.log_message:
            if hasattr(component, "get_session_count"):
                count = component.get_session_count()
            elif hasattr(component, "__len__"):
                count = len(component)
            else:
                count = 0
            logger.debug(config.log_message.format(count=count))
        elif config.key == KVStoreKeys.MEMBER_IDENTITY and hasattr(component, "get_stats"):
            stats = component.get_stats()
            logger.debug(
                f"Loaded member identity data: "
                f"{stats['total_profiles']} profiles, "
                f"{stats['total_groups']} groups"
            )
        else:
            logger.debug(config.log_message)

    async def _load_user_personas(self, get_kv_data: Any) -> None:
        """加载用户画像（特殊处理）"""
        from iris_memory.models.user_persona import UserPersona
        from iris_memory.analysis.persona.persona_logger import persona_log

        personas_data = await get_kv_data(KVStoreKeys.USER_PERSONAS, {})
        if not personas_data:
            return

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
        persona_batch = self._analysis.persona_batch_processor
        if not persona_batch:
            return

        data = await get_kv_data(KVStoreKeys.PERSONA_BATCH_QUEUES, {})
        if data:
            await persona_batch.deserialize_and_clear(data)
            logger.debug("Loaded persona batch processor state (queues cleared)")

    # ── KV 保存 ──

    async def save_to_kv(self, put_kv_data: Any) -> None:
        """保存到 KV 存储（配置驱动）"""
        try:
            self._cached_put_kv_data = put_kv_data
            for config in _KV_SAVERS:
                await self._execute_saver(config, put_kv_data)
        except Exception as e:
            logger.error(f"Failed to save to KV: {e}", exc_info=True)

    async def _execute_saver(self, config: KVSaveConfig, put_kv_data: Any) -> None:
        """执行单个保存配置"""
        if config.config_check and not self._cfg.get(config.config_check, True):
            return

        if config.special_handler:
            await getattr(self, config.special_handler)(put_kv_data)
            return

        component = self._resolve_component(config.component_path)
        if not component:
            return

        method = getattr(component, config.serialize_method)
        if config.is_async:
            data = await method()
        else:
            data = method()

        await put_kv_data(config.key, data)

        if config.log_message:
            self._log_saver_result(config, component)

    def _log_saver_result(self, config: KVSaveConfig, component: Any) -> None:
        """记录保存结果日志"""
        if "{count}" in config.log_message:
            if hasattr(component, "get_session_count"):
                count = component.get_session_count()
            elif hasattr(component, "__len__"):
                count = len(component)
            else:
                count = 0
            logger.debug(config.log_message.format(count=count))
        else:
            logger.debug(config.log_message)

    async def _save_batch_queues(self, put_kv_data: Optional[Any] = None) -> None:
        """保存批量处理器队列（支持无参回调）"""
        if not self._capture.batch_processor:
            return

        put_func = put_kv_data or self._cached_put_kv_data
        if put_func is None:
            return

        batch_queues = await self._capture.batch_processor.serialize_queues()
        await put_func(KVStoreKeys.BATCH_QUEUES, batch_queues)
        logger.debug("Saved batch processor queues")

    async def _save_user_personas(self, put_kv_data: Any) -> None:
        """保存用户画像（特殊处理）"""
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
