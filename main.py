"""
Iris Memory Plugin - 主入口
基于companion-memory框架的三层记忆插件
"""

import sys
from pathlib import Path

# 将插件根目录添加到Python路径，以便能够导入iris_memory模块
plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig, logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from iris_memory.models.memory import Memory
from iris_memory.models.user_persona import UserPersona
from iris_memory.storage.chroma_manager import ChromaManager
from iris_memory.capture.capture_engine import MemoryCaptureEngine
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.storage.session_manager import SessionManager
from iris_memory.storage.lifecycle_manager import SessionLifecycleManager
from iris_memory.storage.cache import CacheManager, WorkingMemoryCache
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.hook_manager import MemoryInjector, InjectionMode, HookPriority


@register("iris_memory", "YourName", "基于companion-memory框架的三层记忆插件", "1.0.0")
class IrisMemoryPlugin(Star):
    """Iris记忆插件
    
    实现三层记忆模型：
    - 工作记忆：会话内临时存储
    - 情景记忆：基于RIF评分动态管理
    - 语义记忆：永久保存用户画像
    
    支持私聊和群聊的完全隔离。
    """
    
    def __init__(self, context: Context, config: AstrBotConfig):
        """初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置对象
        """
        super().__init__(context)
        self.context = context
        self.config = config
        
        # 插件数据目录
        self.plugin_data_path = get_astrbot_data_path() / "plugin_data" / self.name
        
        # 初始化核心组件（将在initialize中完成）
        self.chroma_manager = None
        self.capture_engine = None
        self.retrieval_engine = None
        self.session_manager = None
        self.lifecycle_manager = None
        self.cache_manager = None
        self.working_memory_cache = None
        self.emotion_analyzer = None
        self.rif_scorer = None
        self.memory_injector = None
        
        # 用户情感状态缓存：{user_id: EmotionalState}
        self.user_emotional_states: dict = {}
        
        # 用户画像缓存：{user_id: UserPersona}
        self.user_personas: dict = {}
    
    async def initialize(self):
        """异步初始化插件"""
        try:
            logger.info("IrisMemory plugin initializing...")
            
            # 创建插件数据目录
            self.plugin_data_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化情感分析器
            self.emotion_analyzer = EmotionAnalyzer(self.config)
            
            # 初始化RIF评分器
            self.rif_scorer = RIFScorer()
            
            # 初始化Chroma管理器（传入插件上下文用于嵌入API）
            self.chroma_manager = ChromaManager(self.config, self.plugin_data_path, self.context)
            await self.chroma_manager.initialize()
            
            # 初始化会话管理器
            self.session_manager = SessionManager()
            
            # 初始化缓存管理器
            self.cache_manager = CacheManager({})
            
            # 初始化工作记忆缓存（默认配置，在_apply_config中更新）
            self.working_memory_cache = WorkingMemoryCache(
                max_sessions=100,
                max_memories_per_session=10,
                ttl=86400
            )
            
            # 初始化生命周期管理器
            self.lifecycle_manager = SessionLifecycleManager(
                session_manager=self.session_manager
            )
            await self.lifecycle_manager.start()
            
            # 初始化记忆捕获引擎（传入chroma_manager以启用去重和冲突检测）
            self.capture_engine = MemoryCaptureEngine(
                chroma_manager=self.chroma_manager,
                emotion_analyzer=self.emotion_analyzer,
                rif_scorer=self.rif_scorer
            )
            
            # 初始化记忆检索引擎
            self.retrieval_engine = MemoryRetrievalEngine(
                chroma_manager=self.chroma_manager,
                rif_scorer=self.rif_scorer,
                emotion_analyzer=self.emotion_analyzer
            )
            
            # 初始化记忆注入器
            self.memory_injector = MemoryInjector(
                injection_mode=InjectionMode.SUFFIX,
                priority=HookPriority.NORMAL,
                namespace="iris_memory"
            )
            
            # 配置参数
            self._apply_config()
            
            # 加载会话数据
            await self._load_sessions()
            
            logger.info("IrisMemory plugin initialized successfully")
            
        except Exception as e:
            logger.error(f"IrisMemory plugin initialization failed: {e}")
            raise
    
    def _apply_config(self):
        """应用配置"""
        # 获取记忆配置
        auto_capture = self._get_config("memory_config.auto_capture", True)
        max_working_memory = self._get_config("memory_config.max_working_memory", 10)
        rif_threshold = self._get_config("memory_config.rif_threshold", 0.4)
        
        # 应用缓存配置
        cache_config = self._get_config("cache_config", {})
        if self.cache_manager:
            self.cache_manager = CacheManager({
                'embedding_cache': {
                    'max_size': cache_config.get('embedding_cache_size', 1000),
                    'strategy': cache_config.get('embedding_cache_strategy', 'lru')
                },
                'working_cache': {
                    'max_sessions': cache_config.get('max_sessions', 100),
                    'max_memories_per_session': cache_config.get('max_working_memory', max_working_memory),
                    'ttl': cache_config.get('working_cache_ttl', 86400)
                },
                'compression': {
                    'max_length': cache_config.get('compression_max_length', 200)
                }
            })
        
        # 应用配置
        self.capture_engine.set_config({
            "auto_capture": auto_capture,
            "min_confidence": 0.3,
            "rif_threshold": rif_threshold
        })
        
        self.session_manager.set_max_working_memory(max_working_memory)
        
        # 更新工作记忆缓存配置
        if self.working_memory_cache:
            self.working_memory_cache.max_sessions = cache_config.get('max_sessions', 100)
            self.working_memory_cache.max_memories_per_session = cache_config.get('max_working_memory', max_working_memory)
            self.working_memory_cache.ttl = cache_config.get('working_cache_ttl', 86400)
        
        # 应用生命周期管理器配置
        if self.lifecycle_manager:
            self.lifecycle_manager.cleanup_interval = self._get_config(
                "memory_config.session_cleanup_interval", 3600
            )
            self.lifecycle_manager.session_timeout = self._get_config(
                "memory_config.session_timeout", 86400
            )
            self.lifecycle_manager.inactive_timeout = self._get_config(
                "memory_config.session_inactive_timeout", 1800
            )
        
        # 获取LLM集成配置
        enable_inject = self._get_config("llm_integration.enable_inject", True)
        max_context_memories = self._get_config("llm_integration.max_context_memories", 3)
        
        self.retrieval_engine.set_config({
            "max_context_memories": max_context_memories,
            "enable_time_aware": True,
            "enable_emotion_aware": True,
            "enable_token_budget": enable_inject,  # 启用注入时也启用token预算
            "token_budget": self._get_config("llm_integration.token_budget", 512)
        })
        
        # 配置记忆注入器
        injection_mode_str = self._get_config("llm_integration.injection_mode", "suffix")
        injection_mode = InjectionMode(injection_mode_str.lower())
        self.memory_injector.injection_mode = injection_mode
        
        logger.info(
            f"Config applied: auto_capture={auto_capture}, "
            f"max_working_memory={max_working_memory}, "
            f"injection_mode={injection_mode_str}"
        )
    
    def _get_config(self, key: str, default: any = None) -> any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔）
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = getattr(value, k, value.get(k) if isinstance(value, dict) else default)
            return value if value is not None else default
        except (AttributeError, KeyError):
            return default
    
    async def _load_sessions(self):
        """从KV存储加载会话数据"""
        try:
            # 加载会话数据
            sessions_data = await self.get_kv_data("sessions", {})
            if sessions_data:
                await self.session_manager.deserialize_from_kv_storage(sessions_data)
                logger.info(f"Loaded {self.session_manager.get_session_count()} sessions")
            
            # 加载生命周期状态
            lifecycle_state = await self.get_kv_data("lifecycle_state", {})
            if lifecycle_state and self.lifecycle_manager:
                await self.lifecycle_manager.deserialize_state(lifecycle_state)
                logger.info("Loaded lifecycle state")
            
            # 输出统计信息
            if self.lifecycle_manager:
                stats = self.lifecycle_manager.get_session_statistics()
                logger.info(f"Session statistics: {stats}")
            
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
    
    async def _save_sessions(self):
        """保存会话数据到KV存储"""
        try:
            sessions_data = await self.session_manager.serialize_for_kv_storage()
            await self.put_kv_data("sessions", sessions_data)
            logger.info(f"Saved {self.session_manager.get_session_count()} sessions")
        except Exception as e:
            logger.warning(f"Failed to save sessions: {e}")
    
    def _get_or_create_emotional_state(self, user_id: str) -> EmotionalState:
        """获取或创建用户情感状态
        
        Args:
            user_id: 用户ID
            
        Returns:
            EmotionalState: 情感状态对象
        """
        if user_id not in self.user_emotional_states:
            self.user_emotional_states[user_id] = EmotionalState()
        return self.user_emotional_states[user_id]
    
    def _get_or_create_user_persona(self, user_id: str) -> UserPersona:
        """获取或创建用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            UserPersona: 用户画像对象
        """
        if user_id not in self.user_personas:
            self.user_personas[user_id] = UserPersona(user_id=user_id)
        return self.user_personas[user_id]
    
    # ========== 指令处理器 ==========
    
    @filter.command("memory_save")
    async def save_memory(self, event: AstrMessageEvent):
        """手动保存记忆指令
        
        用法：/memory_save <内容>
        """
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        message = event.message_str.replace("/memory_save", "").strip()
        
        if not message:
            yield event.plain_result("请输入要保存的内容")
            return
        
        # 捕获记忆
        memory = await self.capture_engine.capture_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=True
        )
        
        if memory:
            # 存储到Chroma
            await self.chroma_manager.add_memory(memory)
            
            # 如果是工作记忆，添加到缓存
            if memory.storage_layer.value == "working":
                self.session_manager.add_working_memory(memory)
            
            # 记录保存时间
            await self.put_kv_data(
                f"last_save_{user_id}_{group_id or 'private'}",
                memory.created_time.isoformat()
            )
            
            yield event.plain_result(f"记忆已保存（类型：{memory.type.value}，置信度：{memory.confidence:.2f}）")
        else:
            yield event.plain_result("未能保存记忆，可能不满足捕获条件")
    
    @filter.command("memory_search")
    async def search_memory(self, event: AstrMessageEvent):
        """搜索记忆指令
        
        用法：/memory_search <查询内容>
        """
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        query = event.message_str.replace("/memory_search", "").strip()
        
        if not query:
            yield event.plain_result("请输入搜索内容")
            return
        
        # 获取情感状态
        emotional_state = self._get_or_create_emotional_state(user_id)
        
        # 检索记忆
        memories = await self.retrieval_engine.retrieve(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=5,
            emotional_state=emotional_state
        )
        
        # 格式化结果
        if memories:
            result = f"找到 {len(memories)} 条相关记忆：\n\n"
            for i, memory in enumerate(memories, 1):
                time_str = memory.created_time.strftime("%m-%d %H:%M")
                result += f"{i}. [{memory.type.value.upper()}] {time_str}\n"
                result += f"   {memory.content}\n\n"
        else:
            result = "未找到相关记忆"
        
        yield event.plain_result(result)
    
    @filter.command("memory_clear")
    async def clear_memory(self, event: AstrMessageEvent):
        """清除记忆指令
        
        用法：/memory_clear
        """
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        
        # 清除Chroma中的记忆
        await self.chroma_manager.delete_session(user_id, group_id)
        
        # 清除工作记忆缓存（同时清除SessionManager和WorkingMemoryCache）
        self.session_manager.clear_working_memory(user_id, group_id)
        if self.working_memory_cache:
            await self.working_memory_cache.clear_session(user_id, group_id)
        
        # 删除保存时间记录
        await self.delete_kv_data(f"last_save_{user_id}_{group_id or 'private'}")
        
        yield event.plain_result("记忆已清除")
    
    @filter.command("memory_stats")
    async def memory_stats(self, event: AstrMessageEvent):
        """记忆统计指令
        
        用法：/memory_stats
        """
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        
        # 统计各层记忆数量
        # 同时从SessionManager和WorkingMemoryCache获取
        working_count = len(self.session_manager.get_working_memory(user_id, group_id))
        if self.working_memory_cache:
            memories = await self.working_memory_cache.get_recent_memories(
                user_id, group_id, limit=1000
            )
            working_count = max(working_count, len(memories))
        
        episodic_count = await self.chroma_manager.count_memories(
            user_id=user_id,
            group_id=group_id
        )
        
        # 获取会话信息
        session_key = self.session_manager.get_session_key(user_id, group_id)
        session = self.session_manager.get_session(session_key)
        message_count = session["message_count"] if session else 0
        
        result = f"""记忆统计：
- 工作记忆：{working_count} 条
- 情景记忆：{episodic_count} 条
- 会话消息：{message_count} 条"""
        
        yield event.plain_result(result)
    
    # ========== LLM Hook ==========
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """在LLM请求前注入记忆上下文"""
        enable_inject = self._get_config("llm_integration.enable_inject", True)
        
        if not enable_inject:
            return
        
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        query = event.message_str
        
        # 激活会话（更新生命周期）
        if self.lifecycle_manager:
            await self.lifecycle_manager.activate_session(user_id, group_id)
        
        # 获取情感状态
        emotional_state = self._get_or_create_emotional_state(user_id)
        
        # 更新情感状态
        emotion_result = await self.emotion_analyzer.analyze_emotion(query)
        self.emotion_analyzer.update_emotional_state(
            emotional_state,
            emotion_result["primary"],
            emotion_result["intensity"],
            emotion_result["confidence"],
            emotion_result["secondary"]
        )
        
        # 检索相关记忆
        max_context_memories = self._get_config("llm_integration.max_context_memories", 3)
        memories = await self.retrieval_engine.retrieve(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=max_context_memories,
            emotional_state=emotional_state
        )
        
        # 将记忆注入到system_prompt
        if memories:
            memory_context = self.retrieval_engine.format_memories_for_llm(memories)
            req.system_prompt += f"\n\n{memory_context}\n"
            logger.debug(f"Injected {len(memories)} memories into LLM context")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """在LLM响应后自动捕获新记忆"""
        auto_capture = self._get_config("memory_config.auto_capture", True)
        
        if not auto_capture:
            return
        
        user_id = event.get_sender_id()
        group_id = event.get_sender_group_id()
        message = event.message_str
        
        # 更新会话活动（保持会话活跃）
        self.session_manager.update_session_activity(user_id, group_id)
        
        # 尝试捕获记忆
        memory = await self.capture_engine.capture_memory(
            message=message,
            user_id=user_id,
            group_id=group_id
        )
        
        if memory:
            # 存储到Chroma
            await self.chroma_manager.add_memory(memory)
            
            # 如果是工作记忆，添加到缓存（同时添加到SessionManager和WorkingMemoryCache）
            if memory.storage_layer.value == "working":
                self.session_manager.add_working_memory(memory)
                if self.working_memory_cache:
                    await self.working_memory_cache.add_memory(
                        user_id, group_id, memory.id, memory
                    )
            
            logger.debug(f"Auto-captured memory: {memory.id}")
    
    async def terminate(self):
        """插件销毁"""
        try:
            # 停止生命周期管理器
            if self.lifecycle_manager:
                await self.lifecycle_manager.stop()
            
            # 保存会话状态
            await self._save_sessions()
            
            # 关闭Chroma管理器
            if self.chroma_manager:
                await self.chroma_manager.close()
            
            logger.info("IrisMemory plugin terminated")
            
        except Exception as e:
            logger.error(f"IrisMemory plugin termination error: {e}")