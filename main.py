"""
Iris Memory Plugin - 主入口
基于companion-memory框架的三层记忆插件
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

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
from iris_memory.storage.cache import CacheManager  # WorkingMemoryCache已整合到SessionManager
from iris_memory.analysis.emotion_analyzer import EmotionAnalyzer
from iris_memory.analysis.rif_scorer import RIFScorer
from iris_memory.models.emotion_state import EmotionalState
from iris_memory.utils.hook_manager import MemoryInjector, InjectionMode, HookPriority
from iris_memory.utils.event_utils import get_group_id
from iris_memory.utils.logger import init_logging_from_config, get_logger
from iris_memory.core.config_manager import ConfigManager, init_config_manager
from iris_memory.core.defaults import DEFAULTS

# 新增：分层消息处理和主动回复
from iris_memory.capture.message_classifier import MessageClassifier, ProcessingLayer
from iris_memory.capture.batch_processor import MessageBatchProcessor
from iris_memory.processing.llm_processor import LLMMessageProcessor
from iris_memory.proactive.proactive_reply_detector import ProactiveReplyDetector
from iris_memory.proactive.reply_generator import ProactiveReplyGenerator
from iris_memory.proactive.proactive_manager import ProactiveReplyManager

# 新增：图片智能分析
from iris_memory.multimodal.image_analyzer import ImageAnalyzer, ImageAnalysisLevel


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
        
        # 初始化配置管理器（统一配置访问）
        self.cfg = init_config_manager(config)
        
        # 插件名称（与register装饰器第一个参数一致）
        self.name = "iris_memory"
        
        # 插件数据目录
        self.plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        
        # 初始化核心组件（将在initialize中完成）
        self.chroma_manager = None
        self.capture_engine = None
        self.retrieval_engine = None
        self.session_manager = None
        self.lifecycle_manager = None
        self.cache_manager = None
        self.emotion_analyzer = None
        self.rif_scorer = None
        self.memory_injector = None
        
        # 新增：分层消息处理组件
        self.message_classifier = None
        self.batch_processor = None
        self.llm_processor = None
        
        # 新增：主动回复组件
        self.reply_detector = None
        self.reply_generator = None
        self.proactive_manager = None
        
        # 新增：图片分析器
        self.image_analyzer = None
        
        # 用户情感状态缓存：{user_id: EmotionalState}
        self.user_emotional_states: Dict[str, EmotionalState] = {}
        
        # 用户画像缓存：{user_id: UserPersona}
        self.user_personas: Dict[str, UserPersona] = {}
    
    async def initialize(self):
        """异步初始化插件"""
        try:
            # 创建插件数据目录
            self.plugin_data_path.mkdir(parents=True, exist_ok=True)
            
            # 初始化日志系统（使用配置管理器）
            init_logging_from_config(self.config, self.plugin_data_path)
            
            # 获取模块logger
            self.logger = get_logger("main")
            self.logger.info("IrisMemory plugin initializing...")
            self.logger.debug(f"Plugin data path: {self.plugin_data_path}")
            
            # 初始化情感分析器
            self.emotion_analyzer = EmotionAnalyzer(self.config)
            
            # 初始化RIF评分器
            self.rif_scorer = RIFScorer()
            
            # 初始化Chroma管理器（传入插件上下文用于嵌入API）
            self.chroma_manager = ChromaManager(self.config, self.plugin_data_path, self.context)
            await self.chroma_manager.initialize()
            
            # 初始化会话管理器（使用配置管理器获取配置）
            self.session_manager = SessionManager(
                max_working_memory=self.cfg.max_working_memory,
                max_sessions=DEFAULTS.session.max_sessions,
                ttl=self.cfg.session_timeout
            )
            
            # 初始化缓存管理器（用于嵌入向量缓存）
            self.cache_manager = CacheManager({})
            
            # 初始化生命周期管理器（注入chroma_manager用于记忆升级持久化）
            self.lifecycle_manager = SessionLifecycleManager(
                session_manager=self.session_manager,
                chroma_manager=self.chroma_manager,
                upgrade_mode=self.cfg.upgrade_mode,
                llm_upgrade_batch_size=DEFAULTS.memory.llm_upgrade_batch_size,
                llm_upgrade_threshold=DEFAULTS.memory.llm_upgrade_threshold
            )
            await self.lifecycle_manager.start()
            
            # 初始化记忆捕获引擎（传入chroma_manager以启用去重和冲突检测）
            self.capture_engine = MemoryCaptureEngine(
                chroma_manager=self.chroma_manager,
                emotion_analyzer=self.emotion_analyzer,
                rif_scorer=self.rif_scorer
            )
            
            # 初始化记忆检索引擎（注入session_manager以支持工作记忆合并）
            self.retrieval_engine = MemoryRetrievalEngine(
                chroma_manager=self.chroma_manager,
                rif_scorer=self.rif_scorer,
                emotion_analyzer=self.emotion_analyzer,
                session_manager=self.session_manager
            )
            
            # 初始化记忆注入器
            self.memory_injector = MemoryInjector(
                injection_mode=InjectionMode.SUFFIX,
                priority=HookPriority.NORMAL,
                namespace="iris_memory"
            )
            
            # ========== 新增：分层消息处理初始化 ==========
            await self._init_message_processing()
            
            # ========== 新增：主动回复初始化 ==========
            await self._init_proactive_reply()
            
            # ========== 新增：图片分析器初始化 ==========
            await self._init_image_analyzer()
            
            # 配置参数
            self._apply_config()
            
            # 初始化批量处理器（依赖配置，需要在 _apply_config 之后）
            await self._init_batch_processor()
            
            # 加载会话数据
            await self._load_sessions()
            
            self.logger.info("IrisMemory plugin initialized successfully")
            
        except Exception as e:
            self.logger.error(f"IrisMemory plugin initialization failed: {e}", exc_info=True)
            raise
    
    async def _init_message_processing(self):
        """初始化分层消息处理组件"""
        self.logger.info("Initializing message processing components...")
        
        # 检查是否启用批量处理（默认启用）
        enable_batch = DEFAULTS.message_processing.batch_threshold_count > 0
        use_llm = self.cfg.use_llm
        
        if not enable_batch:
            self.logger.info("Batch processing is disabled")
            return
        
        # 初始化LLM处理器（如果启用）
        if use_llm:
            self.llm_processor = LLMMessageProcessor(
                astrbot_context=self.context,
                max_tokens=DEFAULTS.message_processing.llm_max_tokens_for_summary
            )
            llm_initialized = await self.llm_processor.initialize()
            if not llm_initialized:
                self.logger.warning("LLM processor failed to initialize, using local mode")
                self.llm_processor = None
            else:
                # 如果LLM初始化成功，设置给生命周期管理器用于升级判断
                if self.lifecycle_manager:
                    self.lifecycle_manager.set_llm_provider(self.llm_processor)
                    self.logger.info("LLM provider set for memory upgrade evaluation")
        
        # 初始化消息分类器
        self.message_classifier = MessageClassifier(
            trigger_detector=self.capture_engine.trigger_detector if self.capture_engine else None,
            emotion_analyzer=self.emotion_analyzer,
            llm_processor=self.llm_processor,
            config={
                "llm_processing_mode": DEFAULTS.message_processing.llm_processing_mode,
                "immediate_trigger_confidence": DEFAULTS.message_processing.immediate_trigger_confidence,
                "immediate_emotion_intensity": DEFAULTS.message_processing.immediate_emotion_intensity
            }
        )
        
        self.logger.info("Message classifier initialized")
    
    async def _init_proactive_reply(self):
        """初始化主动回复组件"""
        self.logger.info("Initializing proactive reply components...")
        
        # 检查是否启用主动回复
        enable_proactive = self.cfg.proactive_reply_enabled
        
        if not enable_proactive:
            self.logger.info("Proactive reply is disabled")
            return
        
        # 初始化主动回复检测器
        self.reply_detector = ProactiveReplyDetector(
            emotion_analyzer=self.emotion_analyzer,
            config={
                "high_emotion_threshold": DEFAULTS.proactive_reply.high_emotion_threshold,
                "question_threshold": DEFAULTS.proactive_reply.question_threshold
            }
        )
        
        # 初始化回复生成器
        self.reply_generator = ProactiveReplyGenerator(
            astrbot_context=self.context,
            retrieval_engine=self.retrieval_engine,
            config={
                "max_reply_tokens": DEFAULTS.proactive_reply.max_reply_tokens,
                "reply_temperature": DEFAULTS.proactive_reply.reply_temperature
            }
        )
        await self.reply_generator.initialize()
        
        # 获取每日最大回复数
        max_daily = self.cfg.proactive_reply_max_daily
        
        # 初始化主动回复管理器
        self.proactive_manager = ProactiveReplyManager(
            astrbot_context=self.context,
            reply_detector=self.reply_detector,
            reply_generator=self.reply_generator,
            config={
                "enable_proactive_reply": enable_proactive,
                "reply_cooldown": DEFAULTS.proactive_reply.cooldown_seconds,
                "max_daily_replies": max_daily
            }
        )
        await self.proactive_manager.initialize()
        
        self.logger.info("Proactive reply components initialized")
    
    async def _init_image_analyzer(self):
        """初始化图片分析器"""
        self.logger.info("Initializing image analyzer...")
        
        # 从配置获取图片分析设置
        enable_analysis = self.cfg.image_analysis_enabled
        analysis_mode = self.cfg.image_analysis_mode
        max_images = self.cfg.image_analysis_max_images
        
        # 新增：预算和过滤配置
        daily_budget = self.cfg.image_analysis_daily_budget
        session_budget = self.cfg.image_analysis_session_budget
        require_context = self.cfg.image_analysis_require_context
        
        if not enable_analysis:
            self.logger.info("Image analysis is disabled")
            return
        
        self.image_analyzer = ImageAnalyzer(
            astrbot_context=self.context,
            config={
                "enable_image_analysis": enable_analysis,
                "default_level": analysis_mode,
                "max_images_per_message": max_images,
                "skip_sticker": DEFAULTS.image_analysis.skip_sticker,
                "analysis_cooldown": DEFAULTS.image_analysis.analysis_cooldown,
                "cache_ttl": DEFAULTS.image_analysis.cache_ttl,
                "max_cache_size": DEFAULTS.image_analysis.max_cache_size,
                # 新增配置
                "daily_analysis_budget": daily_budget if daily_budget > 0 else 999999,
                "session_analysis_budget": session_budget if session_budget > 0 else 999999,
                "similar_image_window": DEFAULTS.image_analysis.similar_image_window,
                "recent_image_limit": DEFAULTS.image_analysis.recent_image_limit,
                "require_context_relevance": require_context
            }
        )
        
        self.logger.info(f"Image analyzer initialized: mode={analysis_mode}, max_images={max_images}, "
                        f"daily_budget={daily_budget}, require_context={require_context}")
    
    async def _init_batch_processor(self):
        """初始化批量处理器（在_apply_config后调用）"""
        use_llm = self.cfg.use_llm
        
        self.batch_processor = MessageBatchProcessor(
            capture_engine=self.capture_engine,
            llm_processor=self.llm_processor,
            proactive_manager=self.proactive_manager,
            threshold_count=DEFAULTS.message_processing.batch_threshold_count,
            threshold_interval=DEFAULTS.message_processing.batch_threshold_interval,
            processing_mode=DEFAULTS.message_processing.batch_processing_mode,
            use_llm_summary=use_llm and self.llm_processor is not None,
            on_save_callback=self._save_batch_queues
        )
        await self.batch_processor.start()
        
        self.logger.info("Batch processor initialized")
    
    async def _save_batch_queues(self):
        """保存批量处理器队列到KV存储（供自动保存回调使用）"""
        if self.batch_processor:
            try:
                batch_queues = await self.batch_processor.serialize_queues()
                await self.put_kv_data("batch_queues", batch_queues)
            except Exception as e:
                self.logger.warning(f"Failed to auto-save batch queues: {e}")
    
    def _apply_config(self):
        """应用配置"""
        # 使用配置管理器获取配置
        auto_capture = self.cfg.enable_memory
        max_working_memory = self.cfg.max_working_memory
        rif_threshold = self.cfg.rif_threshold
        
        # 应用缓存配置（使用默认值）
        if self.cache_manager:
            self.cache_manager = CacheManager({
                'embedding_cache': {
                    'max_size': DEFAULTS.cache.embedding_cache_size,
                    'strategy': DEFAULTS.cache.embedding_cache_strategy
                },
                'working_cache': {
                    'max_sessions': DEFAULTS.session.max_sessions,
                    'max_memories_per_session': max_working_memory,
                    'ttl': DEFAULTS.cache.working_cache_ttl
                },
                'compression': {
                    'max_length': DEFAULTS.cache.compression_max_length
                }
            })
        
        # 应用配置到捕获引擎
        self.capture_engine.set_config({
            "auto_capture": auto_capture,
            "min_confidence": DEFAULTS.memory.min_confidence,
            "rif_threshold": rif_threshold
        })
        
        # 更新SessionManager配置
        self.session_manager.max_working_memory = max_working_memory
        self.session_manager.max_sessions = DEFAULTS.session.max_sessions
        self.session_manager.ttl = self.cfg.session_timeout
        
        # 应用生命周期管理器配置
        if self.lifecycle_manager:
            self.lifecycle_manager.cleanup_interval = DEFAULTS.session.session_cleanup_interval
            self.lifecycle_manager.session_timeout = self.cfg.session_timeout
            self.lifecycle_manager.inactive_timeout = DEFAULTS.session.session_inactive_timeout
        
        # 获取LLM集成配置
        enable_inject = self.cfg.enable_inject
        max_context_memories = self.cfg.max_context_memories
        
        self.retrieval_engine.set_config({
            "max_context_memories": max_context_memories,
            "enable_time_aware": DEFAULTS.llm_integration.enable_time_aware,
            "enable_emotion_aware": DEFAULTS.llm_integration.enable_emotion_aware,
            "enable_token_budget": enable_inject,
            "token_budget": self.cfg.token_budget,
            "coordination_strategy": DEFAULTS.llm_integration.coordination_strategy
        })
        
        # 配置记忆注入器
        injection_mode = InjectionMode(DEFAULTS.llm_integration.injection_mode)
        self.memory_injector.injection_mode = injection_mode
        
        self.logger.info(
            f"Config applied: auto_capture={auto_capture}, "
            f"max_working_memory={max_working_memory}, "
            f"max_context_memories={max_context_memories}"
        )
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查用户是否为管理员（使用 AstrBot 全局管理员设置）
        
        Args:
            event: 消息事件对象
            
        Returns:
            bool: 是否为管理员
        """
        return event.is_admin()
    
    def _get_config(self, key: str, default: any = None) -> any:
        """获取配置值（兼容旧代码，内部使用配置管理器）
        
        Args:
            key: 配置键（支持点分隔）
            default: 默认值
            
        Returns:
            配置值
        """
        return self.cfg.get(key, default)
    
    async def _load_sessions(self):
        """从KV存储加载会话数据"""
        try:
            # 加载会话数据
            sessions_data = await self.get_kv_data("sessions", {})
            if sessions_data:
                await self.session_manager.deserialize_from_kv_storage(sessions_data)
                self.logger.info(f"Loaded {self.session_manager.get_session_count()} sessions")
            
            # 加载生命周期状态
            lifecycle_state = await self.get_kv_data("lifecycle_state", {})
            if lifecycle_state and self.lifecycle_manager:
                await self.lifecycle_manager.deserialize_state(lifecycle_state)
                self.logger.info("Loaded lifecycle state")
            
            # 加载批量处理器队列
            batch_queues = await self.get_kv_data("batch_queues", {})
            if batch_queues and self.batch_processor:
                await self.batch_processor.deserialize_queues(batch_queues)
                self.logger.info("Loaded batch processor queues")
            
            # 输出统计信息
            if self.lifecycle_manager:
                stats = self.lifecycle_manager.get_session_statistics()
                self.logger.info(f"Session statistics: {stats}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load sessions: {e}")
    
    async def _save_sessions(self):
        """保存会话数据到KV存储"""
        try:
            # 保存会话数据
            sessions_data = await self.session_manager.serialize_for_kv_storage()
            await self.put_kv_data("sessions", sessions_data)
            self.logger.info(f"Saved {self.session_manager.get_session_count()} sessions")
            
            # 保存批量处理器队列
            if self.batch_processor:
                batch_queues = await self.batch_processor.serialize_queues()
                await self.put_kv_data("batch_queues", batch_queues)
                self.logger.info("Saved batch processor queues")
                
        except Exception as e:
            self.logger.warning(f"Failed to save sessions: {e}")
    
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
        group_id = get_group_id(event)
        # 尝试多种方式移除指令前缀（先strip去除前导空格）
        message = event.message_str.strip()
        for prefix in ["/memory_save", "memory_save"]:
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
                break
        
        self.logger.debug(f"save_memory: raw='{event.message_str}', processed='{message}'")

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
            # 分层存储逻辑：
            # - WORKING 记忆只存内存缓存，不存Chroma
            # - EPISODIC/SEMANTIC 记忆存入Chroma
            if memory.storage_layer.value == "working":
                # 工作记忆仅存入SessionManager（统一缓存）
                await self.session_manager.add_working_memory(memory)
            else:
                # 情景/语义记忆存入Chroma
                await self.chroma_manager.add_memory(memory)
            
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
        group_id = get_group_id(event)
        # 尝试多种方式移除指令前缀（先strip去除前导空格）
        query = event.message_str.strip()
        for prefix in ["/memory_search", "memory_search"]:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break

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
        group_id = get_group_id(event)

        # 清除Chroma中的记忆
        await self.chroma_manager.delete_session(user_id, group_id)
        
        # 清除工作记忆缓存
        await self.session_manager.clear_working_memory(user_id, group_id)
        
        # 删除保存时间记录
        await self.delete_kv_data(f"last_save_{user_id}_{group_id or 'private'}")
        
        yield event.plain_result("记忆已清除")
    
    @filter.command("memory_delete_private")
    async def delete_private_memories(self, event: AstrMessageEvent):
        """删除个人私聊记忆指令

        用法：/memory_delete_private
        功能：删除当前用户在私聊场景下的所有记忆
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 检查是否在私聊场景
        if group_id:
            yield event.plain_result("此命令仅限私聊使用")
            return
        
        # 删除Chroma中的私聊记忆
        success, count = await self.chroma_manager.delete_user_memories(user_id, in_private_only=True)
        
        # 清除工作记忆缓存
        await self.session_manager.clear_working_memory(user_id, None)
        
        # 删除保存时间记录
        await self.delete_kv_data(f"last_save_{user_id}_private")
        
        if success:
            yield event.plain_result(f"已删除 {count} 条个人私聊记忆")
        else:
            yield event.plain_result("未找到记忆或删除失败")
    
    @filter.command("memory_delete_group")
    async def delete_group_memories(self, event: AstrMessageEvent):
        """删除当前群聊记忆指令（仅管理员）

        用法：/memory_delete_group [shared|private|all]
        功能：删除当前群聊的记忆
        - shared: 仅删除群组共享记忆
        - private: 仅删除个人在群聊的记忆
        - all: 删除群组所有记忆（默认）
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 检查是否在群聊场景
        if not group_id:
            yield event.plain_result("此命令仅限群聊使用")
            return
        
        # 检查管理员权限
        if not self._is_admin(event):
            yield event.plain_result("权限不足，仅管理员可以删除群聊记忆")
            return
        
        # 解析参数
        message_parts = event.message_str.split()
        scope_filter = None
        if len(message_parts) > 1:
            param = message_parts[1].lower()
            if param == "shared":
                scope_filter = "group_shared"
            elif param == "private":
                scope_filter = "group_private"
            elif param == "all":
                scope_filter = None
            else:
                yield event.plain_result("参数错误，请使用: shared, private 或 all")
                return
        
        # 删除Chroma中的群聊记忆
        success, count = await self.chroma_manager.delete_group_memories(group_id, scope_filter)
        
        # 清除相关的工作记忆缓存（所有用户在该群的记忆）
        # 注意：这里只能清除当前用户的缓存，其他用户的缓存会在下次访问时自动同步
        if scope_filter != "group_shared":  # 如果不是只删除共享记忆，则清空工作记忆
            await self.session_manager.clear_working_memory(user_id, group_id)
        
        if success:
            scope_desc = "共享" if scope_filter == "group_shared" else "个人" if scope_filter == "group_private" else "所有"
            yield event.plain_result(f"已删除当前群聊的 {count} 条{scope_desc}记忆")
        else:
            yield event.plain_result("未找到记忆或删除失败")
    
    @filter.command("memory_delete_all")
    async def delete_all_memories(self, event: AstrMessageEvent):
        """删除所有记忆指令（仅超级管理员）

        用法：/memory_delete_all confirm
        功能：删除数据库中的所有记忆（危险操作）
        注意：必须添加 'confirm' 参数确认操作
        """
        user_id = event.get_sender_id()
        
        # 检查管理员权限
        if not self._is_admin(event):
            yield event.plain_result("权限不足，仅管理员可以执行此操作")
            return
        
        # 检查确认参数
        message_parts = event.message_str.split()
        if len(message_parts) < 2 or message_parts[1].lower() != "confirm":
            yield event.plain_result("警告：此操作将删除所有记忆！\n请使用 '/memory_delete_all confirm' 确认操作")
            return
        
        # 删除所有记忆
        success, count = await self.chroma_manager.delete_all_memories()
        
        # 重置会话管理器（清除所有缓存）
        self.session_manager = SessionManager(
            max_working_memory=self.cfg.max_working_memory,
            max_sessions=DEFAULTS.session.max_sessions,
            ttl=self.cfg.session_timeout
        )
        
        if success:
            yield event.plain_result(f"已删除所有 {count} 条记忆！")
        else:
            yield event.plain_result("删除失败或数据库为空")
    
    @filter.command("memory_stats")
    async def memory_stats(self, event: AstrMessageEvent):
        """记忆统计指令

        用法：/memory_stats
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)

        # 统计各层记忆数量
        working_memories = await self.session_manager.get_working_memory(user_id, group_id)
        working_count = len(working_memories)
        
        episodic_count = await self.chroma_manager.count_memories(
            user_id=user_id,
            group_id=group_id
        )
        
        # 图片分析统计
        image_stats = ""
        if self.image_analyzer:
            stats = self.image_analyzer.get_statistics()
            image_stats = f"""
- 图片分析：{stats.get('total_analyzed', 0)} 张
- 缓存命中：{stats.get('cache_hits', 0)} 次"""
        
        result = f"""记忆统计：
- 工作记忆：{working_count} 条
- 情景记忆：{episodic_count} 条{image_stats}"""
        
        yield event.plain_result(result)
    
    # ========== LLM Hook ==========
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """在LLM请求前注入记忆上下文
        
        同时分析消息中的图片，将描述注入到上下文
        """
        if not self.cfg.enable_inject:
            return

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        query = event.message_str

        # 激活会话（更新生命周期）
        if self.lifecycle_manager:
            await self.lifecycle_manager.activate_session(user_id, group_id)
        
        # ========== 图片分析（LLM上下文增强） ==========
        image_context = ""
        if self.image_analyzer:
            try:
                message_chain = event.message_obj.message
                session_key = f"{user_id}:{group_id}" if group_id else user_id
                image_results = await self.image_analyzer.analyze_message_images(
                    message_chain=message_chain,
                    user_id=user_id,
                    context_text=query,
                    umo=event.unified_msg_origin,
                    session_id=session_key
                )
                if image_results:
                    image_context = self.image_analyzer.format_for_llm_context(image_results)
            except Exception as e:
                self.logger.warning(f"Image analysis in LLM hook failed: {e}")
        
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
        memories = await self.retrieval_engine.retrieve(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=self.cfg.max_context_memories,
            emotional_state=emotional_state
        )
        
        # 将记忆注入到system_prompt
        if memories:
            # 使用 natural 风格适合群聊真实人设：淡化AI痕迹，强调自然回忆
            memory_context = self.retrieval_engine.format_memories_for_llm(
                memories,
                persona_style="natural"
            )
            req.system_prompt += f"\n\n{memory_context}\n"
            self.logger.debug(f"Injected {len(memories)} memories into LLM context")
        
        # 注入图片上下文描述（如果有）
        if image_context:
            req.system_prompt += f"\n{image_context}\n"
            self.logger.debug("Injected image context into LLM prompt")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """在LLM响应后自动捕获新记忆"""
        if not self.cfg.enable_memory:
            return

        user_id = event.get_sender_id()
        group_id = get_group_id(event)
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
            # 分层存储逻辑：
            # - WORKING 记忆只存内存缓存，不存Chroma
            # - EPISODIC/SEMANTIC 记忆存入Chroma
            if memory.storage_layer.value == "working":
                # 工作记忆仅存入SessionManager
                await self.session_manager.add_working_memory(memory)
            else:
                # 情景/语义记忆存入Chroma
                await self.chroma_manager.add_memory(memory)
            
            self.logger.debug(f"Auto-captured memory: {memory.id}")
    
    # ========== 新增：普通消息分层处理器 ==========
    
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_messages(self, event: AstrMessageEvent):
        """统一处理所有普通消息 - 分层处理策略
        
        三种处理层级：
        1. immediate - 立即捕获高价值消息
        2. batch - 累积批量处理普通消息
        3. discard - 丢弃无价值消息
        
        支持图片智能分析：将图片描述添加到消息内容中
        """
        # 如果批量处理器未初始化，跳过
        if not self.batch_processor:
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        
        # 过滤掉指令消息（以/开头或已知指令）
        if message.strip().startswith('/'):
            return
        
        # 过滤掉其他指令
        known_commands = ['memory_save', 'memory_search', 'memory_clear', 'memory_stats',
                         'memory_delete_private', 'memory_delete_group', 'memory_delete_all']
        for cmd in known_commands:
            if message.strip().startswith(cmd):
                return
        
        # 更新会话活动
        self.session_manager.update_session_activity(user_id, group_id)
        
        # ========== 图片分析 ==========
        image_description = ""
        if self.image_analyzer:
            try:
                message_chain = event.message_obj.message
                session_key = f"{user_id}:{group_id}" if group_id else user_id
                image_results = await self.image_analyzer.analyze_message_images(
                    message_chain=message_chain,
                    user_id=user_id,
                    context_text=message,
                    umo=event.unified_msg_origin,
                    session_id=session_key
                )
                if image_results:
                    image_description = self.image_analyzer.format_for_memory(image_results)
                    self.logger.debug(f"Image analysis result: {image_description}")
            except Exception as e:
                self.logger.warning(f"Image analysis failed: {e}")
        
        # 合并文字和图片描述
        full_message = message
        if image_description:
            full_message = f"{message} {image_description}".strip()
        
        # 构建消息上下文
        context = await self._build_message_context(user_id, group_id)
        
        # 添加图片信息到上下文
        if image_description:
            context["has_image"] = True
            context["image_description"] = image_description
        
        # 分类消息
        classification = await self.message_classifier.classify(full_message, context)
        
        self.logger.debug(f"Message classified: {classification.layer.value} "
                         f"(confidence: {classification.confidence:.2f}, "
                         f"source: {classification.source})")
        
        # 根据层级处理
        if classification.layer == ProcessingLayer.DISCARD:
            # 丢弃层：直接返回
            return
        
        elif classification.layer == ProcessingLayer.IMMEDIATE:
            # 立即处理层：直接捕获（使用包含图片描述的完整消息）
            await self._capture_immediate_memory(
                full_message, user_id, group_id, classification
            )
        
        else:  # BATCH
            # 批量处理层：加入队列（使用包含图片描述的完整消息）
            await self.batch_processor.add_message(
                content=full_message,
                user_id=user_id,
                group_id=group_id,
                context=context,
                umo=event.unified_msg_origin
            )
    
    async def _build_message_context(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> Dict[str, Any]:
        """构建消息上下文"""
        session_key = self.session_manager.get_session_key(user_id, group_id)
        session = self.session_manager.get_session(session_key)
        
        # 获取或创建情感状态
        emotional_state = self._get_or_create_emotional_state(user_id)
        
        return {
            "session_key": session_key,
            "session_message_count": session.get("message_count", 0) if session else 0,
            "user_persona": self.user_personas.get(user_id, {}),
            "emotional_state": emotional_state
        }
    
    async def _capture_immediate_memory(
        self,
        message: str,
        user_id: str,
        group_id: Optional[str],
        classification
    ):
        """捕获立即处理的消息"""
        memory = await self.capture_engine.capture_memory(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context={
                "classification": classification.metadata,
                "source": classification.source
            }
        )
        
        if memory:
            # 分层存储逻辑：
            # - WORKING 记忆只存内存缓存，不存Chroma
            # - EPISODIC/SEMANTIC 记忆存入Chroma
            if memory.storage_layer.value == "working":
                # 工作记忆仅存入SessionManager
                await self.session_manager.add_working_memory(memory)
            else:
                # 情景/语义记忆存入Chroma
                await self.chroma_manager.add_memory(memory)
            
            self.logger.debug(f"Immediate memory captured: {memory.id}")
    
    async def terminate(self):
        """插件销毁"""
        try:
            # 停止批量处理器
            if self.batch_processor:
                await self.batch_processor.stop()
            
            # 停止主动回复管理器
            if self.proactive_manager:
                await self.proactive_manager.stop()
            
            # 停止生命周期管理器
            if self.lifecycle_manager:
                await self.lifecycle_manager.stop()
            
            # 保存会话状态
            await self._save_sessions()
            
            # 关闭Chroma管理器
            if self.chroma_manager:
                await self.chroma_manager.close()
            
            # 输出统计信息
            self._log_final_stats()
            
            self.logger.info("IrisMemory plugin terminated")
            
        except Exception as e:
            self.logger.error(f"IrisMemory plugin termination error: {e}", exc_info=True)
    
    def _log_final_stats(self):
        """输出最终统计信息"""
        self.logger.info("=== Final Statistics ===")
        
        if self.message_classifier:
            stats = self.message_classifier.get_stats()
            self.logger.info(f"Message Classifier: {stats}")
        
        if self.batch_processor:
            stats = self.batch_processor.get_stats()
            self.logger.info(f"Batch Processor: {stats}")
        
        if self.llm_processor:
            stats = self.llm_processor.get_stats()
            self.logger.info(f"LLM Processor: {stats}")
        
        if self.proactive_manager:
            stats = self.proactive_manager.get_stats()
            self.logger.info(f"Proactive Manager: {stats}")
        
        if self.image_analyzer:
            stats = self.image_analyzer.get_statistics()
            self.logger.info(f"Image Analyzer: {stats}")
