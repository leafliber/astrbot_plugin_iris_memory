"""
Iris Memory Plugin - 主入口（重构版）
基于 companion-memory 框架的三层记忆插件

架构：
- Handler 层（本文件）：指令路由、权限检查、消息回发
- Service 层（services/memory_service.py）：业务逻辑封装
- Utils 层（utils/command_utils.py）：工具函数

职责分离原则：
- 本文件只负责 AstrBot 事件处理和响应，不直接操作底层组件
- 所有业务逻辑委托给 MemoryService
"""
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator, Any, List

# 将插件根目录添加到Python路径
plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from iris_memory.services.memory_service import MemoryService
from iris_memory.utils.event_utils import get_group_id
from iris_memory.utils.logger import init_logging_from_config
from iris_memory.utils.command_utils import (
    CommandParser, DeleteScopeParser, StatsFormatter,
    SessionKeyBuilder, MessageFilter
)
from iris_memory.core.constants import (
    CommandPrefix, ErrorMessages, SuccessMessages,
    DeleteScope, NumericDefaults, LogTemplates
)


@register("iris_memory", "YourName", "基于companion-memory框架的三层记忆插件", "1.0.0")
class IrisMemoryPlugin(Star):
    """
    Iris记忆插件 - Handler层
    
    实现三层记忆模型：
    - 工作记忆：会话内临时存储
    - 情景记忆：基于RIF评分动态管理
    - 语义记忆：永久保存用户画像
    
    支持私聊和群聊的完全隔离。
    """
    
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """
        初始化插件
        
        Args:
            context: AstrBot上下文对象
            config: 插件配置对象
        """
        super().__init__(context)
        self.context = context
        self.config = config
        
        # 插件名称
        self.name = "iris_memory"
        
        # 插件数据目录
        data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        
        # 初始化业务服务层
        self._service = MemoryService(context, config, data_path)
    
    async def initialize(self) -> None:
        """异步初始化插件"""
        # 初始化日志系统
        init_logging_from_config(self.config, self._service.plugin_data_path)
        
        # 初始化业务服务
        await self._service.initialize()
        
        # 加载持久化数据
        await self._service.load_from_kv(self.get_kv_data)
    
    # ========== 权限检查 ==========
    
    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """
        检查用户是否为管理员
        
        Args:
            event: 消息事件对象
            
        Returns:
            bool: 是否为管理员
        """
        return event.is_admin()
    
    def _check_private_only(self, event: AstrMessageEvent) -> bool:
        """检查是否在私聊场景"""
        return get_group_id(event) is None
    
    def _check_group_only(self, event: AstrMessageEvent) -> Optional[str]:
        """
        检查是否在群聊场景
        
        Returns:
            Optional[str]: 群聊ID，私聊返回None
        """
        return get_group_id(event)
    
    # ========== 指令处理器 ==========
    
    @filter.command("memory_save")
    async def save_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        手动保存记忆指令
        
        用法：/memory_save <内容>
        """
        # 解析指令
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SAVE)
        
        if not parsed.has_content:
            yield event.plain_result(ErrorMessages.EMPTY_CONTENT)
            return
        
        # 获取上下文信息
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        memory = await self._service.capture_and_store_memory(
            message=parsed.content,
            user_id=user_id,
            group_id=group_id,
            is_user_requested=True
        )
        
        # 响应结果
        if memory:
            result = SuccessMessages.MEMORY_SAVED.format(
                memory_type=memory.type.value,
                confidence=memory.confidence
            )
            # 保存最后保存时间
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await self.put_kv_data(kv_key, memory.created_time.isoformat())
        else:
            result = ErrorMessages.CAPTURE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_search")
    async def search_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        搜索记忆指令
        
        用法：/memory_search <查询内容>
        """
        # 解析指令
        parsed = CommandParser.parse(event.message_str, CommandPrefix.MEMORY_SEARCH)
        
        if not parsed.has_content:
            yield event.plain_result(ErrorMessages.EMPTY_QUERY)
            return
        
        # 获取上下文信息
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        memories = await self._service.search_memories(
            query=parsed.content,
            user_id=user_id,
            group_id=group_id,
            top_k=NumericDefaults.TOP_K_SEARCH
        )
        
        # 格式化并响应
        result = StatsFormatter.format_search_results(memories)
        yield event.plain_result(result)
    
    @filter.command("memory_clear")
    async def clear_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        清除记忆指令
        
        用法：/memory_clear
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        success = await self._service.clear_memories(user_id, group_id)
        
        if success:
            # 删除保存时间记录
            kv_key = SessionKeyBuilder.build_for_kv(user_id, group_id)
            await self.delete_kv_data(kv_key)
            result = SuccessMessages.MEMORY_CLEARED
        else:
            result = ErrorMessages.DELETE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_delete_private")
    async def delete_private_memories(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        删除个人私聊记忆指令
        
        用法：/memory_delete_private
        功能：删除当前用户在私聊场景下的所有记忆
        """
        # 权限检查：私聊场景
        if not self._check_private_only(event):
            yield event.plain_result(ErrorMessages.PRIVATE_ONLY)
            return
        
        user_id = event.get_sender_id()
        
        # 执行业务逻辑
        success, count = await self._service.delete_private_memories(user_id)
        
        # 删除保存时间记录
        kv_key = SessionKeyBuilder.build_for_kv(user_id, None)
        await self.delete_kv_data(kv_key)
        
        # 响应结果
        if success:
            result = SuccessMessages.PRIVATE_DELETED.format(count=count)
        else:
            result = ErrorMessages.DELETE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_delete_group")
    async def delete_group_memories(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        删除当前群聊记忆指令（仅管理员）
        
        用法：/memory_delete_group [shared|private|all]
        功能：删除当前群聊的记忆
        - shared: 仅删除群组共享记忆
        - private: 仅删除个人在群聊的记忆
        - all: 删除群组所有记忆（默认）
        """
        # 权限检查：群聊场景
        group_id = self._check_group_only(event)
        if not group_id:
            yield event.plain_result(ErrorMessages.GROUP_ONLY)
            return
        
        # 权限检查：管理员
        if not self._is_admin(event):
            yield event.plain_result(ErrorMessages.GROUP_ADMIN_REQUIRED)
            return
        
        # 解析参数
        parsed = CommandParser.parse_with_slash(event.message_str, "memory_delete_group")
        
        if parsed.args and not DeleteScopeParser.is_valid(parsed.first_arg):
            yield event.plain_result(ErrorMessages.INVALID_SCOPE_PARAM)
            return
        
        scope_filter, scope_desc = DeleteScopeParser.parse(parsed.first_arg)
        user_id = event.get_sender_id()
        
        # 执行业务逻辑
        success, count = await self._service.delete_group_memories(
            group_id=group_id,
            scope_filter=scope_filter,
            user_id=user_id
        )
        
        # 响应结果
        if success:
            result = SuccessMessages.GROUP_DELETED.format(count=count, scope_desc=scope_desc)
        else:
            result = ErrorMessages.DELETE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_delete_all")
    async def delete_all_memories(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        删除所有记忆指令（仅超级管理员）
        
        用法：/memory_delete_all confirm
        功能：删除数据库中的所有记忆（危险操作）
        注意：必须添加 'confirm' 参数确认操作
        """
        # 权限检查：管理员
        if not self._is_admin(event):
            yield event.plain_result(ErrorMessages.ADMIN_REQUIRED)
            return
        
        # 解析参数
        parsed = CommandParser.parse_with_slash(event.message_str, "memory_delete_all")
        
        # 确认参数检查
        if len(parsed.args) < NumericDefaults.CONFIRM_PARAM_INDEX or \
           parsed.args[NumericDefaults.CONFIRM_PARAM_INDEX - 1].lower() != NumericDefaults.CONFIRM_VALUE:
            yield event.plain_result(ErrorMessages.DELETE_CONFIRM_REQUIRED)
            return
        
        # 执行业务逻辑
        success, count = await self._service.delete_all_memories()
        
        # 响应结果
        if success:
            result = SuccessMessages.ALL_DELETED.format(count=count)
        else:
            result = ErrorMessages.DELETE_FAILED
        
        yield event.plain_result(result)
    
    @filter.command("memory_stats")
    async def memory_stats(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        记忆统计指令
        
        用法：/memory_stats
        """
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        
        # 执行业务逻辑
        stats = await self._service.get_memory_stats(user_id, group_id)
        
        # 格式化响应
        image_stats = ""
        if stats.get("image_analyzed", 0) > 0:
            image_stats = f"\n- 图片分析：{stats['image_analyzed']} 张\n- 缓存命中：{stats['cache_hits']} 次"
        
        result = SuccessMessages.STATS_TEMPLATE.format(
            working_count=stats.get("working_count", 0),
            episodic_count=stats.get("episodic_count", 0),
            image_stats=image_stats
        )
        
        yield event.plain_result(result)
    
    # ========== LLM Hook ==========
    
    @filter.on_llm_request()
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        req: Any
    ) -> None:
        """
        在LLM请求前注入记忆上下文
        
        同时分析消息中的图片，将描述注入到上下文
        """
        # 功能开关检查
        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_inject:
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        query = event.message_str
        
        # 激活会话
        await self._service.activate_session(user_id, group_id)
        
        # 图片分析
        image_context = ""
        if self._service.image_analyzer:
            try:
                llm_ctx, _ = await self._service.analyze_images(
                    message_chain=event.message_obj.message,
                    user_id=user_id,
                    context_text=query,
                    umo=event.unified_msg_origin,
                    session_id=SessionKeyBuilder.build(user_id, group_id)
                )
                image_context = llm_ctx
            except Exception as e:
                self._service.logger.warning(f"Image analysis in LLM hook failed: {e}")
        
        # 准备LLM上下文
        context = await self._service.prepare_llm_context(
            query=query,
            user_id=user_id,
            group_id=group_id,
            image_context=image_context
        )
        
        # 注入上下文
        if context:
            req.system_prompt += f"\n\n{context}\n"
    
    @filter.on_llm_response()
    async def on_llm_response(
        self,
        event: AstrMessageEvent,
        resp: Any
    ) -> None:
        """
        在LLM响应后自动捕获新记忆
        """
        # 功能开关检查
        if not hasattr(self._service, 'cfg') or not self._service.cfg.enable_memory:
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        
        # 更新会话活动
        self._service.update_session_activity(user_id, group_id)
        
        # 捕获记忆
        memory = await self._service.capture_and_store_memory(
            message=message,
            user_id=user_id,
            group_id=group_id
        )
        
        if memory:
            self._service.logger.debug(LogTemplates.MEMORY_CAPTURED.format(memory_id=memory.id))
    
    # ========== 普通消息处理器 ==========
    
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_messages(self, event: AstrMessageEvent) -> None:
        """
        统一处理所有普通消息 - 分层处理策略
        
        三种处理层级：
        1. immediate - 立即捕获高价值消息
        2. batch - 累积批量处理普通消息
        3. discard - 丢弃无价值消息
        """
        # 检查批量处理器是否就绪
        if not self._service.batch_processor:
            return
        
        user_id = event.get_sender_id()
        group_id = get_group_id(event)
        message = event.message_str
        
        # 过滤指令消息
        if MessageFilter.is_command(message):
            return
        
        # 更新会话活动
        self._service.update_session_activity(user_id, group_id)
        
        # 图片分析
        image_description = ""
        if self._service.image_analyzer:
            try:
                _, mem_format = await self._service.analyze_images(
                    message_chain=event.message_obj.message,
                    user_id=user_id,
                    context_text=message,
                    umo=event.unified_msg_origin,
                    session_id=SessionKeyBuilder.build(user_id, group_id)
                )
                image_description = mem_format
            except Exception as e:
                self._service.logger.warning(f"Image analysis failed: {e}")
        
        # 构建上下文
        context = await self._build_message_context(user_id, group_id)
        
        # 处理消息批次
        await self._service.process_message_batch(
            message=message,
            user_id=user_id,
            group_id=group_id,
            context=context,
            umo=event.unified_msg_origin,
            image_description=image_description
        )
    
    async def _build_message_context(
        self,
        user_id: str,
        group_id: Optional[str]
    ) -> dict[str, Any]:
        """
        构建消息上下文
        
        Args:
            user_id: 用户ID
            group_id: 群聊ID
            
        Returns:
            Dict[str, Any]: 上下文字典
        """
        session_key = SessionKeyBuilder.build(user_id, group_id)
        session = None
        
        if self._service.session_manager:
            session = self._service.session_manager.get_session(session_key)
        
        return {
            "session_key": session_key,
            "session_message_count": session.get("message_count", 0) if session else 0,
            "user_persona": self._service.get_or_create_user_persona(user_id),
            "emotional_state": self._service._get_or_create_emotional_state(user_id)
        }
    
    # ========== 生命周期方法 ==========
    
    async def terminate(self) -> None:
        """插件销毁"""
        # 保存数据
        await self._service.save_to_kv(self.put_kv_data)
        
        # 销毁服务
        await self._service.terminate()
