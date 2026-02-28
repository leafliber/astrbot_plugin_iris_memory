"""
Iris Memory Plugin - 主入口
基于 companion-memory 框架的三层记忆插件

架构：
- main.py（本文件）：插件注册、事件装饰器、模块委托
- commands/：命令处理（handlers, permissions, registry）
- message_processor.py：LLM Hook、消息装饰、普通消息处理
- web_ui.py：Web 管理界面
- services/memory_service.py：业务逻辑封装

职责分离原则：
- 本文件只负责 AstrBot 事件绑定和响应发送
- 所有业务逻辑委托给各模块处理
"""
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator, Any

plugin_root = Path(__file__).parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import AstrBotConfig, logger

from iris_memory.services.memory_service import MemoryService
from iris_memory.utils.logger import init_logging_from_config
from iris_memory.commands import CommandHandlers
from iris_memory.web.web_ui import WebUIManager
from iris_memory.processing.message_processor import MessageProcessor, ErrorFriendlyProcessor
from iris_memory.core.constants import PROACTIVE_EXTRA_KEY


@register("iris_memory", "YourName", "基于companion-memory框架的三层记忆插件", "1.9.1")
class IrisMemoryPlugin(Star):
    """
    Iris 记忆插件 - 主入口

    实现三层记忆模型：
    - 工作记忆：会话内临时存储
    - 情景记忆：基于 RIF 评分动态管理
    - 语义记忆：永久保存用户画像

    支持私聊和群聊的完全隔离。
    """

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        """
        初始化插件

        Args:
            context: AstrBot 上下文对象
            config: 插件配置对象
        """
        super().__init__(context)
        self.context = context
        self.config = config
        self.name = "iris_memory"

        data_path = Path(StarTools.get_data_dir())
        self._service = MemoryService(context, config, data_path)

        self._command_handlers: Optional[CommandHandlers] = None
        self._web_ui: Optional[WebUIManager] = None
        self._message_processor: Optional[MessageProcessor] = None
        self._error_processor: Optional[ErrorFriendlyProcessor] = None

    async def initialize(self) -> None:
        """异步初始化插件"""
        init_logging_from_config(self.config, self._service.plugin_data_path)

        await self._service.initialize()

        await self._service.load_from_kv(self.get_kv_data)

        self._command_handlers = CommandHandlers(self._service)
        self._web_ui = WebUIManager(self._service)
        self._message_processor = MessageProcessor(self._service)
        self._error_processor = ErrorFriendlyProcessor(self.config)

        await self._web_ui.initialize()

    @filter.command("memory_save")
    async def save_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """手动保存记忆指令：/memory_save <内容>"""
        result = await self._command_handlers.handle_save_memory(event)
        yield event.plain_result(result)

    @filter.command("memory_search")
    async def search_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """搜索记忆指令：/memory_search <查询内容>"""
        result = await self._command_handlers.handle_search_memory(event)
        yield event.plain_result(result)

    @filter.command("memory_clear")
    async def clear_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """清除当前会话记忆指令：/memory_clear"""
        result = await self._command_handlers.handle_clear_memory(event)
        yield event.plain_result(result)

    @filter.command("memory_stats")
    async def memory_stats(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """记忆统计指令：/memory_stats"""
        result = await self._command_handlers.handle_memory_stats(event)
        yield event.plain_result(result)

    @filter.command("memory_delete")
    async def delete_memory(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """统一删除记忆指令：/memory_delete [scope]"""
        result = await self._command_handlers.handle_delete_memory(
            event, self.delete_kv_data
        )
        yield event.plain_result(result)

    @filter.command("proactive_reply")
    async def proactive_reply_control(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """群聊主动回复开关指令：/proactive_reply <on|off|status|list>"""
        result = await self._command_handlers.handle_proactive_reply(
            event, self.put_kv_data
        )
        yield event.plain_result(result)

    @filter.command("activity_status")
    async def activity_status(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看群活跃度状态指令：/activity_status [all]"""
        result = await self._command_handlers.handle_activity_status(event)
        yield event.plain_result(result)

    @filter.command("iris_reset")
    async def iris_reset(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """重置 Iris Memory 所有数据：/iris_reset confirm"""
        result = await self._command_handlers.handle_iris_reset(
            event, self.delete_kv_data
        )
        yield event.plain_result(result)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """
        消息发送前拦截，替换框架错误消息为友好提示
        """
        if not self._error_processor or not self._error_processor.should_process(event):
            return

        result = event.get_result()
        if result:
            self._error_processor.process_result(result)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        """在 LLM 请求前注入上下文"""
        if self._message_processor:
            await self._message_processor.prepare_llm_context(event, req)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any) -> None:
        """在 LLM 响应后记录回复并捕获记忆"""
        if self._message_processor:
            await self._message_processor.handle_llm_response(event, resp)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_messages(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """
        统一处理所有普通消息

        职责：
        1. 记录消息到聊天缓冲区
        2. 分层处理：immediate/batch/discard
        3. 主动回复事件检测与 LLM 请求转发
        """
        if not self._message_processor:
            return

        prompt = await self._message_processor.process_normal_message(event)

        if prompt is not None:
            yield event.request_llm(prompt=prompt)

    async def terminate(self) -> None:
        """插件销毁"""
        if self._web_ui:
            await self._web_ui.stop()

        try:
            await self._service.save_to_kv(self.put_kv_data)
        except Exception as e:
            self._service.logger.warning(f"[Hot-Reload] Error saving KV data: {e}")

        await self._service.terminate()
