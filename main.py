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
from astrbot.api import AstrBotConfig, llm_tool, logger

from iris_memory.services.memory_service import MemoryService
from iris_memory.utils.logger import init_logging_from_config
from iris_memory.commands import CommandHandlers
from iris_memory.web.web_ui import WebUIManager
from iris_memory.processing.message_processor import MessageProcessor, ErrorFriendlyProcessor
from iris_memory.processing.markdown_stripper import MarkdownStripper
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
        self._markdown_stripper: Optional[MarkdownStripper] = None

    async def initialize(self) -> None:
        """异步初始化插件"""
        init_logging_from_config(self.config, self._service.plugin_data_path)

        await self._service.initialize()

        await self._service.load_from_kv(self.get_kv_data)

        self._command_handlers = CommandHandlers(self._service)
        self._web_ui = WebUIManager(self._service)
        self._message_processor = MessageProcessor(self._service)
        self._error_processor = ErrorFriendlyProcessor(self._service.cfg)
        self._markdown_stripper = MarkdownStripper(
            context=self.context,
            config=self._service.cfg,
        )

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

    @filter.command("cooldown")
    async def cooldown(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """群冷却指令：/cooldown [action] [duration]"""
        result = await self._command_handlers.handle_cooldown(event)
        yield event.plain_result(result)

    # ── LLM 工具：群冷却 ──

    @llm_tool(name="set_group_cooldown")
    async def set_group_cooldown_tool(
        self,
        event: AstrMessageEvent,
        duration_minutes: int = 20,
        reason: str = "群聊较活跃，暂时进入安静模式",
    ) -> str:
        """设置群聊冷却模式，在此期间AI将暂停主动回复。
        适用于群聊过于活跃、用户需要专注时间、深夜时段等场景。

        Args:
            duration_minutes(number): 冷却时长（分钟），范围5-180，默认20
            reason(string): 设置冷却的原因说明
        """
        from iris_memory.utils.event_utils import get_group_id

        group_id = get_group_id(event)
        if not group_id:
            return "冷却模式仅限群聊使用"

        cooldown_mgr = self._service.cooldown.cooldown_manager
        return cooldown_mgr.activate(
            group_id=group_id,
            duration_minutes=duration_minutes,
            reason=reason,
            initiated_by="llm",
        )

    @llm_tool(name="get_cooldown_status")
    async def get_cooldown_status_tool(
        self,
        event: AstrMessageEvent,
    ) -> str:
        """查询当前群聊的冷却状态。在决定是否主动回复前可先调用此工具检查。
        """
        from iris_memory.utils.event_utils import get_group_id

        group_id = get_group_id(event)
        if not group_id:
            return "非群聊环境，无冷却状态"

        cooldown_mgr = self._service.cooldown.cooldown_manager
        return cooldown_mgr.format_status(group_id)

    @llm_tool(name="cancel_group_cooldown")
    async def cancel_group_cooldown_tool(
        self,
        event: AstrMessageEvent,
    ) -> str:
        """取消当前群聊的冷却模式，恢复正常主动回复。
        当用户明确要求恢复时调用。
        """
        from iris_memory.utils.event_utils import get_group_id

        group_id = get_group_id(event)
        if not group_id:
            return "非群聊环境，无冷却状态"

        cooldown_mgr = self._service.cooldown.cooldown_manager
        return cooldown_mgr.deactivate(group_id)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """
        消息发送前拦截，处理链：
        1. 错误消息友好化
        2. Markdown 格式去除
        """
        result = event.get_result()
        if not result:
            return

        # 1. 错误消息友好化
        if self._error_processor and self._error_processor.should_process(event):
            self._error_processor.process_result(result)

        # 2. Markdown 格式去除
        if self._markdown_stripper and self._markdown_stripper.should_process(event):
            self._markdown_stripper.process_result(result)

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
        import asyncio

        if self._web_ui:
            try:
                await asyncio.wait_for(self._web_ui.stop(), timeout=10.0)
            except asyncio.TimeoutError:
                self._service.logger.warning("[Hot-Reload] Web UI stop timed out")
            except Exception as e:
                self._service.logger.warning(f"[Hot-Reload] Error stopping Web UI: {e}")

        try:
            await asyncio.wait_for(
                self._service.save_to_kv(self.put_kv_data), timeout=5.0
            )
        except asyncio.TimeoutError:
            self._service.logger.warning("[Hot-Reload] Save KV data timed out")
        except Exception as e:
            self._service.logger.warning(f"[Hot-Reload] Error saving KV data: {e}")

        try:
            await asyncio.wait_for(self._service.terminate(), timeout=30.0)
        except asyncio.TimeoutError:
            self._service.logger.error("[Hot-Reload] Service terminate timed out")
        except Exception as e:
            self._service.logger.error(f"[Hot-Reload] Error terminating service: {e}")
