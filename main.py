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


@register("astrbot_plugin_iris_memory", "Iris Memory", "基于 companion-memory 框架的三层记忆插件", "1.10.3")
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

        self._service: Optional[MemoryService] = None
        self._command_handlers: Optional[CommandHandlers] = None
        self._web_ui: Optional[WebUIManager] = None
        self._message_processor: Optional[MessageProcessor] = None
        self._error_processor: Optional[ErrorFriendlyProcessor] = None
        self._markdown_stripper: Optional[MarkdownStripper] = None

    async def initialize(self) -> None:
        """异步初始化插件"""
        data_path = Path(StarTools.get_data_dir())
        self._service = MemoryService(self.context, self.config, data_path)
        
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

    @filter.command("memory")
    async def memory_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """记忆管理统一入口：/memory <子命令> [参数]"""
        result = await self._command_handlers.handle_memory_command(
            event, self.delete_kv_data, self.put_kv_data
        )
        yield event.plain_result(result)

    @filter.command("iris")
    async def iris_command(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """Iris 管理统一入口：/iris <子命令> [参数]"""
        result = await self._command_handlers.handle_iris_command(
            event, self.delete_kv_data, self.put_kv_data
        )
        yield event.plain_result(result)

    # ── LLM 工具：群冷却 ──

    @llm_tool(name="set_group_cooldown")
    async def set_group_cooldown_tool(
        self,
        event: AstrMessageEvent,
        duration_minutes: int = 20,
        reason: str = "群聊较活跃，暂时进入安静模式",
    ) -> str:
        """设置群聊冷却模式，在此期间 AI 将暂停主动回复。
        适用于群聊过于活跃、用户需要专注时间、深夜时段等场景。

        Args:
            duration_minutes(number): 冷却时长（分钟），范围 5-180，默认 20
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
        """取消当前群聊的冷却模式，立即恢复 AI 的主动回复能力。
        """
        from iris_memory.utils.event_utils import get_group_id

        group_id = get_group_id(event)
        if not group_id:
            return "非群聊环境，无需取消冷却"

        cooldown_mgr = self._service.cooldown.cooldown_manager
        return cooldown_mgr.deactivate(group_id=group_id, initiated_by="llm")

    # ── LLM 工具：记忆操作 ──

    @llm_tool(name="save_memory")
    async def save_memory_tool(
        self,
        event: AstrMessageEvent,
        content: str,
    ) -> str:
        """手动保存记忆。当用户明确表达重要信息（如喜好、习惯、身份）时调用。

        Args:
            content(string): 要保存的记忆内容，应该是一个完整的陈述句
        """
        from iris_memory.models.memory import Memory
        from iris_memory.core.memory_scope import MemoryScope
        from iris_memory.utils.event_utils import get_group_id

        scope = MemoryScope.PRIVATE if not get_group_id(event) else MemoryScope.GROUP
        memory = Memory(
            content=content,
            scope=scope,
            confidence=0.9,
            source="llm_tool",
        )
        await self._service.capture.capture_memory(
            event,
            [memory],
            from_llm=True,
        )
        return f"已保存记忆：{content}"

    @llm_tool(name="search_memory")
    async def search_memory_tool(
        self,
        event: AstrMessageEvent,
        query: str,
    ) -> str:
        """搜索相关记忆。在需要回忆用户信息时调用。

        Args:
            query(string): 搜索关键词或问题
        """
        results = await self._service.retrieval.search(query, event, top_k=3)
        if not results:
            return "未找到相关记忆"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r.memory.content} (置信度：{r.memory.confidence})")
        return "\n".join(formatted)

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_messages(self, event: AstrMessageEvent) -> None:
        """处理所有消息（包括不触发 LLM 的消息）

        职责：
        1. 记录用户消息到聊天缓冲区
        2. 处理不触发 LLM 的普通消息（分层处理：immediate/batch/discard）

        注意：
        - 触发 LLM 的消息会同时触发 on_llm_request 和 on_llm_response
        - on_llm_response 中也会捕获用户消息，但使用的是 capture_and_store_memory（立即存储）
        - 这里使用 process_normal_message 进行分层处理（可能立即、批量或丢弃）
        - 两者不冲突，因为触发 LLM 的消息不会进入 process_normal_message 的批量处理逻辑
        """
        if self._message_processor:
            await self._message_processor.process_normal_message(event)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req) -> None:
        """消息预处理 Hook：在 LLM 请求前注入记忆"""
        if self._message_processor:
            await self._message_processor.prepare_llm_context(event, req)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp) -> None:
        """消息后处理 Hook：在 LLM 响应后捕获记忆"""
        if self._message_processor:
            await self._message_processor.handle_llm_response(event, resp)

    async def terminate(self) -> None:
        """插件终止时的清理工作"""
        if self._service:
            await self._service.save_to_kv(self.put_kv_data)
            await self._service.terminate()

        if self._web_ui:
            await self._web_ui.stop()
