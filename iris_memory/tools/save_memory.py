"""保存记忆 LLM Tool"""

from datetime import datetime
from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.l2_memory.adapter import L2MemoryAdapter

logger = get_logger("tools")


@dataclass
class SaveMemoryTool(FunctionTool[AstrAgentContext]):
    """保存记忆到L2记忆库的Tool

    允许LLM主动保存重要记忆到长期记忆库。
    """

    name: str = "save_memory"
    description: str = (
        "保存重要记忆到长期记忆库，用于存储用户偏好、重要事件、关键信息等"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "记忆内容（简洁明确，不超过500字）",
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度（0.0-1.0，表示记忆的可靠性）",
                    "default": 1.0,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表（可选，用于分类记忆）",
                },
            },
            "required": ["content"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """执行保存记忆操作

        Args:
            context: AstrBot执行上下文
            **kwargs: Tool参数
                - content: 记忆内容
                - confidence: 置信度（可选）
                - tags: 标签列表（可选）

        Returns:
            str: 包含操作结果的执行结果
        """
        try:
            # 获取参数
            content = kwargs.get("content", "").strip()
            confidence = kwargs.get("confidence", 1.0)
            tags = kwargs.get("tags", [])

            if not content:
                return "记忆内容不能为空"

            from iris_memory.utils import sanitize_input

            content = sanitize_input(content, source="tool:save_memory")

            # 获取event对象
            event = context.context.event

            # 使用Platform适配器获取上下文
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            user_id = adapter.get_user_id(event)
            group_id = adapter.get_group_id(event)
            user_name = adapter.get_user_name(event) or "未知用户"

            # 始终保留真实 group_id：检索侧根据 enable_group_memory_isolation
            # 决定是否过滤，写入侧无需剥离，以便用户后续开启隔离时历史记忆可按群过滤
            # 获取L2记忆适配器
            manager = get_component_manager()
            l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

            if not l2_adapter or not l2_adapter.is_available:
                return "L2记忆库当前不可用"

            from iris_memory.core.persona import resolve_persona

            persona_id = await resolve_persona(manager, event)

            now = datetime.now().isoformat()

            metadata = {
                "user_id": user_id,
                "user_name": user_name,
                "group_id": group_id,
                "timestamp": now,
                "access_count": 1,
                "last_access_time": now,
                "confidence": confidence,
                "source": "tool",
                "tags": tags,
            }

            memory_id = await l2_adapter.add_memory(
                content, metadata, persona_id=persona_id
            )

            if not memory_id:
                return "保存记忆失败：可能存在重复记忆或写入异常"

            logger.info(
                f"LLM保存记忆: user={user_id}, group={group_id}, "
                f"content={content[:50]}..., confidence={confidence}"
            )

            return (
                f"✓ 已保存记忆到长期记忆库\n"
                f"ID: {memory_id}\n"
                f"内容: {content[:100]}{'...' if len(content) > 100 else ''}\n"
                f"置信度: {confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"保存记忆失败：{e}", exc_info=True)
            return f"保存记忆失败：{str(e)}"
