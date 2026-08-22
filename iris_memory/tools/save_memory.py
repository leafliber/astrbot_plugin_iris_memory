"""保存记忆 LLM Tool"""

from datetime import datetime, timedelta
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
                "importance": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "重要度：high=长期稳定的核心信息（身份/职业/亲密关系/重大事件/持久偏好），"
                        "medium=一般性事实与阶段性信息，low=边缘信息"
                    ),
                    "default": "medium",
                },
                "ttl_hours": {
                    "type": "number",
                    "description": (
                        "存活时长（小时，可选）。仅对确实会过期的临时事实设置，"
                        "如「明天考试」「这周末搬家」；长期信息不要设置"
                    ),
                },
                "scope": {
                    "type": "string",
                    "enum": ["group", "global"],
                    "description": (
                        "作用域：group=仅当前群可见可检索（默认）；"
                        "global=全局共享，所有群与私聊均可检索。"
                        "仅 AstrBot 机器人管理员可创建 global；其他用户会自动降级为 group"
                    ),
                    "default": "group",
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
                - importance: 重要度high/medium/low（可选）
                - tags: 标签列表（可选）

        Returns:
            str: 包含操作结果的执行结果
        """
        try:
            # 获取参数
            content = kwargs.get("content", "").strip()
            confidence = kwargs.get("confidence", 1.0)
            importance = kwargs.get("importance", "medium")
            ttl_hours = kwargs.get("ttl_hours")
            scope = kwargs.get("scope", "group")
            tags = kwargs.get("tags", [])

            if not content:
                return "记忆内容不能为空"

            if importance not in ("high", "medium", "low"):
                importance = "medium"
            if scope not in ("group", "global"):
                scope = "group"

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

            # global 会跨群/私聊共享，不能把 LLM 的参数选择当作权限边界。
            # 只接受 AstrBot 配置的机器人管理员；群主/群管理员并不自动获得
            # 跨群写权限。无法证明管理员身份时一律保守降级为 group。
            scope_downgraded = False
            if scope == "global":
                try:
                    global_allowed = event.is_admin() is True
                except Exception:
                    global_allowed = False
                if not global_allowed:
                    scope = "group"
                    scope_downgraded = True
                    logger.warning(
                        "拒绝未授权的全局记忆写入，已降级为群作用域: "
                        f"user={user_id}, group={group_id}"
                    )

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

            from iris_memory.l1_buffer.summarizer import importance_to_float

            metadata = {
                "user_id": user_id,
                "user_name": user_name,
                "group_id": group_id,
                "timestamp": now,
                "access_count": 1,
                "last_access_time": now,
                "confidence": confidence,
                "importance": importance_to_float(importance),
                "importance_level": importance,
                "source": "tool",
                "tags": tags,
                # 与 L1 总结写入形态对齐：用户级清理按 user_id/active_users
                # 双条件命中，缺失 active_users 的历史工具记忆曾被漏删
                "active_users": user_id,
            }

            # 全局共享记忆显式标记，供检索隔离豁免与清理保护
            if scope == "global":
                metadata["scope"] = "global"

            # TTL：仅当提供合法正数 ttl_hours 时写入 expires_at
            expires_at = None
            if ttl_hours is not None:
                try:
                    ttl_hours = float(ttl_hours)
                except (TypeError, ValueError):
                    ttl_hours = None
                if ttl_hours is not None and ttl_hours > 0:
                    expires_at = (
                        datetime.now() + timedelta(hours=ttl_hours)
                    ).isoformat()
                    metadata["expires_at"] = expires_at

            memory_id = await l2_adapter.add_memory(
                content, metadata, persona_id=persona_id
            )

            if not memory_id:
                return "保存记忆失败：可能存在重复记忆或写入异常"

            logger.info(
                f"LLM保存记忆: user={user_id}, group={group_id}, "
                f"content={content[:50]}..., confidence={confidence}"
                f"{f', expires_at={expires_at}' if expires_at else ''}"
            )

            ttl_line = f"\n过期时间: {expires_at}" if expires_at else ""
            scope_line = (
                "\n作用域: group（未授权创建全局记忆，已自动降级）"
                if scope_downgraded
                else f"\n作用域: {scope}"
            )
            return (
                f"✓ 已保存记忆到长期记忆库\n"
                f"ID: {memory_id}\n"
                f"内容: {content[:100]}{'...' if len(content) > 100 else ''}\n"
                f"置信度: {confidence:.2f}{ttl_line}{scope_line}"
            )

        except Exception as e:
            logger.error(f"保存记忆失败：{e}", exc_info=True)
            return f"保存记忆失败：{str(e)}"
