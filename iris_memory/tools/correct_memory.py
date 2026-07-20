"""修正记忆 LLM Tool"""

from datetime import datetime
from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter

logger = get_logger("tools")


@dataclass
class CorrectMemoryTool(FunctionTool[AstrAgentContext]):
    """修正错误记忆的Tool

    允许用户纠正LLM产生的错误记忆或幻觉。
    同时更新L2记忆库和L3知识图谱中的相关节点。
    """

    name: str = "correct_memory"
    description: str = "修正错误记忆或幻觉，用户主动纠正不准确的信息"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "要修正的记忆ID（格式：mem_xxxxxxxxxx）",
                },
                "correction": {
                    "type": "string",
                    "description": "修正后的正确内容",
                },
                "reason": {
                    "type": "string",
                    "description": "修正原因（为什么原记忆是错误的）",
                },
            },
            "required": ["memory_id", "correction", "reason"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        try:
            memory_id = kwargs.get("memory_id", "").strip()
            correction = kwargs.get("correction", "").strip()
            reason = kwargs.get("reason", "").strip()

            if not all([memory_id, correction, reason]):
                return "参数不完整：需要提供 memory_id、correction 和 reason"

            from iris_memory.utils import sanitize_input

            correction = sanitize_input(correction, source="tool:correct_memory")
            reason = sanitize_input(reason, source="tool:correct_memory")

            event = context.context.event
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            user_id = adapter.get_user_id(event)
            group_id = adapter.get_group_id(event)

            manager = get_component_manager()
            l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)
            l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

            if not l2_adapter or not l2_adapter._is_available:
                return "L2记忆库当前不可用"

            from iris_memory.core.persona import resolve_persona

            persona_id = await resolve_persona(manager, event)

            # 按 ID 精确定位原记忆（而非语义检索），避免 memory_id 无语义导致误命中不相关记忆
            original_entry = await l2_adapter.get_entry_by_id(memory_id, persona_id)
            if not original_entry:
                return f"未找到ID为 {memory_id} 的记忆"

            original_content = original_entry.content

            # 群聊隔离校验：开启群记忆隔离时仅允许修正本群记忆，防止跨群越权改写
            from iris_memory.config import get_config

            config = get_config()
            if (
                config.get("isolation_config.enable_group_memory_isolation")
                and group_id
                and original_entry.group_id
                and original_entry.group_id != group_id
            ):
                return "无权修正其他群聊的记忆"

            now = datetime.now().isoformat()
            new_metadata = original_entry.metadata.copy()
            new_metadata.update(
                {
                    "corrected": True,
                    "correction_time": now,
                    "correction_reason": reason,
                    "corrected_by": user_id,
                    "confidence": 1.0,
                }
            )

            # 先写入修正后的记忆，确认成功后再删除原记忆。
            # 这样即使写入失败，原记忆仍在，不会造成数据丢失。
            # skip_dedup=True：修正内容可能与原记忆或其他记忆相似，不应被去重跳过。
            new_id = await l2_adapter.add_memory(
                content=correction,
                metadata=new_metadata,
                persona_id=persona_id,
                skip_dedup=True,
            )
            if not new_id:
                logger.error(f"写入修正记忆失败，原记忆未改动: memory_id={memory_id}")
                return "写入修正记忆失败，原记忆未改动"

            # 新记忆已入库，删除旧记忆。此时即使失败也仅留下重复条目，不会丢数据。
            try:
                await l2_adapter.delete_entries([memory_id])
            except Exception as e:
                logger.warning(
                    f"删除旧记忆失败（新记忆已写入，存在重复）: "
                    f"memory_id={memory_id}, {e}"
                )

            logger.info(
                f"用户修正记忆: user={user_id}, memory_id={memory_id} -> {new_id}, "
                f"original={original_content[:30]}..., "
                f"corrected={correction[:30]}..."
            )

            kg_message = ""

            if l3_adapter and l3_adapter._is_available:
                try:
                    node_id = await l3_adapter.update_node_content_by_source_memory(
                        memory_id, correction, new_source_memory_id=new_id
                    )
                    if node_id:
                        kg_message = "已更新知识图谱中的相关节点"
                        logger.info(f"已更新图谱节点: node_id={node_id}")
                    else:
                        kg_message = "知识图谱中未找到相关节点"

                except Exception as e:
                    logger.warning(f"更新L3图谱失败：{e}")
                    kg_message = f"更新知识图谱失败：{str(e)}"
            else:
                kg_message = "知识图谱未启用或不可用"

            result_lines = [
                "✓ 记忆修正完成",
                "",
                f"记忆ID: {memory_id}",
                f"原始内容: {original_content}",
                f"修正内容: {correction}",
                f"修正原因: {reason}",
                "",
                "L2记忆库: 已更新",
                f"L3知识图谱: {kg_message}",
            ]

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"修正记忆失败：{e}", exc_info=True)
            return f"修正记忆失败：{str(e)}"
