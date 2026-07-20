"""搜索记忆 LLM Tool"""

from typing import List
from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.l2_memory import MemorySearchResult
from iris_memory.l2_memory.adapter import L2MemoryAdapter
from iris_memory.l3_kg.adapter import L3KGAdapter

logger = get_logger("tools")


@dataclass
class SearchMemoryTool(FunctionTool[AstrAgentContext]):
    """搜索L2记忆库的Tool

    允许LLM主动检索相关记忆，可选附带知识图谱上下文。
    """

    name: str = "search_memory"
    description: str = (
        "从长期记忆库检索相关记忆，用于回忆用户偏好、历史事件、关键信息等。"
        "可同时从知识图谱获取关联实体的结构化上下文。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询文本（描述你想查找的记忆）",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的记忆数量（默认5条，最多20条）",
                    "default": 5,
                },
                "with_graph_context": {
                    "type": "boolean",
                    "description": "是否同时从知识图谱获取关联实体的上下文（默认false）",
                    "default": False,
                },
            },
            "required": ["query"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        try:
            query = kwargs.get("query", "").strip()
            top_k = min(kwargs.get("top_k", 5), 20)
            with_graph = kwargs.get("with_graph_context", False)

            if not query:
                return "查询内容不能为空"

            from iris_memory.utils import sanitize_input

            query = sanitize_input(query, source="tool:search_memory")

            event = context.context.event
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            user_id = adapter.get_user_id(event)
            group_id = adapter.get_group_id(event)

            from iris_memory.config import get_config

            config = get_config()
            if not config.get("isolation_config.enable_group_memory_isolation"):
                group_id = None

            manager = get_component_manager()
            l2_adapter = manager.get_component("l2_memory", L2MemoryAdapter)

            if not l2_adapter or not l2_adapter._is_available:
                return "L2记忆库当前不可用"

            from iris_memory.core.persona import resolve_persona

            persona_id = await resolve_persona(manager, event)

            results: List[MemorySearchResult] = await l2_adapter.retrieve(
                query=query,
                top_k=top_k,
                group_id=group_id,
                persona_id=persona_id,
            )

            if not results:
                return f"未找到与「{query}」相关的记忆"

            output_lines = [f"找到 {len(results)} 条相关记忆：\n"]

            for idx, result in enumerate(results, 1):
                entry = result.entry
                output_lines.append(
                    f"{idx}. [{entry.id}] {entry.content}\n"
                    f"   相似度: {result.score:.2f} | 置信度: {entry.confidence:.2f}\n"
                    f"   时间: {entry.timestamp or '未知'}\n"
                )

            if with_graph:
                graph_context = await self._get_graph_context(
                    manager, results, group_id
                )
                if graph_context:
                    output_lines.append(graph_context)

            logger.info(
                f"LLM检索记忆: user={user_id}, group={group_id}, "
                f"query={query[:30]}..., results={len(results)}, "
                f"with_graph={with_graph}"
            )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"检索记忆失败：{e}", exc_info=True)
            return f"检索记忆失败：{str(e)}"

    async def _get_graph_context(
        self, manager, results: List[MemorySearchResult], group_id
    ) -> str:
        try:
            l3_adapter = manager.get_component("l3_kg", L3KGAdapter)
            if not l3_adapter or not l3_adapter._is_available:
                return ""

            from iris_memory.l3_kg import GraphRetriever

            retriever = GraphRetriever(l3_adapter)

            # L2 条目的 metadata 中不含 source_memory_id（该字段是 L3 图节点的
            # 属性，不是 L2 元数据）。用 L2 条目自身的 id 反查 L3 图节点。
            memory_ids = [r.entry.id for r in results if r.entry.id]
            if not memory_ids:
                return ""

            # 通过 source_memory_id 反查图节点 ID
            graph_node_ids = await l3_adapter.get_node_ids_by_source_memory_ids(
                memory_ids
            )
            if not graph_node_ids:
                return ""

            nodes, edges = await retriever.retrieve_with_expansion(
                graph_node_ids, group_id=group_id
            )

            if not nodes:
                return ""

            return retriever.format_for_context(nodes, edges)

        except Exception as e:
            logger.warning(f"获取图谱上下文失败：{e}")
            return ""
