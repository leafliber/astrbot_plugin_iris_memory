"""搜索知识图谱 LLM Tool"""

from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.l3_kg.adapter import L3KGAdapter

logger = get_logger("tools")


@dataclass
class SearchKnowledgeGraphTool(FunctionTool[AstrAgentContext]):
    """搜索知识图谱的Tool

    允许LLM主动搜索L3知识图谱中的实体和关系，
    用于获取结构化的长期知识（人物关系、事件关联、概念联系等）。
    """

    name: str = "search_knowledge_graph"
    description: str = (
        "搜索知识图谱中的实体和关系，用于查找人物关系、事件关联、概念联系等结构化知识。"
        "当你需要了解某个实体的详细信息、实体之间的关系，或者想从知识图谱中获取更多上下文时使用此工具。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（实体名称或描述）",
                },
                "label": {
                    "type": "string",
                    "description": "节点类型过滤（可选，如 Person, Event, Concept, Location, Item, Topic）",
                },
                "expand_depth": {
                    "type": "integer",
                    "description": "关系扩展深度（默认1层，最多2层。1层=直接关联，2层=间接关联）",
                    "default": 1,
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
            label = kwargs.get("label", "").strip()
            expand_depth = min(kwargs.get("expand_depth", 1), 2)

            if not query:
                return "搜索关键词不能为空"

            from iris_memory.utils import sanitize_input

            query = sanitize_input(query, source="tool:search_knowledge_graph")

            event = context.context.event
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            group_id = adapter.get_group_id(event)

            from iris_memory.config import get_config

            config = get_config()
            if not config.get("isolation_config.enable_group_memory_isolation"):
                group_id = None

            manager = get_component_manager()
            l3_adapter = manager.get_component("l3_kg", L3KGAdapter)

            if not l3_adapter or not l3_adapter._is_available:
                return "知识图谱当前不可用"

            matched_nodes = await self._search_nodes(l3_adapter, query, label, group_id)

            if not matched_nodes:
                return f"未在知识图谱中找到与「{query}」相关的实体"

            expanded_nodes = []
            expanded_edges = []

            if expand_depth >= 1 and matched_nodes:
                node_ids = [n["id"] for n in matched_nodes if "id" in n]
                if node_ids:
                    try:
                        exp_nodes, exp_edges = await l3_adapter.expand_from_nodes(
                            node_ids=node_ids,
                            max_depth=expand_depth,
                            group_id=group_id,
                        )
                        expanded_nodes = exp_nodes
                        expanded_edges = exp_edges
                    except Exception as e:
                        logger.warning(f"知识图谱关系扩展失败：{e}")

            result_text = self._format_results(
                matched_nodes, expanded_nodes, expanded_edges, query
            )

            user_id = adapter.get_user_id(event)
            logger.info(
                f"LLM搜索知识图谱: user={user_id}, group={group_id}, "
                f"query={query[:30]}..., matched={len(matched_nodes)}, "
                f"expanded_nodes={len(expanded_nodes)}, expanded_edges={len(expanded_edges)}"
            )

            return result_text

        except Exception as e:
            logger.error(f"搜索知识图谱失败：{e}", exc_info=True)
            return f"搜索知识图谱失败：{str(e)}"

    async def _search_nodes(
        self, l3_adapter, query: str, label: str, group_id
    ) -> list[dict]:
        try:
            nodes = await l3_adapter.search_nodes_detailed(
                query=query, label=label or None, group_id=group_id, limit=15
            )
            if len(nodes) > 15:
                logger.debug(f"KG Tool 搜索节点截断：原始 {len(nodes)} 个 → 保留 15 个")
            return nodes[:15]

        except Exception as e:
            logger.warning(f"搜索知识图谱节点失败：{e}")
            return []

    def _format_results(
        self,
        matched_nodes: list[dict],
        expanded_nodes: list[dict],
        expanded_edges: list[dict],
        query: str,
    ) -> str:
        lines = [f"## 知识图谱搜索结果 - 「{query}」", ""]

        lines.append(f"**匹配实体**（{len(matched_nodes)} 个）：")
        for idx, node in enumerate(matched_nodes, 1):
            name = node.get("name", "未知")
            content = node.get("content", "")
            label = node.get("label", "")
            confidence = node.get("confidence", 0)
            if len(content) > 150:
                logger.debug(
                    f"KG Tool 匹配实体内容截断：节点 '{name}'，"
                    f"原始 {len(content)} 字符 → 150 字符"
                )
                content = content[:150] + "..."
            lines.append(
                f"{idx}. [{label}] {name}"
                f"{f'：{content}' if content else ''}"
                f"（置信度: {confidence:.2f}）"
            )

        if expanded_nodes:
            additional_nodes = [
                n
                for n in expanded_nodes
                if n.get("id") not in {m.get("id") for m in matched_nodes}
            ]
            if additional_nodes:
                lines.append("")
                lines.append(f"**关联实体**（{len(additional_nodes)} 个）：")
                if len(additional_nodes) > 10:
                    logger.debug(
                        f"KG Tool 关联实体截断：原始 {len(additional_nodes)} 个 → 保留 10 个"
                    )
                for idx, node in enumerate(additional_nodes[:10], 1):
                    name = node.get("name", "未知")
                    content = node.get("content", "")
                    label = node.get("label", "")
                    if len(content) > 100:
                        logger.debug(
                            f"KG Tool 关联实体内容截断：节点 '{name}'，"
                            f"原始 {len(content)} 字符 → 100 字符"
                        )
                        content = content[:100] + "..."
                    lines.append(
                        f"{idx}. [{label}] {name}{f'：{content}' if content else ''}"
                    )

        if expanded_edges:
            lines.append("")
            lines.append(f"**关联关系**（{len(expanded_edges)} 条）：")
            if len(expanded_edges) > 15:
                logger.debug(
                    f"KG Tool 关联关系截断：原始 {len(expanded_edges)} 条 → 保留 15 条"
                )
            for idx, edge in enumerate(expanded_edges[:15], 1):
                source = edge.get("source_name", edge.get("_src", "未知"))
                target = edge.get("target_name", edge.get("_dst", "未知"))
                relation = edge.get("relation_type", "相关")
                lines.append(f"{idx}. {source} —[{relation}]→ {target}")

        return "\n".join(lines)
