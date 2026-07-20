"""保存知识 LLM Tool"""

from pydantic import Field
from pydantic.dataclasses import dataclass
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from iris_memory.core import get_logger, get_component_manager
from iris_memory.l3_kg import GraphNode, GraphEdge
from iris_memory.l3_kg.adapter import L3KGAdapter

logger = get_logger("tools")


@dataclass
class SaveKnowledgeTool(FunctionTool[AstrAgentContext]):
    """保存知识到图谱的Tool

    允许LLM手动添加实体和关系到知识图谱中。
    """

    name: str = "save_knowledge"
    description: str = "保存知识到知识图谱，添加实体节点和关系边"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "节点列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "节点类型（如 Person, Event, Concept）",
                            },
                            "name": {"type": "string", "description": "实体名称"},
                            "content": {"type": "string", "description": "实体描述"},
                            "confidence": {
                                "type": "number",
                                "description": "置信度（0.0-1.0）",
                                "default": 1.0,
                            },
                        },
                        "required": ["label", "name", "content"],
                    },
                },
                "edges": {
                    "type": "array",
                    "description": "边列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_name": {
                                "type": "string",
                                "description": "源实体名称（必须在nodes中定义）",
                            },
                            "target_name": {
                                "type": "string",
                                "description": "目标实体名称（必须在nodes中定义）",
                            },
                            "relation_type": {
                                "type": "string",
                                "description": "关系类型（如 KNOWS, RELATED_TO）",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "置信度（0.0-1.0）",
                                "default": 1.0,
                            },
                        },
                        "required": ["source_name", "target_name", "relation_type"],
                    },
                },
            },
            "required": ["nodes"],
        }
    )

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """执行保存知识操作

        Args:
            context: AstrBot执行上下文
            **kwargs: Tool参数
                - nodes: 节点列表
                - edges: 边列表

        Returns:
            str: 包含操作结果的执行结果
        """
        try:
            # 获取参数
            nodes = kwargs.get("nodes", [])
            edges = kwargs.get("edges", [])

            from iris_memory.utils import sanitize_input

            for node_data in nodes:
                if "content" in node_data:
                    node_data["content"] = sanitize_input(
                        node_data["content"], source="tool:save_knowledge"
                    )
                if "name" in node_data:
                    node_data["name"] = sanitize_input(
                        node_data["name"], source="tool:save_knowledge"
                    )
            for edge_data in edges:
                if "relation_type" in edge_data:
                    edge_data["relation_type"] = sanitize_input(
                        edge_data["relation_type"], source="tool:save_knowledge"
                    )

            # 获取图谱适配器
            component_manager = get_component_manager()
            kg_adapter = component_manager.get_component("l3_kg", L3KGAdapter)

            if not kg_adapter or not kg_adapter._is_available:
                return "知识图谱不可用"

            if not nodes:
                return "未提供任何节点"

            # 解析群聊上下文：始终将知识节点绑定到来源群，
            # 检索侧根据 enable_group_memory_isolation 决定是否跨群共享
            event = context.context.event
            from iris_memory.platform import get_adapter

            adapter = get_adapter(event)
            group_id = adapter.get_group_id(event)

            # 构建 GraphNode 对象
            graph_nodes = []
            for node_data in nodes:
                # 钳制 confidence 到 [0.0, 1.0]，防止 LLM 返回越界值
                # 经 max() 合并后被永久固化，破坏遗忘评分语义
                raw_conf = node_data.get("confidence", 1.0)
                clamped_conf = max(0.0, min(1.0, float(raw_conf)))
                node = GraphNode(
                    id="",
                    label=node_data["label"],
                    name=node_data["name"],
                    content=node_data["content"],
                    confidence=clamped_conf,
                    group_id=group_id,
                )
                node.id = node.generate_id()
                graph_nodes.append(node)

            # 构建 GraphEdge 对象
            node_name_to_id = {n.name: n.id for n in graph_nodes}
            graph_edges = []
            for edge_data in edges:
                source_id = node_name_to_id.get(edge_data["source_name"])
                target_id = node_name_to_id.get(edge_data["target_name"])

                if source_id and target_id:
                    edge = GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=edge_data["relation_type"],
                        confidence=max(0.0, min(1.0, float(edge_data.get("confidence", 1.0)))),
                    )
                    graph_edges.append(edge)

            # 存储到图谱
            added_nodes = 0
            added_edges = 0

            # 主体关联检查：对缺少 Person 边的主体绑定类型节点降级置信度
            # 不阻止保存——这些节点可能仍有价值，降级后由梦境遗忘清洗处理
            _SUBJECT_LABELS = {"Preference", "Trait", "Belief", "Goal", "Skill"}
            nodes_with_person_edge: set[str] = set()
            for edge in graph_edges:
                source_node = next((n for n in graph_nodes if n.id == edge.source_id), None)
                target_node = next((n for n in graph_nodes if n.id == edge.target_id), None)
                if source_node and target_node:
                    if source_node.label == "Person" and target_node.label in _SUBJECT_LABELS:
                        nodes_with_person_edge.add(target_node.id)
                    if target_node.label == "Person" and source_node.label in _SUBJECT_LABELS:
                        nodes_with_person_edge.add(source_node.id)

            orphaned = [
                n for n in graph_nodes
                if n.label in _SUBJECT_LABELS and n.id not in nodes_with_person_edge
            ]
            if orphaned:
                for n in orphaned:
                    n.confidence = min(n.confidence, 0.4)
                    n.properties["orphaned_subject"] = "true"
                logger.warning(
                    f"保存知识：{len(orphaned)} 个 {_SUBJECT_LABELS} 节点缺少 "
                    f"Person 关联边，已降级置信度：{[n.name for n in orphaned]}"
                )

            for node in graph_nodes:
                if await kg_adapter.add_node(node):
                    added_nodes += 1

            for edge in graph_edges:
                if await kg_adapter.add_edge(edge):
                    added_edges += 1

            msg = f"成功保存 {added_nodes} 个节点和 {added_edges} 条边到知识图谱"
            if orphaned:
                msg += f"（{len(orphaned)} 个节点因缺少主体关联已降级置信度）"
            logger.info(msg)
            return msg

        except Exception as e:
            logger.error(f"保存知识失败：{e}")
            return f"保存失败：{str(e)}"
