"""实体和关系提取器"""

from typing import List
from iris_memory.core import get_logger
from iris_memory.config import get_config
from .models import (
    GraphNode,
    GraphEdge,
    ExtractionResult,
    NODE_TYPE_WHITELIST,
    NODE_TYPE_DESCRIPTIONS,
    RELATION_TYPE_WHITELIST,
    RELATION_TYPE_DESCRIPTIONS,
)
from iris_memory.l2_memory import MemoryEntry
import json

logger = get_logger("l3_kg")

_MIN_NODE_CONFIDENCE = 0.5

# 需要绑定到 Person 主体的节点类型——这些类型描述的是"某人的属性"，
# 若没有关联的 Person 边则成为无主节点（如"有特定角色偏好"不知道是谁的偏好）
_SUBJECT_BOUND_LABELS = {"Preference", "Trait", "Belief", "Goal", "Skill"}
_MIN_EDGE_CONFIDENCE = 0.4
_MAX_CONTENT_LENGTH = 120


class EntityExtractor:
    """实体和关系提取器

    使用 LLM 从总结文本中提取高度抽象的实体（节点）和关系（边）。
    L3 定位：高度浓缩的结构化知识，场景弱相关，聚焦高层次内容和关联。
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.config = get_config()

    async def extract_from_text(
        self, text: str, context: dict = None
    ) -> ExtractionResult:
        """从文本中提取实体和关系

        Args:
            text: 待提取的文本（总结内容）
            context: 上下文信息（group_id, source_memory_id 等）

        Returns:
            ExtractionResult: 提取结果，包含节点和边
        """
        if context is None:
            context = {}

        prompt = self._build_extraction_prompt(text)

        try:
            response = await self.llm_manager.generate_direct(
                prompt=prompt, module="l3_kg_extraction"
            )

            result = self._parse_extraction_result(response, context)

            result = self._filter_low_quality(result)

            logger.info(
                f"实体提取完成：{len(result.nodes)} 个节点，{len(result.edges)} 条边"
            )

            return result
        except Exception as e:
            logger.error(f"实体提取失败：{e}")
            return ExtractionResult()

    async def extract_from_memories(
        self, memories: List[MemoryEntry], context: dict = None
    ) -> ExtractionResult:
        """从多条记忆中批量提取实体和关系

        将多条记忆合并后提取，用于 L3 知识图谱定时提取任务。

        Args:
            memories: 记忆条目列表
            context: 上下文信息

        Returns:
            ExtractionResult: 提取结果，包含节点和边
        """
        if not memories:
            return ExtractionResult()

        if context is None:
            context = {}

        combined_text = self._combine_memories(memories)

        if context.get("source_memory_id") is None and memories:
            context["source_memory_ids"] = [m.id for m in memories]

        if context.get("group_id") is None and memories:
            context["group_id"] = memories[0].group_id

        active_users = set()
        for mem in memories:
            user_id = mem.metadata.get("user_id")
            if user_id:
                active_users.add(user_id)
        if active_users and not context.get("active_users"):
            context["active_users"] = list(active_users)

        logger.info(f"从 {len(memories)} 条记忆中批量提取实体和关系")

        return await self.extract_from_text(combined_text, context)

    def _combine_memories(self, memories: List[MemoryEntry]) -> str:
        """合并多条记忆内容

        Args:
            memories: 记忆条目列表

        Returns:
            合并后的文本
        """
        lines = []
        for i, mem in enumerate(memories, 1):
            user_info = ""
            user_id = mem.metadata.get("user_id")
            if user_id:
                user_info = f"[用户:{user_id}] "

            lines.append(f"{i}. {user_info}{mem.content}")

        return "\n".join(lines)

    def _build_extraction_prompt(self, text: str) -> str:
        """构建提取 prompt

        L3 定位为高度浓缩的抽象知识，与场景弱相关。
        Prompt 核心指导：提取抽象实体和深层关联，而非具体对话内容。

        Args:
            text: 待提取的文本

        Returns:
            构建好的 prompt
        """
        enable_whitelist = self.config.get("l3_enable_type_whitelist", True)

        if enable_whitelist:
            node_types_desc = "\n".join(
                f"  - {t}: {NODE_TYPE_DESCRIPTIONS.get(t, '')}"
                for t in sorted(NODE_TYPE_WHITELIST)
            )
            rel_types_desc = "\n".join(
                f"  - {t}: {RELATION_TYPE_DESCRIPTIONS.get(t, '')}"
                for t in sorted(RELATION_TYPE_WHITELIST)
            )
            whitelist_hint = f"""
## 可用节点类型（优先使用，如不匹配可创建新类型，PascalCase）
{node_types_desc}

## 可用关系类型（优先使用，如不匹配可创建新类型，UPPER_CASE）
{rel_types_desc}
"""
        else:
            whitelist_hint = ""

        return f"""从对话总结中提取高度抽象的结构化知识。

## 核心原则
提取脱离具体对话场景仍具有长期价值的知识，而非对话内容的简单索引。
- ✅ 偏好、技能、性格特征、目标、信念、深层关联、因果逻辑
- ❌ 临时性内容、寒暄、即时指令、纯情绪、一次性问答

## 提取规则
1. **抽象优先**："张三说喜欢Python"→ Person(张三) -HAS_PREFERENCE→ Preference(Python编程)
2. **关联优先**：重点提取实体间深层关系（因果、支持、矛盾），而非浅层提及
3. **内容浓缩**：content 为一句话概括（≤{_MAX_CONTENT_LENGTH}字），不复制原文
4. **关系精准**：优先使用具体关系类型，RELATED_TO 仅作最后手段
5. **置信度诚实**：不确定 0.3-0.6，确定 0.7-1.0
6. **宁缺毋滥**：无值得保留的抽象知识则返回空结果
7. **主体完整**：Preference/Trait/Belief/Goal/Skill 必须同时提取对应 Person 节点并用边连接，缺少主体的属性节点无价值
{whitelist_hint}
## 输出格式（JSON）
{{
  "nodes": [
    {{
      "label": "Person",
      "name": "实体名称",
      "content": "一句话概括（≤{_MAX_CONTENT_LENGTH}字）",
      "confidence": 0.9
    }}
  ],
  "edges": [
    {{
      "source_label": "Person",
      "source_name": "源实体名称",
      "target_label": "Preference",
      "target_name": "目标实体名称",
      "relation_type": "HAS_PREFERENCE",
      "confidence": 0.8
    }}
  ],
  "extraction_confidence": 0.85
}}

## 待提取文本
{text}

严格按 JSON 格式输出，不要添加其他内容。"""

    def _parse_extraction_result(
        self, response: str, context: dict
    ) -> ExtractionResult:
        """解析 LLM 提取结果

        Args:
            response: LLM 返回的 JSON 字符串
            context: 上下文信息

        Returns:
            解析后的 ExtractionResult 对象
        """
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response.strip())

            nodes = []
            node_key_to_id = {}

            for node_data in data.get("nodes", []):
                properties = node_data.get("properties", {})

                active_users = context.get("active_users", [])
                if active_users:
                    properties["active_users"] = ",".join(active_users)

                source_memory_ids = context.get("source_memory_ids", [])
                if source_memory_ids:
                    properties["source_memory_ids"] = ",".join(source_memory_ids)
                elif context.get("source_memory_id"):
                    properties["source_memory_ids"] = context["source_memory_id"]

                content = node_data.get("content", "")
                if len(content) > _MAX_CONTENT_LENGTH:
                    logger.debug(
                        f"KG 实体提取内容截断：节点 '{node_data.get('name', '?')}'，"
                        f"原始 {len(content)} 字符 → {_MAX_CONTENT_LENGTH} 字符"
                    )
                    content = content[:_MAX_CONTENT_LENGTH]

                node = GraphNode(
                    id="",
                    label=node_data["label"],
                    name=node_data["name"],
                    content=content,
                    confidence=node_data.get("confidence", 1.0),
                    source_memory_id=(
                        ",".join(context["source_memory_ids"])
                        if context.get("source_memory_ids")
                        else context.get("source_memory_id")
                    ),
                    group_id=context.get("group_id"),
                    properties=properties,
                )
                node.id = node.generate_id()
                nodes.append(node)
                node_key_to_id[f"{node.label}:{node.name}"] = node.id

            edges = []
            for edge_data in data.get("edges", []):
                source_label = edge_data.get("source_label")
                source_name = edge_data.get("source_name")
                target_label = edge_data.get("target_label")
                target_name = edge_data.get("target_name")

                source_id = self._resolve_node_id(
                    source_label, source_name, node_key_to_id
                )
                target_id = self._resolve_node_id(
                    target_label, target_name, node_key_to_id
                )

                if source_id and target_id:
                    edge = GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=edge_data["relation_type"],
                        confidence=edge_data.get("confidence", 1.0),
                        source_memory_id=(
                            context.get("source_memory_ids", [None])[0]
                            if context.get("source_memory_ids")
                            else context.get("source_memory_id")
                        ),
                    )
                    edges.append(edge)

            return ExtractionResult(
                nodes=nodes,
                edges=edges,
                extraction_confidence=data.get("extraction_confidence", 1.0),
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败：{e}")
            logger.debug(f"原始响应：{response}")
            return ExtractionResult()
        except Exception as e:
            logger.error(f"解析提取结果失败：{e}")
            return ExtractionResult()

    def _resolve_node_id(
        self,
        label: str | None,
        name: str | None,
        node_key_to_id: dict[str, str],
    ) -> str | None:
        """解析节点 ID，优先精确匹配，回退到名称匹配

        Args:
            label: 节点类型标签
            name: 节点名称
            node_key_to_id: 节点键到 ID 的映射

        Returns:
            节点 ID，未找到返回 None
        """
        if not name:
            return None

        if label and name:
            exact_id = node_key_to_id.get(f"{label}:{name}")
            if exact_id:
                return exact_id

        for key, nid in node_key_to_id.items():
            if label and key.endswith(f":{name}"):
                # 回退匹配时优先要求 label 一致，避免同名不同 label 的节点错连
                # （如 Skill:Python 与 Preference:Python）
                key_label = key.rsplit(":", 1)[0] if ":" in key else ""
                if key_label == label:
                    return nid

        # label 不一致时仍允许按 name 回退，但记录警告
        for key, nid in node_key_to_id.items():
            if key.endswith(f":{name}"):
                logger.warning(
                    f"节点名称回退匹配：'{name}' 存在同名不同 label 的节点，"
                    f"使用 {key}（可能非预期）"
                )
                return nid

        return None

    def _filter_low_quality(self, result: ExtractionResult) -> ExtractionResult:
        """过滤低质量提取结果

        过滤规则：
        - 置信度低于阈值的节点
        - 置信度低于阈值的边
        - 引用已过滤节点的边

        对于 Preference/Trait/Belief/Goal/Skill 节点缺少 Person 关联边
        的情况，不做硬删除（避免误伤有用信息），而是降级置信度并标记，
        交由梦境遗忘清洗阶段按综合评分决定是否淘汰。

        Args:
            result: 原始提取结果

        Returns:
            过滤后的提取结果
        """
        valid_nodes = [n for n in result.nodes if n.confidence >= _MIN_NODE_CONFIDENCE]
        valid_node_ids = {n.id for n in valid_nodes}

        valid_edges = [
            e
            for e in result.edges
            if e.confidence >= _MIN_EDGE_CONFIDENCE
            and e.source_id in valid_node_ids
            and e.target_id in valid_node_ids
        ]

        # 主体关联检查：对缺少 Person 边的主体绑定类型节点降级置信度
        # 不做硬删除——这些节点可能仍有价值，只是主体关联未被 LLM 提取到。
        # 降级后由梦境遗忘清洗按综合评分处理。
        person_node_ids = {
            n.id for n in valid_nodes if n.label == "Person"
        }
        nodes_with_person_edge: set[str] = set()
        for e in valid_edges:
            if e.source_id in person_node_ids and e.target_id not in person_node_ids:
                nodes_with_person_edge.add(e.target_id)
            if e.target_id in person_node_ids and e.source_id not in person_node_ids:
                nodes_with_person_edge.add(e.source_id)

        downgraded_count = 0
        for n in valid_nodes:
            if (
                n.label in _SUBJECT_BOUND_LABELS
                and n.id not in nodes_with_person_edge
            ):
                # 降级置信度，使遗忘评分更容易触发淘汰
                original = n.confidence
                n.confidence = min(n.confidence, 0.4)
                n.properties["orphaned_subject"] = "true"
                downgraded_count += 1
                logger.debug(
                    f"主体关联缺失：节点 '{n.name}' ({n.label}) "
                    f"无 Person 边，置信度 {original} → {n.confidence}"
                )

        if downgraded_count > 0:
            logger.info(
                f"主体关联检查：{downgraded_count} 个 {_SUBJECT_BOUND_LABELS} "
                f"节点缺少 Person 边，已降级置信度并标记"
            )

        filtered_nodes = len(result.nodes) - len(valid_nodes)
        filtered_edges = len(result.edges) - len(valid_edges)

        if filtered_nodes > 0 or filtered_edges > 0:
            logger.debug(
                f"过滤低质量提取：移除 {filtered_nodes} 个节点，{filtered_edges} 条边"
            )

        return ExtractionResult(
            nodes=valid_nodes,
            edges=valid_edges,
            extraction_confidence=result.extraction_confidence,
        )
