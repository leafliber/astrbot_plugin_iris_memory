"""图谱检索器"""

from typing import Optional

from iris_memory.core import get_logger
from iris_memory.config import get_config
from .adapter import L3KGAdapter
from collections import defaultdict
import asyncio

logger = get_logger("l3_kg")

_RELATION_TYPE_LABELS = {
    "KNOWS": "认识",
    "HAS_PREFERENCE": "偏好",
    "HAS_SKILL": "掌握",
    "HAS_TRAIT": "具有",
    "HAS_GOAL": "追求",
    "HAS_BELIEF": "相信",
    "PARTICIPATED_IN": "参与",
    "LOCATED_AT": "位于",
    "HAPPENED_AT": "发生在",
    "PART_OF": "属于",
    "LEADS_TO": "导致",
    "CONTRADICTS": "矛盾",
    "SUPPORTS": "支持",
    "RELATED_TO": "相关",
    "MENTIONED": "提及",
    "DISCUSSED": "讨论过",
}

_NODE_TYPE_LABELS = {
    "Person": "人物",
    "Preference": "偏好",
    "Skill": "技能",
    "Trait": "性格特征",
    "Goal": "目标",
    "Belief": "信念",
    "Event": "事件",
    "Concept": "概念",
    "Location": "地点",
    "Item": "物品",
    "Topic": "话题",
    "Group": "群体",
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)


class GraphRetriever:
    """图谱检索器

    提供：
    - 路径扩展检索
    - 超时保护
    - 访问计数更新
    - 结果格式化
    """

    def __init__(self, adapter: L3KGAdapter):
        self.adapter = adapter
        self.config = get_config()

    async def retrieve_with_expansion(
        self, memory_node_ids: list[str], group_id: Optional[str] = None
    ) -> tuple[list[dict], list[dict]]:
        if not self.adapter.is_available:
            return [], []

        try:
            max_depth = self.config.get("l3_expansion_depth", 2)
            timeout_ms = self.config.get("l3_timeout_ms", 1500)

            nodes, edges = await asyncio.wait_for(
                self.adapter.expand_from_nodes(
                    node_ids=memory_node_ids, max_depth=max_depth, group_id=group_id
                ),
                timeout=timeout_ms / 1000,
            )

            logger.info(
                f"图增强检索完成：{len(nodes)} 个节点，"
                f"{len(edges)} 条边，深度 {max_depth}"
            )

            return nodes, edges
        except asyncio.TimeoutError:
            logger.warning(f"图增强检索超时（{timeout_ms}ms），跳过")
            return [], []
        except Exception as e:
            logger.error(f"图增强检索失败：{e}")
            return [], []

    async def retrieve_by_keywords(
        self, keywords: list[str], group_id: Optional[str] = None, limit: int = 10
    ) -> tuple[list[dict], list[dict]]:
        """基于关键词搜索图谱节点并扩展

        Args:
            keywords: 搜索关键词列表
            group_id: 群聊ID
            limit: 每个关键词最大返回节点数

        Returns:
            (节点列表, 边列表)
        """
        if not self.adapter.is_available or not keywords:
            return [], []

        matched_node_ids: set[str] = set()

        for keyword in keywords:
            try:
                found = await self.adapter.search_nodes(
                    keyword, limit=limit, group_id=group_id
                )
                for node in found:
                    node_id = node.get("id")
                    if node_id:
                        matched_node_ids.add(node_id)
            except Exception as e:
                logger.debug(f"关键词 '{keyword}' 搜索失败：{e}")

        if not matched_node_ids:
            return [], []

        if len(matched_node_ids) > 20:
            logger.debug(
                f"L3 关键词检索节点截断：原始 {len(matched_node_ids)} 个 → 保留 20 个"
            )
            matched_node_ids = set(list(matched_node_ids)[:20])

        return await self.retrieve_with_expansion(
            memory_node_ids=list(matched_node_ids), group_id=group_id
        )

    async def update_access_count(self, node_ids: list[str]):
        await self.adapter.update_node_access(node_ids)

    def format_for_context(
        self,
        nodes: list[dict],
        edges: list[dict],
        max_tokens: int = 400,
        max_content_length: int = 150,
    ) -> str:
        """格式化图谱结果为上下文文本

        按节点类型分组展示实体，使用自然语言描述关系，
        支持 token 预算控制。

        Args:
            nodes: 节点列表
            edges: 边列表
            max_tokens: 最大 token 预算（估算）
            max_content_length: 节点描述最大字符数

        Returns:
            (格式化的文本, 实际纳入文本的节点 ID 集合)，如果为空则返回 ("", set())
        """
        if not nodes:
            return "", set()

        node_map: dict[str, dict] = {}
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                node_map[node_id] = node

        type_groups: dict[str, list[dict]] = defaultdict(list)
        for node in nodes:
            label = node.get("label", "Entity")
            type_groups[label].append(node)

        lines: list[str] = []
        token_budget = max_tokens
        included_node_ids: set[str] = set()

        header = "【长期知识】以下是关于相关人物和概念的深层知识，在回答时自然参考："
        lines.append(header)
        token_budget -= _estimate_tokens(header)

        _PRIORITY_ORDER = {
            "Person": 0,
            "Trait": 1,
            "Preference": 2,
            "Skill": 3,
            "Goal": 4,
            "Belief": 5,
            "Concept": 6,
            "Event": 7,
            "Group": 8,
            "Topic": 9,
            "Location": 10,
            "Item": 11,
        }

        ordered_types = sorted(
            type_groups.keys(),
            key=lambda t: (_PRIORITY_ORDER.get(t, 99), t),
        )

        for node_type in ordered_types:
            if token_budget <= 0:
                break

            group = type_groups[node_type]
            type_label = _NODE_TYPE_LABELS.get(node_type, node_type)

            entity_lines: list[str] = []
            included_in_group: list[str] = []
            for node in group:
                name = node.get("name", "")
                content = node.get("content", "")
                node_id = node.get("id", "")
                if name and content:
                    if len(content) > max_content_length:
                        logger.debug(
                            f"L3 图谱节点内容截断：节点 '{name}' ({node_type})，"
                            f"原始 {len(content)} 字符 → {max_content_length} 字符"
                        )
                        truncated = content[:max_content_length]
                        for sep in ("。", "；", "，", " "):
                            last_sep = truncated.rfind(sep)
                            if last_sep > max_content_length // 2:
                                truncated = truncated[:last_sep]
                                break
                        content = truncated + "…"
                    entity_lines.append(f"  - {name}：{content}")
                    included_in_group.append(node_id)
                elif name:
                    entity_lines.append(f"  - {name}")
                    included_in_group.append(node_id)

            if not entity_lines:
                continue

            section = f"{type_label}：\n" + "\n".join(entity_lines)
            section_tokens = _estimate_tokens(section)

            if section_tokens > token_budget:
                remaining = max(1, token_budget // 30)
                logger.debug(
                    f"L3 图谱实体截断：类型 {node_type}，"
                    f"原始 {len(entity_lines)} 条 → 保留 {remaining} 条（token 预算不足）"
                )
                section = f"{type_label}：\n" + "\n".join(entity_lines[:remaining])
                section_tokens = _estimate_tokens(section)
                included_in_group = included_in_group[:remaining]

            if section_tokens <= token_budget:
                lines.append(section)
                token_budget -= section_tokens
                included_node_ids.update(
                    nid for nid in included_in_group if nid
                )

        if edges and token_budget > 20:
            edge_lines: list[str] = []
            for edge in edges:
                source_id = edge.get("source", edge.get("_src", ""))
                target_id = edge.get("target", edge.get("_dst", ""))
                if isinstance(source_id, dict):
                    source_id = source_id.get("id", "")
                if isinstance(target_id, dict):
                    target_id = target_id.get("id", "")
                relation = edge.get("relation_type", "")

                source_node = node_map.get(source_id, {})
                target_node = node_map.get(target_id, {})

                source_name = source_node.get("name", source_id)
                target_name = target_node.get("name", target_id)

                if source_name and target_name and relation:
                    rel_label = _RELATION_TYPE_LABELS.get(relation, relation)
                    edge_lines.append(f"  - {source_name} {rel_label} {target_name}")

            if edge_lines:
                rel_section = "关系：\n" + "\n".join(edge_lines)
                rel_tokens = _estimate_tokens(rel_section)

                if rel_tokens > token_budget:
                    remaining = max(1, token_budget // 20)
                    logger.debug(
                        f"L3 图谱关系截断：原始 {len(edge_lines)} 条 → 保留 {remaining} 条（token 预算不足）"
                    )
                    rel_section = "关系：\n" + "\n".join(edge_lines[:remaining])

                if _estimate_tokens(rel_section) <= token_budget:
                    lines.append(rel_section)

        result = "\n".join(lines)
        if result.strip() == header.strip():
            return "", set()

        return result, included_node_ids
