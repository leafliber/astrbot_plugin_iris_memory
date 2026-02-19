"""
知识图谱结果格式化器

将知识图谱查询/推理结果格式化为可注入 LLM 上下文的文本。
与 MemoryRetrievalEngine.format_memories_for_llm() 协同工作，
生成 【知识关联】 段落。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGPath,
)
from iris_memory.knowledge_graph.kg_reasoning import ReasoningResult
from iris_memory.utils.logger import get_logger

logger = get_logger("kg_context")


class KGContextFormatter:
    """知识图谱上下文格式化器"""

    def __init__(self, max_facts: int = 8, max_chars: int = 500) -> None:
        """
        Args:
            max_facts: 最大注入的事实条数
            max_chars: 最大字符数（粗略的 token 预算控制）
        """
        self.max_facts = max_facts
        self.max_chars = max_chars

    def format_reasoning_result(
        self,
        result: ReasoningResult,
        group_id: Optional[str] = None,
    ) -> str:
        """将多跳推理结果格式化为 LLM 上下文

        Args:
            result: 推理结果
            group_id: 群组 ID（用于标注来源）

        Returns:
            格式化后的文本，如果没有内容返回空字符串
        """
        if not result.has_results:
            return ""

        lines = ["【知识关联】"]
        lines.append("以下是你从记忆中推理出的实体关系，可以自然地在对话中体现，不要机械地列举：")

        facts = result.get_fact_summary()
        char_count = sum(len(l) for l in lines)

        for i, fact in enumerate(facts):
            if i >= self.max_facts:
                break
            line = f"· {fact}"
            if char_count + len(line) > self.max_chars:
                break
            lines.append(line)
            char_count += len(line)

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def format_entity_relations(
        self,
        entity_name: str,
        relations: List[Tuple[KGEdge, KGNode]],
        group_id: Optional[str] = None,
    ) -> str:
        """将单一实体的直接关系格式化

        用于 1 跳快速查询场景。

        Args:
            entity_name: 查询的实体名称
            relations: (边, 邻居节点) 列表
            group_id: 群组 ID

        Returns:
            格式化文本
        """
        if not relations:
            return ""

        lines = ["【知识关联】"]
        lines.append(f"关于「{entity_name}」你了解到：")

        char_count = sum(len(l) for l in lines)
        seen: set = set()

        for edge, neighbor in relations:
            label = edge.relation_label or edge.relation_type.value
            display = neighbor.display_name or neighbor.name
            fact = f"· {entity_name} {label} {display}"

            if fact in seen:
                continue
            seen.add(fact)

            if char_count + len(fact) > self.max_chars:
                break
            if len(seen) > self.max_facts:
                break

            lines.append(fact)
            char_count += len(fact)

        if len(lines) <= 2:
            return ""

        return "\n".join(lines)

    def format_mixed(
        self,
        reasoning_result: Optional[ReasoningResult] = None,
        direct_relations: Optional[List[Tuple[KGEdge, KGNode]]] = None,
        entity_name: str = "",
        group_id: Optional[str] = None,
    ) -> str:
        """混合格式化（多跳 + 直接关系）

        优先使用多跳推理结果，再补充直接关系。
        """
        parts: List[str] = []

        if reasoning_result and reasoning_result.has_results:
            text = self.format_reasoning_result(reasoning_result, group_id)
            if text:
                parts.append(text)

        if direct_relations and entity_name and not parts:
            text = self.format_entity_relations(entity_name, direct_relations, group_id)
            if text:
                parts.append(text)

        return "\n".join(parts) if parts else ""
