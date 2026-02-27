"""
多跳推理引擎 — 受限 BFS

在知识图谱上进行受限广度优先搜索（Bounded BFS），支持：
- 从查询中提取种子实体
- 多跳遍历（默认最多 3 跳）
- 每跳节点数限制（默认 10）
- 路径置信度衰减
- scope 隔离（只遍历可见节点/边）
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGPath,
    KGRelationType,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.utils.logger import get_logger

logger = get_logger("kg_reasoning")

# ── 默认参数 ──
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_NODES_PER_HOP = 10
DEFAULT_MIN_CONFIDENCE = 0.2
CONFIDENCE_DECAY = 0.85            # 每跳置信度衰减因子
BFS_TIMEOUT = 2.0                  # BFS 遍历超时（秒）


@dataclass
class ReasoningResult:
    """推理结果"""
    paths: List[KGPath] = field(default_factory=list)
    seed_nodes: List[KGNode] = field(default_factory=list)
    explored_count: int = 0
    max_depth_reached: int = 0

    @property
    def has_results(self) -> bool:
        return len(self.paths) > 0

    def get_all_nodes(self) -> List[KGNode]:
        """获取所有涉及的节点（去重）"""
        seen: Set[str] = set()
        nodes: List[KGNode] = []
        for path in self.paths:
            for node in path.nodes:
                if node.id not in seen:
                    seen.add(node.id)
                    nodes.append(node)
        return nodes

    def get_all_edges(self) -> List[KGEdge]:
        """获取所有涉及的边（去重）"""
        seen: Set[str] = set()
        edges: List[KGEdge] = []
        for path in self.paths:
            for edge in path.edges:
                if edge.id not in seen:
                    seen.add(edge.id)
                    edges.append(edge)
        return edges

    def get_fact_summary(self) -> List[str]:
        """将推理结果转为事实摘要列表"""
        facts: List[str] = []
        for path in self.paths:
            text = path.to_text()
            if text:
                facts.append(text)
        return facts


class KGReasoning:
    """多跳推理引擎

    基于受限 BFS (Bounded Breadth-First Search) 在知识图谱上进行多跳推理。

    算法流程:
    1. 从查询文本中提取种子实体（通过 FTS5 搜索节点）
    2. 以种子实体为起点，进行 BFS 遍历
    3. 每跳限制探索节点数，并对置信度进行衰减
    4. 收集所有发现的路径，按置信度排序返回
    """

    def __init__(
        self,
        storage: KGStorage,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes_per_hop: int = DEFAULT_MAX_NODES_PER_HOP,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.storage = storage
        self.max_depth = max_depth
        self.max_nodes_per_hop = max_nodes_per_hop
        self.min_confidence = min_confidence

    def estimate_query_depth(self, query: str) -> int:
        """根据查询复杂度动态估算推理深度
        
        规则：
        - 包含多跳关键词（"之间的关系"、"有什么联系"）→ max_depth + 1
        - 包含比较/链式关键词（"通过谁认识"、"间接"）→ max_depth + 2
        - 查询中提及 2+ 实体 → max_depth + 1
        - 简单单实体查询 → max_depth
        
        上限为 5，下限为 1。
        
        Args:
            query: 查询文本
            
        Returns:
            int: 建议推理深度
        """
        depth = self.max_depth
        
        # 多跳关系查询
        multi_hop_keywords = ["之间", "联系", "关系", "关联", "连接", "有关"]
        if any(kw in query for kw in multi_hop_keywords):
            depth += 1
        
        # 间接/链式推理查询
        chain_keywords = ["通过", "间接", "经过", "中间", "怎么认识"]
        if any(kw in query for kw in chain_keywords):
            depth += 2
        
        return max(1, min(depth, 5))

    async def reason(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_results: int = 10,
        persona_id: Optional[str] = None,
    ) -> ReasoningResult:
        """执行多跳推理

        Args:
            query: 查询文本
            user_id: 用户 ID（scope 隔离）
            group_id: 群组 ID（scope 隔离）
            max_depth: 最大跳数（覆盖默认值）
            max_results: 最大返回路径数
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            ReasoningResult: 推理结果
        """
        depth = max_depth or self.estimate_query_depth(query)
        result = ReasoningResult()

        # ── Step 1: 提取种子实体 ──
        seed_nodes = await self._find_seed_nodes(query, user_id, group_id, persona_id)
        if not seed_nodes:
            logger.debug(f"No seed nodes found for query: {query[:50]}")
            return result
        result.seed_nodes = seed_nodes

        # ── Step 2: BFS 遍历 ──
        all_paths: List[KGPath] = []
        for seed in seed_nodes:
            paths = await self._bfs_from_node(
                start_node=seed,
                max_depth=depth,
                user_id=user_id,
                group_id=group_id,
                persona_id=persona_id,
            )
            all_paths.extend(paths)

        # ── Step 3: 去重和排序 ──
        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda p: p.total_confidence, reverse=True)
        result.paths = unique_paths[:max_results]

        if result.paths:
            result.max_depth_reached = max(p.hop_count for p in result.paths)

        logger.debug(
            f"Reasoning complete: seeds={len(seed_nodes)}, "
            f"paths={len(result.paths)}, max_depth={result.max_depth_reached}"
        )
        return result

    async def query_entity_relations(
        self,
        entity_name: str,
        user_id: str,
        group_id: Optional[str] = None,
        relation_type: Optional[KGRelationType] = None,
        persona_id: Optional[str] = None,
    ) -> List[Tuple[KGEdge, KGNode]]:
        """查询指定实体的直接关系（1 跳）

        Args:
            entity_name: 实体名称
            user_id: 用户 ID
            group_id: 群组 ID
            relation_type: 过滤关系类型
            persona_id: 人格 ID（非 None 时启用 persona 过滤）

        Returns:
            (边, 邻居节点) 列表
        """
        nodes = await self.storage.search_nodes(
            entity_name, user_id=user_id, group_id=group_id, limit=3,
            persona_id=persona_id,
        )
        if not nodes:
            return []

        results: List[Tuple[KGEdge, KGNode]] = []
        for node in nodes:
            neighbors = await self.storage.get_neighbors(node.id)
            for edge, neighbor in neighbors:
                if relation_type and edge.relation_type != relation_type:
                    continue
                # scope 检查
                if self._is_visible(neighbor, user_id, group_id, persona_id):
                    results.append((edge, neighbor))

        return results

    # ================================================================
    # BFS 核心
    # ================================================================

    async def _bfs_from_node(
        self,
        start_node: KGNode,
        max_depth: int,
        user_id: str,
        group_id: Optional[str],
        persona_id: Optional[str] = None,
    ) -> List[KGPath]:
        """从单个节点开始 BFS

        使用 deque 实现标准 BFS，每跳限制 max_nodes_per_hop 个节点。
        同时收集路径信息。设有超时机制防止大规模图谱查询阻塞事件循环。
        """
        paths: List[KGPath] = []
        visited: Set[str] = {start_node.id}

        # BFS 队列：(当前节点, 当前路径节点列表, 当前路径边列表, 当前置信度, 当前深度)
        queue: deque[Tuple[KGNode, List[KGNode], List[KGEdge], float, int]] = deque()
        queue.append((start_node, [start_node], [], start_node.confidence, 0))

        explored = 0
        start_time = time.monotonic()

        while queue:
            # 超时检查
            if time.monotonic() - start_time > BFS_TIMEOUT:
                logger.warning(
                    f"BFS timeout reached ({BFS_TIMEOUT}s), "
                    f"explored={explored}, paths={len(paths)}"
                )
                break

            current, path_nodes, path_edges, path_conf, depth = queue.popleft()

            # 深度限制
            if depth >= max_depth:
                continue

            # 获取邻居
            neighbors = self.storage.get_neighbors_sync(current.id, limit=self.max_nodes_per_hop * 2)
            hop_count = 0

            for edge, neighbor in neighbors:
                if neighbor.id in visited:
                    continue

                # scope 检查
                if not self._is_visible(neighbor, user_id, group_id, persona_id):
                    continue

                # 置信度衰减
                new_conf = path_conf * CONFIDENCE_DECAY * edge.confidence
                if new_conf < self.min_confidence:
                    continue

                visited.add(neighbor.id)
                explored += 1
                hop_count += 1

                new_path_nodes = path_nodes + [neighbor]
                new_path_edges = path_edges + [edge]

                # 记录路径（每个中间节点和终点都产生一条路径）
                path = KGPath(
                    nodes=list(new_path_nodes),
                    edges=list(new_path_edges),
                    total_confidence=new_conf,
                    hop_count=depth + 1,
                )
                paths.append(path)

                # 继续扩展
                if depth + 1 < max_depth:
                    queue.append((
                        neighbor, new_path_nodes, new_path_edges,
                        new_conf, depth + 1,
                    ))

                if hop_count >= self.max_nodes_per_hop:
                    break

        return paths

    # ================================================================
    # 种子实体提取
    # ================================================================

    async def _find_seed_nodes(
        self,
        query: str,
        user_id: str,
        group_id: Optional[str],
        persona_id: Optional[str] = None,
    ) -> List[KGNode]:
        """从查询中提取种子实体

        策略：
        1. 提取查询中可能的实体词（关键词/专有名词）
        2. 在 KG 中搜索匹配节点
        3. 按匹配度和提及次数排序
        """
        # 提取候选关键词
        candidates = self._extract_query_entities(query)

        seed_nodes: List[KGNode] = []
        seen_ids: Set[str] = set()

        # 先尝试每个候选词单独搜索
        for candidate in candidates:
            nodes = await self.storage.search_nodes(
                candidate, user_id=user_id, group_id=group_id, limit=5,
                persona_id=persona_id,
            )
            for node in nodes:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    seed_nodes.append(node)

        # 如果没找到，用整个查询搜索
        if not seed_nodes:
            nodes = await self.storage.search_nodes(
                query, user_id=user_id, group_id=group_id, limit=5,
                persona_id=persona_id,
            )
            for node in nodes:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    seed_nodes.append(node)

        # 按 mention_count 排序（高优先）
        seed_nodes.sort(key=lambda n: n.mention_count, reverse=True)
        return seed_nodes[:5]  # 最多 5 个种子

    def _extract_query_entities(self, query: str) -> List[str]:
        """从查询文本提取候选实体词

        简单规则：
        - 中文人名模式（姓 + 1~2 字 / 昵称前缀）
        - 英文专有名词（首字母大写词）
        - 引号内容
        - 关系查询中的关键主体
        """
        candidates: List[str] = []

        # 引号中的内容
        for m in re.finditer(r'[""「」『』](.*?)[""「」『』]', query):
            candidates.append(m.group(1))

        # 中文人名/昵称
        for m in re.finditer(r'[小老阿][\u4e00-\u9fa5]', query):
            candidates.append(m.group())
        # 常见姓氏
        surnames = '王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹'
        # 2 字名
        for m in re.finditer(f'[{surnames}][\u4e00-\u9fa5]', query):
            candidates.append(m.group())
        # 3 字名
        for m in re.finditer(f'[{surnames}][\u4e00-\u9fa5]{{2}}(?![\u4e00-\u9fa5])', query):
            candidates.append(m.group())

        # 英文专有名词
        for m in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', query):
            candidates.append(m.group())

        # 关系查询结构提取
        # "XXX的YYY是谁" / "谁是XXX的YYY"
        for m in re.finditer(r'([\u4e00-\u9fa5A-Za-z]+)的(?:[上下]司|同事|朋友|老板)', query):
            candidates.append(m.group(1))
        for m in re.finditer(r'谁是([\u4e00-\u9fa5A-Za-z]+)', query):
            candidates.append(m.group(1))
        for m in re.finditer(r'([\u4e00-\u9fa5A-Za-z]+)是谁', query):
            candidates.append(m.group(1))

        # 去重，保持顺序
        seen: Set[str] = set()
        unique: List[str] = []
        for c in candidates:
            c = c.strip()
            if c and c not in seen and len(c) >= 1:
                seen.add(c)
                unique.append(c)

        return unique

    # ================================================================
    # 辅助
    # ================================================================

    def _is_visible(
        self,
        node: KGNode,
        user_id: str,
        group_id: Optional[str],
        persona_id: Optional[str] = None,
    ) -> bool:
        """检查节点对当前 scope 是否可见"""
        # persona 过滤
        if persona_id is not None and node.persona_id != persona_id:
            return False
        if group_id:
            return node.user_id == user_id or node.group_id == group_id
        else:
            return node.user_id == user_id and (node.group_id is None or node.group_id == "")

    def _deduplicate_paths(self, paths: List[KGPath]) -> List[KGPath]:
        """路径去重：按终点+起点去重，保留置信度最高的"""
        best: Dict[str, KGPath] = {}
        for path in paths:
            if len(path.nodes) < 2:
                continue
            key = f"{path.nodes[0].id}->{path.nodes[-1].id}:{path.hop_count}"
            if key not in best or path.total_confidence > best[key].total_confidence:
                best[key] = path
        return list(best.values())
