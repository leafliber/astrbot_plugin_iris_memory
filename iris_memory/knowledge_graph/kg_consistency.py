"""
知识图谱一致性检测器

提供图谱一致性检查：
- 检测矛盾关系（如 A 喜欢 B vs A 讨厌 B）
- 验证边的源/目标节点是否存在
- 检测自引用边（source_id == target_id）
- 检测短循环依赖

设计原则：
- 检测逻辑与存储层分离
- 每个检测方法独立可测、可单独调用
- 仅检测不修复，修复由 KGMaintenanceManager 负责
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

from iris_memory.knowledge_graph.kg_models import KGEdge, KGRelationType
from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage

logger = get_logger("kg_consistency")


# ── 矛盾关系映射 ──
# 若同源同目标节点之间同时存在 key 和 value 中的任一关系类型，则视为矛盾
CONTRADICTORY_RELATIONS: Dict[KGRelationType, FrozenSet[KGRelationType]] = {
    KGRelationType.LIKES: frozenset({KGRelationType.DISLIKES}),
    KGRelationType.DISLIKES: frozenset({KGRelationType.LIKES}),
    KGRelationType.FRIEND_OF: frozenset({KGRelationType.DISLIKES}),
    KGRelationType.BOSS_OF: frozenset({KGRelationType.SUBORDINATE_OF}),
    KGRelationType.SUBORDINATE_OF: frozenset({KGRelationType.BOSS_OF}),
}


@dataclass
class ContradictionIssue:
    """矛盾关系问题"""
    edge_a_id: str
    edge_b_id: str
    source_id: str
    target_id: str
    relation_a: str
    relation_b: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            self.description = (
                f"矛盾关系: {self.relation_a} 与 {self.relation_b} "
                f"(source={self.source_id[:8]}..., target={self.target_id[:8]}...)"
            )


@dataclass
class DanglingEdgeIssue:
    """悬空边问题"""
    edge_id: str
    missing_node_id: str
    is_source_missing: bool
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            side = "源" if self.is_source_missing else "目标"
            self.description = (
                f"悬空边: {side}节点 {self.missing_node_id[:8]}... 不存在 "
                f"(edge={self.edge_id[:8]}...)"
            )


@dataclass
class SelfReferenceIssue:
    """自引用边问题"""
    edge_id: str
    node_id: str
    relation_type: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            self.description = (
                f"自引用边: 节点 {self.node_id[:8]}... "
                f"--[{self.relation_type}]--> 自身 (edge={self.edge_id[:8]}...)"
            )


@dataclass
class CycleIssue:
    """短循环问题"""
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    cycle_length: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            ids_preview = " → ".join(nid[:8] + "..." for nid in self.node_ids[:4])
            self.description = f"循环依赖 (长度={self.cycle_length}): {ids_preview}"


@dataclass
class ConsistencyReport:
    """一致性检测报告"""
    contradictions: List[ContradictionIssue] = field(default_factory=list)
    dangling_edges: List[DanglingEdgeIssue] = field(default_factory=list)
    self_references: List[SelfReferenceIssue] = field(default_factory=list)
    cycles: List[CycleIssue] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (
            len(self.contradictions)
            + len(self.dangling_edges)
            + len(self.self_references)
            + len(self.cycles)
        )

    @property
    def is_consistent(self) -> bool:
        return self.total_issues == 0

    def summary(self) -> str:
        """生成可读摘要"""
        if self.is_consistent:
            return "一致性检测通过：未发现问题"
        lines = [f"一致性检测：共发现 {self.total_issues} 个问题"]
        if self.contradictions:
            lines.append(f"  - 矛盾关系: {len(self.contradictions)} 个")
        if self.dangling_edges:
            lines.append(f"  - 悬空边: {len(self.dangling_edges)} 条")
        if self.self_references:
            lines.append(f"  - 自引用: {len(self.self_references)} 条")
        if self.cycles:
            lines.append(f"  - 循环依赖: {len(self.cycles)} 个")
        return "\n".join(lines)


class KGConsistencyDetector:
    """知识图谱一致性检测器

    职责：
    - 检测矛盾关系（同源同目标之间的冲突关系类型）
    - 验证边引用的节点是否存在
    - 检测自引用边
    - 检测短循环依赖（2-3 跳）

    仅检测不修复。修复操作由 KGMaintenanceManager 执行。
    """

    def __init__(self, storage: "KGStorage") -> None:
        self._storage = storage

    # ================================================================
    # 矛盾关系检测
    # ================================================================

    async def detect_contradictions(self) -> List[ContradictionIssue]:
        """检测矛盾关系

        在同源同目标的边集合中，查找定义在 CONTRADICTORY_RELATIONS 中的冲突对。

        Returns:
            矛盾关系问题列表
        """
        all_edges = await self._storage.get_all_edges()

        # 按 (source_id, target_id) 分组
        edge_groups: Dict[Tuple[str, str], List[KGEdge]] = defaultdict(list)
        for edge in all_edges:
            key = (edge.source_id, edge.target_id)
            edge_groups[key].append(edge)

        issues: List[ContradictionIssue] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for (src, tgt), edges in edge_groups.items():
            if len(edges) < 2:
                continue

            # 检查每对边是否矛盾
            for i, edge_a in enumerate(edges):
                contradictory_types = CONTRADICTORY_RELATIONS.get(edge_a.relation_type)
                if not contradictory_types:
                    continue
                for edge_b in edges[i + 1:]:
                    if edge_b.relation_type in contradictory_types:
                        # 避免重复记录
                        pair_key = tuple(sorted([edge_a.id, edge_b.id]))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        issues.append(ContradictionIssue(
                            edge_a_id=edge_a.id,
                            edge_b_id=edge_b.id,
                            source_id=src,
                            target_id=tgt,
                            relation_a=edge_a.relation_type.value,
                            relation_b=edge_b.relation_type.value,
                        ))

        logger.debug(f"Contradiction detection: found {len(issues)} issues")
        return issues

    # ================================================================
    # 悬空边检测
    # ================================================================

    async def validate_edge_references(self) -> List[DanglingEdgeIssue]:
        """验证所有边的源/目标节点是否存在

        Returns:
            悬空边问题列表
        """
        all_edges = await self._storage.get_all_edges()
        node_ids = await self._storage.get_node_ids_set()

        issues: List[DanglingEdgeIssue] = []
        for edge in all_edges:
            if edge.source_id not in node_ids:
                issues.append(DanglingEdgeIssue(
                    edge_id=edge.id,
                    missing_node_id=edge.source_id,
                    is_source_missing=True,
                ))
            if edge.target_id not in node_ids:
                issues.append(DanglingEdgeIssue(
                    edge_id=edge.id,
                    missing_node_id=edge.target_id,
                    is_source_missing=False,
                ))

        logger.debug(f"Edge reference validation: found {len(issues)} dangling edges")
        return issues

    # ================================================================
    # 自引用检测
    # ================================================================

    async def detect_self_references(self) -> List[SelfReferenceIssue]:
        """检测自引用边（source_id == target_id）

        Returns:
            自引用边问题列表
        """
        self_ref_edges = await self._storage.get_self_referencing_edges()

        issues: List[SelfReferenceIssue] = []
        for edge in self_ref_edges:
            issues.append(SelfReferenceIssue(
                edge_id=edge.id,
                node_id=edge.source_id,
                relation_type=edge.relation_type.value,
            ))

        logger.debug(f"Self-reference detection: found {len(issues)} issues")
        return issues

    # ================================================================
    # 短循环检测
    # ================================================================

    async def detect_cycles(self, max_cycle_length: int = 3) -> List[CycleIssue]:
        """检测短循环依赖（2 ~ max_cycle_length 跳）

        使用有限深度 DFS 检测从每个节点出发的短循环。
        注意：仅检测同方向的循环（A→B→C→A），不检测双向关系（A→B, B→A 视为合法双向关系）。

        Args:
            max_cycle_length: 最大循环长度（默认 3）

        Returns:
            循环问题列表
        """
        all_edges = await self._storage.get_all_edges()

        # 构建邻接表：node_id -> [(edge, target_id)]
        adj: Dict[str, List[Tuple[KGEdge, str]]] = defaultdict(list)
        for edge in all_edges:
            # 跳过自引用（已由 detect_self_references 处理）
            if edge.source_id != edge.target_id:
                adj[edge.source_id].append((edge, edge.target_id))

        issues: List[CycleIssue] = []
        seen_cycles: Set[frozenset] = set()

        for start_node in adj:
            # DFS 检测从 start_node 出发的短循环
            cycles = self._dfs_find_cycles(
                start_node, adj, max_cycle_length
            )
            for cycle_nodes, cycle_edges in cycles:
                # 用 frozenset 去重（不同起点发现的同一循环）
                cycle_key = frozenset(cycle_nodes)
                if cycle_key in seen_cycles:
                    continue
                seen_cycles.add(cycle_key)

                issues.append(CycleIssue(
                    node_ids=cycle_nodes,
                    edge_ids=cycle_edges,
                    cycle_length=len(cycle_nodes),
                ))

        logger.debug(f"Cycle detection: found {len(issues)} cycles (max_length={max_cycle_length})")
        return issues

    @staticmethod
    def _dfs_find_cycles(
        start: str,
        adj: Dict[str, List[Tuple[KGEdge, str]]],
        max_depth: int,
    ) -> List[Tuple[List[str], List[str]]]:
        """有限深度 DFS 检测从 start 出发回到 start 的循环

        Args:
            start: 起始节点 ID
            adj: 邻接表
            max_depth: 最大搜索深度

        Returns:
            [(node_id_list, edge_id_list)] 循环列表
        """
        cycles: List[Tuple[List[str], List[str]]] = []

        # 栈：(当前节点, 路径上的节点列表, 路径上的边列表, 深度)
        stack: List[Tuple[str, List[str], List[str], int]] = [
            (start, [start], [], 0)
        ]

        while stack:
            current, path_nodes, path_edges, depth = stack.pop()

            if depth >= max_depth:
                continue

            for edge, neighbor in adj.get(current, []):
                if neighbor == start and depth >= 1:
                    # 找到回到起点的循环（至少 2 跳）
                    cycles.append((
                        list(path_nodes),
                        path_edges + [edge.id],
                    ))
                elif neighbor not in path_nodes and depth + 1 < max_depth:
                    stack.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge.id],
                        depth + 1,
                    ))

        return cycles

    # ================================================================
    # 完整一致性检查
    # ================================================================

    async def run_all_checks(
        self,
        max_cycle_length: int = 3,
    ) -> ConsistencyReport:
        """执行所有一致性检查

        Args:
            max_cycle_length: 循环检测最大长度

        Returns:
            完整一致性报告
        """
        report = ConsistencyReport()

        report.contradictions = await self.detect_contradictions()
        report.dangling_edges = await self.validate_edge_references()
        report.self_references = await self.detect_self_references()
        report.cycles = await self.detect_cycles(max_cycle_length)

        logger.info(
            f"Consistency check complete: {report.total_issues} issues found "
            f"(contradictions={len(report.contradictions)}, "
            f"dangling={len(report.dangling_edges)}, "
            f"self_ref={len(report.self_references)}, "
            f"cycles={len(report.cycles)})"
        )
        return report
