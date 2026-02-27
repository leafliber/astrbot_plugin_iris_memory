"""
知识图谱质量报告生成器

提供图谱质量统计与报告：
- 孤立节点比例
- 低置信度节点/边占比
- 关系类型分布
- 节点类型分布
- 平均置信度等综合指标

设计原则：
- 只读操作，不修改图谱数据
- 每个统计方法独立可测
- 通过 KGStorage 公共接口获取数据
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage

logger = get_logger("kg_quality")

# ── 默认阈值 ──
DEFAULT_LOW_CONFIDENCE_THRESHOLD: float = 0.3


@dataclass
class LowConfidenceStats:
    """低置信度统计"""
    low_confidence_node_count: int = 0
    low_confidence_node_ratio: float = 0.0
    low_confidence_edge_count: int = 0
    low_confidence_edge_ratio: float = 0.0
    threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD


@dataclass
class QualityReport:
    """图谱质量报告"""
    # 基础统计
    total_nodes: int = 0
    total_edges: int = 0

    # 孤立节点
    orphan_node_count: int = 0
    orphan_node_ratio: float = 0.0

    # 低置信度
    low_confidence_stats: LowConfidenceStats = field(default_factory=LowConfidenceStats)

    # 平均置信度
    avg_node_confidence: float = 0.0
    avg_edge_confidence: float = 0.0

    # 分布
    relation_type_distribution: Dict[str, int] = field(default_factory=dict)
    node_type_distribution: Dict[str, int] = field(default_factory=dict)

    # 连通性（可选扩展）
    avg_edges_per_node: float = 0.0

    def summary(self) -> str:
        """生成可读摘要"""
        lines = ["图谱质量报告："]
        lines.append(f"  节点总数: {self.total_nodes}")
        lines.append(f"  边总数: {self.total_edges}")
        lines.append(f"  孤立节点: {self.orphan_node_count} ({self.orphan_node_ratio:.1%})")
        lines.append(
            f"  低置信度节点: {self.low_confidence_stats.low_confidence_node_count} "
            f"({self.low_confidence_stats.low_confidence_node_ratio:.1%})"
        )
        lines.append(
            f"  低置信度边: {self.low_confidence_stats.low_confidence_edge_count} "
            f"({self.low_confidence_stats.low_confidence_edge_ratio:.1%})"
        )
        lines.append(f"  平均节点置信度: {self.avg_node_confidence:.3f}")
        lines.append(f"  平均边置信度: {self.avg_edge_confidence:.3f}")
        lines.append(f"  平均每节点边数: {self.avg_edges_per_node:.2f}")

        if self.node_type_distribution:
            lines.append("  节点类型分布:")
            for ntype, count in sorted(
                self.node_type_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    - {ntype}: {count}")

        if self.relation_type_distribution:
            lines.append("  关系类型分布:")
            for rtype, count in sorted(
                self.relation_type_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    - {rtype}: {count}")

        return "\n".join(lines)


class KGQualityReporter:
    """知识图谱质量报告生成器

    职责：
    - 统计孤立节点比例
    - 统计低置信度节点/边占比
    - 统计关系类型与节点类型分布
    - 计算平均置信度等综合指标

    所有方法都是只读的，不会修改图谱数据。
    优先使用 SQL 聚合查询避免全量加载到 Python 内存。
    """

    def __init__(self, storage: "KGStorage") -> None:
        self._storage = storage

    # ================================================================
    # 孤立节点统计
    # ================================================================

    async def get_orphan_node_ratio(self) -> float:
        """计算孤立节点比例

        Returns:
            孤立节点占总节点数的比例 (0.0 ~ 1.0)，无节点时返回 0.0
        """
        orphan_ids = await self._storage.get_orphan_node_ids()
        total_nodes = await self._storage.get_node_count()

        if total_nodes == 0:
            return 0.0
        return len(orphan_ids) / total_nodes

    # ================================================================
    # 低置信度统计
    # ================================================================

    async def get_low_confidence_stats(
        self,
        threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    ) -> LowConfidenceStats:
        """统计低置信度节点和边的占比（使用 SQL 聚合）

        Args:
            threshold: 低置信度阈值（低于此值视为低置信度）

        Returns:
            低置信度统计数据
        """
        low_counts = await self._storage.get_low_confidence_counts(threshold)
        total_nodes = await self._storage.get_node_count()
        total_edges = await self._storage.get_edge_count()

        low_conf_nodes = low_counts["nodes"]
        low_conf_edges = low_counts["edges"]

        return LowConfidenceStats(
            low_confidence_node_count=low_conf_nodes,
            low_confidence_node_ratio=low_conf_nodes / total_nodes if total_nodes > 0 else 0.0,
            low_confidence_edge_count=low_conf_edges,
            low_confidence_edge_ratio=low_conf_edges / total_edges if total_edges > 0 else 0.0,
            threshold=threshold,
        )

    # ================================================================
    # 关系类型分布
    # ================================================================

    async def get_relation_type_distribution(self) -> Dict[str, int]:
        """统计关系类型分布（使用 SQL GROUP BY）

        Returns:
            {relation_type_value: count} 的字典
        """
        return await self._storage.get_relation_type_distribution()

    # ================================================================
    # 节点类型分布
    # ================================================================

    async def get_node_type_distribution(self) -> Dict[str, int]:
        """统计节点类型分布（使用 SQL GROUP BY）

        Returns:
            {node_type_value: count} 的字典
        """
        return await self._storage.get_node_type_distribution()

    # ================================================================
    # 平均置信度
    # ================================================================

    async def get_avg_confidence(self) -> Dict[str, float]:
        """计算节点和边的平均置信度（使用 SQL AVG）

        Returns:
            {"nodes": avg_node_conf, "edges": avg_edge_conf}
        """
        return await self._storage.get_avg_confidence()

    # ================================================================
    # 完整质量报告
    # ================================================================

    async def generate_report(
        self,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    ) -> QualityReport:
        """生成完整质量报告

        使用 SQL 聚合查询替代全量加载，大幅降低内存占用。

        Args:
            low_confidence_threshold: 低置信度阈值

        Returns:
            完整质量报告
        """
        # 使用 SQL 聚合获取统计数据（避免 get_all_nodes / get_all_edges）
        total_nodes = await self._storage.get_node_count()
        total_edges = await self._storage.get_edge_count()
        orphan_ids = await self._storage.get_orphan_node_ids()
        avg_conf = await self._storage.get_avg_confidence()
        low_counts = await self._storage.get_low_confidence_counts(low_confidence_threshold)
        node_type_dist = await self._storage.get_node_type_distribution()
        relation_type_dist = await self._storage.get_relation_type_distribution()

        low_conf_nodes = low_counts["nodes"]
        low_conf_edges = low_counts["edges"]

        # 每节点平均边数
        avg_edges = total_edges / total_nodes if total_nodes > 0 else 0.0

        report = QualityReport(
            total_nodes=total_nodes,
            total_edges=total_edges,
            orphan_node_count=len(orphan_ids),
            orphan_node_ratio=len(orphan_ids) / total_nodes if total_nodes > 0 else 0.0,
            low_confidence_stats=LowConfidenceStats(
                low_confidence_node_count=low_conf_nodes,
                low_confidence_node_ratio=low_conf_nodes / total_nodes if total_nodes > 0 else 0.0,
                low_confidence_edge_count=low_conf_edges,
                low_confidence_edge_ratio=low_conf_edges / total_edges if total_edges > 0 else 0.0,
                threshold=low_confidence_threshold,
            ),
            avg_node_confidence=avg_conf["nodes"],
            avg_edge_confidence=avg_conf["edges"],
            relation_type_distribution=relation_type_dist,
            node_type_distribution=node_type_dist,
            avg_edges_per_node=avg_edges,
        )

        logger.info(
            f"Quality report: nodes={total_nodes}, edges={total_edges}, "
            f"orphans={len(orphan_ids)}, "
            f"low_conf_nodes={low_conf_nodes}, low_conf_edges={low_conf_edges}"
        )
        return report
