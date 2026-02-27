"""
知识图谱维护管理器

提供定时清理任务：
- 检测并移除孤立节点（无任何边连接）
- 清理低置信度关系（confidence < 阈值且长期未更新）
- 移除指向不存在节点的悬空边

设计原则：
- 维护逻辑与存储层分离，通过 KGStorage 公共接口完成所有查询/删除
- 每个清理方法独立可测、可单独调用
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING

from iris_memory.utils.logger import get_logger

if TYPE_CHECKING:
    from iris_memory.knowledge_graph.kg_storage import KGStorage

logger = get_logger("kg_maintenance")


# ── 默认阈值 ──
DEFAULT_LOW_CONFIDENCE_THRESHOLD: float = 0.2
DEFAULT_STALENESS_DAYS: int = 30


@dataclass
class CleanupResult:
    """单次清理任务结果"""
    task_name: str
    removed_count: int = 0
    details: List[str] = field(default_factory=list)


@dataclass
class MaintenanceReport:
    """完整维护报告"""
    results: List[CleanupResult] = field(default_factory=list)
    total_removed: int = 0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """生成可读摘要"""
        lines = [f"维护报告（耗时 {self.duration_seconds:.2f}s，共清理 {self.total_removed} 项）："]
        for r in self.results:
            lines.append(f"  - {r.task_name}: 移除 {r.removed_count} 项")
            for detail in r.details[:5]:
                lines.append(f"    · {detail}")
        return "\n".join(lines)


class KGMaintenanceManager:
    """知识图谱维护管理器

    职责：
    - 孤立节点检测与清理
    - 低置信度过期边清理
    - 悬空边（外键失效）清理

    所有操作通过 KGStorage 公共接口完成，不直接操作数据库。
    """

    def __init__(self, storage: "KGStorage") -> None:
        self._storage = storage

    # ================================================================
    # 孤立节点清理
    # ================================================================

    async def find_orphan_nodes(self) -> List[str]:
        """检测孤立节点（无任何边连接的节点）

        Returns:
            孤立节点 ID 列表
        """
        return await self._storage.get_orphan_node_ids()

    async def remove_orphan_nodes(self) -> CleanupResult:
        """检测并移除所有孤立节点

        Returns:
            清理结果
        """
        result = CleanupResult(task_name="孤立节点清理")

        orphan_ids = await self.find_orphan_nodes()
        if not orphan_ids:
            result.details.append("未发现孤立节点")
            return result

        # 记录被删除的节点信息（用于日志）
        for node_id in orphan_ids[:10]:
            node = await self._storage.get_node(node_id)
            if node:
                result.details.append(
                    f"移除孤立节点: {node.display_name or node.name} "
                    f"(type={node.node_type.value}, user={node.user_id})"
                )

        count = await self._storage.delete_nodes_by_ids(orphan_ids)
        result.removed_count = count

        if len(orphan_ids) > 10:
            result.details.append(f"... 及其他 {len(orphan_ids) - 10} 个孤立节点")

        logger.info(f"Orphan node cleanup: removed {count} nodes")
        return result

    # ================================================================
    # 低置信度边清理
    # ================================================================

    async def find_low_confidence_stale_edges(
        self,
        confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        staleness_days: int = DEFAULT_STALENESS_DAYS,
    ) -> List[str]:
        """检测低置信度且长期未更新的边

        仅当同时满足以下条件时才标记为待清理：
        1. confidence < confidence_threshold
        2. updated_time 距今超过 staleness_days 天

        Args:
            confidence_threshold: 置信度阈值（低于此值视为低置信度）
            staleness_days: 过期天数（超过此天数未更新视为过期）

        Returns:
            待清理的边 ID 列表
        """
        all_edges = await self._storage.get_all_edges()
        cutoff = datetime.now() - timedelta(days=staleness_days)

        stale_edge_ids: List[str] = []
        for edge in all_edges:
            if edge.confidence < confidence_threshold and edge.updated_time < cutoff:
                stale_edge_ids.append(edge.id)

        return stale_edge_ids

    async def clean_low_confidence_edges(
        self,
        confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        staleness_days: int = DEFAULT_STALENESS_DAYS,
    ) -> CleanupResult:
        """清理低置信度且长期未更新的边

        Args:
            confidence_threshold: 置信度阈值
            staleness_days: 过期天数

        Returns:
            清理结果
        """
        result = CleanupResult(task_name="低置信度边清理")

        stale_ids = await self.find_low_confidence_stale_edges(
            confidence_threshold, staleness_days
        )
        if not stale_ids:
            result.details.append(
                f"未发现低置信度过期边 (threshold={confidence_threshold}, "
                f"staleness={staleness_days}d)"
            )
            return result

        count = await self._storage.delete_edges_by_ids(stale_ids)
        result.removed_count = count
        result.details.append(
            f"移除 {count} 条低置信度过期边 "
            f"(confidence < {confidence_threshold}, "
            f"未更新超过 {staleness_days} 天)"
        )

        logger.info(
            f"Low confidence edge cleanup: removed {count} edges "
            f"(threshold={confidence_threshold}, staleness={staleness_days}d)"
        )
        return result

    # ================================================================
    # 悬空边清理
    # ================================================================

    async def find_dangling_edges(self) -> List[str]:
        """检测悬空边（指向不存在节点的边）

        Returns:
            悬空边 ID 列表
        """
        dangling = await self._storage.get_dangling_edges()
        return [e.id for e in dangling]

    async def remove_dangling_edges(self) -> CleanupResult:
        """移除所有悬空边

        Returns:
            清理结果
        """
        result = CleanupResult(task_name="悬空边清理")

        dangling = await self._storage.get_dangling_edges()
        if not dangling:
            result.details.append("未发现悬空边")
            return result

        node_ids_set = await self._storage.get_node_ids_set()

        for edge in dangling[:10]:
            missing_parts = []
            if edge.source_id not in node_ids_set:
                missing_parts.append(f"源节点 {edge.source_id[:8]}... 不存在")
            if edge.target_id not in node_ids_set:
                missing_parts.append(f"目标节点 {edge.target_id[:8]}... 不存在")
            result.details.append(
                f"悬空边: {edge.relation_type.value} ({', '.join(missing_parts)})"
            )

        edge_ids = [e.id for e in dangling]
        count = await self._storage.delete_edges_by_ids(edge_ids)
        result.removed_count = count

        if len(dangling) > 10:
            result.details.append(f"... 及其他 {len(dangling) - 10} 条悬空边")

        logger.info(f"Dangling edge cleanup: removed {count} edges")
        return result

    # ================================================================
    # 完整清理流程
    # ================================================================

    async def run_full_cleanup(
        self,
        confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        staleness_days: int = DEFAULT_STALENESS_DAYS,
    ) -> MaintenanceReport:
        """执行完整清理流程

        清理顺序：
        1. 悬空边 → 2. 低置信度过期边 → 3. 孤立节点

        先清理边，再清理因边被删而产生的新孤立节点。

        Args:
            confidence_threshold: 低置信度阈值
            staleness_days: 过期天数

        Returns:
            完整维护报告
        """
        report = MaintenanceReport()
        start = time.monotonic()

        try:
            # Step 1: 清理悬空边
            r1 = await self.remove_dangling_edges()
            report.results.append(r1)
            report.total_removed += r1.removed_count

            # Step 2: 清理低置信度过期边
            r2 = await self.clean_low_confidence_edges(
                confidence_threshold, staleness_days
            )
            report.results.append(r2)
            report.total_removed += r2.removed_count

            # Step 3: 清理孤立节点（边清理后可能产生新的孤立节点）
            r3 = await self.remove_orphan_nodes()
            report.results.append(r3)
            report.total_removed += r3.removed_count

        except Exception as e:
            logger.error(f"Full cleanup failed: {e}")
            raise
        finally:
            report.duration_seconds = time.monotonic() - start

        logger.info(
            f"Full cleanup complete: total_removed={report.total_removed}, "
            f"duration={report.duration_seconds:.2f}s"
        )
        return report
