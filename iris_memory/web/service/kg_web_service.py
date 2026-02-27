"""Web 知识图谱管理服务

封装面向 Web 的 KG 节点/边管理和图可视化数据构建。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.knowledge_graph.kg_models import KGEdge, KGNode, KGNodeType
from iris_memory.web.service.audit import audit_log
from iris_memory.web.service.dto.converters import (
    edge_to_graph_dict,
    edge_to_web_dict,
    node_to_graph_dict,
    node_to_web_dict,
)
from iris_memory.utils.logger import get_logger

logger = get_logger("kg_web_service")


class KgWebService:
    """Web 端知识图谱管理服务"""

    def __init__(self, memory_service: Any) -> None:
        self._service = memory_service

    # ================================================================
    # 节点
    # ================================================================

    async def search_kg_nodes(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """搜索知识图谱节点"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            nt = KGNodeType(node_type) if node_type else None

            if query:
                nodes = await kg.storage.search_nodes(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    node_type=nt,
                    limit=limit,
                )
            else:
                nodes = await self._list_all_kg_nodes(
                    user_id=user_id,
                    group_id=group_id,
                    node_type=nt,
                    limit=limit,
                )

            return [node_to_web_dict(n) for n in nodes]

        except Exception as e:
            logger.warning(f"Search KG nodes error: {e}")
            return []

    async def delete_kg_node(self, node_id: str) -> Tuple[bool, str]:
        """删除知识图谱节点及关联边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute(
                        "DELETE FROM kg_edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id),
                    )
                    edge_count = cur.rowcount
                    cur.execute("DELETE FROM kg_nodes WHERE id = ?", (node_id,))
                    node_count = cur.rowcount

                storage._invalidate_cache()

                if node_count > 0:
                    audit_log("delete_kg_node", f"id={node_id} edges_removed={edge_count}")
                    return True, f"已删除节点及 {edge_count} 条关联边"
                return False, "节点不存在"

        except Exception as e:
            logger.error(f"Delete KG node error: {e}")
            return False, f"删除失败: {e}"

    # ================================================================
    # 边
    # ================================================================

    async def list_kg_edges(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        node_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """列出知识图谱边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return []

            storage = kg.storage
            async with storage._lock:
                assert storage._conn

                sql = "SELECT * FROM kg_edges"
                conditions: List[str] = []
                params: List[Any] = []

                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if group_id:
                    conditions.append("(group_id = ? OR group_id IS NULL)")
                    params.append(group_id)
                if relation_type:
                    conditions.append("relation_type = ?")
                    params.append(relation_type)
                if node_id:
                    conditions.append("(source_id = ? OR target_id = ?)")
                    params.extend([node_id, node_id])

                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += f" ORDER BY created_time DESC LIMIT {limit}"

                rows = storage._conn.execute(sql, params).fetchall()
                edges = [KGEdge.from_row(dict(r)) for r in rows]

                # 获取节点名称映射
                node_ids = set()
                for e in edges:
                    node_ids.add(e.source_id)
                    node_ids.add(e.target_id)

                node_names: Dict[str, str] = {}
                if node_ids:
                    placeholders = ",".join(["?"] * len(node_ids))
                    nrows = storage._conn.execute(
                        f"SELECT id, display_name, name FROM kg_nodes WHERE id IN ({placeholders})",
                        list(node_ids),
                    ).fetchall()
                    for nr in nrows:
                        nrd = dict(nr)
                        node_names[nrd["id"]] = nrd.get("display_name") or nrd.get("name") or nrd["id"]

            return [edge_to_web_dict(e, node_names) for e in edges]

        except Exception as e:
            logger.warning(f"List KG edges error: {e}")
            return []

    async def delete_kg_edge(self, edge_id: str) -> Tuple[bool, str]:
        """删除知识图谱边"""
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return False, "知识图谱未启用"

            storage = kg.storage
            async with storage._lock:
                assert storage._conn
                with storage._tx() as cur:
                    cur.execute("DELETE FROM kg_edges WHERE id = ?", (edge_id,))
                    if cur.rowcount > 0:
                        audit_log("delete_kg_edge", f"id={edge_id}")
                        return True, "删除成功"
                    return False, "边不存在"

        except Exception as e:
            logger.error(f"Delete KG edge error: {e}")
            return False, f"删除失败: {e}"

    # ================================================================
    # 图谱可视化
    # ================================================================

    async def get_kg_graph_data(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        center_node_id: Optional[str] = None,
        depth: int = 2,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """获取知识图谱可视化数据

        Returns:
            {nodes: [...], edges: [...]}
        """
        result: Dict[str, Any] = {"nodes": [], "edges": []}

        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return result

            storage = kg.storage

            if center_node_id:
                return await self._get_graph_from_center(storage, center_node_id, depth, max_nodes)

            nodes = await self._list_all_kg_nodes(
                user_id=user_id,
                group_id=group_id,
                limit=max_nodes,
            )
            if not nodes:
                return result

            node_ids = {n.id for n in nodes}
            result["nodes"] = [node_to_graph_dict(n) for n in nodes]

            async with storage._lock:
                assert storage._conn
                if len(node_ids) > 0:
                    placeholders = ",".join(["?"] * len(node_ids))
                    ids_list = list(node_ids)
                    rows = storage._conn.execute(
                        f"""SELECT * FROM kg_edges
                            WHERE source_id IN ({placeholders})
                            OR target_id IN ({placeholders})
                            LIMIT 500""",
                        ids_list + ids_list,
                    ).fetchall()

                    for r in rows:
                        edge = KGEdge.from_row(dict(r))
                        if edge.source_id in node_ids and edge.target_id in node_ids:
                            result["edges"].append(edge_to_graph_dict(edge))

        except Exception as e:
            logger.warning(f"Get KG graph data error: {e}")

        return result

    # ================================================================
    # 内部方法
    # ================================================================

    async def _list_all_kg_nodes(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        node_type: Optional[KGNodeType] = None,
        limit: int = 50,
    ) -> List[KGNode]:
        """列出所有节点（带过滤）"""
        kg = self._service.kg
        if not kg or not kg.enabled:
            return []

        storage = kg.storage
        async with storage._lock:
            assert storage._conn
            sql = "SELECT * FROM kg_nodes"
            conditions: List[str] = []
            params: List[Any] = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if group_id:
                conditions.append("(group_id = ? OR group_id IS NULL)")
                params.append(group_id)
            if node_type:
                conditions.append("node_type = ?")
                params.append(node_type.value)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += f" ORDER BY mention_count DESC LIMIT {limit}"

            rows = storage._conn.execute(sql, params).fetchall()
            return [KGNode.from_row(dict(r)) for r in rows]

    async def _get_graph_from_center(
        self,
        storage: Any,
        center_node_id: str,
        depth: int,
        max_nodes: int,
    ) -> Dict[str, Any]:
        """从中心节点出发获取子图"""
        visited_nodes: Dict[str, KGNode] = {}
        edges_list: List[KGEdge] = []
        to_visit = [center_node_id]

        for _ in range(depth):
            if not to_visit or len(visited_nodes) >= max_nodes:
                break

            next_visit: List[str] = []
            for nid in to_visit:
                if nid in visited_nodes:
                    continue
                node = await storage.get_node(nid)
                if node:
                    visited_nodes[nid] = node
                neighbors = await storage.get_neighbors(nid, limit=20)
                for edge, neighbor in neighbors:
                    edges_list.append(edge)
                    if neighbor.id not in visited_nodes and len(visited_nodes) < max_nodes:
                        next_visit.append(neighbor.id)
                        visited_nodes[neighbor.id] = neighbor

            to_visit = next_visit

        # 去重边
        seen_edges: set[str] = set()
        unique_edges: List[KGEdge] = []
        for e in edges_list:
            if e.id not in seen_edges:
                seen_edges.add(e.id)
                if e.source_id in visited_nodes and e.target_id in visited_nodes:
                    unique_edges.append(e)

        return {
            "nodes": [node_to_graph_dict(n) for n in visited_nodes.values()],
            "edges": [edge_to_graph_dict(e) for e in unique_edges],
        }

    # ================================================================
    # 维护 / 一致性 / 质量
    # ================================================================

    async def run_maintenance(self) -> Dict[str, Any]:
        """执行图谱维护清理

        Returns:
            维护报告字典
        """
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return {"error": "知识图谱未启用"}

            report = await kg.run_maintenance()
            return {
                "total_removed": report.total_removed,
                "duration_seconds": round(report.duration_seconds, 2),
                "summary": report.summary(),
                "results": [
                    {
                        "task": r.task_name,
                        "removed_count": r.removed_count,
                        "details": r.details[:10],
                    }
                    for r in report.results
                ],
            }
        except Exception as e:
            logger.error(f"Run maintenance error: {e}")
            return {"error": f"维护执行失败: {e}"}

    async def check_consistency(self) -> Dict[str, Any]:
        """执行一致性检查

        Returns:
            一致性报告字典
        """
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return {"error": "知识图谱未启用"}

            report = await kg.check_consistency()
            return {
                "is_consistent": report.is_consistent,
                "total_issues": report.total_issues,
                "summary": report.summary(),
                "contradictions": len(report.contradictions),
                "dangling_edges": len(report.dangling_edges),
                "self_references": len(report.self_references),
                "cycles": len(report.cycles),
                "duplicate_relations": len(report.duplicate_relations),
                "details": {
                    "contradictions": [
                        {
                            "edge_a_id": c.edge_a_id,
                            "edge_b_id": c.edge_b_id,
                            "source_id": c.source_id,
                            "target_id": c.target_id,
                            "relation_a": c.relation_a,
                            "relation_b": c.relation_b,
                            "description": c.description,
                        }
                        for c in report.contradictions[:20]
                    ],
                    "dangling_edges": [
                        {
                            "edge_id": d.edge_id,
                            "missing_node_id": d.missing_node_id,
                            "is_source_missing": d.is_source_missing,
                            "description": d.description,
                        }
                        for d in report.dangling_edges[:20]
                    ],
                    "self_references": [
                        {
                            "edge_id": s.edge_id,
                            "node_id": s.node_id,
                            "relation_type": s.relation_type,
                            "description": s.description,
                        }
                        for s in report.self_references[:20]
                    ],
                    "cycles": [
                        {
                            "node_ids": cy.node_ids,
                            "edge_ids": cy.edge_ids,
                            "cycle_length": cy.cycle_length,
                            "description": cy.description,
                        }
                        for cy in report.cycles[:20]
                    ],
                    "duplicate_relations": [
                        {
                            "source_id": dr.source_id,
                            "relation_type": dr.relation_type,
                            "edge_ids": dr.edge_ids,
                            "target_ids": dr.target_ids,
                            "description": dr.description,
                        }
                        for dr in report.duplicate_relations[:20]
                    ],
                },
            }
        except Exception as e:
            logger.error(f"Check consistency error: {e}")
            return {"error": f"一致性检查失败: {e}"}

    async def get_quality_report(self) -> Dict[str, Any]:
        """获取图谱质量报告

        Returns:
            质量报告字典
        """
        try:
            kg = self._service.kg
            if not kg or not kg.enabled:
                return {"error": "知识图谱未启用"}

            report = await kg.generate_quality_report()
            return {
                "total_nodes": report.total_nodes,
                "total_edges": report.total_edges,
                "orphan_node_count": report.orphan_node_count,
                "orphan_node_ratio": round(report.orphan_node_ratio, 4),
                "avg_node_confidence": round(report.avg_node_confidence, 4),
                "avg_edge_confidence": round(report.avg_edge_confidence, 4),
                "avg_edges_per_node": round(report.avg_edges_per_node, 2),
                "low_confidence_stats": {
                    "threshold": report.low_confidence_stats.threshold,
                    "low_confidence_node_count": report.low_confidence_stats.low_confidence_node_count,
                    "low_confidence_node_ratio": round(
                        report.low_confidence_stats.low_confidence_node_ratio, 4
                    ),
                    "low_confidence_edge_count": report.low_confidence_stats.low_confidence_edge_count,
                    "low_confidence_edge_ratio": round(
                        report.low_confidence_stats.low_confidence_edge_ratio, 4
                    ),
                },
                "node_type_distribution": report.node_type_distribution,
                "relation_type_distribution": report.relation_type_distribution,
                "summary": report.summary(),
            }
        except Exception as e:
            logger.error(f"Get quality report error: {e}")
            return {"error": f"质量报告生成失败: {e}"}
